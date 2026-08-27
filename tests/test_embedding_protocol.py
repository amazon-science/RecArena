"""Tests for the unified embedding protocol + output-embedding accessor.

Verifies:
  1. nn.Embedding and HierarchicalLoRAEmbedding satisfy the ItemEmbedding protocol.
  2. create_embedding produces protocol-compliant layers.
  3. Every sequential model routes logits through get_output_embeddings(), so a
     swapped embedding (different .weight) actually changes the logits — proving
     no model bypasses the accessor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rec_arena.modules.layer_utils.embedding_factory import (  # noqa: E402
    ItemEmbedding,  # noqa: F401  (kept to assert the protocol is importable)
    create_embedding,
)


# --------------------------------------------------------------------------- #
# 1 & 2. Protocol compliance (structural duck-typing; runtime_checkable
# isinstance is unreliable for protocols with data members, so we check the
# concrete members the contract requires).
# --------------------------------------------------------------------------- #
def _is_item_embedding(emb) -> bool:
    return (
        callable(emb)
        and hasattr(emb, "weight")
        and hasattr(emb, "num_embeddings")
        and hasattr(emb, "embedding_dim")
    )


def test_nn_embedding_satisfies_protocol():
    emb = nn.Embedding(10, 4)
    assert _is_item_embedding(emb)
    assert emb.num_embeddings == 10 and emb.embedding_dim == 4
    assert emb.weight.shape == (10, 4)


def test_create_embedding_standard():
    emb = create_embedding(
        "standard", num_embeddings=12, embedding_dim=8, padding_idx=0
    )
    assert _is_item_embedding(emb)
    out = emb(torch.tensor([1, 2, 3]))
    assert out.shape == (3, 8)


def test_create_embedding_hierarchical_lora_protocol():
    mapping = torch.arange(20) % 5  # 20 items -> 5 parents
    emb = create_embedding(
        "hierarchical_lora",
        num_embeddings=20,
        embedding_dim=8,
        num_parents=5,
        item_to_parent_mapping=mapping,
        lora_rank=4,
    )
    assert _is_item_embedding(emb)
    assert emb.weight.shape == (20, 8)
    assert emb(torch.tensor([0, 7, 19])).shape == (3, 8)


def test_create_embedding_unknown_raises():
    with pytest.raises(ValueError):
        create_embedding("does_not_exist", num_embeddings=5, embedding_dim=4)


# --------------------------------------------------------------------------- #
# 3. Every sequential model uses get_output_embeddings()
# --------------------------------------------------------------------------- #
def _seq_models():
    from rec_arena.models import (
        SASRec,
        BERT4Rec,
        GRU4Rec,
        Caser,
        FMLPRec,
        HSTU,
        FuXi,
        FuXiGamma,
        MLP4Rec,
    )
    from rec_arena.configs.defaults.sasrec import SASRecConfig
    from rec_arena.configs.defaults.bert4rec import BERT4RecConfig
    from rec_arena.configs.defaults.gru4rec import GRU4RecConfig
    from rec_arena.configs.defaults.caser import CaserConfig
    from rec_arena.configs.defaults.fmlprec import FMLPRecConfig
    from rec_arena.configs.defaults.hstu import HSTUConfig
    from rec_arena.configs.defaults.fuxi import FuXiConfig
    from rec_arena.configs.defaults.fuxi_gamma import FuXiGammaConfig
    from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig

    V, S = 30, 12
    common = dict(
        vocab_size=V, max_seq_length=S, embedding_dim=16, loss_type="cross_entropy"
    )
    return [
        ("SASRec", SASRec, SASRecConfig(**common, num_heads=2, num_layers=1)),
        ("BERT4Rec", BERT4Rec, BERT4RecConfig(**common, num_heads=2, num_layers=1)),
        ("GRU4Rec", GRU4Rec, GRU4RecConfig(**common, hidden_size=16, num_layers=1)),
        ("Caser", Caser, CaserConfig(**common, vertical_filter_size=S)),
        ("FMLPRec", FMLPRec, FMLPRecConfig(**common, num_blocks=1)),
        ("HSTU", HSTU, HSTUConfig(**common, num_heads=2, num_layers=1)),
        (
            "FuXi",
            FuXi,
            FuXiConfig(
                **common, num_heads=2, num_layers=1, attention_dim=8, linear_dim=8
            ),
        ),
        (
            "FuXiGamma",
            FuXiGamma,
            FuXiGammaConfig(
                **common, num_heads=2, num_layers=1, attention_dim=8, linear_dim=8
            ),
        ),
        ("MLP4Rec", MLP4Rec, MLP4RecConfig(**common)),
    ]


@pytest.mark.parametrize(
    "name,cls,cfg", _seq_models(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_model_logits_depend_on_output_embeddings(name, cls, cfg):
    """If a model bypassed get_output_embeddings(), zeroing the embedding's
    .weight would not change its logits. We assert it does, proving the
    accessor is on the logit path."""
    torch.manual_seed(0)
    model = cls(cfg).eval()
    B, S = 2, cfg.max_seq_length
    seqs = torch.randint(3, cfg.vocab_size, (B, S))
    lengths = torch.full((B,), S, dtype=torch.long)

    # Directly prove the output projection reads get_output_embeddings():
    # monkeypatch it to return a different table and confirm predictions change.
    with torch.no_grad():
        out1 = model.predict_next(seqs, lengths)

        orig = model.get_output_embeddings()
        perturbed = orig + torch.randn_like(orig)
        model.get_output_embeddings = lambda: perturbed  # type: ignore[assignment]
        out2 = model.predict_next(seqs, lengths)

    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2, atol=1e-5), (
        f"{name} predictions unchanged when get_output_embeddings() returns a "
        f"different table -> the model bypasses the accessor on its output path"
    )


def test_get_output_embeddings_shape():
    from rec_arena.models import SASRec
    from rec_arena.configs.defaults.sasrec import SASRecConfig

    cfg = SASRecConfig(
        vocab_size=30,
        max_seq_length=12,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        loss_type="cross_entropy",
    )
    model = SASRec(cfg)
    w = model.get_output_embeddings()
    assert w.shape == (30, 16)
