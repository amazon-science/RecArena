"""SimpleX: A Simple and Strong Baseline for Collaborative Filtering (CIKM 2021).

Faithful reimplementation matching the paper and RecBole's SimpleX. The earlier
RecArena version was NOT SimpleX -- it was cosine-MF trained with BPR, missing
the two defining contributions:
  * the Cosine-Contrastive Loss (CCL): relu(1-pos_cos) + w * mean(relu(neg_cos-m))
  * user-history aggregation: UI_aggregation = g*user + (1-g)*UI_map(agg(history))

This version implements both. Per-user history is provided by the training
harness via ``set_user_history`` (built from the train interactions), mirroring
RecBole's ``dataset.history_item_matrix``. If history is not set the model
degrades to gamma=1 (user embedding only) rather than crashing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deep import DeepModel
from ...configs.defaults.simplex import SimpleXConfig


class SimpleX(DeepModel):
    """SimpleX with Cosine-Contrastive Loss + history aggregation.

    Paper: "SimpleX: A Simple and Strong Baseline for Collaborative Filtering"
    Link: https://arxiv.org/abs/2109.12613

    Aggregators: "mean" (default), "user_attention", "self_attention".
    """

    def __init__(self, config: SimpleXConfig):
        super().__init__(config)
        self.save_hyperparameters()

        self.embedding_dim = config.get("embedding_dim", 64)
        self.margin = config.get("margin", 0.5)
        self.negative_weight = config.get("negative_weight", 10.0)
        self.gamma = config.get("gamma", 0.5)
        self.reg_weight = config.get("reg_weight", 1e-5)
        self.aggregator = config.get("aggregator", "mean")
        if self.aggregator not in ["mean", "user_attention", "self_attention"]:
            raise ValueError(
                "aggregator must be mean, user_attention or self_attention"
            )
        self.dropout = nn.Dropout(config.get("dropout_prob", 0.0))

        # user + item embeddings (item padding_idx=0 so PAD history contributes 0)
        self.user_embedding = nn.Embedding(
            self.config.num_users, self.embedding_dim, sparse=self._sparse_embeddings
        )
        self.item_embedding = nn.Embedding(
            self.config.num_items,
            self.embedding_dim,
            padding_idx=0,
            sparse=self._sparse_embeddings,
        )
        # feature-space mapping applied to the aggregated history
        self.UI_map = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh()
            )
            if self.aggregator == "self_attention":
                self.W_q = nn.Linear(self.embedding_dim, 1, bias=False)

        # Per-user history, set by the harness (see set_user_history). Registered
        # as buffers so they move with .to(device) and are saved with the model.
        self.register_buffer(
            "history_item_id", torch.zeros(self.config.num_users, 1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "history_item_len", torch.zeros(self.config.num_users, dtype=torch.float),
            persistent=False,
        )
        self._has_history = False

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()  # PAD row = 0
        nn.init.xavier_normal_(self.UI_map.weight)

    # ------------------------------------------------------------------ #
    # History wiring (called by the training/eval harness)
    # ------------------------------------------------------------------ #
    def set_user_history(self, history_item_id: torch.Tensor, history_item_len: torch.Tensor):
        """Provide per-user interacted-item history.

        Args:
            history_item_id: [num_users, max_history_len] padded item ids (0=PAD).
            history_item_len: [num_users] number of real history items per user.
        """
        self.history_item_id = history_item_id.long()
        self.history_item_len = history_item_len.float()
        self._has_history = True

    # ------------------------------------------------------------------ #
    # Core SimpleX pieces (mirror RecBole)
    # ------------------------------------------------------------------ #
    def get_UI_aggregation(self, user_e, history_item_e, history_len):
        """Combine user embedding with aggregated history: g*user + (1-g)*UI_map(agg)."""
        if self.aggregator == "mean":
            pos_item_sum = history_item_e.sum(dim=1)
            out = pos_item_sum / (history_len + 1.0e-10).unsqueeze(1)
        else:  # user_attention / self_attention
            key = self.W_k(history_item_e)
            if self.aggregator == "user_attention":
                attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2)
            else:
                attention = self.W_q(key).squeeze(2)
            e_attention = torch.exp(attention)
            mask = (history_item_e.sum(dim=-1) != 0).int()
            e_attention = e_attention * mask
            attention_weight = e_attention / (
                e_attention.sum(dim=1, keepdim=True) + 1.0e-10
            )
            out = torch.matmul(attention_weight.unsqueeze(1), history_item_e).squeeze(1)
        out = self.UI_map(out)
        g = self.gamma
        return g * user_e + (1 - g) * out

    def _user_representation(self, user_ids):
        """Aggregated (pre-normalization) user representation for `user_ids`."""
        user_e = self.user_embedding(user_ids)
        if not self._has_history:
            return user_e  # graceful fallback: gamma=1 (user only)
        history_item = self.history_item_id[user_ids]
        history_len = self.history_item_len[user_ids]
        history_item_e = self.item_embedding(history_item)
        return self.get_UI_aggregation(user_e, history_item_e, history_len)

    @staticmethod
    def get_cos(user_e, item_e):
        """Cosine similarity. user_e:[B,D], item_e:[B,I,D] -> [B,I]."""
        user_e = F.normalize(user_e, dim=1).unsqueeze(2)  # [B,D,1]
        item_e = F.normalize(item_e, dim=2)               # [B,I,D]
        return torch.matmul(item_e, user_e).squeeze(2)     # [B,I]

    # ------------------------------------------------------------------ #
    # Training: Cosine-Contrastive Loss
    # ------------------------------------------------------------------ #
    def compute_loss(self, batch):
        user_ids = batch["user_id"]
        pos_item = batch["item_id"]
        neg_item_seq = batch["neg_items"]  # [B, num_neg]

        # Raw user embedding kept separately for the reg term (RecBole regularizes
        # the raw user_e, not the history-aggregated / dropped-out representation).
        user_e = self.user_embedding(user_ids)              # [B, D]
        user_rep = self.dropout(self._user_representation(user_ids))
        pos_item_e = self.item_embedding(pos_item)          # [B, D]
        neg_item_seq_e = self.item_embedding(neg_item_seq)  # [B, num_neg, D]

        pos_cos = self.get_cos(user_rep, pos_item_e.unsqueeze(1))  # [B,1]
        neg_cos = self.get_cos(user_rep, neg_item_seq_e)           # [B,num_neg]

        # CCL: pull positive toward cos=1, push negatives below `margin`,
        # with negatives down-weighted by `negative_weight`.
        pos_loss = torch.relu(1 - pos_cos)
        neg_loss = torch.relu(neg_cos - self.margin).mean(dim=1, keepdim=True)
        neg_loss = neg_loss * self.negative_weight
        ccl_loss = (pos_loss + neg_loss).mean()

        # L2 embedding regularization, matching RecBole's EmbLoss (require_pow=
        # False): sum of per-group L2 norms normalized by batch size, over the
        # user, positive-item, history-item and negative-item embeddings.
        if self.reg_weight > 0:
            batch_size = user_e.shape[0]
            reg = torch.norm(user_e, p=2) + torch.norm(pos_item_e, p=2) + torch.norm(
                neg_item_seq_e, p=2
            )
            if self._has_history:
                history_item_e = self.item_embedding(self.history_item_id[user_ids])
                reg = reg + torch.norm(history_item_e, p=2)
            reg_loss = reg / batch_size
            return ccl_loss + self.reg_weight * reg_loss
        return ccl_loss

    # ------------------------------------------------------------------ #
    # Scoring / eval
    # ------------------------------------------------------------------ #
    def forward(self, user_ids, item_ids):
        """Cosine score for given user-item pairs (uses history aggregation)."""
        user_rep = self._user_representation(user_ids)
        item_e = self.item_embedding(item_ids)
        return self.get_cos(user_rep, item_e.unsqueeze(1)).squeeze(1)

    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def full_sort_scores(self, user_ids):
        """[B, num_items] cosine scores over all items (history-aware).

        The implicit eval harness calls this when present so SimpleX is scored
        with its true history-aggregated cosine, not a plain user-item dot.
        """
        user_rep = F.normalize(self._user_representation(user_ids), dim=1)
        all_item = F.normalize(self.item_embedding.weight, dim=1)
        return torch.matmul(user_rep, all_item.t())

    def recommend(self, user_ids, k=10):
        scores = self.full_sort_scores(user_ids)
        top_scores, top_items = torch.topk(scores, k, dim=-1)
        return top_items, top_scores
