"""Configuration classes for all models."""

from .defaults.sasrec import SASRecConfig
from .defaults.gru4rec import GRU4RecConfig
from .defaults.bert4rec import BERT4RecConfig
from .defaults.mamba4rec import Mamba4RecConfig
from .defaults.recm import RecMConfig
from .defaults.ncf import NCFConfig
from .defaults.bprmf import BPRMFConfig
from .defaults.hstu import HSTUConfig
from .defaults.ease import EASEConfig
from .defaults.slim import SLIMConfig
from .defaults.itemknn import ItemKNNConfig
from .base import BaseModelConfig

__all__ = [
    "BaseModelConfig",
    "SASRecConfig",
    "GRU4RecConfig",
    "BERT4RecConfig",
    "Mamba4RecConfig",
    "RecMConfig",
    "NCFConfig",
    "BPRMFConfig",
    "HSTUConfig",
    "EASEConfig",
    "SLIMConfig",
    "ItemKNNConfig",
]
