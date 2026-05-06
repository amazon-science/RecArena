from .base import BaseModel
from .deep import DeepModel
from .traditional import TraditionalModel
from .sequential import SequentialModel, DeepSequentialModel

# Sequential models
from .sequential_models.sasrec import SASRec
from .sequential_models.gru4rec import GRU4Rec
from .sequential_models.bert4rec import BERT4Rec
from .sequential_models.llada4rec import LLaDA4Rec
from .sequential_models.recm import RecM
from .sequential_models.caser import Caser
from .sequential_models.fmlprec import FMLPRec
from .sequential_models.hstu import HSTU
from .sequential_models.fuxi import FuXi
from .sequential_models.fuxi_gamma import FuXiGamma
from .sequential_models.mlp4rec import MLP4Rec

# Deep models
from .deep_models.ncf import NCF
from .deep_models.twotower import TwoTower
from .deep_models.simplex import SimpleX

# Traditional models
from .traditional_models.bpr_mf import BPRMF
from .traditional_models.ease import EASE
from .traditional_models.slim import SLIM
from .traditional_models.itemknn import ItemKNN

# Graph models (optional - requires torch_geometric)
try:
    from .graph_models.lightgcn import PyGLightGCN
except ImportError:
    PyGLightGCN = None


# Add get_item_embedding to non-sequential models
BaseModel.get_item_embedding = lambda self, item_ids: None  # Default implementation

__all__ = [
    "BaseModel",
    "DeepModel",
    "TraditionalModel",
    "SequentialModel",
    "DeepSequentialModel",
    "SASRec",
    "GRU4Rec",
    "BERT4Rec",
    "LLaDA4Rec",
    "RecM",
    "Caser",
    "FMLPRec",
    "HSTU",
    "FuXi",
    "FuXiGamma",
    "MLP4Rec",
    "NCF",
    "TwoTower",
    "SimpleX",
    "BPRMF",
    "EASE",
    "SLIM",
    "ItemKNN",
    "PyGLightGCN",
]
