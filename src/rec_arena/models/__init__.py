from .base import BaseModel
from .deep import DeepModel
from .traditional import TraditionalModel
from .sequential import SequentialModel, DeepSequentialModel

# Sequential models
from .sequential_models.sasrec import SASRec
from .sequential_models.gru4rec import GRU4Rec
from .sequential_models.bert4rec import BERT4Rec
from .sequential_models.recm import RecM
from .sequential_models.caser import Caser
from .sequential_models.fmlprec import FMLPRec
from .sequential_models.hstu import HSTU
from .sequential_models.fuxi import FuXi
from .sequential_models.fuxi_gamma import FuXiGamma
from .sequential_models.fuxi_linear import FuXiLinear
from .sequential_models.fuxi_beta import FuXiBeta
from .sequential_models.mlp4rec import MLP4Rec
from .sequential_models.sasrec_pack import SASRecPackModel, train as train_sasrec_pack

# Optional models with heavy/optional dependencies. These are not part of the
# default benchmark and require extra packages (mamba-ssm, torch_geometric,
# etc.), so import failures are tolerated to keep the core package importable.
_OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}

try:
    from .sequential_models.llada4rec import LLaDA4Rec
except Exception as _e:  # noqa: BLE001
    LLaDA4Rec = None
    _OPTIONAL_IMPORT_ERRORS["LLaDA4Rec"] = str(_e)

try:
    from .sequential_models.mamba4rec import Mamba4Rec  # needs mamba-ssm
except Exception as _e:  # noqa: BLE001
    Mamba4Rec = None
    _OPTIONAL_IMPORT_ERRORS["Mamba4Rec"] = str(_e)

# Deep models
from .deep_models.ncf import NCF
from .deep_models.twotower import TwoTower
from .deep_models.simplex import SimpleX

# Traditional models
from .traditional_models.bpr_mf import BPRMF
from .traditional_models.ease import EASE
from .traditional_models.slim import SLIM
from .traditional_models.itemknn import ItemKNN

# Graph models (needs torch_geometric)
try:
    from .graph_models.lightgcn import PyGLightGCN
except Exception as _e:  # noqa: BLE001
    PyGLightGCN = None
    _OPTIONAL_IMPORT_ERRORS["PyGLightGCN"] = str(_e)


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
    "Mamba4Rec",
    "RecM",
    "Caser",
    "FMLPRec",
    "HSTU",
    "FuXi",
    "FuXiGamma",
    "FuXiBeta",
    "FuXiLinear",
    "MLP4Rec",
    "SASRecPackModel",
    "train_sasrec_pack",
    "NCF",
    "TwoTower",
    "SimpleX",
    "BPRMF",
    "EASE",
    "SLIM",
    "ItemKNN",
    "PyGLightGCN",
]
