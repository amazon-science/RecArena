"""BERT4Rec hyperparameter search space."""

from typing import Dict, Any


class BERT4RecSearchSpace:
    """BERT4Rec hyperparameter search space for HPO."""

    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        """Get search space definition."""
        return {
            "embedding_dim": [32, 64, 128, 256],
            "num_heads": [1, 2, 4, 8],
            "num_layers": [1, 2, 3, 4],
            "dropout_rate": (0.0, 0.5),
            "mask_prob": (0.1, 0.3),
            "lr": (1e-5, 1e-2),
            "loss_type": ["cross_entropy", "gbce", "bce", "sampled_softmax"],
            "transformer_activation": ["gelu", "relu", "tanh", "silu"],
            "use_ligr": [True, False],
            "layer_norm_first": [True, False],
            "init_std": (0.01, 0.05),
            "weight_decay": (0.0, 1e-3),
            "label_smoothing": (0.0, 0.3),
        }

    @staticmethod
    def suggest_hyperparameters(trial) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial."""
        return {
            "embedding_dim": trial.suggest_categorical(
                "embedding_dim", [32, 64, 128, 256]
            ),
            "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4, 8]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3, 4]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "mask_prob": trial.suggest_float("mask_prob", 0.1, 0.3),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "loss_type": trial.suggest_categorical(
                "loss_type", ["cross_entropy", "gbce", "bce", "sampled_softmax"]
            ),
            "transformer_activation": trial.suggest_categorical(
                "transformer_activation", ["gelu", "relu", "tanh", "silu"]
            ),
            "use_ligr": trial.suggest_categorical("use_ligr", [True, False]),
            "layer_norm_first": trial.suggest_categorical(
                "layer_norm_first", [True, False]
            ),
            "init_std": trial.suggest_float("init_std", 0.01, 0.05),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.3),
        }
