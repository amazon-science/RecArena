"""SASRec hyperparameter search space."""

from typing import Dict, Any


class SASRecSearchSpace:
    """SASRec hyperparameter search space for HPO."""
    
    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        """Get search space definition.

        NOTE: `loss_type` values MUST stay in sync with the sequential-model
        validator allowlist (see configs/validation.py:
        ConfigValidator.validate_sequential_config -> ["cross_entropy", "bce",
        "bpr", "sampled_softmax", "gbce"]). The old 'contrastive' entry was NOT
        in that allowlist and every sampled config using it would have crashed;
        it has been removed so this dict agrees with suggest_hyperparameters().
        """
        return {
            'embedding_dim': [32, 64, 128, 256],
            'num_heads': [1, 2, 4, 8],
            'num_layers': [1, 2, 3, 4],
            'dropout_rate': (0.0, 0.5),
            'lr': (1e-5, 1e-2),
            'loss_type': ['cross_entropy', 'bpr', 'bce', 'sampled_softmax', 'gbce'],
            'transformer_activation': ['gelu', 'relu', 'tanh', 'silu'],
            'use_ligr': [True, False],
            'layer_norm_first': [True, False],
            'init_std': (0.001, 0.1),
            'weight_decay': (0.0, 1e-3),
        }
    
    @staticmethod
    def suggest_hyperparameters(trial) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial.

        batch_size is included (the confirmed dominant lever for RecArena
        sequential models; it counts USERS per batch). loss_type is restricted
        to the sequential validator's allowlist -- 'contrastive' is NOT valid
        (validation.py) and previously crashed every trial. weight_decay uses a
        non-zero log lower bound (log(0) is invalid for log=True).
        """
        return {
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256]),
            'embedding_dim': trial.suggest_categorical('embedding_dim', [32, 64, 128, 256]),
            'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4, 8]),
            'num_layers': trial.suggest_categorical('num_layers', [1, 2, 3, 4]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'loss_type': trial.suggest_categorical('loss_type', ['cross_entropy', 'bpr', 'bce', 'sampled_softmax', 'gbce']),
            'transformer_activation': trial.suggest_categorical('transformer_activation', ['gelu', 'relu', 'tanh', 'silu']),
            'use_ligr': trial.suggest_categorical('use_ligr', [True, False]),
            'layer_norm_first': trial.suggest_categorical('layer_norm_first', [True, False]),
            'init_std': trial.suggest_float('init_std', 0.001, 0.1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
        }