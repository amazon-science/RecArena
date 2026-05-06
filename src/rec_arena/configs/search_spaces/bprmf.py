"""BPRMF hyperparameter search space."""

from typing import Dict, Any


class BPRMFSearchSpace:
    """BPRMF hyperparameter search space for HPO."""
    
    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        """Get search space definition."""
        return {
            'embedding_dim': [32, 64, 128, 256],
            'reg_weight': (1e-5, 1e-1),
            'lr': (1e-4, 1e-1),
            'init_std': (0.01, 0.2),
            'normalize_embeddings': [True, False],
            'use_bias': [True, False],
            'weight_decay': (0.0, 1e-3),
        }
    
    @staticmethod
    def suggest_hyperparameters(trial) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial."""
        return {
            'embedding_dim': trial.suggest_categorical('embedding_dim', [32, 64, 128, 256]),
            'reg_weight': trial.suggest_float('reg_weight', 1e-5, 1e-1, log=True),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'init_std': trial.suggest_float('init_std', 0.01, 0.2),
            'normalize_embeddings': trial.suggest_categorical('normalize_embeddings', [True, False]),
            'use_bias': trial.suggest_categorical('use_bias', [True, False]),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 1e-3, log=True),
        }