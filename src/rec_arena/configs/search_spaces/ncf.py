"""NCF hyperparameter search space."""

from typing import Dict, Any


class NCFSearchSpace:
    """NCF hyperparameter search space for HPO."""
    
    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        """Get search space definition."""
        return {
            'embedding_dim': [32, 64, 128, 256],
            'hidden_dims': [
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128],
                [64, 32, 16],
                [256, 128, 64, 32],
            ],
            'dropout_rate': (0.0, 0.5),
            'lr': (1e-5, 1e-2),
            'activation': ['relu', 'gelu', 'swish', 'tanh'],
            'use_batch_norm': [True, False],
            'init_std': (0.01, 0.2),
            'weight_decay': (0.0, 1e-3),
        }
    
    @staticmethod
    def suggest_hyperparameters(trial) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial."""
        return {
            'embedding_dim': trial.suggest_categorical('embedding_dim', [32, 64, 128, 256]),
            'hidden_dims': trial.suggest_categorical('hidden_dims', [
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128],
                [64, 32, 16],
                [256, 128, 64, 32],
            ]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish', 'tanh']),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'init_std': trial.suggest_float('init_std', 0.01, 0.2),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 1e-3, log=True),
        }