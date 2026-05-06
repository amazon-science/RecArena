from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ExperimentConfig:
    """Configuration for a benchmark experiment."""
    name: str
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'model_config': self.model_config,
            'dataset_config': self.dataset_config,
            'training_config': self.training_config,
            'evaluation_config': self.evaluation_config
        }


@dataclass
class ExperimentResult:
    """Results from a benchmark experiment."""
    config: ExperimentConfig
    metrics: Dict[str, float]
    training_time: float
    inference_time: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }


class Experiment:
    """Single benchmark experiment runner."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.result: Optional[ExperimentResult] = None
        
    def run(self, model, dataset, metric_calculator) -> ExperimentResult:
        """Run the experiment and return results."""
        import time
        
        start_time = time.time()
        
        # Training
        train_start = time.time()
        model.fit(dataset.train_dataloader(), dataset.val_dataloader())
        training_time = time.time() - train_start
        
        # Evaluation
        eval_start = time.time()
        metrics = self._evaluate_model(model, dataset, metric_calculator)
        inference_time = time.time() - eval_start
        
        self.result = ExperimentResult(
            config=self.config,
            metrics=metrics,
            training_time=training_time,
            inference_time=inference_time,
            timestamp=datetime.now().isoformat()
        )
        
        return self.result
    
    def _evaluate_model(self, model, dataset, metric_calculator) -> Dict[str, float]:
        """Evaluate model on test set."""
        import torch
        from ..models.sequential import SequentialModel
        
        test_loader = dataset.test_dataloader()
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(model, SequentialModel):
                    sequences = batch['sequence']
                    targets = batch['target']
                    seq_lengths = batch['sequence_length']
                    
                    predictions = model.predict_next(sequences, seq_lengths)
                    all_predictions.append(predictions)
                    all_targets.append(targets)
                else:
                    user_ids = batch['user_id']
                    item_ids = batch['item_id']
                    
                    predictions = model.predict(user_ids, item_ids)
                    targets = batch.get('label', torch.ones_like(predictions))
                    
                    all_predictions.append(predictions)
                    all_targets.append(targets)
        
        if not all_predictions:
            return {}
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return metric_calculator.calculate_batch(all_predictions.unsqueeze(0), all_targets.unsqueeze(0))
    
    def save_result(self, path: str) -> None:
        """Save experiment result to file."""
        if self.result:
            with open(path, 'w') as f:
                json.dump(self.result.to_dict(), f, indent=2)