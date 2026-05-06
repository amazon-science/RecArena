import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .experiment import Experiment, ExperimentConfig, ExperimentResult
from ..metrics import MetricCalculator


class BenchmarkSuite:
    """Main benchmark suite for running multiple experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.experiments: List[Experiment] = []
        self.results: List[ExperimentResult] = []
        self.metric_calculator = MetricCalculator()
        
        if config_path:
            self.load_config(config_path)
    
    def add_experiment(self, config: ExperimentConfig) -> None:
        """Add an experiment to the suite."""
        experiment = Experiment(config)
        self.experiments.append(experiment)
    
    def run_all(self, models: Dict[str, Any], datasets: Dict[str, Any]) -> List[ExperimentResult]:
        """Run all experiments in the suite."""
        self.results = []
        
        for experiment in self.experiments:
            print(f"Running experiment: {experiment.config.name}")
            
            # Get model and dataset instances
            model_name = experiment.config.model_config['name']
            dataset_name = experiment.config.dataset_config['name']
            
            model = models[model_name]
            dataset = datasets[dataset_name]
            
            # Run experiment
            result = experiment.run(model, dataset, self.metric_calculator)
            self.results.append(result)
            
            print(f"Completed: {experiment.config.name}")
        
        return self.results
    
    def run_single(self, experiment_name: str, models: Dict[str, Any], 
                   datasets: Dict[str, Any]) -> Optional[ExperimentResult]:
        """Run a single experiment by name."""
        for experiment in self.experiments:
            if experiment.config.name == experiment_name:
                model_name = experiment.config.model_config['name']
                dataset_name = experiment.config.dataset_config['name']
                
                model = models[model_name]
                dataset = datasets[dataset_name]
                
                result = experiment.run(model, dataset, self.metric_calculator)
                self.results.append(result)
                return result
        
        return None
    
    def get_leaderboard(self, metric: str = 'ndcg@10') -> List[Dict[str, Any]]:
        """Get leaderboard sorted by specified metric."""
        leaderboard = []
        
        for result in self.results:
            if metric in result.metrics:
                leaderboard.append({
                    'name': result.config.name,
                    'score': result.metrics[metric],
                    'all_metrics': result.metrics
                })
        
        return sorted(leaderboard, key=lambda x: x['score'], reverse=True)
    
    def save_results(self, path: str) -> None:
        """Save all results to file."""
        import os
        
        try:
            if '..' in path:
                raise ValueError("Invalid path: path traversal detected")
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            results_data = [result.to_dict() for result in self.results]
            
            with open(path, 'w') as f:
                json.dump(results_data, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save results: {e}")
    
    def load_config(self, config_path: str) -> None:
        """Load experiment configurations from file."""
        import os
        
        try:
            if '..' in config_path or not os.path.exists(config_path):
                raise ValueError("Invalid or non-existent config path")
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for exp_config in config_data['experiments']:
                config = ExperimentConfig(**exp_config)
                self.add_experiment(config)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def export_leaderboard(self, path: str, metric: str = 'ndcg@10') -> None:
        """Export leaderboard to CSV or JSON."""
        import os
        
        try:
            if '..' in path:
                raise ValueError("Invalid path: path traversal detected")
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            leaderboard = self.get_leaderboard(metric)
            
            if path.endswith('.json'):
                with open(path, 'w') as f:
                    json.dump(leaderboard, f, indent=2)
            elif path.endswith('.csv'):
                import pandas as pd
                df = pd.DataFrame(leaderboard)
                df.to_csv(path, index=False)
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")
        except Exception as e:
            raise RuntimeError(f"Failed to export leaderboard: {e}")