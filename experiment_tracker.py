#!/usr/bin/env python3
"""
Experiment tracking system for systematic documentation of model runs.
Tracks hyperparameters, metrics, and results for comparison.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd


class ExperimentTracker:
    """Track experiments with hyperparameters and results."""
    
    def __init__(self, experiment_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiment results
        """
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.current_experiment = None
        self.experiments_log = os.path.join(experiment_dir, "experiments.json")
        
        # Load existing experiments
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> List[Dict]:
        """Load existing experiments from file."""
        if os.path.exists(self.experiments_log):
            try:
                with open(self.experiments_log, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def _save_experiments(self):
        """Save experiments to file."""
        with open(self.experiments_log, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def start_experiment(self, 
                        model_name: str,
                        hyperparameters: Dict[str, Any],
                        description: str = "") -> str:
        """
        Start a new experiment.
        
        Args:
            model_name: Name of the model being tested
            hyperparameters: Dictionary of hyperparameters
            description: Optional description of the experiment
            
        Returns:
            Experiment ID
        """
        experiment_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_experiment = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "description": description,
            "hyperparameters": hyperparameters,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration": None,
            "metrics": {},
            "training_history": {},
            "status": "running"
        }
        
        print(f"Starting experiment: {experiment_id}")
        print(f"Model: {model_name}")
        print(f"Hyperparameters: {json.dumps(hyperparameters, indent=2, default=str)}")
        
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log evaluation metrics for the current experiment.
        
        Args:
            metrics: Dictionary of metrics (accuracy, f1_score, precision, recall, etc.)
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["metrics"].update(metrics)
        print(f"Logged metrics: {metrics}")
    
    def log_training_history(self, history: Dict[str, List]):
        """
        Log training history for the current experiment.
        
        Args:
            history: Dictionary with training history (train_loss, val_accuracy, etc.)
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["training_history"] = history
        print(f"Logged training history with {len(history)} metrics")
    
    def end_experiment(self, status: str = "completed"):
        """
        End the current experiment.
        
        Args:
            status: Final status of the experiment (completed, failed, interrupted)
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment to end.")
        
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.current_experiment["start_time"])
        duration = (end_time - start_time).total_seconds()
        
        self.current_experiment["end_time"] = end_time.isoformat()
        self.current_experiment["duration"] = duration
        self.current_experiment["status"] = status
        
        # Add to experiments list
        self.experiments.append(self.current_experiment.copy())
        self._save_experiments()
        
        print(f"Experiment {self.current_experiment['experiment_id']} ended.")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Status: {status}")
        
        self.current_experiment = None
    
    def get_best_experiments(self, metric: str = "f1_score", top_k: int = 5) -> List[Dict]:
        """
        Get the best experiments by a specific metric.
        
        Args:
            metric: Metric to sort by
            top_k: Number of top experiments to return
            
        Returns:
            List of best experiments
        """
        # Filter experiments that have the specified metric
        valid_experiments = [exp for exp in self.experiments 
                           if metric in exp.get("metrics", {})]
        
        # Sort by metric (descending)
        valid_experiments.sort(key=lambda x: x["metrics"][metric], reverse=True)
        
        return valid_experiments[:top_k]
    
    def get_experiments_summary(self) -> pd.DataFrame:
        """
        Get a summary of all experiments as a DataFrame.
        
        Returns:
            DataFrame with experiment summaries
        """
        if not self.experiments:
            return pd.DataFrame()
        
        summary_data = []
        for exp in self.experiments:
            row = {
                "experiment_id": exp["experiment_id"],
                "model_name": exp["model_name"],
                "status": exp["status"],
                "duration": exp.get("duration", 0),
                "start_time": exp["start_time"]
            }
            
            # Add hyperparameters
            for key, value in exp.get("hyperparameters", {}).items():
                row[f"hp_{key}"] = value
            
            # Add metrics
            for key, value in exp.get("metrics", {}).items():
                row[f"metric_{key}"] = value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def export_results(self, filename: str = None):
        """
        Export experiment results to CSV.
        
        Args:
            filename: Output filename (optional)
        """
        if filename is None:
            filename = os.path.join(self.experiment_dir, "experiments_summary.csv")
        
        df = self.get_experiments_summary()
        df.to_csv(filename, index=False)
        print(f"Exported {len(df)} experiments to {filename}")
    
    def compare_models(self, model_names: List[str], metric: str = "f1_score"):
        """
        Compare different models by their best performance on a metric.
        
        Args:
            model_names: List of model names to compare
            metric: Metric to compare by
        """
        print(f"\n=== Model Comparison by {metric} ===")
        
        for model_name in model_names:
            model_experiments = [exp for exp in self.experiments 
                               if exp["model_name"] == model_name and 
                               metric in exp.get("metrics", {})]
            
            if model_experiments:
                best_exp = max(model_experiments, key=lambda x: x["metrics"][metric])
                best_score = best_exp["metrics"][metric]
                print(f"{model_name}: {best_score:.4f} (Experiment: {best_exp['experiment_id']})")
            else:
                print(f"{model_name}: No experiments with {metric}")


def create_enhanced_training_script():
    """Create an enhanced training script that uses experiment tracking."""
    
    script_content = '''#!/usr/bin/env python3
"""
Enhanced training script with pre-trained embeddings, improved regularization,
gradient clipping, and experiment tracking.
"""

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd

from models.lstm_variants import LSTMWithPretrainedEmbeddingsModel
from models.gru_variants import GRUWithPretrainedEmbeddingsModel
from embedding_utils import get_pretrained_embeddings
from experiment_tracker import ExperimentTracker
from train import train_model_epochs
from evaluate import evaluate_model_comprehensive
from utils import tokenize_texts, simple_tokenizer


def enhanced_training_experiment(
    model_class,
    model_name: str,
    hyperparameters: dict,
    texts: list,
    labels: list,
    vocab: dict,
    use_pretrained_embeddings: bool = True,
    embedding_type: str = "glove"
):
    """Run a complete training experiment with tracking."""
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Start experiment
    experiment_id = tracker.start_experiment(
        model_name=model_name,
        hyperparameters=hyperparameters,
        description=f"Enhanced training with {embedding_type} embeddings" if use_pretrained_embeddings else "Enhanced training without pre-trained embeddings"
    )
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Get pre-trained embeddings if requested
        pretrained_embeddings = None
        if use_pretrained_embeddings:
            pretrained_embeddings = get_pretrained_embeddings(
                vocab, embedding_type, hyperparameters['embed_dim']
            )
        
        # Initialize model
        model = model_class(
            vocab_size=len(vocab),
            embed_dim=hyperparameters['embed_dim'],
            hidden_dim=hyperparameters['hidden_dim'],
            num_classes=3,
            pretrained_embeddings=pretrained_embeddings,
            dropout_rate=hyperparameters.get('dropout_rate', 0.3)
        )
        model.to(device)
        
        # Prepare data loaders
        train_loader = prepare_data(X_train, y_train, 'lstm', vocab, hyperparameters['batch_size'])
        test_loader = prepare_data(X_test, y_test, 'lstm', vocab, hyperparameters['batch_size'])
        
        # Setup optimizer with L2 regularization (weight decay)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparameters['learning_rate'],
            weight_decay=hyperparameters.get('weight_decay', 1e-4)
        )
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Train model with enhanced features
        history = train_model_epochs(
            model, train_loader, test_loader, optimizer, loss_fn, device,
            num_epochs=hyperparameters.get('num_epochs', 20),
            scheduler=scheduler,
            gradient_clip_value=hyperparameters.get('gradient_clip_value', 1.0)
        )
        
        # Evaluate model
        eval_results = evaluate_model_comprehensive(model, test_loader, device)
        
        # Log results
        tracker.log_training_history(history)
        tracker.log_metrics(eval_results)
        
        # End experiment
        tracker.end_experiment("completed")
        
        return experiment_id, eval_results
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        tracker.end_experiment("failed")
        raise


def prepare_data(texts, labels, model_type, vocab, batch_size=32):
    """Prepare data for training."""
    input_ids, _ = tokenize_texts(texts, model_type, vocab)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, labels_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # This would be the main enhanced training script
    print("Enhanced training script created!")
'''
    
    with open("enhanced_training.py", 'w') as f:
        f.write(script_content)
    
    print("Created enhanced_training.py")


if __name__ == "__main__":
    # Demonstrate experiment tracking
    tracker = ExperimentTracker()
    
    # Example experiment
    experiment_id = tracker.start_experiment(
        model_name="LSTM_with_GloVe",
        hyperparameters={
            "learning_rate": 0.001,
            "batch_size": 32,
            "embed_dim": 100,
            "hidden_dim": 128,
            "dropout_rate": 0.3,
            "weight_decay": 1e-4,
            "gradient_clip_value": 1.0
        },
        description="LSTM with GloVe embeddings and enhanced regularization"
    )
    
    # Simulate logging metrics
    tracker.log_metrics({
        "accuracy": 0.76,
        "f1_score": 0.78,
        "precision": 0.75,
        "recall": 0.81
    })
    
    tracker.end_experiment("completed")
    
    # Export results
    tracker.export_results()
    print("Experiment tracking demonstration complete!")