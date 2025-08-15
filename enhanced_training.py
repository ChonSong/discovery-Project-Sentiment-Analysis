#!/usr/bin/env python3
"""
Enhanced training script with pre-trained embeddings, improved regularization,
gradient clipping, and experiment tracking.
"""

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from models.lstm_variants import LSTMWithPretrainedEmbeddingsModel
from models.gru_variants import GRUWithPretrainedEmbeddingsModel
from embedding_utils import get_pretrained_embeddings
from experiment_tracker import ExperimentTracker
from train import train_model_epochs
from evaluate import evaluate_model_comprehensive
from utils import tokenize_texts, simple_tokenizer


def categorize_sentiment(score):
    """Convert continuous sentiment score to categorical label."""
    try:
        score = float(score)
        if score < -0.1:
            return 0  # Negative
        elif score > 0.1:
            return 2  # Positive 
        else:
            return 1  # Neutral
    except:
        return 1  # Default to neutral


def prepare_data(texts, labels, model_type, vocab, batch_size=32):
    """Prepare data for training."""
    input_ids, _ = tokenize_texts(texts, model_type, vocab)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, labels_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
        print(f"\nTraining {model_name} with enhanced regularization...")
        print(f"Pre-trained embeddings: {use_pretrained_embeddings}")
        print(f"Gradient clipping: {hyperparameters.get('gradient_clip_value', 1.0)}")
        print(f"Weight decay: {hyperparameters.get('weight_decay', 1e-4)}")
        
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
        
        print(f"\n✅ Experiment {experiment_id} completed!")
        print(f"Final F1 Score: {eval_results.get('f1_score', 'N/A'):.4f}")
        print(f"Final Accuracy: {eval_results.get('accuracy', 'N/A'):.4f}")
        
        return experiment_id, eval_results
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        tracker.end_experiment("failed")
        raise


def run_comprehensive_experiments():
    """Run comprehensive experiments with different configurations."""
    
    print("=" * 80)
    print("COMPREHENSIVE ENHANCED TRAINING EXPERIMENTS")
    print("=" * 80)
    
    # Load dataset
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Use larger subset for better results
        df = df.head(5000)
        
        texts = df['original_text'].astype(str).tolist()
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
        print(f"Dataset: {len(texts)} samples")
        
    except FileNotFoundError:
        print("Dataset file not found. Creating dummy data for testing...")
        texts = [
            "I love this product! It's amazing!",
            "This is terrible and awful",
            "It's okay I guess, nothing special",
            "Fantastic quality and great value",
            "Worst purchase ever, very disappointed"
        ] * 200
        labels = [2, 0, 1, 2, 0] * 200
    
    # Build vocabulary
    all_tokens = []
    for text in texts:
        tokens = simple_tokenizer(text)
        all_tokens.extend(tokens)
    
    vocab = {"<PAD>": 0}
    for token in set(all_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Define hyperparameter configurations
    base_hyperparameters = {
        'embed_dim': 100,
        'hidden_dim': 128,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 15,
        'dropout_rate': 0.3,
        'weight_decay': 1e-4,
        'gradient_clip_value': 1.0
    }
    
    # Enhanced configurations for better performance
    enhanced_hyperparameters = {
        'embed_dim': 150,
        'hidden_dim': 256,
        'batch_size': 64,
        'learning_rate': 0.0005,
        'num_epochs': 25,
        'dropout_rate': 0.4,
        'weight_decay': 5e-4,
        'gradient_clip_value': 0.5
    }
    
    experiments = []
    
    # Experiment 1: LSTM with GloVe embeddings
    print(f"\n{'='*50}")
    print("Experiment 1: LSTM with GloVe embeddings")
    print(f"{'='*50}")
    
    exp_id, results = enhanced_training_experiment(
        model_class=LSTMWithPretrainedEmbeddingsModel,
        model_name="LSTM_with_GloVe",
        hyperparameters=base_hyperparameters,
        texts=texts,
        labels=labels,
        vocab=vocab,
        use_pretrained_embeddings=True,
        embedding_type="glove"
    )
    experiments.append(("LSTM_with_GloVe", results))
    
    # Experiment 2: LSTM without pre-trained embeddings
    print(f"\n{'='*50}")
    print("Experiment 2: LSTM without pre-trained embeddings")
    print(f"{'='*50}")
    
    exp_id, results = enhanced_training_experiment(
        model_class=LSTMWithPretrainedEmbeddingsModel,
        model_name="LSTM_baseline",
        hyperparameters=base_hyperparameters,
        texts=texts,
        labels=labels,
        vocab=vocab,
        use_pretrained_embeddings=False
    )
    experiments.append(("LSTM_baseline", results))
    
    # Experiment 3: GRU with FastText embeddings
    print(f"\n{'='*50}")
    print("Experiment 3: GRU with FastText embeddings")
    print(f"{'='*50}")
    
    exp_id, results = enhanced_training_experiment(
        model_class=GRUWithPretrainedEmbeddingsModel,
        model_name="GRU_with_FastText",
        hyperparameters=base_hyperparameters,
        texts=texts,
        labels=labels,
        vocab=vocab,
        use_pretrained_embeddings=True,
        embedding_type="fasttext"
    )
    experiments.append(("GRU_with_FastText", results))
    
    # Experiment 4: Enhanced LSTM with optimized hyperparameters
    print(f"\n{'='*50}")
    print("Experiment 4: Enhanced LSTM with optimized hyperparameters")
    print(f"{'='*50}")
    
    exp_id, results = enhanced_training_experiment(
        model_class=LSTMWithPretrainedEmbeddingsModel,
        model_name="LSTM_enhanced",
        hyperparameters=enhanced_hyperparameters,
        texts=texts,
        labels=labels,
        vocab=vocab,
        use_pretrained_embeddings=True,
        embedding_type="glove"
    )
    experiments.append(("LSTM_enhanced", results))
    
    # Print final comparison
    print(f"\n{'='*80}")
    print("FINAL EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    for model_name, results in experiments:
        f1_score = results.get('f1_score', 0)
        accuracy = results.get('accuracy', 0)
        print(f"{model_name:25} | F1: {f1_score:.4f} | Accuracy: {accuracy:.4f}")
    
    # Find best model
    best_experiment = max(experiments, key=lambda x: x[1].get('f1_score', 0))
    best_f1 = best_experiment[1].get('f1_score', 0)
    
    print(f"\n🏆 Best model: {best_experiment[0]} with F1 score: {best_f1:.4f}")
    
    if best_f1 > 0.75:
        print("✅ SUCCESS: Achieved F1 score above 75%!")
    else:
        print(f"⚠️  F1 score {best_f1:.4f} is below target of 75%. Consider further tuning.")
    
    # Generate experiment report
    tracker = ExperimentTracker()
    tracker.export_results()
    print(f"\n📊 Full experiment results exported to experiments/experiments_summary.csv")


if __name__ == "__main__":
    run_comprehensive_experiments()