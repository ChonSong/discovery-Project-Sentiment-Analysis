#!/usr/bin/env python3
"""
Quick test of enhanced training with pre-trained embeddings.
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


def quick_test():
    """Quick test of enhanced features."""
    
    print("=" * 60)
    print("QUICK TEST: Enhanced Training with Pre-trained Embeddings")
    print("=" * 60)
    
    # Create test data
    texts = [
        "I love this product! It's amazing and fantastic!",
        "This is terrible and awful, worst experience ever",
        "It's okay I guess, nothing special but not bad",
        "Excellent quality and great value for money",
        "Poor quality, very disappointed with purchase",
        "Amazing service and wonderful staff",
        "Horrible experience, will never buy again",
        "Good product but could be better",
        "Outstanding quality, highly recommend",
        "Bad service, not satisfied at all"
    ] * 50  # 500 samples total
    
    labels = [2, 0, 1, 2, 0, 2, 0, 1, 2, 0] * 50
    
    # Build vocabulary
    all_tokens = []
    for text in texts:
        tokens = simple_tokenizer(text)
        all_tokens.extend(tokens)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in set(all_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"Dataset: {len(texts)} samples")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Test 1: LSTM with GloVe embeddings
    print(f"\n{'='*40}")
    print("Test 1: LSTM with GloVe embeddings")
    print(f"{'='*40}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    experiment_id = tracker.start_experiment(
        model_name="LSTM_GloVe_Test",
        hyperparameters={
            'embed_dim': 50,
            'hidden_dim': 64,
            'batch_size': 16,
            'learning_rate': 0.001,
            'num_epochs': 8,
            'dropout_rate': 0.3,
            'weight_decay': 1e-4,
            'gradient_clip_value': 1.0
        },
        description="Quick test with GloVe embeddings"
    )
    
    # Get pre-trained embeddings
    pretrained_embeddings = get_pretrained_embeddings(vocab, "glove", 50)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMWithPretrainedEmbeddingsModel(
        vocab_size=len(vocab),
        embed_dim=50,
        hidden_dim=64,
        num_classes=3,
        pretrained_embeddings=pretrained_embeddings,
        dropout_rate=0.3
    )
    model.to(device)
    
    # Prepare data
    train_loader = prepare_data(X_train, y_train, 'lstm', vocab, 16)
    test_loader = prepare_data(X_test, y_test, 'lstm', vocab, 16)
    
    # Setup training with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Train with enhanced features
    print("Training with gradient clipping and enhanced regularization...")
    history = train_model_epochs(
        model, train_loader, test_loader, optimizer, loss_fn, device,
        num_epochs=8, scheduler=scheduler, gradient_clip_value=1.0
    )
    
    # Evaluate
    eval_results = evaluate_model_comprehensive(model, test_loader, device)
    
    # Log experiment
    tracker.log_training_history(history)
    tracker.log_metrics(eval_results)
    tracker.end_experiment("completed")
    
    print(f"\n✅ Results:")
    print(f"   Accuracy: {eval_results.get('accuracy', 0):.4f}")
    print(f"   F1 Score: {eval_results.get('f1_score', 0):.4f}")
    print(f"   Precision: {eval_results.get('precision', 0):.4f}")
    print(f"   Recall: {eval_results.get('recall', 0):.4f}")
    
    # Test 2: GRU without pre-trained embeddings for comparison
    print(f"\n{'='*40}")
    print("Test 2: GRU without pre-trained embeddings")
    print(f"{'='*40}")
    
    experiment_id2 = tracker.start_experiment(
        model_name="GRU_Baseline_Test",
        hyperparameters={
            'embed_dim': 50,
            'hidden_dim': 64,
            'batch_size': 16,
            'learning_rate': 0.001,
            'num_epochs': 8,
            'dropout_rate': 0.3,
            'weight_decay': 1e-4,
            'gradient_clip_value': 1.0
        },
        description="Quick test without pre-trained embeddings"
    )
    
    # Initialize model without pre-trained embeddings
    model2 = GRUWithPretrainedEmbeddingsModel(
        vocab_size=len(vocab),
        embed_dim=50,
        hidden_dim=64,
        num_classes=3,
        pretrained_embeddings=None,  # No pre-trained embeddings
        dropout_rate=0.3
    )
    model2.to(device)
    
    # Setup training
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', factor=0.5, patience=2)
    
    # Train
    print("Training without pre-trained embeddings...")
    history2 = train_model_epochs(
        model2, train_loader, test_loader, optimizer2, loss_fn, device,
        num_epochs=8, scheduler=scheduler2, gradient_clip_value=1.0
    )
    
    # Evaluate
    eval_results2 = evaluate_model_comprehensive(model2, test_loader, device)
    
    # Log experiment
    tracker.log_training_history(history2)
    tracker.log_metrics(eval_results2)
    tracker.end_experiment("completed")
    
    print(f"\n✅ Results:")
    print(f"   Accuracy: {eval_results2.get('accuracy', 0):.4f}")
    print(f"   F1 Score: {eval_results2.get('f1_score', 0):.4f}")
    print(f"   Precision: {eval_results2.get('precision', 0):.4f}")
    print(f"   Recall: {eval_results2.get('recall', 0):.4f}")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    f1_improvement = eval_results.get('f1_score', 0) - eval_results2.get('f1_score', 0)
    acc_improvement = eval_results.get('accuracy', 0) - eval_results2.get('accuracy', 0)
    
    print(f"LSTM with GloVe:     F1={eval_results.get('f1_score', 0):.4f}, Acc={eval_results.get('accuracy', 0):.4f}")
    print(f"GRU without GloVe:   F1={eval_results2.get('f1_score', 0):.4f}, Acc={eval_results2.get('accuracy', 0):.4f}")
    print(f"Improvement:         F1={f1_improvement:+.4f}, Acc={acc_improvement:+.4f}")
    
    if f1_improvement > 0:
        print("✅ Pre-trained embeddings show improvement!")
    else:
        print("⚠️  No improvement from pre-trained embeddings in this test")
    
    # Export results
    tracker.export_results()
    print(f"\n📊 Experiment results saved to experiments/experiments_summary.csv")
    
    return eval_results, eval_results2


if __name__ == "__main__":
    quick_test()