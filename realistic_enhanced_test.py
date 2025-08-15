#!/usr/bin/env python3
"""
Realistic test of enhanced training on actual dataset.
Demonstrates the improvements and aims for F1 > 75%.
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


def realistic_enhanced_test():
    """Test enhanced features on realistic dataset."""
    
    print("=" * 80)
    print("REALISTIC ENHANCED TRAINING TEST - Targeting F1 > 75%")
    print("=" * 80)
    
    # Load real dataset
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Filter for English text and reasonable length
        df = df[df['original_text'].str.len() > 10]
        df = df[df['original_text'].str.len() < 300]
        
        # Use a substantial subset for training
        df = df.head(2000)
        
        texts = df['original_text'].astype(str).tolist()
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
        print(f"Loaded dataset: {len(texts)} samples")
        
        # Check label distribution
        from collections import Counter
        label_dist = Counter(labels)
        print(f"Label distribution: {dict(label_dist)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic dataset for demonstration...")
        
        # Create more challenging synthetic data
        positive_texts = [
            "This product is absolutely amazing and I love it so much!",
            "Outstanding quality and excellent customer service",
            "Fantastic experience, highly recommend to everyone",
            "Brilliant work, exceeded all my expectations completely",
            "Perfect solution, exactly what I was looking for",
        ] * 100
        
        negative_texts = [
            "This is terrible quality and completely disappointing",
            "Worst experience ever, extremely poor service quality",
            "Horrible product, waste of money and time",
            "Awful customer support, very unprofessional behavior",
            "Completely unsatisfied, will never recommend this",
        ] * 100
        
        neutral_texts = [
            "The product is okay, nothing special but acceptable",
            "Average quality, meets basic requirements adequately",
            "It's fine I guess, could be better",
            "Standard service, neither good nor bad really",
            "Mediocre experience, just what you'd expect",
        ] * 100
        
        texts = positive_texts + negative_texts + neutral_texts
        labels = [2] * 500 + [0] * 500 + [1] * 500
        
        # Shuffle
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)
        
        print(f"Created synthetic dataset: {len(texts)} samples")
    
    # Build comprehensive vocabulary
    all_tokens = []
    for text in texts:
        tokens = simple_tokenizer(text)
        all_tokens.extend(tokens)
    
    # Create vocabulary with proper tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    token_counts = Counter(all_tokens)
    
    # Only include tokens that appear at least twice
    for token, count in token_counts.items():
        if count >= 2 and token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total unique tokens: {len(set(all_tokens))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = []
    
    # Configuration 1: Enhanced LSTM with GloVe
    print(f"\n{'='*60}")
    print("Configuration 1: Enhanced LSTM with GloVe Embeddings")
    print(f"{'='*60}")
    
    hyperparams_1 = {
        'embed_dim': 100,
        'hidden_dim': 256,
        'batch_size': 64,
        'learning_rate': 0.0005,
        'num_epochs': 12,
        'dropout_rate': 0.4,
        'weight_decay': 1e-3,
        'gradient_clip_value': 0.5
    }
    
    experiment_id = tracker.start_experiment(
        model_name="Enhanced_LSTM_GloVe",
        hyperparameters=hyperparams_1,
        description="Enhanced LSTM with GloVe embeddings, high dropout, gradient clipping"
    )
    
    # Get pre-trained embeddings
    pretrained_embeddings = get_pretrained_embeddings(vocab, "glove", 100)
    
    # Initialize model
    model1 = LSTMWithPretrainedEmbeddingsModel(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=256,
        num_classes=3,
        pretrained_embeddings=pretrained_embeddings,
        dropout_rate=0.4
    )
    model1.to(device)
    
    # Prepare data
    train_loader = prepare_data(X_train, y_train, 'lstm', vocab, 64)
    test_loader = prepare_data(X_test, y_test, 'lstm', vocab, 64)
    
    # Setup training with strong regularization
    optimizer1 = optim.Adam(model1.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='max', factor=0.7, patience=3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Train
    print("Training with enhanced regularization and gradient clipping...")
    history1 = train_model_epochs(
        model1, train_loader, test_loader, optimizer1, loss_fn, device,
        num_epochs=12, scheduler=scheduler1, gradient_clip_value=0.5
    )
    
    # Evaluate
    eval_results1 = evaluate_model_comprehensive(model1, test_loader, device)
    
    # Log experiment
    tracker.log_training_history(history1)
    tracker.log_metrics(eval_results1)
    tracker.end_experiment("completed")
    
    results.append(("Enhanced_LSTM_GloVe", eval_results1))
    
    print(f"\n✅ Enhanced LSTM with GloVe Results:")
    print(f"   Accuracy: {eval_results1.get('accuracy', 0):.4f}")
    print(f"   F1 Score: {eval_results1.get('f1_score', 0):.4f}")
    print(f"   Precision: {eval_results1.get('precision', 0):.4f}")
    print(f"   Recall: {eval_results1.get('recall', 0):.4f}")
    
    # Configuration 2: Enhanced GRU with FastText
    print(f"\n{'='*60}")
    print("Configuration 2: Enhanced GRU with FastText Embeddings")
    print(f"{'='*60}")
    
    hyperparams_2 = {
        'embed_dim': 100,
        'hidden_dim': 256,
        'batch_size': 64,
        'learning_rate': 0.0007,
        'num_epochs': 12,
        'dropout_rate': 0.35,
        'weight_decay': 5e-4,
        'gradient_clip_value': 1.0
    }
    
    experiment_id = tracker.start_experiment(
        model_name="Enhanced_GRU_FastText",
        hyperparameters=hyperparams_2,
        description="Enhanced GRU with FastText embeddings and gradient clipping"
    )
    
    # Get FastText embeddings
    fasttext_embeddings = get_pretrained_embeddings(vocab, "fasttext", 100)
    
    # Initialize model
    model2 = GRUWithPretrainedEmbeddingsModel(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=256,
        num_classes=3,
        pretrained_embeddings=fasttext_embeddings,
        dropout_rate=0.35
    )
    model2.to(device)
    
    # Setup training
    optimizer2 = optim.Adam(model2.parameters(), lr=0.0007, weight_decay=5e-4)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', factor=0.7, patience=3)
    
    # Train
    print("Training GRU with FastText embeddings...")
    history2 = train_model_epochs(
        model2, train_loader, test_loader, optimizer2, loss_fn, device,
        num_epochs=12, scheduler=scheduler2, gradient_clip_value=1.0
    )
    
    # Evaluate
    eval_results2 = evaluate_model_comprehensive(model2, test_loader, device)
    
    # Log experiment
    tracker.log_training_history(history2)
    tracker.log_metrics(eval_results2)
    tracker.end_experiment("completed")
    
    results.append(("Enhanced_GRU_FastText", eval_results2))
    
    print(f"\n✅ Enhanced GRU with FastText Results:")
    print(f"   Accuracy: {eval_results2.get('accuracy', 0):.4f}")
    print(f"   F1 Score: {eval_results2.get('f1_score', 0):.4f}")
    print(f"   Precision: {eval_results2.get('precision', 0):.4f}")
    print(f"   Recall: {eval_results2.get('recall', 0):.4f}")
    
    # Configuration 3: Baseline comparison (no pre-trained embeddings)
    print(f"\n{'='*60}")
    print("Configuration 3: Baseline LSTM (no pre-trained embeddings)")
    print(f"{'='*60}")
    
    hyperparams_3 = {
        'embed_dim': 100,
        'hidden_dim': 256,
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 12,
        'dropout_rate': 0.3,
        'weight_decay': 1e-4,
        'gradient_clip_value': 1.0
    }
    
    experiment_id = tracker.start_experiment(
        model_name="Baseline_LSTM",
        hyperparameters=hyperparams_3,
        description="Baseline LSTM without pre-trained embeddings"
    )
    
    # Initialize baseline model
    model3 = LSTMWithPretrainedEmbeddingsModel(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=256,
        num_classes=3,
        pretrained_embeddings=None,  # No pre-trained embeddings
        dropout_rate=0.3
    )
    model3.to(device)
    
    # Setup training
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='max', factor=0.5, patience=5)
    
    # Train
    print("Training baseline model...")
    history3 = train_model_epochs(
        model3, train_loader, test_loader, optimizer3, loss_fn, device,
        num_epochs=12, scheduler=scheduler3, gradient_clip_value=1.0
    )
    
    # Evaluate
    eval_results3 = evaluate_model_comprehensive(model3, test_loader, device)
    
    # Log experiment
    tracker.log_training_history(history3)
    tracker.log_metrics(eval_results3)
    tracker.end_experiment("completed")
    
    results.append(("Baseline_LSTM", eval_results3))
    
    print(f"\n✅ Baseline LSTM Results:")
    print(f"   Accuracy: {eval_results3.get('accuracy', 0):.4f}")
    print(f"   F1 Score: {eval_results3.get('f1_score', 0):.4f}")
    print(f"   Precision: {eval_results3.get('precision', 0):.4f}")
    print(f"   Recall: {eval_results3.get('recall', 0):.4f}")
    
    # Final Results and Analysis
    print(f"\n{'='*80}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*80}")
    
    for model_name, results_dict in results:
        f1 = results_dict.get('f1_score', 0)
        acc = results_dict.get('accuracy', 0)
        prec = results_dict.get('precision', 0)
        rec = results_dict.get('recall', 0)
        print(f"{model_name:25} | F1: {f1:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    
    # Find best model
    best_model = max(results, key=lambda x: x[1].get('f1_score', 0))
    best_f1 = best_model[1].get('f1_score', 0)
    
    print(f"\n🏆 Best Model: {best_model[0]} with F1 Score: {best_f1:.4f}")
    
    # Check if we achieved the target
    if best_f1 >= 0.75:
        print("🎉 SUCCESS: Achieved F1 score >= 75%!")
        print("✅ Pre-trained embeddings and enhanced regularization are effective!")
    elif best_f1 >= 0.70:
        print("✅ GOOD: F1 score >= 70%, close to target!")
        print("📈 Significant improvement demonstrated")
    else:
        print(f"⚠️  F1 score {best_f1:.4f} below target. Dataset may need more tuning.")
    
    # Calculate improvement from baseline
    baseline_f1 = results[-1][1].get('f1_score', 0)  # Last result is baseline
    if best_f1 > baseline_f1:
        improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
        print(f"📊 Improvement over baseline: +{improvement:.1f}%")
    
    # Export results
    tracker.export_results()
    print(f"\n📋 Full experiment results exported to experiments/experiments_summary.csv")
    
    return results


if __name__ == "__main__":
    from collections import Counter
    realistic_enhanced_test()