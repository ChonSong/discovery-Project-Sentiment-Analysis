#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Key Models

This script focuses on tuning hyperparameters for Bidirectional LSTM and GRU with Attention models
to find optimal learning rates and batch sizes for the foundational improvements.
"""

import pandas as pd
import torch
import torch.optim as optim
import time
import itertools
from sklearn.model_selection import train_test_split

# Import models and utilities
from models.lstm_variants import BidirectionalLSTMModel, LSTMWithAttentionModel
from models.gru_variants import BidirectionalGRUModel, GRUWithAttentionModel
from utils import tokenize_texts, simple_tokenizer
from train import train_model_epochs
from evaluate import evaluate_model_comprehensive

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
    """Prepare data for training with configurable batch size."""
    input_ids, _ = tokenize_texts(texts, model_type, vocab)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, labels_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def tune_hyperparameters():
    """Run hyperparameter tuning for key models."""
    print("=" * 70)
    print("HYPERPARAMETER TUNING FOR FOUNDATIONAL IMPROVEMENTS")
    print("=" * 70)
    
    # Load and prepare data
    print("Loading dataset...")
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Use subset for faster tuning
        df = df.head(5000)
        
        texts = df['original_text'].astype(str).tolist()
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
        print(f"Dataset loaded: {len(texts)} samples")
        
    except FileNotFoundError:
        print("Dataset file not found. Please run getdata.py first.")
        return
    
    # Build vocabulary
    all_tokens = []
    for text in texts:
        all_tokens.extend(simple_tokenizer(text))
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for token in set(all_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Models to tune
    models_to_tune = {
        'Bidirectional_LSTM': {'class': BidirectionalLSTMModel, 'type': 'lstm'},
        'LSTM_Attention': {'class': LSTMWithAttentionModel, 'type': 'lstm'},
        'Bidirectional_GRU': {'class': BidirectionalGRUModel, 'type': 'gru'},
        'GRU_Attention': {'class': GRUWithAttentionModel, 'type': 'gru'},
    }
    
    # Hyperparameter grid
    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [32, 64]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    all_results = {}
    
    for model_name, model_config in models_to_tune.items():
        print(f"\n{'='*50}")
        print(f"TUNING {model_name}")
        print(f"{'='*50}")
        
        model_results = []
        best_f1 = 0.0
        best_config = None
        
        # Grid search
        for lr, batch_size in itertools.product(learning_rates, batch_sizes):
            print(f"\nTesting LR={lr}, Batch Size={batch_size}")
            
            try:
                # Initialize model
                model = model_config['class'](
                    vocab_size=len(vocab), embed_dim=64, 
                    hidden_dim=64, num_classes=3
                )
                model.to(device)
                
                # Prepare data with current batch size
                train_loader = prepare_data(X_train, y_train, model_config['type'], vocab, batch_size)
                test_loader = prepare_data(X_test, y_test, model_config['type'], vocab, batch_size)
                
                # Setup training
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=2, verbose=False
                )
                loss_fn = torch.nn.CrossEntropyLoss()
                
                # Train for limited epochs for tuning
                start_time = time.time()
                history = train_model_epochs(
                    model, train_loader, test_loader, optimizer, loss_fn, device, 
                    num_epochs=15, scheduler=scheduler
                )
                training_time = time.time() - start_time
                
                # Evaluate
                eval_results = evaluate_model_comprehensive(model, test_loader, device)
                
                result = {
                    'model': model_name,
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'accuracy': eval_results['accuracy'],
                    'f1_score': eval_results['f1_score'],
                    'precision': eval_results['precision'],
                    'recall': eval_results['recall'],
                    'training_time': training_time,
                    'final_val_acc': max(history['val_accuracy']) if history['val_accuracy'] else 0.0
                }
                
                model_results.append(result)
                
                print(f"  Results: F1={eval_results['f1_score']:.4f}, "
                      f"Acc={eval_results['accuracy']:.4f}, Time={training_time:.1f}s")
                
                # Track best configuration
                if eval_results['f1_score'] > best_f1:
                    best_f1 = eval_results['f1_score']
                    best_config = result.copy()
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        all_results[model_name] = {
            'results': model_results,
            'best_config': best_config
        }
        
        # Display best configuration for this model
        if best_config:
            print(f"\n🏆 Best configuration for {model_name}:")
            print(f"  Learning Rate: {best_config['learning_rate']}")
            print(f"  Batch Size: {best_config['batch_size']}")
            print(f"  F1 Score: {best_config['f1_score']:.4f}")
            print(f"  Accuracy: {best_config['accuracy']:.4f}")
        else:
            print(f"\n❌ No successful runs for {model_name}")
    
    # Generate summary report
    print(f"\n{'='*70}")
    print("HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*70}")
    
    print(f"{'Model':<20} {'Best LR':<10} {'Best Batch':<12} {'Best F1':<10} {'Best Acc':<10}")
    print("-" * 70)
    
    for model_name, data in all_results.items():
        if data['best_config']:
            bc = data['best_config']
            print(f"{model_name:<20} {bc['learning_rate']:<10} {bc['batch_size']:<12} "
                  f"{bc['f1_score']:<10.4f} {bc['accuracy']:<10.4f}")
        else:
            print(f"{model_name:<20} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
    
    # Save detailed results
    all_results_flat = []
    for model_name, data in all_results.items():
        all_results_flat.extend(data['results'])
    
    if all_results_flat:
        results_df = pd.DataFrame(all_results_flat)
        results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
        print(f"\n💾 Detailed results saved to hyperparameter_tuning_results.csv")
    
    # Generate recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR BASELINE V2")
    print(f"{'='*70}")
    
    for model_name, data in all_results.items():
        if data['best_config']:
            bc = data['best_config']
            improvement_estimate = (bc['f1_score'] - 0.35) / 0.35 * 100  # Estimate vs V1 baseline
            print(f"\n{model_name}:")
            print(f"  Recommended LR: {bc['learning_rate']}")
            print(f"  Recommended Batch Size: {bc['batch_size']}")
            print(f"  Expected F1: {bc['f1_score']:.4f}")
            print(f"  Estimated improvement over V1: {improvement_estimate:+.1f}%")
    
    return all_results

if __name__ == "__main__":
    tune_hyperparameters()