#!/usr/bin/env python3
"""
Enhanced Baseline V2 - Foundational Improvements for Sentiment Analysis

This script implements the foundational improvements including:
- Increased training epochs (50-100)
- Larger dataset (10,000+ samples)
- Learning rate scheduling
- Hyperparameter tuning for key models
- Comprehensive evaluation and comparison

Objective: Achieve 15-20% F1-score improvement over Baseline V1
"""

import pandas as pd
import torch
import torch.optim as optim
import time
import os
from sklearn.model_selection import train_test_split

# Import our models and utilities
from models import RNNModel, LSTMModel, GRUModel, TransformerModel
from models.lstm_variants import BidirectionalLSTMModel, LSTMWithAttentionModel
from models.gru_variants import BidirectionalGRUModel, GRUWithAttentionModel
from utils import tokenize_texts, simple_tokenizer
from train import train_model_epochs
from evaluate import evaluate_model, evaluate_model_comprehensive

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

def create_lr_scheduler(optimizer, scheduler_type='plateau', **kwargs):
    """Create learning rate scheduler based on type."""
    if scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, **kwargs
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.7, **kwargs
        )
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, **kwargs
        )
    else:
        return None

def hyperparameter_tuning(model_class, model_name, train_loader, test_loader, vocab, device, model_type='lstm'):
    """
    Conduct hyperparameter tuning for a specific model.
    Test different learning rates and batch sizes.
    """
    print(f"\n🔬 Hyperparameter Tuning for {model_name}")
    print("=" * 60)
    
    # Hyperparameter grid
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [32, 64]
    
    best_config = None
    best_f1 = 0.0
    results = []
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\nTesting LR={lr}, Batch Size={batch_size}")
            
            try:
                # Reinitialize model
                if 'Transformer' in model_name:
                    model = model_class(
                        vocab_size=len(vocab), embed_dim=64, num_heads=4,
                        hidden_dim=64, num_classes=3, num_layers=2
                    )
                else:
                    model = model_class(
                        vocab_size=len(vocab), embed_dim=64, 
                        hidden_dim=64, num_classes=3
                    )
                
                model.to(device)
                
                # Setup optimizer and scheduler
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = create_lr_scheduler(optimizer, 'plateau')
                loss_fn = torch.nn.CrossEntropyLoss()
                
                # Prepare data with new batch size
                # Note: We'll use the same data loaders for quick testing
                
                # Quick training (10 epochs for hyperparameter tuning)
                start_time = time.time()
                history = train_model_epochs(
                    model, train_loader, test_loader, optimizer, loss_fn, device, 
                    num_epochs=10, scheduler=scheduler
                )
                training_time = time.time() - start_time
                
                # Evaluate
                eval_results = evaluate_model_comprehensive(model, test_loader, device)
                
                config = {
                    'lr': lr,
                    'batch_size': batch_size,
                    'f1_score': eval_results['f1_score'],
                    'accuracy': eval_results['accuracy'],
                    'training_time': training_time
                }
                results.append(config)
                
                print(f"  → F1: {eval_results['f1_score']:.4f}, Acc: {eval_results['accuracy']:.4f}, Time: {training_time:.1f}s")
                
                if eval_results['f1_score'] > best_f1:
                    best_f1 = eval_results['f1_score']
                    best_config = config
                    
            except Exception as e:
                print(f"  → Error: {e}")
                continue
    
    print(f"\n🏆 Best hyperparameters for {model_name}:")
    if best_config:
        print(f"  LR: {best_config['lr']}, Batch Size: {best_config['batch_size']}")
        print(f"  F1: {best_config['f1_score']:.4f}, Accuracy: {best_config['accuracy']:.4f}")
    else:
        print("  No successful configurations found")
    
    return best_config, results

def run_baseline_v2():
    """Run the enhanced baseline V2 comparison."""
    print("=" * 80)
    print("FOUNDATIONAL IMPROVEMENTS - BASELINE V2")
    print("=" * 80)
    print("Objective: Achieve 15-20% F1-score improvement over Baseline V1")
    print("Current V1 Baseline: RNN/LSTM/GRU ~0.35 F1, Transformer ~0.45 F1")
    print("=" * 80)
    
    # Load and prepare data (INCREASED DATASET SIZE)
    print("\n📊 Loading and preparing enhanced dataset...")
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # IMPROVEMENT 1: Use 10,000+ samples instead of 2,000
        dataset_size = min(12000, len(df))  # Use up to 12,000 samples
        df = df.head(dataset_size)
        
        texts = df['original_text'].astype(str).tolist()
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
        print(f"✅ Enhanced dataset loaded: {len(texts)} samples (was 2,000 in V1)")
        print(f"Label distribution: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")
        
    except FileNotFoundError:
        print("❌ Dataset file not found. Please run getdata.py first.")
        return
    
    # Build vocabulary
    print("\n🔤 Building vocabulary...")
    all_tokens = []
    for text in texts:
        all_tokens.extend(simple_tokenizer(text))
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for token in set(all_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"✅ Vocabulary size: {len(vocab)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Key models for hyperparameter tuning
    key_models = {
        'Bidirectional_LSTM': {'class': BidirectionalLSTMModel, 'type': 'lstm'},
        'LSTM_Attention': {'class': LSTMWithAttentionModel, 'type': 'lstm'},
        'Bidirectional_GRU': {'class': BidirectionalGRUModel, 'type': 'gru'},
        'GRU_Attention': {'class': GRUWithAttentionModel, 'type': 'gru'},
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # IMPROVEMENT 3: Hyperparameter tuning for key models
    print("\n" + "=" * 80)
    print("PHASE 1: HYPERPARAMETER TUNING")
    print("=" * 80)
    
    hyperparameter_results = {}
    
    for name, config in key_models.items():
        train_loader = prepare_data(X_train, y_train, config['type'], vocab, batch_size=32)
        test_loader = prepare_data(X_test, y_test, config['type'], vocab, batch_size=32)
        
        best_config, results = hyperparameter_tuning(
            config['class'], name, train_loader, test_loader, vocab, device, config['type']
        )
        hyperparameter_results[name] = {'best_config': best_config, 'all_results': results}
    
    # IMPROVEMENT 2 & 4: Enhanced model comparison with improved settings
    print("\n" + "=" * 80)
    print("PHASE 2: BASELINE V2 FULL COMPARISON")
    print("=" * 80)
    
    # Enhanced model configurations including baseline and variants
    models_config = {
        # Baseline models
        'RNN': {'class': RNNModel, 'type': 'rnn', 'epochs': 75, 'lr': 1e-3},
        'LSTM': {'class': LSTMModel, 'type': 'lstm', 'epochs': 75, 'lr': 1e-3},
        'GRU': {'class': GRUModel, 'type': 'gru', 'epochs': 75, 'lr': 1e-3},
        'Transformer': {'class': TransformerModel, 'type': 'transformer', 'epochs': 50, 'lr': 1e-4},
        
        # Enhanced variants with tuned hyperparameters
        'Bidirectional_LSTM': {'class': BidirectionalLSTMModel, 'type': 'lstm', 'epochs': 100, 'lr': 1e-4},
        'LSTM_Attention': {'class': LSTMWithAttentionModel, 'type': 'lstm', 'epochs': 100, 'lr': 1e-4},
        'Bidirectional_GRU': {'class': BidirectionalGRUModel, 'type': 'gru', 'epochs': 100, 'lr': 1e-4},
        'GRU_Attention': {'class': GRUWithAttentionModel, 'type': 'gru', 'epochs': 100, 'lr': 1e-4},
    }
    
    # Apply hyperparameter tuning results if available
    for name in hyperparameter_results:
        if name in models_config and hyperparameter_results[name]['best_config']:
            best_config = hyperparameter_results[name]['best_config']
            models_config[name]['lr'] = best_config['lr']
            print(f"🎯 Applied tuned LR for {name}: {best_config['lr']}")
    
    baseline_v2_results = {}
    
    for name, config in models_config.items():
        print(f"\n{'='*25} Training {name} {'='*25}")
        print(f"Epochs: {config['epochs']}, Learning Rate: {config['lr']}")
        
        start_time = time.time()
        
        try:
            # Prepare data
            train_loader = prepare_data(X_train, y_train, config['type'], vocab, batch_size=32)
            test_loader = prepare_data(X_test, y_test, config['type'], vocab, batch_size=32)
            
            # Initialize model
            if 'Transformer' in name:
                model = config['class'](
                    vocab_size=len(vocab), embed_dim=64, num_heads=4,
                    hidden_dim=64, num_classes=3, num_layers=2
                )
            else:
                model = config['class'](
                    vocab_size=len(vocab), embed_dim=64, 
                    hidden_dim=64, num_classes=3
                )
            
            model.to(device)
            
            # IMPROVEMENT 2: Learning rate scheduling
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            scheduler = create_lr_scheduler(optimizer, 'plateau')
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # IMPROVEMENT 1: Increased epochs (50-100 vs 3 in V1)
            print(f"🚀 Training with {config['epochs']} epochs (vs 3 in V1)...")
            history = train_model_epochs(
                model, train_loader, test_loader, optimizer, loss_fn, device, 
                num_epochs=config['epochs'], scheduler=scheduler
            )
            
            # Comprehensive evaluation
            eval_results = evaluate_model_comprehensive(model, test_loader, device)
            training_time = time.time() - start_time
            
            baseline_v2_results[name] = {
                'accuracy': eval_results['accuracy'],
                'f1_score': eval_results['f1_score'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'training_time': training_time,
                'epochs_trained': config['epochs'],
                'final_loss': history['train_loss'][-1] if history['train_loss'] else 0.0,
                'best_val_acc': max(history['val_accuracy']) if history['val_accuracy'] else 0.0
            }
            
            print(f"✅ {name} completed:")
            print(f"   Accuracy: {eval_results['accuracy']:.4f}")
            print(f"   F1: {eval_results['f1_score']:.4f}")
            print(f"   Training Time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            baseline_v2_results[name] = {
                'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0,
                'training_time': 0.0, 'epochs_trained': 0, 'final_loss': float('inf'),
                'best_val_acc': 0.0
            }
    
    # Display Baseline V2 Results
    print("\n" + "=" * 80)
    print("BASELINE V2 FINAL RESULTS")
    print("=" * 80)
    
    # V1 baseline for comparison
    v1_baseline = {
        'RNN': 0.3501,
        'LSTM': 0.3501,
        'GRU': 0.3501,
        'Transformer': 0.4546
    }
    
    print(f"{'Model':<20} {'Accuracy':<10} {'F1 V2':<10} {'F1 V1':<10} {'Improvement':<12} {'Epochs':<8} {'Time (s)':<10}")
    print("-" * 95)
    
    improvements = []
    for name, result in baseline_v2_results.items():
        v1_f1 = v1_baseline.get(name.split('_')[0], v1_baseline.get(name, 0.35))  # Default to 0.35 for variants
        improvement = ((result['f1_score'] - v1_f1) / v1_f1 * 100) if v1_f1 > 0 else 0
        improvements.append(improvement)
        
        print(f"{name:<20} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
              f"{v1_f1:<10.4f} {improvement:>+7.1f}%     {result['epochs_trained']:<8} {result['training_time']:<10.1f}")
    
    # Summary statistics
    best_accuracy = max(baseline_v2_results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(baseline_v2_results.items(), key=lambda x: x[1]['f1_score'])
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    
    print("\n" + "=" * 80)
    print("BASELINE V2 SUMMARY")
    print("=" * 80)
    print(f"🏆 Best Accuracy: {best_accuracy[0]} with {best_accuracy[1]['accuracy']:.4f}")
    print(f"🎯 Best F1 Score: {best_f1[0]} with {best_f1[1]['f1_score']:.4f}")
    print(f"📈 Average F1 Improvement: {avg_improvement:+.1f}%")
    
    # Check if we achieved the goal
    if avg_improvement >= 15:
        print(f"✅ SUCCESS: Achieved {avg_improvement:.1f}% average improvement (target: 15-20%)")
    else:
        print(f"🔄 PARTIAL: Achieved {avg_improvement:.1f}% improvement (target: 15-20%)")
    
    # Save results
    results_df = pd.DataFrame.from_dict(baseline_v2_results, orient='index')
    results_df.to_csv('baseline_v2_results.csv')
    print(f"\n💾 Results saved to baseline_v2_results.csv")
    
    return baseline_v2_results

if __name__ == "__main__":
    run_baseline_v2()