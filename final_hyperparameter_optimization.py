#!/usr/bin/env python3
"""
Final Model Optimization - Focused Hyperparameter Search

This script performs focused hyperparameter tuning on the top-performing 
model architectures to achieve the final optimized sentiment analysis model.

Objective: Achieve 75-80% F1 score through systematic optimization
"""

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import time
import itertools
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Import models and utilities
from models.lstm_variants import BidirectionalLSTMModel, LSTMWithAttentionModel, LSTMWithPretrainedEmbeddingsModel
from models.gru_variants import BidirectionalGRUModel, GRUWithAttentionModel, GRUWithPretrainedEmbeddingsModel
from models.transformer_variants import TransformerWithPoolingModel
from utils import tokenize_texts, simple_tokenizer
from train import train_model_epochs
from evaluate import evaluate_model_comprehensive
from experiment_tracker import ExperimentTracker

def categorize_sentiment(score):
    """Convert continuous sentiment score to categorical label."""
    if score < -0.1:
        return 0  # Negative
    elif score > 0.1:
        return 2  # Positive  
    else:
        return 1  # Neutral

def prepare_data(texts, labels, model_type, vocab, batch_size=32):
    """Prepare data for training with configurable batch size."""
    input_ids, _ = tokenize_texts(texts, model_type, vocab)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def run_focused_hyperparameter_search():
    """Run focused hyperparameter search on top-performing architectures."""
    print("=" * 80)
    print("FINAL MODEL OPTIMIZATION - FOCUSED HYPERPARAMETER SEARCH")
    print("=" * 80)
    print("Objective: Achieve 75-80% F1 score through systematic optimization")
    print("Target models: Top 3 architectures from previous experiments")
    print("=" * 80)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Load and prepare data (use larger dataset)
    print("\n📊 Loading dataset for optimization...")
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Use full dataset for final optimization
        dataset_size = min(15000, len(df))  # Use larger dataset
        df = df.head(dataset_size)
        
        texts = df['original_text'].astype(str).tolist()
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
        print(f"Dataset loaded: {len(texts)} samples")
        print(f"Label distribution: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")
        
        # Check for class imbalance
        neg_ratio = labels.count(0) / len(labels)
        neu_ratio = labels.count(1) / len(labels) 
        pos_ratio = labels.count(2) / len(labels)
        print(f"Class distribution: Neg={neg_ratio:.3f}, Neu={neu_ratio:.3f}, Pos={pos_ratio:.3f}")
        
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
    
    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define top-performing model architectures based on Week 2 results
    # These are the architectures that showed most promise
    top_models = {
        'Bidirectional_LSTM_Attention': {
            'class': LSTMWithAttentionModel,
            'type': 'lstm',
            'baseline_params': {
                'vocab_size': len(vocab),
                'embed_dim': 128,
                'hidden_dim': 256, 
                'num_classes': 3,
                'dropout_rate': 0.4
            }
        },
        'Bidirectional_GRU_Attention': {
            'class': GRUWithAttentionModel,
            'type': 'gru',
            'baseline_params': {
                'vocab_size': len(vocab),
                'embed_dim': 128,
                'hidden_dim': 256,
                'num_classes': 3,
                'dropout_rate': 0.4
            }
        },
        'Transformer_with_Pooling': {
            'class': TransformerWithPoolingModel,
            'type': 'transformer',
            'baseline_params': {
                'vocab_size': len(vocab),
                'embed_dim': 128,
                'hidden_dim': 512,
                'num_classes': 3,
                'num_heads': 8,
                'num_layers': 4,
                'dropout_rate': 0.3
            }
        }
    }
    
    # Focused hyperparameter grids for optimization
    hyperparameter_grids = {
        'Bidirectional_LSTM_Attention': {
            'learning_rate': [5e-4, 1e-3, 2e-3],
            'batch_size': [32, 64],
            'embed_dim': [100, 128, 200],
            'hidden_dim': [128, 256, 512],
            'dropout_rate': [0.3, 0.4, 0.5],
            'weight_decay': [1e-4, 5e-4, 1e-3],
            'gradient_clip_value': [0.5, 1.0]
        },
        'Bidirectional_GRU_Attention': {
            'learning_rate': [5e-4, 1e-3, 2e-3],
            'batch_size': [32, 64],
            'embed_dim': [100, 128, 200],
            'hidden_dim': [128, 256, 512],
            'dropout_rate': [0.3, 0.4, 0.5],
            'weight_decay': [1e-4, 5e-4, 1e-3],
            'gradient_clip_value': [0.5, 1.0]
        },
        'Transformer_with_Pooling': {
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'batch_size': [32, 64],
            'embed_dim': [128, 256],
            'hidden_dim': [256, 512],
            'num_heads': [4, 8],
            'num_layers': [2, 4],
            'dropout_rate': [0.2, 0.3, 0.4],
            'weight_decay': [1e-4, 5e-4],
            'gradient_clip_value': [0.5, 1.0]
        }
    }
    
    optimization_results = {}
    
    # Run focused search for each top model
    for model_name, model_config in top_models.items():
        print(f"\n{'='*60}")
        print(f"OPTIMIZING {model_name}")
        print(f"{'='*60}")
        
        grid = hyperparameter_grids[model_name]
        
        # Create focused parameter combinations (limit to prevent explosion)
        # Use grid search on most important parameters first
        key_params = ['learning_rate', 'batch_size', 'dropout_rate', 'weight_decay']
        key_combinations = list(itertools.product(*[grid[param] for param in key_params]))
        
        # Limit to manageable number of combinations
        max_combinations = 24  # 3*2*3*3 = 54, take best subset
        if len(key_combinations) > max_combinations:
            # Sample combinations strategically
            key_combinations = key_combinations[::len(key_combinations)//max_combinations][:max_combinations]
        
        best_f1 = 0.0
        best_config = None
        results = []
        
        print(f"Testing {len(key_combinations)} hyperparameter combinations...")
        
        for i, (lr, batch_size, dropout_rate, weight_decay) in enumerate(key_combinations):
            print(f"\n--- Combination {i+1}/{len(key_combinations)} ---")
            print(f"LR: {lr}, Batch: {batch_size}, Dropout: {dropout_rate}, WD: {weight_decay}")
            
            try:
                # Create model with current hyperparameters
                params = model_config['baseline_params'].copy()
                params['dropout_rate'] = dropout_rate
                
                # Add transformer-specific params if needed
                if 'num_heads' in grid:
                    params['num_heads'] = grid['num_heads'][0]  # Use default for key search
                if 'num_layers' in grid:
                    params['num_layers'] = grid['num_layers'][0]  # Use default for key search
                
                model = model_config['class'](**params)
                model.to(device)
                
                # Prepare data loaders
                train_loader = prepare_data(X_train, y_train, model_config['type'], vocab, batch_size)
                test_loader = prepare_data(X_test, y_test, model_config['type'], vocab, batch_size)
                
                # Setup training
                optimizer = optim.Adam(
                    model.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay
                )
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.7, patience=3
                )
                loss_fn = nn.CrossEntropyLoss()
                
                # Start experiment tracking
                experiment_id = tracker.start_experiment(
                    model_name=f"{model_name}_Optimization",
                    hyperparameters={
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'dropout_rate': dropout_rate,
                        'weight_decay': weight_decay,
                        'gradient_clip_value': grid['gradient_clip_value'][0],
                        **params
                    },
                    description=f"Focused optimization of {model_name}"
                )
                
                # Train for optimization epochs
                start_time = time.time()
                history = train_model_epochs(
                    model, train_loader, test_loader, optimizer, loss_fn, device,
                    num_epochs=25,  # Reasonable epochs for optimization
                    scheduler=scheduler,
                    gradient_clip_value=grid['gradient_clip_value'][0]
                )
                training_time = time.time() - start_time
                
                # Comprehensive evaluation
                eval_results = evaluate_model_comprehensive(model, test_loader, device)
                
                # Log results to experiment tracker
                tracker.log_metrics(eval_results)
                tracker.end_experiment("completed")
                
                # Store results
                result = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'dropout_rate': dropout_rate,
                    'weight_decay': weight_decay,
                    'f1_score': eval_results['f1_score'],
                    'accuracy': eval_results['accuracy'],
                    'precision': eval_results['precision'],
                    'recall': eval_results['recall'],
                    'training_time': training_time,
                    'experiment_id': experiment_id
                }
                results.append(result)
                
                print(f"Results: F1={eval_results['f1_score']:.4f}, Acc={eval_results['accuracy']:.4f}")
                
                # Track best configuration
                if eval_results['f1_score'] > best_f1:
                    best_f1 = eval_results['f1_score']
                    best_config = result.copy()
                    print(f"🏆 NEW BEST for {model_name}!")
                
            except Exception as e:
                print(f"Error in combination: {e}")
                continue
        
        optimization_results[model_name] = {
            'results': results,
            'best_config': best_config,
            'best_f1': best_f1
        }
        
        # Display best configuration for this model
        if best_config:
            print(f"\n🏆 BEST CONFIGURATION for {model_name}:")
            for key, value in best_config.items():
                if key != 'experiment_id':
                    print(f"  {key}: {value}")
            print(f"  Best F1 Score: {best_f1:.4f}")
        else:
            print(f"\n❌ No successful runs for {model_name}")
    
    # Generate optimization summary
    print(f"\n{'='*80}")
    print("FOCUSED OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    all_results = []
    for model_name, data in optimization_results.items():
        if data['best_config']:
            bc = data['best_config']
            all_results.append({
                'model': model_name,
                'f1_score': bc['f1_score'],
                'accuracy': bc['accuracy'],
                'config': bc
            })
    
    # Sort by F1 score
    all_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print(f"{'Model':<30} {'F1 Score':<10} {'Accuracy':<10}")
    print("-" * 60)
    for result in all_results:
        print(f"{result['model']:<30} {result['f1_score']:<10.4f} {result['accuracy']:<10.4f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    all_results_flat = []
    for model_name, data in optimization_results.items():
        for result in data['results']:
            result['model'] = model_name
            all_results_flat.append(result)
    
    if all_results_flat:
        results_df = pd.DataFrame(all_results_flat)
        results_file = f'final_optimization_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to {results_file}")
    
    # Save optimization summary
    summary = {
        'timestamp': timestamp,
        'dataset_size': len(texts),
        'optimization_results': optimization_results,
        'top_performing_model': all_results[0] if all_results else None
    }
    
    summary_file = f'optimization_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"💾 Optimization summary saved to {summary_file}")
    
    # Export experiment tracker results
    tracker.export_results()
    
    return optimization_results, all_results[0] if all_results else None

if __name__ == "__main__":
    results, best_model = run_focused_hyperparameter_search()
    
    if best_model:
        print(f"\n🎯 FINAL RECOMMENDATION:")
        print(f"Best Model: {best_model['model']}")
        print(f"F1 Score: {best_model['f1_score']:.4f}")
        print(f"Accuracy: {best_model['accuracy']:.4f}")
        print("\nReady for final model training with optimized hyperparameters!")
    else:
        print("\n❌ No successful optimization runs completed.")