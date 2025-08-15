#!/usr/bin/env python3
"""
Enhanced Model Architecture Comparison for Sentiment Analysis.

This script compares all available model architectures including:
- RNN variants (Vanilla, Deep, Bidirectional, Attention)
- LSTM variants (Single, Stacked, Bidirectional, Attention, Pretrained embeddings)
- GRU variants (Single, Stacked, Bidirectional, Attention, Pretrained embeddings)  
- Transformer variants (Standard, Lightweight, Deep, Pooling)

Provides comprehensive metrics, timing, and visualizations.
"""

import pandas as pd
import torch
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import all models including new variants
from models import (
    # Original models
    RNNModel, LSTMModel, GRUModel, TransformerModel,
    # RNN variants
    DeepRNNModel, BidirectionalRNNModel, RNNWithAttentionModel,
    # LSTM variants
    StackedLSTMModel, BidirectionalLSTMModel, LSTMWithAttentionModel, LSTMWithPretrainedEmbeddingsModel,
    # GRU variants
    StackedGRUModel, BidirectionalGRUModel, GRUWithAttentionModel, GRUWithPretrainedEmbeddingsModel,
    # Transformer variants
    LightweightTransformerModel, DeepTransformerModel, TransformerWithPoolingModel
)

from utils import tokenize_texts, simple_tokenizer
from train import train_model
from evaluate import evaluate_model_comprehensive
from visualize_models import visualize_all_models

def simple_train_model(model, train_loader, device, num_epochs=5):
    """Simple training function for model comparison."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

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

def prepare_data(texts, labels, model_type, vocab):
    """Prepare data for training."""
    input_ids, _ = tokenize_texts(texts, model_type, vocab)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, labels_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def create_performance_visualization(results, save_path="enhanced_model_comparison.png"):
    """Create comprehensive performance visualization."""
    # Prepare data for plotting
    model_names = list(results.keys())
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Model Architecture Comparison', fontsize=16)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results[model][metric] for model in model_names]
        
        bars = ax.bar(range(len(model_names)), values)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Model Architecture')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performance visualization saved to {save_path}")
    return save_path

def create_timing_visualization(results, save_path="enhanced_timing_comparison.png"):
    """Create timing comparison visualization."""
    model_names = list(results.keys())
    training_times = [results[model]['training_time'] for model in model_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(model_names)), training_times)
    plt.title('Training Time Comparison Across Architectures')
    plt.xlabel('Model Architecture')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Timing visualization saved to {save_path}")
    return save_path

def main():
    print("Enhanced Model Architecture Comparison for Sentiment Analysis")
    print("=" * 80)
    
    # Load and prepare data
    print("Loading data...")
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Use a subset for faster comparison (increase for production)
        df = df.head(1500)
        
        texts = df['original_text'].astype(str).tolist()
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
        print(f"Loaded {len(texts)} samples")
        print(f"Label distribution: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")
        
    except FileNotFoundError:
        print("Dataset file not found. Please run getdata.py first.")
        return
    
    # Build vocabulary
    print("Building vocabulary...")
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
    
    # Enhanced model configurations
    models_config = {
        # Original models
        'RNN': {'class': RNNModel, 'type': 'rnn'},
        'LSTM': {'class': LSTMModel, 'type': 'lstm'},
        'GRU': {'class': GRUModel, 'type': 'gru'},
        'Transformer': {'class': TransformerModel, 'type': 'transformer'},
        
        # RNN variants
        'Deep_RNN': {'class': DeepRNNModel, 'type': 'rnn'},
        'Bidirectional_RNN': {'class': BidirectionalRNNModel, 'type': 'rnn'},
        'RNN_Attention': {'class': RNNWithAttentionModel, 'type': 'rnn'},
        
        # LSTM variants
        'Stacked_LSTM': {'class': StackedLSTMModel, 'type': 'lstm'},
        'Bidirectional_LSTM': {'class': BidirectionalLSTMModel, 'type': 'lstm'},
        'LSTM_Attention': {'class': LSTMWithAttentionModel, 'type': 'lstm'},
        'LSTM_Pretrained': {'class': LSTMWithPretrainedEmbeddingsModel, 'type': 'lstm'},
        
        # GRU variants
        'Stacked_GRU': {'class': StackedGRUModel, 'type': 'gru'},
        'Bidirectional_GRU': {'class': BidirectionalGRUModel, 'type': 'gru'},
        'GRU_Attention': {'class': GRUWithAttentionModel, 'type': 'gru'},
        'GRU_Pretrained': {'class': GRUWithPretrainedEmbeddingsModel, 'type': 'gru'},
        
        # Transformer variants
        'Lightweight_Transformer': {'class': LightweightTransformerModel, 'type': 'transformer'},
        'Deep_Transformer': {'class': DeepTransformerModel, 'type': 'transformer'},
        'Transformer_Pooling': {'class': TransformerWithPoolingModel, 'type': 'transformer'}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    for name, config in models_config.items():
        print(f"\n{'='*30} Training {name} {'='*30}")
        
        start_time = time.time()
        
        try:
            # Prepare data
            train_loader = prepare_data(X_train, y_train, config['type'], vocab)
            test_loader = prepare_data(X_test, y_test, config['type'], vocab)
            
            # Initialize model with proper parameters
            if 'Transformer' in name:
                if name == 'Lightweight_Transformer':
                    model = config['class'](
                        vocab_size=len(vocab), embed_dim=32, num_heads=2,
                        hidden_dim=32, num_classes=3, num_layers=2
                    )
                elif name == 'Deep_Transformer':
                    model = config['class'](
                        vocab_size=len(vocab), embed_dim=64, num_heads=4,
                        hidden_dim=64, num_classes=3, num_layers=6
                    )
                else:
                    model = config['class'](
                        vocab_size=len(vocab), embed_dim=64, num_heads=4,
                        hidden_dim=64, num_classes=3, num_layers=4
                    )
            else:
                model = config['class'](
                    vocab_size=len(vocab), embed_dim=64, 
                    hidden_dim=64, num_classes=3
                )
            
            model = model.to(device)
            
            # Train model
            print(f"Training {name}...")
            model = simple_train_model(model, train_loader, device, num_epochs=5)
            
            training_time = time.time() - start_time
            
            # Evaluate model
            print(f"Evaluating {name}...")
            eval_results = evaluate_model_comprehensive(model, test_loader, device)
            
            # Store results
            results[name] = {
                'accuracy': eval_results['accuracy'],
                'f1_score': eval_results['f1_score'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'training_time': training_time
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy: {eval_results['accuracy']:.4f}")
            print(f"  F1 Score: {eval_results['f1_score']:.4f}")
            print(f"  Precision: {eval_results['precision']:.4f}")
            print(f"  Recall: {eval_results['recall']:.4f}")
            print(f"  Training Time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    # Create results summary
    print(f"\n{'='*80}")
    print("ENHANCED MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    
    if results:
        # Create formatted table
        print(f"{'Model':<25} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<11} {'Recall':<8} {'Time (s)':<8}")
        print("-" * 80)
        
        for name, metrics in results.items():
            print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} "
                  f"{metrics['precision']:<11.4f} {metrics['recall']:<8.4f} {metrics['training_time']:<8.1f}")
        
        # Find best models
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
        fastest = min(results.items(), key=lambda x: x[1]['training_time'])
        
        print(f"\n🏆 Best Accuracy: {best_accuracy[0]} with {best_accuracy[1]['accuracy']:.4f}")
        print(f"🎯 Best F1 Score: {best_f1[0]} with {best_f1[1]['f1_score']:.4f}")
        print(f"⚡ Fastest Model: {fastest[0]} trained in {fastest[1]['training_time']:.1f} seconds")
        
        # Create visualizations
        print("\nCreating visualizations...")
        perf_path = create_performance_visualization(results)
        timing_path = create_timing_visualization(results)
        
        # Generate model architecture visualizations
        print("\nGenerating model architecture visualizations...")
        try:
            viz_paths = visualize_all_models(
                vocab_size=len(vocab), embed_dim=64, hidden_dim=64, 
                num_classes=3, save_dir="enhanced_model_visualizations"
            )
            print("Architecture visualizations completed!")
        except Exception as e:
            print(f"Error creating architecture visualizations: {e}")
    
    else:
        print("No models were successfully trained.")

if __name__ == "__main__":
    main()