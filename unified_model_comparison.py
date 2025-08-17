#!/usr/bin/env python3
"""
Unified Model Comparison Script

This script provides comprehensive comparison of all enhanced model architectures
for both sentiment and emotion analysis, replacing multiple comparison scripts.

Features:
- Tests all enhanced unified models
- Comprehensive metrics including F1, precision, recall, accuracy
- Timing analysis for performance comparison
- Support for both synthetic and real data testing
- Backward compatibility with original model interfaces
"""

import pandas as pd
import torch
import torch.optim as optim
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import enhanced models
from models.enhanced import (
    # Enhanced base models
    EnhancedLSTMModel, EnhancedGRUModel, EnhancedTransformerModel, EnhancedRNNModel,
    # Convenience classes
    LSTMModel, LSTMModelEmotion, LSTMWithAttentionModel,
    GRUModel, GRUModelEmotion, GRUWithAttentionModel,
    TransformerModel, TransformerWithPoolingModel,
    RNNModel, RNNModelEmotion
)

# Import utilities
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
    """Prepare data for training."""
    input_ids, _ = tokenize_texts(texts, model_type, vocab)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, labels_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_enhanced_models_config():
    """Get configuration for all enhanced models to test."""
    return {
        'Enhanced_LSTM': {
            'class': EnhancedLSTMModel,
            'args': {'dropout': 0.3, 'attention': False},
            'description': 'Enhanced LSTM with configurable architecture'
        },
        'LSTM_Emotion': {
            'class': LSTMModelEmotion, 
            'args': {},
            'description': 'LSTM optimized for emotion detection'
        },
        'LSTM_Attention': {
            'class': LSTMWithAttentionModel,
            'args': {},
            'description': 'LSTM with attention mechanism'
        },
        'Enhanced_GRU': {
            'class': EnhancedGRUModel,
            'args': {'dropout': 0.3, 'bidirectional': False},
            'description': 'Enhanced GRU with configurable features'
        },
        'GRU_Emotion': {
            'class': GRUModelEmotion,
            'args': {},
            'description': 'GRU optimized for emotion detection'
        },
        'GRU_Attention': {
            'class': GRUWithAttentionModel,
            'args': {},
            'description': 'GRU with attention mechanism'
        },
        'Enhanced_Transformer': {
            'class': EnhancedTransformerModel,
            'args': {'num_heads': 4, 'pooling_strategy': 'last'},
            'description': 'Enhanced Transformer with positional encoding'
        },
        'Transformer_Pooling': {
            'class': TransformerWithPoolingModel,
            'args': {'num_heads': 4},
            'description': 'Transformer with global average pooling'
        },
        'Enhanced_RNN': {
            'class': EnhancedRNNModel,
            'args': {'dropout': 0.3, 'attention': True},
            'description': 'Enhanced RNN with attention and regularization'
        }
    }

def run_unified_comparison(use_real_data=False, quick_test=True):
    """Run comprehensive comparison of all enhanced models."""
    
    print("=" * 80)
    print("UNIFIED ENHANCED MODEL COMPARISON")
    print("=" * 80)
    print(f"Mode: {'Real data' if use_real_data else 'Synthetic data'}")
    print(f"Test type: {'Quick test' if quick_test else 'Full evaluation'}")
    print("=" * 80)
    
    # Prepare data
    if use_real_data and os.path.exists('exorde_raw_sample.csv'):
        print("📊 Loading real dataset...")
        df = pd.read_csv('exorde_raw_sample.csv')
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Convert sentiment scores to categories
        df['sentiment_label'] = df['sentiment'].apply(categorize_sentiment)
        
        # Use subset for quick testing
        if quick_test:
            df = df.head(500)
        else:
            df = df.head(2000)
            
        texts = df['original_text'].astype(str).tolist()
        labels = df['sentiment_label'].tolist()
        print(f"   Dataset size: {len(texts)} samples")
        
    else:
        print("📊 Generating synthetic test data...")
        # Create synthetic data
        positive_texts = [
            "I love this! It's amazing and wonderful!",
            "Excellent quality and great value for money",
            "Outstanding performance, highly recommend",
            "Fantastic experience, will buy again",
            "Perfect product, exceeded expectations"
        ] * (20 if quick_test else 100)
        
        negative_texts = [
            "This is terrible and awful, hate it",
            "Poor quality, very disappointed",
            "Horrible experience, never again",
            "Worst product ever, total waste",
            "Completely unsatisfied, terrible service"
        ] * (20 if quick_test else 100)
        
        neutral_texts = [
            "It's okay, nothing special but works",
            "Average product, does what it says",
            "Decent quality for the price",
            "Not bad, could be better though",
            "Acceptable performance, meets needs"
        ] * (20 if quick_test else 100)
        
        texts = positive_texts + negative_texts + neutral_texts
        labels = [2] * len(positive_texts) + [0] * len(negative_texts) + [1] * len(neutral_texts)
        print(f"   Dataset size: {len(texts)} synthetic samples")
    
    # Build vocabulary
    all_tokens = []
    for text in texts:
        tokens = simple_tokenizer(text)
        all_tokens.extend(tokens)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in set(all_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"   Vocabulary size: {len(vocab)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Model configuration
    vocab_size = len(vocab)
    embed_dim = 64
    hidden_dim = 128
    num_classes = 3
    batch_size = 16
    num_epochs = 3 if quick_test else 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    print()
    
    # Get models to test
    models_config = get_enhanced_models_config()
    
    # Run comparison
    results = []
    
    for model_name, config in models_config.items():
        print(f"🚀 Testing {model_name}: {config['description']}")
        
        try:
            # Initialize model
            model_class = config['class']
            base_args = {'vocab_size': vocab_size, 'embed_dim': embed_dim, 
                        'hidden_dim': hidden_dim, 'num_classes': num_classes}
            
            # Add transformer-specific args
            if 'Transformer' in model_name:
                base_args['num_heads'] = config['args'].get('num_heads', 4)
                if 'pooling_strategy' in config['args']:
                    base_args['pooling_strategy'] = config['args']['pooling_strategy']
            
            # Add model-specific args
            base_args.update(config['args'])
            
            model = model_class(**base_args)
            model.to(device)
            
            # Prepare data loaders
            model_type = 'lstm'  # Use LSTM tokenization for all models
            train_loader = prepare_data(X_train, y_train, model_type, vocab, batch_size)
            test_loader = prepare_data(X_test, y_test, model_type, vocab, batch_size)
            
            # Setup training
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # Train model
            start_time = time.time()
            print(f"   Training for {num_epochs} epochs...")
            
            history = train_model_epochs(
                model, train_loader, test_loader, optimizer, loss_fn, device,
                num_epochs=num_epochs, scheduler=scheduler, gradient_clip_value=1.0
            )
            
            training_time = time.time() - start_time
            
            # Evaluate model
            eval_results = evaluate_model_comprehensive(model, test_loader, device)
            
            # Store results
            result = {
                'Model': model_name,
                'Description': config['description'],
                'F1_Score': eval_results.get('f1_score', 0),
                'Accuracy': eval_results.get('accuracy', 0),
                'Precision': eval_results.get('precision', 0),
                'Recall': eval_results.get('recall', 0),
                'Training_Time': training_time,
                'Final_Loss': history['train_loss'][-1] if history['train_loss'] else 0
            }
            results.append(result)
            
            print(f"   ✅ Results: F1={result['F1_Score']:.4f}, "
                  f"Acc={result['Accuracy']:.4f}, Time={training_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results.append({
                'Model': model_name,
                'Description': config['description'],
                'F1_Score': 0,
                'Accuracy': 0,
                'Precision': 0,
                'Recall': 0,
                'Training_Time': 0,
                'Final_Loss': float('inf'),
                'Error': str(e)
            })
        
        print()
    
    # Create results summary
    results_df = pd.DataFrame(results)
    
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Sort by F1 score
    results_df_sorted = results_df.sort_values('F1_Score', ascending=False)
    
    print(results_df_sorted[['Model', 'F1_Score', 'Accuracy', 'Training_Time']].to_string(index=False))
    
    print(f"\n🏆 Best performing model: {results_df_sorted.iloc[0]['Model']}")
    print(f"   F1 Score: {results_df_sorted.iloc[0]['F1_Score']:.4f}")
    print(f"   Description: {results_df_sorted.iloc[0]['Description']}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"unified_model_comparison_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    return results_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Model Comparison")
    parser.add_argument("--real-data", action="store_true", help="Use real dataset if available")
    parser.add_argument("--full-test", action="store_true", help="Run full evaluation (slower)")
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_unified_comparison(
        use_real_data=args.real_data,
        quick_test=not args.full_test
    )
    
    print("\n🎉 Unified model comparison completed!")