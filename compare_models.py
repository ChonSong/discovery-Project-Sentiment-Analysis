#!/usr/bin/env python3
"""
Compare performance of different model architectures.

This script trains all available models and compares their performance.
"""

import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split

# Import our models and utilities
from models import RNNModel, LSTMModel, GRUModel, TransformerModel
from utils import tokenize_texts, simple_tokenizer
from train import train_model_epochs
from evaluate import evaluate_model

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

def main():
    print("Model Comparison for Sentiment Analysis")
    print("=" * 50)
    
    # Load and prepare data
    print("Loading data...")
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Use a subset for faster comparison
        df = df.head(2000)
        
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
    
    # Model configurations
    models_config = {
        'RNN': {'class': RNNModel, 'type': 'rnn'},
        'LSTM': {'class': LSTMModel, 'type': 'lstm'},
        'GRU': {'class': GRUModel, 'type': 'gru'},
        'Transformer': {'class': TransformerModel, 'type': 'transformer'}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    for name, config in models_config.items():
        print(f"\n{'='*20} Training {name} {'='*20}")
        
        start_time = time.time()
        
        try:
            # Prepare data
            train_loader = prepare_data(X_train, y_train, config['type'], vocab)
            test_loader = prepare_data(X_test, y_test, config['type'], vocab)
            
            # Initialize model
            if name == 'Transformer':
                model = config['class'](
                    vocab_size=len(vocab),
                    embed_dim=64,
                    num_heads=4,
                    hidden_dim=64,
                    num_classes=3,
                    num_layers=2
                )
            else:
                model = config['class'](
                    vocab_size=len(vocab),
                    embed_dim=64,
                    hidden_dim=64,
                    num_classes=3
                )
            
            model.to(device)
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # Train model (fewer epochs for comparison)
            history = train_model_epochs(
                model, train_loader, test_loader, optimizer, loss_fn, device, num_epochs=3
            )
            
            # Final evaluation
            final_accuracy = evaluate_model(model, test_loader, None, device)
            training_time = time.time() - start_time
            
            results[name] = {
                'accuracy': final_accuracy,
                'time': training_time,
                'final_loss': history['train_loss'][-1] if history['train_loss'] else 0.0
            }
            
            print(f"{name} completed - Accuracy: {final_accuracy:.4f}, Time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {'accuracy': 0.0, 'time': 0.0, 'final_loss': float('inf')}
    
    # Display results
    print("\n" + "=" * 50)
    print("FINAL COMPARISON RESULTS")
    print("=" * 50)
    print(f"{'Model':<12} {'Accuracy':<10} {'Time (s)':<10} {'Final Loss':<12}")
    print("-" * 50)
    
    for name, result in results.items():
        print(f"{name:<12} {result['accuracy']:<10.4f} {result['time']:<10.1f} {result['final_loss']:<12.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")
    
    # Find fastest model
    fastest_model = min(results.items(), key=lambda x: x[1]['time'])
    print(f"⚡ Fastest Model: {fastest_model[0]} trained in {fastest_model[1]['time']:.1f} seconds")

if __name__ == "__main__":
    main()