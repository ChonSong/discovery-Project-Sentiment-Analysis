#!/usr/bin/env python3
"""
Example script demonstrating sentiment analysis with different model architectures.

This script shows how to:
1. Load and preprocess data
2. Train different models
3. Evaluate model performance
4. Compare results across architectures
"""

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import our models and utilities
from models import RNNModel, LSTMModel, GRUModel, TransformerModel
from utils import tokenize_texts, simple_tokenizer
from train import train_model_epochs
from evaluate import evaluate_model

def create_sample_data(num_samples=1000):
    """Create sample data for testing if CSV files are not available."""
    print("Creating sample sentiment data for testing...")
    
    # Sample texts with different sentiments
    positive_texts = [
        "I love this product! It's amazing!",
        "Great service and friendly staff",
        "Excellent quality and fast delivery",
        "Highly recommend this to everyone",
        "Perfect! Exactly what I needed"
    ]
    
    negative_texts = [
        "Terrible experience, very disappointed",
        "Poor quality and bad customer service", 
        "Not worth the money, very poor",
        "Waste of time and money",
        "Completely useless product"
    ]
    
    neutral_texts = [
        "It's okay, nothing special",
        "Average product, decent price",
        "Not bad but not great either",
        "Could be better, could be worse",
        "Standard quality as expected"
    ]
    
    # Generate random samples
    texts = []
    labels = []
    
    for _ in range(num_samples):
        sentiment = np.random.choice([0, 1, 2])  # 0=negative, 1=neutral, 2=positive
        if sentiment == 0:
            text = np.random.choice(negative_texts)
        elif sentiment == 1:
            text = np.random.choice(neutral_texts)
        else:
            text = np.random.choice(positive_texts)
        
        # Add some variation
        text = text + f" Sample {len(texts)}"
        texts.append(text)
        labels.append(sentiment)
    
    return texts, labels

def prepare_data(texts, labels, model_type, vocab, max_len=50):
    """Prepare data for training."""
    if model_type == "transformer":
        # For transformer, we'll use simple tokenization
        input_ids, _ = tokenize_texts(texts, model_type, vocab)
    else:
        # For RNN/LSTM/GRU
        input_ids, _ = tokenize_texts(texts, model_type, vocab)
    
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def build_vocabulary(texts, min_freq=1):
    """Build vocabulary from texts."""
    all_tokens = []
    for text in texts:
        tokens = simple_tokenizer(text)
        all_tokens.extend(tokens)
    
    # Count tokens
    token_counts = {}
    for token in all_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    # Build vocab
    vocab = {'<pad>': 0, '<unk>': 1}
    for token, count in token_counts.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    
    return vocab

def train_and_evaluate_model(model_type, texts, labels, vocab, device, num_epochs=5):
    """Train and evaluate a specific model type."""
    print(f"\n=== Training {model_type.upper()} Model ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Prepare data loaders
    train_loader = prepare_data(X_train, y_train, model_type, vocab)
    test_loader = prepare_data(X_test, y_test, model_type, vocab)
    
    # Model parameters
    vocab_size = len(vocab)
    embed_dim = 64
    hidden_dim = 64
    num_classes = len(set(labels))
    
    # Initialize model
    if model_type == "rnn":
        model = RNNModel(vocab_size, embed_dim, hidden_dim, num_classes)
    elif model_type == "lstm":
        model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_classes)
    elif model_type == "gru":
        model = GRUModel(vocab_size, embed_dim, hidden_dim, num_classes)
    elif model_type == "transformer":
        model = TransformerModel(vocab_size, embed_dim, 4, hidden_dim, num_classes, 2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Train model
    history = train_model_epochs(
        model, train_loader, test_loader, optimizer, loss_fn, device, num_epochs
    )
    
    # Final evaluation
    final_accuracy = evaluate_model(model, test_loader, None, device)
    print(f"Final {model_type.upper()} Test Accuracy: {final_accuracy:.4f}")
    
    return model, final_accuracy, history

def main():
    """Main function to run the example."""
    print("Sentiment Analysis Example")
    print("=" * 40)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Try to load real data, otherwise create sample data
    try:
        print("Attempting to load real data...")
        df = pd.read_csv("exorde_raw_sample.csv")
        
        # Check if required columns exist
        if 'text' not in df.columns:
            # Try to find text column
            text_columns = [col for col in df.columns if 'text' in col.lower()]
            if text_columns:
                df['text'] = df[text_columns[0]]
            else:
                raise KeyError("No text column found")
        
        if 'sentiment' not in df.columns:
            # Try to find sentiment/label column
            sentiment_columns = [col for col in df.columns if any(x in col.lower() for x in ['sentiment', 'label', 'emotion'])]
            if sentiment_columns:
                df['sentiment'] = df[sentiment_columns[0]]
            else:
                raise KeyError("No sentiment column found")
        
        # Clean and prepare data
        df = df.dropna(subset=['text', 'sentiment'])
        texts = df['text'].astype(str).tolist()[:1000]  # Limit for demo
        
        # Convert sentiment labels to categorical
        # Group continuous sentiment scores into 3 categories
        sentiment_values = df['sentiment'].values
        
        # Create 3 sentiment categories: negative, neutral, positive
        def categorize_sentiment(score):
            if score < -0.1:
                return 0  # Negative
            elif score > 0.1:
                return 2  # Positive 
            else:
                return 1  # Neutral
        
        labels = [categorize_sentiment(float(s)) for s in sentiment_values[:1000]]
        
        # Create sentiment mapping for display
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        print(f"Loaded {len(texts)} samples from CSV")
        print(f"Sentiment mapping: {sentiment_map}")
        
    except (FileNotFoundError, KeyError, Exception) as e:
        print(f"Could not load real data ({e}), creating sample data...")
        texts, labels = create_sample_data(1000)
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocabulary(texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Train and compare different models
    models_to_test = ["rnn", "lstm", "gru", "transformer"]
    results = {}
    
    for model_type in models_to_test:
        try:
            model, accuracy, history = train_and_evaluate_model(
                model_type, texts, labels, vocab, device, num_epochs=3
            )
            results[model_type] = accuracy
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = 0.0
    
    # Display final results
    print("\n" + "=" * 40)
    print("FINAL RESULTS SUMMARY")
    print("=" * 40)
    for model_type, accuracy in results.items():
        print(f"{model_type.upper():12}: {accuracy:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\nBest model: {best_model[0].upper()} with accuracy {best_model[1]:.4f}")

if __name__ == "__main__":
    main()