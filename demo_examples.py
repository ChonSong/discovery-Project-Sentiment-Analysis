#!/usr/bin/env python3
"""
Example sentences demonstration for sentiment analysis.

This script demonstrates sentiment analysis on example sentences using trained models.
It shows predictions with confidence scores and provides a variety of sample texts
representing different sentiments.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models import RNNModel, LSTMModel, GRUModel, TransformerModel
from utils import tokenize_texts, simple_tokenizer
from train import train_model_epochs
from evaluate import evaluate_model_comprehensive
import matplotlib.pyplot as plt
import seaborn as sns

# Sample sentences for demonstration
EXAMPLE_SENTENCES = [
    # Positive sentiment examples
    "I absolutely love this product! It's amazing and works perfectly.",
    "This is the best experience I've ever had. Highly recommend!",
    "Fantastic quality and excellent customer service. Five stars!",
    "I'm so happy with my purchase. It exceeded my expectations.",
    "Outstanding performance and great value for money.",
    
    # Negative sentiment examples
    "This is terrible. I hate it and want my money back.",
    "Worst product ever. Complete waste of money.",
    "Very disappointed with the quality. Poor customer service.",
    "I regret buying this. It doesn't work at all.",
    "Absolutely awful experience. Would not recommend to anyone.",
    
    # Neutral sentiment examples
    "The product is okay. Nothing special but it works.",
    "Average quality for the price. Could be better.",
    "It's fine, does what it's supposed to do.",
    "Standard product with decent features.",
    "Not bad, but not great either. Just average.",
    
    # Mixed/ambiguous examples
    "Good product but delivery was slow.",
    "Great features but a bit expensive for what you get.",
    "The design is nice but the quality could be improved.",
    "Fast shipping but the product had some minor issues.",
    "Excellent customer service but the product is just okay."
]

# Expected sentiments for evaluation (0=negative, 1=neutral, 2=positive)
EXPECTED_SENTIMENTS = [
    2, 2, 2, 2, 2,  # Positive examples
    0, 0, 0, 0, 0,  # Negative examples  
    1, 1, 1, 1, 1,  # Neutral examples
    1, 1, 1, 1, 1   # Mixed examples (treating as neutral)
]

SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']

def prepare_single_text(text, vocab, max_len=50):
    """
    Prepare a single text for model prediction.
    
    Args:
        text: Input text string
        vocab: Vocabulary dictionary
        max_len: Maximum sequence length
    
    Returns:
        torch.Tensor: Tokenized and padded input tensor
    """
    tokens = simple_tokenizer(text)
    # Convert tokens to ids
    token_ids = [vocab.get(token, vocab.get('<unk>', 1)) for token in tokens]
    
    # Pad or truncate to max_len
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids.extend([vocab.get('<pad>', 0)] * (max_len - len(token_ids)))
    
    return torch.tensor([token_ids], dtype=torch.long)

def predict_sentiment(model, text, vocab, device):
    """
    Predict sentiment for a single text.
    
    Args:
        model: Trained PyTorch model
        text: Input text string
        vocab: Vocabulary dictionary
        device: Device to run prediction on
    
    Returns:
        tuple: (predicted_class, confidence_scores, predicted_label)
    """
    model.eval()
    
    # Prepare input
    input_tensor = prepare_single_text(text, vocab).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence_scores = probabilities.squeeze().cpu().numpy()
    
    predicted_label = SENTIMENT_LABELS[predicted_class]
    
    return predicted_class, confidence_scores, predicted_label

def create_prediction_visualization(sentences, predictions, expected, model_name, save_path=None):
    """
    Create a visualization of predictions vs expected sentiments.
    
    Args:
        sentences: List of input sentences
        predictions: List of predicted sentiment classes
        expected: List of expected sentiment classes
        model_name: Name of the model
        save_path: Path to save the visualization
    """
    # Create confusion matrix data
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(expected, predictions)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS, ax=ax1)
    ax1.set_title(f'{model_name} - Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Prediction accuracy by category
    correct_by_class = np.diag(cm)
    total_by_class = np.sum(cm, axis=1)
    accuracy_by_class = correct_by_class / total_by_class
    
    bars = ax2.bar(SENTIMENT_LABELS, accuracy_by_class, color=['red', 'gray', 'green'], alpha=0.7)
    ax2.set_title(f'{model_name} - Accuracy by Sentiment')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracy_by_class):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")
    
    plt.show()

def demonstrate_sentiment_analysis(model_type='lstm', num_epochs=5):
    """
    Demonstrate sentiment analysis with example sentences.
    
    Args:
        model_type: Type of model to train and use ('rnn', 'lstm', 'gru', 'transformer')
        num_epochs: Number of training epochs
    """
    print(f"Sentiment Analysis Demonstration with {model_type.upper()} Model")
    print("=" * 60)
    
    # Load and prepare training data
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        df = df.head(1000)  # Use smaller dataset for demo
        
        texts = df['original_text'].astype(str).tolist()
        
        def categorize_sentiment(score):
            if score < -0.1:
                return 0  # Negative
            elif score > 0.1:
                return 2  # Positive
            else:
                return 1  # Neutral
        
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
    except FileNotFoundError:
        print("Dataset not found. Using synthetic data for demonstration.")
        # Create simple synthetic data
        texts = [
            "I love this product", "This is great", "Amazing quality",
            "Terrible experience", "Very bad product", "I hate this",
            "It's okay", "Average product", "Nothing special"
        ] * 20
        labels = ([2] * 3 + [0] * 3 + [1] * 3) * 20
    
    # Build vocabulary
    all_tokens = []
    for text in texts:
        all_tokens.extend(simple_tokenizer(text))
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for token in set(all_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"Training data: {len(texts)} samples")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Prepare data
    def prepare_dataloader(texts, labels):
        input_ids, _ = tokenize_texts(texts, model_type, vocab)
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(input_ids, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    train_loader = prepare_dataloader(X_train, y_train)
    test_loader = prepare_dataloader(X_test, y_test)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_classes = {
        'rnn': RNNModel,
        'lstm': LSTMModel,
        'gru': GRUModel,
        'transformer': TransformerModel
    }
    
    if model_type == 'transformer':
        model = model_classes[model_type](
            vocab_size=len(vocab), embed_dim=64, num_heads=4,
            hidden_dim=64, num_classes=3, num_layers=2
        )
    else:
        model = model_classes[model_type](
            vocab_size=len(vocab), embed_dim=64, 
            hidden_dim=64, num_classes=3
        )
    
    model.to(device)
    
    # Train model
    print(f"\nTraining {model_type.upper()} model for {num_epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    history = train_model_epochs(model, train_loader, test_loader, optimizer, loss_fn, device, num_epochs)
    
    # Comprehensive evaluation
    print("\nModel Evaluation:")
    print("-" * 30)
    eval_results = evaluate_model_comprehensive(model, test_loader, device, SENTIMENT_LABELS)
    
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"F1 Score: {eval_results['f1_score']:.4f}")
    print(f"Precision: {eval_results['precision']:.4f}")
    print(f"Recall: {eval_results['recall']:.4f}")
    
    # Predict on example sentences
    print("\nExample Sentence Predictions:")
    print("=" * 60)
    
    predictions = []
    
    for i, sentence in enumerate(EXAMPLE_SENTENCES):
        predicted_class, confidence_scores, predicted_label = predict_sentiment(
            model, sentence, vocab, device
        )
        expected_label = SENTIMENT_LABELS[EXPECTED_SENTIMENTS[i]]
        
        predictions.append(predicted_class)
        
        print(f"\nSentence {i+1}: {sentence}")
        print(f"Expected: {expected_label} | Predicted: {predicted_label}")
        print(f"Confidence: Neg={confidence_scores[0]:.3f}, Neu={confidence_scores[1]:.3f}, Pos={confidence_scores[2]:.3f}")
        
        # Color code the result
        if predicted_class == EXPECTED_SENTIMENTS[i]:
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
    
    # Calculate accuracy on examples
    correct = sum(1 for p, e in zip(predictions, EXPECTED_SENTIMENTS) if p == e)
    accuracy = correct / len(EXAMPLE_SENTENCES)
    
    print(f"\nExample Sentences Accuracy: {accuracy:.2f} ({correct}/{len(EXAMPLE_SENTENCES)})")
    
    # Create visualization
    create_prediction_visualization(
        EXAMPLE_SENTENCES, predictions, EXPECTED_SENTIMENTS, 
        model_type.upper(), f"example_predictions_{model_type}.png"
    )
    
    return model, vocab, eval_results

if __name__ == "__main__":
    print("Sentiment Analysis Example Demonstration")
    print("=" * 50)
    
    # Run demonstration with LSTM model
    model, vocab, results = demonstrate_sentiment_analysis('lstm', num_epochs=10)
    
    print("\n" + "=" * 50)
    print("Demonstration completed!")
    print(f"Final model performance: {results['f1_score']:.4f} F1 score")