#!/usr/bin/env python3
"""
Error Analysis and Qualitative Model Assessment

This script conducts comprehensive error analysis of the best-performing model
to identify patterns in misclassified sentences and guide final improvements.
"""

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Import models and utilities
from models.lstm_variants import LSTMWithAttentionModel
from models.gru_variants import GRUWithAttentionModel  
from models.transformer_variants import TransformerWithPoolingModel
from utils import tokenize_texts, simple_tokenizer
from evaluate import evaluate_model_comprehensive

def categorize_sentiment(score):
    """Convert continuous sentiment score to categorical label."""
    if score < -0.1:
        return 0  # Negative
    elif score > 0.1:
        return 2  # Positive  
    else:
        return 1  # Neutral

def prepare_data(texts, labels, model_type, vocab, batch_size=32):
    """Prepare data for evaluation."""
    input_ids, _ = tokenize_texts(texts, model_type, vocab)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_model_predictions(model, dataloader, device):
    """Get detailed predictions from model."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_true_labels)

def analyze_prediction_confidence(probabilities, predictions, true_labels):
    """Analyze prediction confidence patterns."""
    correct_mask = predictions == true_labels
    incorrect_mask = ~correct_mask
    
    # Get confidence scores (max probability)
    confidence_scores = np.max(probabilities, axis=1)
    
    correct_confidence = confidence_scores[correct_mask]
    incorrect_confidence = confidence_scores[incorrect_mask]
    
    return {
        'correct_confidence_mean': np.mean(correct_confidence),
        'correct_confidence_std': np.std(correct_confidence),
        'incorrect_confidence_mean': np.mean(incorrect_confidence),
        'incorrect_confidence_std': np.std(incorrect_confidence),
        'low_confidence_threshold': np.percentile(confidence_scores, 25),
        'high_confidence_threshold': np.percentile(confidence_scores, 75)
    }

def analyze_text_characteristics(texts, labels, predictions):
    """Analyze characteristics of misclassified texts."""
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    
    analysis = {
        'length_analysis': {},
        'word_patterns': {},
        'misclassification_patterns': {}
    }
    
    # Text length analysis
    lengths = [len(text.split()) for text in texts]
    
    for true_label in range(3):
        for pred_label in range(3):
            mask = (labels == true_label) & (predictions == pred_label)
            if np.any(mask):
                masked_lengths = np.array(lengths)[mask]
                key = f"{sentiment_labels[true_label]}_predicted_as_{sentiment_labels[pred_label]}"
                analysis['length_analysis'][key] = {
                    'count': len(masked_lengths),
                    'mean_length': np.mean(masked_lengths),
                    'median_length': np.median(masked_lengths),
                    'std_length': np.std(masked_lengths)
                }
    
    # Word pattern analysis for misclassifications
    misclassified_mask = labels != predictions
    misclassified_texts = np.array(texts)[misclassified_mask]
    misclassified_true = labels[misclassified_mask]
    misclassified_pred = predictions[misclassified_mask]
    
    # Extract common words in misclassified samples
    for true_label in range(3):
        for pred_label in range(3):
            if true_label == pred_label:
                continue
                
            mask = (misclassified_true == true_label) & (misclassified_pred == pred_label)
            if np.any(mask):
                texts_subset = misclassified_texts[mask]
                all_words = []
                for text in texts_subset:
                    words = simple_tokenizer(text.lower())
                    all_words.extend(words)
                
                word_freq = Counter(all_words)
                key = f"{sentiment_labels[true_label]}_misclassified_as_{sentiment_labels[pred_label]}"
                analysis['word_patterns'][key] = {
                    'top_words': word_freq.most_common(20),
                    'unique_words': len(set(all_words)),
                    'total_words': len(all_words),
                    'sample_count': len(texts_subset)
                }
    
    return analysis

def create_error_analysis_visualizations(confusion_mat, confidence_analysis, text_analysis, save_prefix="error_analysis"):
    """Create comprehensive error analysis visualizations."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Error Analysis', fontsize=16)
    
    # 1. Confusion Matrix Heatmap
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    
    # 2. Confidence Distribution
    axes[0, 1].hist([confidence_analysis['correct_confidence_mean']], 
                   alpha=0.7, label=f'Correct (μ={confidence_analysis["correct_confidence_mean"]:.3f})', 
                   bins=20)
    axes[0, 1].hist([confidence_analysis['incorrect_confidence_mean']], 
                   alpha=0.7, label=f'Incorrect (μ={confidence_analysis["incorrect_confidence_mean"]:.3f})', 
                   bins=20)
    axes[0, 1].set_title('Prediction Confidence Analysis')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Text Length Distribution by Error Type
    length_data = []
    length_labels = []
    for key, data in text_analysis['length_analysis'].items():
        if 'predicted_as' in key and data['count'] > 0:
            length_data.append(data['mean_length'])
            length_labels.append(key.replace('_predicted_as_', '→').replace('_', ' '))
    
    if length_data:
        axes[0, 2].bar(range(len(length_data)), length_data)
        axes[0, 2].set_title('Average Text Length by Misclassification Type')
        axes[0, 2].set_xlabel('Error Type')
        axes[0, 2].set_ylabel('Average Length (words)')
        axes[0, 2].set_xticks(range(len(length_data)))
        axes[0, 2].set_xticklabels(length_labels, rotation=45, ha='right')
    
    # 4. Error Rate by Class
    total_per_class = confusion_mat.sum(axis=1)
    correct_per_class = np.diag(confusion_mat)
    error_rates = 1 - (correct_per_class / total_per_class)
    
    axes[1, 0].bar(['Negative', 'Neutral', 'Positive'], error_rates, 
                  color=['red', 'gray', 'green'], alpha=0.7)
    axes[1, 0].set_title('Error Rate by Sentiment Class')
    axes[1, 0].set_ylabel('Error Rate')
    axes[1, 0].set_ylim(0, 1)
    
    # 5. Misclassification Flow (Sankey-like visualization)
    misclass_counts = defaultdict(int)
    for i in range(3):
        for j in range(3):
            if i != j:
                misclass_counts[f"{['Neg', 'Neu', 'Pos'][i]}→{['Neg', 'Neu', 'Pos'][j]}"] = confusion_mat[i, j]
    
    if misclass_counts:
        labels = list(misclass_counts.keys())
        values = list(misclass_counts.values())
        axes[1, 1].bar(labels, values, color='coral', alpha=0.7)
        axes[1, 1].set_title('Misclassification Patterns')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Performance Metrics Summary
    precision = correct_per_class / confusion_mat.sum(axis=0)
    recall = correct_per_class / total_per_class
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    x = np.arange(3)
    width = 0.25
    axes[1, 2].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[1, 2].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[1, 2].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    axes[1, 2].set_title('Performance by Class')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(['Negative', 'Neutral', 'Positive'])
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def run_comprehensive_error_analysis():
    """Run comprehensive error analysis on best model."""
    print("=" * 80)
    print("COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 80)
    print("Analyzing prediction patterns and error characteristics...")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading dataset...")
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Use subset for analysis (manageable size)
        df = df.head(5000)
        
        texts = df['original_text'].astype(str).tolist()
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
        print(f"Dataset loaded: {len(texts)} samples")
        print(f"Label distribution: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")
        
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
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # For demonstration, we'll analyze a LSTM with Attention model
    # In practice, you'd load your best-performing model from optimization
    print("\n🤖 Loading model for analysis...")
    model = LSTMWithAttentionModel(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=3,
        dropout_rate=0.4
    )
    model.to(device)
    
    # Quick training for demonstration (in practice, load optimized model)
    print("Training model for analysis...")
    from train import train_model_epochs
    import torch.optim as optim
    import torch.nn as nn
    
    train_loader = prepare_data(X_train, y_train, 'lstm', vocab, 32)
    test_loader = prepare_data(X_test, y_test, 'lstm', vocab, 32)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train for analysis
    history = train_model_epochs(
        model, train_loader, test_loader, optimizer, loss_fn, device,
        num_epochs=15, scheduler=scheduler, gradient_clip_value=1.0
    )
    
    # Get predictions for analysis
    print("\n🔍 Analyzing model predictions...")
    predictions, probabilities, true_labels = get_model_predictions(model, test_loader, device)
    
    # Comprehensive evaluation
    eval_results = evaluate_model_comprehensive(model, test_loader, device)
    print(f"\nModel Performance:")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"F1 Score: {eval_results['f1_score']:.4f}")
    print(f"Precision: {eval_results['precision']:.4f}")
    print(f"Recall: {eval_results['recall']:.4f}")
    
    # Error Analysis
    print("\n" + "="*60)
    print("DETAILED ERROR ANALYSIS")
    print("="*60)
    
    # 1. Confusion Matrix Analysis
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print("        Pred:  Neg  Neu  Pos")
    for i, (true_class, row) in enumerate(zip(['Neg', 'Neu', 'Pos'], conf_matrix)):
        print(f"True {true_class}: {row}")
    
    # 2. Confidence Analysis
    confidence_analysis = analyze_prediction_confidence(probabilities, predictions, true_labels)
    print(f"\nConfidence Analysis:")
    print(f"Correct predictions confidence: {confidence_analysis['correct_confidence_mean']:.3f} ± {confidence_analysis['correct_confidence_std']:.3f}")
    print(f"Incorrect predictions confidence: {confidence_analysis['incorrect_confidence_mean']:.3f} ± {confidence_analysis['incorrect_confidence_std']:.3f}")
    
    # 3. Text Characteristics Analysis
    text_analysis = analyze_text_characteristics(X_test, true_labels, predictions)
    
    print(f"\nText Length Analysis (by error type):")
    for error_type, stats in text_analysis['length_analysis'].items():
        if stats['count'] > 0:
            print(f"{error_type}: {stats['count']} samples, avg length: {stats['mean_length']:.1f} words")
    
    print(f"\nCommon Words in Misclassifications:")
    for pattern, word_data in text_analysis['word_patterns'].items():
        if word_data['sample_count'] > 0:
            print(f"\n{pattern} ({word_data['sample_count']} samples):")
            top_words = [f"{word}({count})" for word, count in word_data['top_words'][:10]]
            print(f"  Top words: {', '.join(top_words)}")
    
    # 4. Specific Error Examples
    print(f"\n" + "="*60)
    print("SPECIFIC ERROR EXAMPLES")
    print("="*60)
    
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    error_mask = predictions != true_labels
    error_indices = np.where(error_mask)[0]
    
    # Show examples of each type of error
    for true_class in range(3):
        for pred_class in range(3):
            if true_class == pred_class:
                continue
                
            # Find examples of this error type
            specific_errors = error_indices[(true_labels[error_indices] == true_class) & 
                                          (predictions[error_indices] == pred_class)]
            
            if len(specific_errors) > 0:
                print(f"\n{sentiment_labels[true_class]} → {sentiment_labels[pred_class]} errors:")
                
                # Show top 3 examples with highest confidence (most confident mistakes)
                error_confidences = np.max(probabilities[specific_errors], axis=1)
                top_confident_errors = specific_errors[np.argsort(error_confidences)[-3:]]
                
                for idx in top_confident_errors:
                    text = X_test[idx]
                    confidence = np.max(probabilities[idx])
                    pred_probs = probabilities[idx]
                    
                    print(f"  Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
                    print(f"  Confidence: {confidence:.3f}")
                    print(f"  Probabilities: Neg={pred_probs[0]:.3f}, Neu={pred_probs[1]:.3f}, Pos={pred_probs[2]:.3f}")
                    print("  ---")
    
    # 5. Create visualizations
    print(f"\n📊 Creating error analysis visualizations...")
    create_error_analysis_visualizations(conf_matrix, confidence_analysis, text_analysis)
    
    # 6. Generate recommendations
    print(f"\n" + "="*60)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    # Check class imbalance issues
    class_accuracies = np.diag(conf_matrix) / conf_matrix.sum(axis=1)
    if min(class_accuracies) < 0.5:
        worst_class = np.argmin(class_accuracies)
        recommendations.append(f"• Address poor performance on {sentiment_labels[worst_class]} class (accuracy: {class_accuracies[worst_class]:.3f})")
    
    # Check confidence patterns
    if confidence_analysis['incorrect_confidence_mean'] > 0.7:
        recommendations.append("• Model is overconfident in wrong predictions - consider calibration techniques")
    
    if confidence_analysis['correct_confidence_mean'] < 0.8:
        recommendations.append("• Model shows low confidence even in correct predictions - may need more training")
    
    # Check text length patterns
    length_patterns = text_analysis['length_analysis']
    if length_patterns:
        short_text_errors = [k for k, v in length_patterns.items() if v['mean_length'] < 10 and v['count'] > 5]
        if short_text_errors:
            recommendations.append("• Consider special handling for short texts (< 10 words)")
        
        long_text_errors = [k for k, v in length_patterns.items() if v['mean_length'] > 50 and v['count'] > 5]
        if long_text_errors:
            recommendations.append("• Consider truncation strategy for very long texts (> 50 words)")
    
    # Check confusion patterns
    off_diagonal_sum = conf_matrix.sum() - np.trace(conf_matrix)
    if off_diagonal_sum > np.trace(conf_matrix):
        recommendations.append("• Overall error rate is high - consider ensemble methods or architecture changes")
    
    if recommendations:
        print("\nKey Recommendations:")
        for rec in recommendations:
            print(rec)
    else:
        print("\nModel performance appears well-balanced. Consider fine-tuning hyperparameters for final optimization.")
    
    # Save analysis results
    analysis_summary = {
        'model_performance': eval_results,
        'confusion_matrix': conf_matrix.tolist(),
        'confidence_analysis': confidence_analysis,
        'text_analysis': text_analysis,
        'recommendations': recommendations
    }
    
    # Save to file
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'error_analysis_summary_{timestamp}.json', 'w') as f:
        json.dump(analysis_summary, f, indent=2, default=str)
    
    print(f"\n💾 Analysis summary saved to error_analysis_summary_{timestamp}.json")
    print("📊 Error analysis visualizations saved as error_analysis_comprehensive.png")
    
    return analysis_summary

if __name__ == "__main__":
    run_comprehensive_error_analysis()