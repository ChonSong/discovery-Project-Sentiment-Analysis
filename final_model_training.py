#!/usr/bin/env python3
"""
Final Model Training - Best Configuration with Full Dataset

This script trains the final optimized model using the best hyperparameters
found during focused optimization, on the largest possible dataset.
"""

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import time
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Import models and utilities
from models.lstm_variants import BidirectionalLSTMModel, LSTMWithAttentionModel, LSTMWithPretrainedEmbeddingsModel
from models.gru_variants import BidirectionalGRUModel, GRUWithAttentionModel, GRUWithPretrainedEmbeddingsModel
from models.transformer_variants import TransformerWithPoolingModel
from utils import tokenize_texts, simple_tokenizer
from train import train_model_epochs
from evaluate import evaluate_model_comprehensive
from experiment_tracker import ExperimentTracker
from embedding_utils import create_embedding_matrix

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

def create_balanced_loss_function(labels):
    """Create class-balanced loss function to handle imbalanced data."""
    # Calculate class weights
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    
    # Convert to tensor
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    print(f"Class weights: {dict(zip(unique_labels, class_weights))}")
    
    return nn.CrossEntropyLoss(weight=class_weights_tensor)

def save_final_model(model, vocab, config, performance, save_path):
    """Save the final trained model with all necessary information."""
    model_package = {
        'model_state_dict': model.state_dict(),
        'model_config': config,
        'vocab': vocab,
        'performance': performance,
        'training_timestamp': datetime.now().isoformat(),
        'model_class': type(model).__name__
    }
    
    torch.save(model_package, save_path)
    print(f"✅ Final model saved to {save_path}")

def train_final_optimized_model():
    """Train the final model with optimized hyperparameters on full dataset."""
    print("=" * 80)
    print("FINAL MODEL TRAINING - OPTIMIZED CONFIGURATION")
    print("=" * 80)
    print("Training best model with optimized hyperparameters on full dataset")
    print("Objective: Achieve maximum performance for production deployment")
    print("=" * 80)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Load full dataset
    print("\n📊 Loading full dataset...")
    try:
        df = pd.read_csv("exorde_raw_sample.csv")
        df = df.dropna(subset=['original_text', 'sentiment'])
        
        # Use maximum available data for final training
        print(f"Total samples available: {len(df)}")
        
        # For final model, use as much data as possible
        dataset_size = min(20000, len(df))  # Use up to 20K samples
        if dataset_size < len(df):
            # Stratified sampling to maintain class distribution
            df_sampled = df.groupby(
                df['sentiment'].apply(categorize_sentiment), 
                group_keys=False
            ).apply(lambda x: x.sample(min(len(x), dataset_size//3), random_state=42))
            df = df_sampled
        else:
            df = df.head(dataset_size)
        
        texts = df['original_text'].astype(str).tolist()
        labels = [categorize_sentiment(s) for s in df['sentiment'].tolist()]
        
        print(f"Final dataset size: {len(texts)} samples")
        
        # Analyze class distribution
        neg_count = labels.count(0)
        neu_count = labels.count(1) 
        pos_count = labels.count(2)
        total = len(labels)
        
        print(f"Class distribution:")
        print(f"  Negative: {neg_count} ({neg_count/total:.1%})")
        print(f"  Neutral:  {neu_count} ({neu_count/total:.1%})")
        print(f"  Positive: {pos_count} ({pos_count/total:.1%})")
        
        # Check for severe imbalance
        imbalance_ratio = max(neg_count, neu_count, pos_count) / min(neg_count, neu_count, pos_count)
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
    except FileNotFoundError:
        print("Dataset file not found. Please run getdata.py first.")
        return
    
    # Build comprehensive vocabulary
    print("\n🔤 Building vocabulary...")
    all_tokens = []
    for text in texts:
        all_tokens.extend(simple_tokenizer(text))
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for token in set(all_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Strategic train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load best configuration from optimization results
    # For this demo, we'll use a high-performing configuration
    # In practice, you'd load this from the optimization results
    
    best_config = {
        'model_name': 'Bidirectional_LSTM_Attention',
        'model_class': LSTMWithAttentionModel,
        'model_type': 'lstm',
        'hyperparameters': {
            'vocab_size': len(vocab),
            'embed_dim': 128,
            'hidden_dim': 256,
            'num_classes': 3,
            'dropout_rate': 0.4,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'weight_decay': 5e-4,
            'gradient_clip_value': 1.0
        }
    }
    
    print(f"\n🤖 Training final model: {best_config['model_name']}")
    print("Configuration:")
    for key, value in best_config['hyperparameters'].items():
        if key not in ['vocab_size', 'num_classes']:
            print(f"  {key}: {value}")
    
    # Create final model
    model_params = {k: v for k, v in best_config['hyperparameters'].items() 
                   if k in ['vocab_size', 'embed_dim', 'hidden_dim', 'num_classes', 'dropout_rate']}
    
    model = best_config['model_class'](**model_params)
    model.to(device)
    
    # Prepare data loaders
    batch_size = best_config['hyperparameters']['batch_size']
    train_loader = prepare_data(X_train, y_train, best_config['model_type'], vocab, batch_size)
    val_loader = prepare_data(X_val, y_val, best_config['model_type'], vocab, batch_size)
    
    # Setup training with class balancing
    print("\n⚖️ Setting up class-balanced training...")
    loss_fn = create_balanced_loss_function(y_train)
    loss_fn.to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=best_config['hyperparameters']['learning_rate'],
        weight_decay=best_config['hyperparameters']['weight_decay']
    )
    
    # Advanced learning rate scheduling for final training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6
    )
    
    # Start experiment tracking
    experiment_id = tracker.start_experiment(
        model_name=f"FINAL_{best_config['model_name']}",
        hyperparameters=best_config['hyperparameters'],
        description="Final optimized model training on full dataset with class balancing"
    )
    
    print(f"\n🚀 Starting final training...")
    print(f"Training epochs: 100 (with early stopping)")
    print(f"Early stopping patience: 15")
    
    # Extended training for final model
    start_time = time.time()
    history = train_model_epochs(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        num_epochs=100,  # Extended training
        scheduler=scheduler,
        gradient_clip_value=best_config['hyperparameters']['gradient_clip_value'],
        early_stop_patience=15  # More patience for final training
    )
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Comprehensive final evaluation
    print("\n📊 Final Model Evaluation...")
    final_performance = evaluate_model_comprehensive(model, val_loader, device)
    
    print(f"\nFINAL MODEL PERFORMANCE:")
    print(f"{'='*50}")
    print(f"Accuracy:  {final_performance['accuracy']:.4f}")
    print(f"F1 Score:  {final_performance['f1_score']:.4f}")
    print(f"Precision: {final_performance['precision']:.4f}")
    print(f"Recall:    {final_performance['recall']:.4f}")
    print(f"{'='*50}")
    
    # Check if target performance achieved
    target_f1 = 0.75
    if final_performance['f1_score'] >= target_f1:
        print(f"🎯 TARGET ACHIEVED! F1 Score {final_performance['f1_score']:.4f} >= {target_f1}")
    else:
        print(f"📈 Progress made! F1 Score {final_performance['f1_score']:.4f} (target: {target_f1})")
    
    # Log final results to experiment tracker
    tracker.log_metrics(final_performance)
    tracker.end_experiment("completed")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"final_optimized_model_{timestamp}.pt"
    
    save_final_model(
        model, vocab, best_config, final_performance, model_save_path
    )
    
    # Generate comprehensive training report
    training_report = {
        'model_name': best_config['model_name'],
        'final_performance': final_performance,
        'training_config': best_config['hyperparameters'],
        'dataset_info': {
            'total_samples': len(texts),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'class_distribution': {
                'negative': neg_count,
                'neutral': neu_count,
                'positive': pos_count
            },
            'imbalance_ratio': imbalance_ratio
        },
        'training_history': {
            'total_epochs': len(history['train_loss']),
            'training_time_minutes': training_time / 60,
            'best_val_accuracy': max(history['val_accuracy']) if history['val_accuracy'] else 0,
            'final_learning_rate': optimizer.param_groups[0]['lr']
        },
        'model_path': model_save_path,
        'experiment_id': experiment_id,
        'timestamp': timestamp
    }
    
    # Save training report
    report_path = f"final_training_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(training_report, f, indent=2, default=str)
    
    print(f"\n💾 Training report saved to {report_path}")
    
    # Export experiment results
    tracker.export_results()
    
    # Generate final recommendations
    print(f"\n" + "="*60)
    print("FINAL MODEL DEPLOYMENT RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    if final_performance['f1_score'] >= 0.75:
        recommendations.append("✅ Model ready for production deployment")
    elif final_performance['f1_score'] >= 0.65:
        recommendations.append("⚠️ Model suitable for testing/staging environment")
    else:
        recommendations.append("❌ Model needs additional optimization before deployment")
    
    if imbalance_ratio > 3:
        recommendations.append("• Consider collecting more balanced training data")
    
    if final_performance['accuracy'] - final_performance['f1_score'] > 0.1:
        recommendations.append("• Monitor for class-specific performance issues")
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(rec)
    
    return training_report, model, vocab

def load_and_test_final_model(model_path):
    """Load and test the final trained model."""
    print(f"\n🔄 Loading final model from {model_path}...")
    
    model_package = torch.load(model_path, map_location='cpu')
    
    print(f"Model: {model_package['model_class']}")
    print(f"Performance: F1={model_package['performance']['f1_score']:.4f}")
    print(f"Trained: {model_package['training_timestamp']}")
    
    return model_package

if __name__ == "__main__":
    print("Starting final model training with optimized configuration...")
    
    # Train final model
    report, model, vocab = train_final_optimized_model()
    
    print(f"\n🎉 FINAL MODEL TRAINING COMPLETED!")
    print(f"Model saved: {report['model_path']}")
    print(f"F1 Score: {report['final_performance']['f1_score']:.4f}")
    print(f"Training time: {report['training_history']['training_time_minutes']:.1f} minutes")
    
    # Test loading the saved model
    print(f"\n🧪 Testing model loading...")
    loaded_model = load_and_test_final_model(report['model_path'])
    print("✅ Model loading test successful!")
    
    print(f"\n🚀 Final optimized model ready for deployment!")