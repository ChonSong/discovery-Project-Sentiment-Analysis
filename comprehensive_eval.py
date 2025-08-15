#!/usr/bin/env python3
"""
Comprehensive evaluation and demonstration script for sentiment analysis models.

This script combines model training, evaluation with multiple metrics, visualization,
and demonstration with example sentences all in one place.
"""

import os
import sys
import argparse
from models import RNNModel, LSTMModel, GRUModel, TransformerModel
from evaluate import evaluate_model_comprehensive
from visualize_models import visualize_all_models
from demo_examples import demonstrate_sentiment_analysis
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from utils import tokenize_texts, simple_tokenizer
from train import train_model_epochs

def main():
    parser = argparse.ArgumentParser(description='Comprehensive sentiment analysis evaluation')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'rnn', 'lstm', 'gru', 'transformer'],
                       help='Model type to train and evaluate')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate model architecture visualizations')
    parser.add_argument('--demo', action='store_true',
                       help='Run example sentence demonstrations')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results and visualizations')
    
    args = parser.parse_args()
    
    print("🚀 Comprehensive Sentiment Analysis Evaluation")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations if requested
    if args.visualize:
        print("\n📊 Generating Model Visualizations...")
        viz_dir = os.path.join(args.output_dir, "visualizations")
        try:
            paths = visualize_all_models(save_dir=viz_dir)
            print(f"✅ Visualizations saved to: {viz_dir}")
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
    
    # Run demonstrations if requested
    if args.demo:
        print("\n🎯 Running Example Sentence Demonstrations...")
        if args.model == 'all':
            for model_type in ['rnn', 'lstm', 'gru', 'transformer']:
                print(f"\n--- {model_type.upper()} Model ---")
                try:
                    demonstrate_sentiment_analysis(model_type, args.epochs)
                except Exception as e:
                    print(f"❌ Error with {model_type}: {e}")
        else:
            demonstrate_sentiment_analysis(args.model, args.epochs)
    
    print("\n✨ Evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()