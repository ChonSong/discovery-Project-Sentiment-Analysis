#!/usr/bin/env python3
"""
Quick start script for training sentiment analysis models.

Usage:
    python quickstart.py --model rnn
    python quickstart.py --model lstm  
    python quickstart.py --model gru
    python quickstart.py --model transformer
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--model', 
                       choices=['rnn', 'lstm', 'gru', 'transformer'],
                       default='rnn',
                       help='Model architecture to train (default: rnn)')
    parser.add_argument('--epochs', 
                       type=int,
                       default=5,
                       help='Number of training epochs (default: 5)')
    
    args = parser.parse_args()
    
    print(f"Starting training with {args.model.upper()} model for {args.epochs} epochs...")
    
    # Modify the exorde_train_eval.py script to use the selected model
    with open('exorde_train_eval.py', 'r') as f:
        content = f.read()
    
    # Replace model type
    content = content.replace('model_type = "rnn"', f'model_type = "{args.model}"')
    
    # Replace epochs 
    content = content.replace('num_epochs=10', f'num_epochs={args.epochs}')
    
    # Write temporary script
    with open('temp_training.py', 'w') as f:
        f.write(content)
    
    # Run the training
    import subprocess
    result = subprocess.run([sys.executable, 'temp_training.py'], 
                          capture_output=False)
    
    # Clean up
    if os.path.exists('temp_training.py'):
        os.remove('temp_training.py')
    
    if result.returncode == 0:
        print(f"\n✅ Successfully trained {args.model.upper()} model!")
        print(f"📁 Model saved as: trained_{args.model}_model.pt")
    else:
        print(f"\n❌ Training failed with return code {result.returncode}")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    sys.exit(main())