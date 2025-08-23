# Automated Sentiment Analysis of Social Media Data

This project evaluates and compares multiple deep learning architectures for sentiment and emotion classification on large-scale, noisy social media data, aiming to create a reproducible analysis framework.

## Overview

This project provides a complete pipeline for sentiment analysis using various deep learning models including RNN, LSTM, GRU, and Transformer architectures. The system is designed to classify social media posts by sentiment and emotion.

## Features

- **Multiple Model Architectures**: RNN, LSTM, GRU, and Transformer models
- **Emotion Detection**: Both sentiment and emotion classification capabilities
- **Data Pipeline**: Automated data loading and preprocessing from Exorde dataset
- **Flexible Training**: Configurable training and evaluation scripts
- **Pre-trained Models**: Support for saving and loading trained models

## Dataset

The project uses the [Exorde social media dataset](https://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1) which contains:
- Social media posts with text content
- Sentiment labels
- Emotion classifications
- Multiple languages support
- Metadata including timestamps and themes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ChonSong/discovery-Project-Sentiment-Analysis.git
cd discovery-Project-Sentiment-Analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data
```bash
python getdata.py
```

### 2. Train and Evaluate Models
```bash
python exorde_train_eval.py
```

## Usage

### Data Loading
The `getdata.py` script downloads a sample of the Exorde dataset:
```python
from getdata import download_exorde_sample
df = download_exorde_sample(sample_size=10000)
```

### Model Training
Choose from enhanced model architectures with improved emotion detection:
```python
from models.enhanced import (
    LSTMModelEmotion, 
    GRUWithAttentionModel, 
    TransformerWithPoolingModel
)

# Initialize enhanced model for emotion detection
model = LSTMModelEmotion(vocab_size=1000, embed_dim=64, hidden_dim=64, num_classes=3)

# Or use attention-based model
model = GRUWithAttentionModel(vocab_size=1000, embed_dim=64, hidden_dim=64, num_classes=3)

# Train
train_model(model, train_loader, optimizer, loss_fn, device)
```

### Model Evaluation
```python
from evaluate import evaluate_model
accuracy = evaluate_model(model, test_loader, metric_fn, device)
print(f"Model accuracy: {accuracy:.4f}")
```

## Model Architectures

### Enhanced Unified Models
The project now features consolidated, enhanced model architectures that improve upon the original implementations:

- **EnhancedLSTMModel**: Unified LSTM with configurable layers, attention, and emotion detection
- **EnhancedGRUModel**: Advanced GRU with bidirectional processing and attention mechanisms  
- **EnhancedTransformerModel**: Enhanced transformer with multiple pooling strategies and positional encoding
- **EnhancedRNNModel**: Improved RNN with regularization and attention support

### Model Variants
Each enhanced model supports multiple configurations for different use cases:

- **Basic variants**: `LSTMModel`, `GRUModel`, `TransformerModel` (backward compatible)
- **Emotion variants**: `LSTMModelEmotion`, `GRUModelEmotion` (enhanced regularization)
- **Attention variants**: `LSTMWithAttentionModel`, `GRUWithAttentionModel` (focus on relevant words)
- **Advanced variants**: `BidirectionalLSTMModel`, `TransformerWithPoolingModel` (improved context)

### Key Improvements
- **Unified Architecture**: Single configurable models replace multiple specialized files
- **Enhanced Emotion Detection**: Improved attention mechanisms for better emotion classification
- **Better Regularization**: Advanced dropout strategies and gradient clipping
- **Pretrained Embeddings**: Support for GloVe, Word2Vec, and FastText
- **Backward Compatibility**: Existing code continues to work without changes

## Project Structure

```
├── models/                 # Enhanced neural network implementations
│   ├── enhanced.py         # Unified enhanced model imports
│   ├── enhanced_lstm.py    # Enhanced LSTM with emotion detection
│   ├── enhanced_gru.py     # Enhanced GRU with attention
│   ├── enhanced_transformer.py # Enhanced transformer with pooling
│   ├── enhanced_rnn.py     # Enhanced RNN with regularization
│   ├── base.py            # Base model class
│   └── [legacy files]     # Original model implementations (maintained for compatibility)
├── getdata.py             # Data downloading script
├── exorde_train_eval.py   # Main training and evaluation pipeline
├── train.py               # Enhanced training functions with scheduling
├── evaluate.py            # Comprehensive evaluation functions
├── utils.py               # Utility functions
├── FINAL_PROJECT_REPORT.md # Consolidated comprehensive final report
└── *.csv                  # Data files
```

## Configuration

The training script supports various configuration options:
- Model type selection (rnn, lstm, gru, transformer)
- Hyperparameter tuning (learning rate, batch size, etc.)
- Data preprocessing options
- Training epochs and validation splits

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Exorde team for providing the social media dataset
- PyTorch community for the deep learning framework
- Contributors to the open source libraries used in this project
