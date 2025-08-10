# Discovery Project: Sentiment Analysis

A comprehensive sentiment analysis project that implements multiple neural network architectures for analyzing social media data from the Exorde dataset.

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
Choose from multiple model architectures:
```python
from models.rnn import RNNModel
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.transformer import TransformerModel

# Initialize model
model = RNNModel(vocab_size=1000, embed_dim=64, hidden_dim=64, num_classes=3)

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

- **RNN**: Basic recurrent neural network for sequence modeling
- **LSTM**: Long Short-Term Memory for handling long dependencies
- **GRU**: Gated Recurrent Unit as a lighter alternative to LSTM
- **Transformer**: Self-attention based model for state-of-the-art performance

## Project Structure

```
├── models/                 # Neural network model implementations
│   ├── base.py            # Base model class
│   ├── rnn.py             # RNN implementation
│   ├── lstm.py            # LSTM implementation
│   ├── gru.py             # GRU implementation
│   └── transformer.py     # Transformer implementation
├── getdata.py             # Data downloading script
├── exorde_train_eval.py   # Main training and evaluation pipeline
├── train.py               # Training functions
├── evaluate.py            # Evaluation functions
├── utils.py               # Utility functions
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
