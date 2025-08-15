"""
Neural network models for sentiment analysis.

This module provides various deep learning architectures for text classification
including RNN, LSTM, GRU, and Transformer models with multiple variants.
"""

from .base import BaseModel
from .rnn import RNNModel
from .lstm import LSTMModel
from .gru import GRUModel
from .transformer import TransformerModel

# Import enhanced architecture variants
from .rnn_variants import DeepRNNModel, BidirectionalRNNModel, RNNWithAttentionModel
from .lstm_variants import StackedLSTMModel, BidirectionalLSTMModel, LSTMWithAttentionModel, LSTMWithPretrainedEmbeddingsModel
from .gru_variants import StackedGRUModel, BidirectionalGRUModel, GRUWithAttentionModel, GRUWithPretrainedEmbeddingsModel
from .transformer_variants import LightweightTransformerModel, DeepTransformerModel, TransformerWithPoolingModel

__all__ = [
    'BaseModel',
    # Original models
    'RNNModel', 
    'LSTMModel',
    'GRUModel',
    'TransformerModel',
    # RNN variants
    'DeepRNNModel',
    'BidirectionalRNNModel', 
    'RNNWithAttentionModel',
    # LSTM variants
    'StackedLSTMModel',
    'BidirectionalLSTMModel',
    'LSTMWithAttentionModel',
    'LSTMWithPretrainedEmbeddingsModel',
    # GRU variants
    'StackedGRUModel',
    'BidirectionalGRUModel',
    'GRUWithAttentionModel',
    'GRUWithPretrainedEmbeddingsModel',
    # Transformer variants
    'LightweightTransformerModel',
    'DeepTransformerModel',
    'TransformerWithPoolingModel'
]