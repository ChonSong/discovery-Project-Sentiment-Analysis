"""
Enhanced Models Module

This module provides consolidated, enhanced versions of all sentiment and emotion
analysis models with improved architecture and performance optimizations.

All models support:
- Configurable dropout and regularization
- Optional pretrained embeddings
- Attention mechanisms where applicable
- Bidirectional processing options
- Multiple pooling strategies (for transformers)

Usage:
    from models.enhanced import (
        EnhancedLSTMModel, 
        EnhancedGRUModel, 
        EnhancedTransformerModel,
        LSTMModelEmotion,
        GRUWithAttentionModel,
        TransformerWithPoolingModel
    )
"""

# Enhanced base models
from .enhanced_rnn import EnhancedRNNModel
from .enhanced_lstm import EnhancedLSTMModel
from .enhanced_gru import EnhancedGRUModel  
from .enhanced_transformer import EnhancedTransformerModel

# Convenience classes for backward compatibility
from .enhanced_rnn import (
    RNNModel, RNNModelEmotion, BidirectionalRNNModel, RNNWithAttentionModel
)
from .enhanced_lstm import (
    LSTMModel, LSTMModelEmotion, StackedLSTMModel, 
    BidirectionalLSTMModel, LSTMWithAttentionModel, LSTMWithPretrainedEmbeddingsModel
)
from .enhanced_gru import (
    GRUModel, GRUModelEmotion, StackedGRUModel,
    BidirectionalGRUModel, GRUWithAttentionModel, GRUWithPretrainedEmbeddingsModel
)
from .enhanced_transformer import (
    TransformerModel, TransformerModelEmotion, LightweightTransformerModel,
    DeepTransformerModel, TransformerWithPoolingModel
)

__all__ = [
    # Enhanced base models
    'EnhancedRNNModel',
    'EnhancedLSTMModel', 
    'EnhancedGRUModel',
    'EnhancedTransformerModel',
    
    # RNN variants
    'RNNModel',
    'RNNModelEmotion',
    'BidirectionalRNNModel',
    'RNNWithAttentionModel',
    
    # LSTM variants
    'LSTMModel',
    'LSTMModelEmotion',
    'StackedLSTMModel',
    'BidirectionalLSTMModel',
    'LSTMWithAttentionModel',
    'LSTMWithPretrainedEmbeddingsModel',
    
    # GRU variants
    'GRUModel',
    'GRUModelEmotion',
    'StackedGRUModel',
    'BidirectionalGRUModel',
    'GRUWithAttentionModel',
    'GRUWithPretrainedEmbeddingsModel',
    
    # Transformer variants
    'TransformerModel',
    'TransformerModelEmotion',
    'LightweightTransformerModel',
    'DeepTransformerModel',
    'TransformerWithPoolingModel',
]