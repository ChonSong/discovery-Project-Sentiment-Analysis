import torch
import torch.nn as nn
from .base import BaseModel

class EnhancedLSTMModel(BaseModel):
    """
    Enhanced LSTM model that consolidates basic and emotion variants
    with advanced features for improved sentiment and emotion analysis.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 num_layers=1, dropout=0.0, bidirectional=False, 
                 attention=False, pretrained_embeddings=None):
        super().__init__()
        
        # Embedding layer with optional pretrained weights
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
        
        # Dropout for embeddings (lighter dropout)
        self.embedding_dropout = nn.Dropout(dropout * 0.5) if dropout > 0 else nn.Identity()
        
        # LSTM layer with configurable architecture
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, 
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism (optional)
        self.use_attention = attention
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        if attention:
            self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Output layers with dropout
        self.hidden_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * directions)
        
        if self.use_attention:
            # Attention mechanism for emotionally relevant words
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
            attended_output = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_dim * directions)
            out = attended_output
        else:
            # Use last output
            out = lstm_out[:, -1, :]
        
        # Apply dropout and final classification
        out = self.hidden_dropout(out)
        out = self.fc(out)
        return out


# Convenience classes for backward compatibility and common configurations
class LSTMModel(EnhancedLSTMModel):
    """Basic LSTM for sentiment analysis"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes)


class LSTMModelEmotion(EnhancedLSTMModel):
    """Enhanced LSTM for emotion detection with improved regularization"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         num_layers=3, dropout=0.3)


class StackedLSTMModel(EnhancedLSTMModel):
    """Stacked LSTM with multiple layers for deeper representations"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=3):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         num_layers=num_layers, dropout=0.3)


class BidirectionalLSTMModel(EnhancedLSTMModel):
    """Bidirectional LSTM to capture forward and backward dependencies"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         bidirectional=True, dropout=0.3)


class LSTMWithAttentionModel(EnhancedLSTMModel):
    """LSTM with attention mechanism to focus on emotionally relevant words"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         attention=True, dropout=0.3)


class LSTMWithPretrainedEmbeddingsModel(EnhancedLSTMModel):
    """LSTM with pretrained embeddings support (GloVe, Word2Vec, FastText)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 pretrained_embeddings=None, dropout_rate=0.3):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         pretrained_embeddings=pretrained_embeddings, 
                         dropout=dropout_rate)