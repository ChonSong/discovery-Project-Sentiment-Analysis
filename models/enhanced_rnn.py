import torch
import torch.nn as nn
from .base import BaseModel

class EnhancedRNNModel(BaseModel):
    """
    Enhanced RNN model that supports both sentiment and emotion classification
    with configurable architecture for improved performance.
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
        
        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(dropout * 0.5) if dropout > 0 else nn.Identity()
        
        # RNN layer with configurable architecture
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, batch_first=True, 
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism (optional)
        self.use_attention = attention
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        if attention:
            self.attention = nn.Linear(rnn_output_dim, 1)
        
        # Output layers with dropout
        self.hidden_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(rnn_output_dim, num_classes)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        # RNN processing
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim * directions)
        
        if self.use_attention:
            # Attention mechanism for better emotion/sentiment detection
            attention_weights = torch.softmax(self.attention(rnn_out), dim=1)  # (batch, seq_len, 1)
            attended_output = torch.sum(attention_weights * rnn_out, dim=1)  # (batch, hidden_dim * directions)
            out = attended_output
        else:
            # Use last output
            out = rnn_out[:, -1, :]
        
        # Apply dropout and final classification
        out = self.hidden_dropout(out)
        out = self.fc(out)
        return out


# Convenience classes for common configurations
class RNNModel(EnhancedRNNModel):
    """Basic RNN for sentiment analysis"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes)


class RNNModelEmotion(EnhancedRNNModel):
    """Enhanced RNN for emotion detection with improved regularization"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         num_layers=3, dropout=0.3)


class BidirectionalRNNModel(EnhancedRNNModel):
    """Bidirectional RNN for better context understanding"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         bidirectional=True, dropout=0.3)


class RNNWithAttentionModel(EnhancedRNNModel):
    """RNN with attention mechanism for emotion detection"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         attention=True, dropout=0.3)