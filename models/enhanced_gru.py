import torch
import torch.nn as nn
from .base import BaseModel

class EnhancedGRUModel(BaseModel):
    """
    Enhanced GRU model that consolidates basic and emotion variants
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
        
        # GRU layer with configurable architecture
        self.gru = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, 
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism (optional)
        self.use_attention = attention
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        if attention:
            self.attention = nn.Linear(gru_output_dim, 1)
        
        # Output layers with dropout
        self.hidden_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(gru_output_dim, num_classes)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        # GRU processing
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim * directions)
        
        if self.use_attention:
            # Attention mechanism for emotionally relevant words
            attention_weights = torch.softmax(self.attention(gru_out), dim=1)  # (batch, seq_len, 1)
            attended_output = torch.sum(attention_weights * gru_out, dim=1)  # (batch, hidden_dim * directions)
            out = attended_output
        else:
            # Use last output
            out = gru_out[:, -1, :]
        
        # Apply dropout and final classification
        out = self.hidden_dropout(out)
        out = self.fc(out)
        return out


# Convenience classes for backward compatibility and common configurations
class GRUModel(EnhancedGRUModel):
    """Basic GRU for sentiment analysis"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes)


class GRUModelEmotion(EnhancedGRUModel):
    """Enhanced GRU for emotion detection with improved regularization"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         num_layers=3, dropout=0.3)


class StackedGRUModel(EnhancedGRUModel):
    """Stacked GRU with multiple layers for deeper representations"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=3):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         num_layers=num_layers, dropout=0.3)


class BidirectionalGRUModel(EnhancedGRUModel):
    """Bidirectional GRU to capture forward and backward dependencies"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         bidirectional=True, dropout=0.3)


class GRUWithAttentionModel(EnhancedGRUModel):
    """GRU with attention mechanism to focus on emotionally relevant words"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         attention=True, dropout=0.3)


class GRUWithPretrainedEmbeddingsModel(EnhancedGRUModel):
    """GRU with pretrained embeddings support (GloVe, Word2Vec, FastText)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 pretrained_embeddings=None, dropout_rate=0.3):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                         pretrained_embeddings=pretrained_embeddings, 
                         dropout=dropout_rate)