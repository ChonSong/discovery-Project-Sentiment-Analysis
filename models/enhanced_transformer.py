import torch
import torch.nn as nn
from .base import BaseModel

class EnhancedTransformerModel(BaseModel):
    """
    Enhanced Transformer model that consolidates basic and emotion variants
    with advanced features for improved sentiment and emotion analysis.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, 
                 num_layers=2, dropout=0.1, pooling_strategy='last', 
                 pretrained_embeddings=None):
        super().__init__()
        
        # Embedding layer with optional pretrained weights
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
        
        # Positional encoding for better sequence understanding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling strategy
        self.pooling_strategy = pooling_strategy
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Embedding with positional encoding
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        out = self.transformer_encoder(x)  # (batch, seq_len, embed_dim)
        
        # Apply pooling strategy
        if self.pooling_strategy == 'last':
            out = out[:, -1, :]  # Take last token
        elif self.pooling_strategy == 'mean':
            out = torch.mean(out, dim=1)  # Global average pooling
        elif self.pooling_strategy == 'max':
            out, _ = torch.max(out, dim=1)  # Global max pooling
        elif self.pooling_strategy == 'cls':
            out = out[:, 0, :]  # Take first token (CLS-like)
        
        # Final classification
        out = self.dropout(out)
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


# Convenience classes for backward compatibility and common configurations
class TransformerModel(EnhancedTransformerModel):
    """Basic Transformer for sentiment analysis"""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=2):
        super().__init__(vocab_size, embed_dim, num_heads, hidden_dim, num_classes, 
                         num_layers=num_layers, dropout=0.1, pooling_strategy='last')


class TransformerModelEmotion(EnhancedTransformerModel):
    """Enhanced Transformer for emotion detection with improved regularization"""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=4):
        super().__init__(vocab_size, embed_dim, num_heads, hidden_dim, num_classes, 
                         num_layers=num_layers, dropout=0.3, pooling_strategy='last')


class LightweightTransformerModel(EnhancedTransformerModel):
    """Lightweight Transformer with fewer parameters for faster inference"""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=2):
        super().__init__(vocab_size, embed_dim, num_heads, hidden_dim, num_classes, 
                         num_layers=num_layers, dropout=0.1, pooling_strategy='last')


class DeepTransformerModel(EnhancedTransformerModel):
    """Deeper Transformer with more layers for better representation"""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=6):
        super().__init__(vocab_size, embed_dim, num_heads, hidden_dim, num_classes, 
                         num_layers=num_layers, dropout=0.3, pooling_strategy='last')


class TransformerWithPoolingModel(EnhancedTransformerModel):
    """Transformer with global average pooling instead of last token"""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=4):
        super().__init__(vocab_size, embed_dim, num_heads, hidden_dim, num_classes, 
                         num_layers=num_layers, dropout=0.3, pooling_strategy='mean')