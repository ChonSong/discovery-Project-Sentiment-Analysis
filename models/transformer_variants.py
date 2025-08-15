import torch.nn as nn
import torch
from .base import BaseModel

class LightweightTransformerModel(BaseModel):
    """Lightweight Transformer with fewer parameters for faster inference"""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Smaller embedding dimension for lightweight model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True, 
            dropout=0.1  # Lower dropout for smaller model
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]  # Take last token
        out = self.fc(out)
        return out

class DeepTransformerModel(BaseModel):
    """Deeper Transformer with more layers for better representation"""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True, 
            dropout=0.3
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]  # Take last token
        out = self.fc(out)
        return out

class TransformerWithPoolingModel(BaseModel):
    """Transformer with global average pooling instead of last token"""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True, 
            dropout=0.3
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        out = torch.mean(out, dim=1)  # (batch, embed_dim)
        
        out = self.fc(out)
        return out