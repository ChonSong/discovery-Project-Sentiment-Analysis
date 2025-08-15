import torch.nn as nn
import torch
from .base import BaseModel

class StackedGRUModel(BaseModel):
    """Stacked GRU with multiple layers for more complexity"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last output
        out = self.fc(out)
        return out

class BidirectionalGRUModel(BaseModel):
    """Bidirectional GRU to capture context from both directions"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.3)
        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last output (concatenated forward and backward)
        out = self.fc(out)
        return out

class GRUWithAttentionModel(BaseModel):
    """GRU with attention mechanism to focus on important features"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, dropout=0.3)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        
        # Compute attention weights
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * gru_out, dim=1)  # (batch, hidden_dim)
        
        out = self.fc(attended_output)
        return out

class GRUWithPretrainedEmbeddingsModel(BaseModel):
    """GRU with pretrained embeddings support"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Allow fine-tuning
        
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last output
        out = self.fc(out)
        return out