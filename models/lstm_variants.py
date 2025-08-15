import torch.nn as nn
import torch
from .base import BaseModel

class StackedLSTMModel(BaseModel):
    """Stacked LSTM with multiple layers for deeper representations"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.fc(out)
        return out

class BidirectionalLSTMModel(BaseModel):
    """Bidirectional LSTM to capture forward and backward dependencies"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.3)
        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output (concatenated forward and backward)
        out = self.fc(out)
        return out

class LSTMWithAttentionModel(BaseModel):
    """LSTM with attention mechanism to focus on emotionally relevant words"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.3)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        
        # Compute attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_dim)
        
        out = self.fc(attended_output)
        return out

class LSTMWithPretrainedEmbeddingsModel(BaseModel):
    """LSTM with pretrained embeddings support (GloVe, Word2Vec, FastText)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pretrained_embeddings=None, dropout_rate=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Allow fine-tuning
        
        # Enhanced regularization with multiple dropout layers
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)  # Lighter dropout on embeddings
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout_rate)
        self.hidden_dropout = nn.Dropout(dropout_rate)  # Additional dropout after LSTM
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)  # Dropout on embeddings
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.hidden_dropout(out)  # Dropout before final layer
        out = self.fc(out)
        return out