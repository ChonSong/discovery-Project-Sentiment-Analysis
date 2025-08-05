import torch.nn as nn
from .base import BaseModel

class TransformerModelEmotion(BaseModel):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
