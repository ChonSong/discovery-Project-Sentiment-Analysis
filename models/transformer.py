import torch.nn as nn
from .base import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        out = self.transformer_encoder(x)  # (batch, seq_len, embed_dim)
        out = out[:, -1, :]  # take last token
        out = self.fc(out)
        return out
