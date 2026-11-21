"""
model.py
--------
Attention-based transformer classifier, in the spirit of Khan et al. (2024):
stroke-point sequence -> linear embedding -> positional encoding ->
multi-head self-attention transformer encoder layers -> attention pooling
-> fused with global handcrafted features -> softmax over emotion classes.
"""

import math
import torch
import torch.nn as nn

from feature_extraction import POINT_FEATURE_DIM, GLOBAL_FEATURE_DIM, MAX_SEQ_LEN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AttentionPooling(nn.Module):
    """Learns a query vector and attends over the sequence to produce one pooled vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.scale = d_model ** 0.5

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T) with 1 = valid, 0 = padding
        scores = (x @ self.query) / self.scale               # (B, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)                     # (B, D)
        return pooled, weights.squeeze(-1)


class HandwritingEmotionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dff: int = 128,
        dropout: float = 0.2,
        global_hidden: int = 32,
    ):
        super().__init__()
        self.input_proj = nn.Linear(POINT_FEATURE_DIM, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = AttentionPooling(d_model)

        self.global_mlp = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, global_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = d_model + global_hidden
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, point_seq, seq_mask, global_feats, return_attention=False):
        # point_seq: (B, T, POINT_FEATURE_DIM), seq_mask: (B, T), global_feats: (B, GLOBAL_FEATURE_DIM)
        x = self.input_proj(point_seq)
        x = self.pos_enc(x)
        key_padding_mask = seq_mask == 0  # True = ignore
        enc_out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled, attn_weights = self.pool(enc_out, seq_mask)

        g = self.global_mlp(global_feats)
        fused = torch.cat([pooled, g], dim=-1)
        logits = self.classifier(fused)
        if return_attention:
            return logits, attn_weights
        return logits
