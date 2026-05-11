"""BiLSTM encoder-decoder with scaled dot-product attention.

Parallel decoder (not autoregressive) — the output sequence has the same length
as the input PPG, so we can decode in parallel using attention over encoder
outputs. This is faster than autoregressive decoding and works fine for
sample-aligned reconstruction.

Attention uses scaled dot-product (query @ keys.T / sqrt(H)) rather than
additive Bahdanau formulation. Additive attention expands a (B, T_q, T_k, H)
intermediate that OOMs at T=500, B=64, H=128 on 8 GB VRAM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotAttention(nn.Module):
    """Scaled dot-product attention over encoder outputs.

    Produces context vectors of the same shape as the query sequence.
    Memory: O(B * T_q * T_k) — no hidden-dim broadcast.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # query: (B, T_q, H), keys: (B, T_k, H)
        # Returns context: (B, T_q, H)
        q = self.W_q(query)                              # (B, T_q, H)
        k = self.W_k(keys)                               # (B, T_k, H)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, T_q, T_k)
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, keys)                     # (B, T_q, H)


class BiLSTMSeq2Seq(nn.Module):
    """BiLSTM encoder, LSTM decoder with scaled dot-product attention, parallel decoding.

    Args:
        config: dict with model.{hidden_dim, num_layers, dropout, use_attention}.
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        hidden = m["hidden_dim"]
        layers = m["num_layers"]
        dropout = m.get("dropout", 0.1)
        self.use_attention = m.get("use_attention", True)

        # Encoder: BiLSTM over the input PPG
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        # Encoder output dim is 2*hidden (bidir). Project to hidden for decoder.
        self.enc_proj = nn.Linear(2 * hidden, hidden)

        # Decoder: unidirectional LSTM. Input at each step is previous output (we
        # parallelize by feeding the encoder output at each timestep instead).
        self.decoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if layers > 1 else 0.0,
        )

        if self.use_attention:
            self.attention = ScaledDotAttention(hidden)
            self.combine = nn.Linear(2 * hidden, hidden)

        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) -> (B, T, 1)
        x_seq = x.transpose(1, 2)
        enc_out, _ = self.encoder(x_seq)        # (B, T, 2*H)
        enc_out = self.enc_proj(enc_out)        # (B, T, H)

        # Decode: feed the encoder output through the decoder LSTM
        dec_out, _ = self.decoder(enc_out)      # (B, T, H)

        if self.use_attention:
            ctx = self.attention(dec_out, enc_out)        # (B, T, H)
            dec_out = torch.tanh(self.combine(torch.cat([dec_out, ctx], dim=-1)))  # (B, T, H)

        out = self.head(dec_out)                # (B, T, 1)
        return out.transpose(1, 2)              # (B, 1, T)
