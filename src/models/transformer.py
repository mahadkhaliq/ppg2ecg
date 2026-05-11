"""Patch-based Transformer encoder for PPG-to-ECG reconstruction.

The input PPG is split into non-overlapping patches (patch_size=25 → 20 patches
for a 500-sample window). Each patch is linearly projected to d_model, processed
by a stack of Transformer encoder layers with self-attention, then projected back
to patch_size samples via a linear head.

Encoder-only design: for same-length reconstruction (PPG → ECG, T_in = T_out),
a separate decoder with cross-attention adds no benefit over encoder self-attention
and, critically, decoder designs that use learned (input-independent) target queries
degenerate to outputting a fixed template regardless of the input signal. The
encoder-only architecture avoids this failure mode while remaining closely related
to PatchTST (Nie et al. 2023) applied to cross-modal synthesis.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerSeq2Seq(nn.Module):
    """Patch-based Transformer encoder for PPG → ECG reconstruction.

    Args:
        config: dict with model.{patch_size, d_model, num_heads, num_encoder_layers,
                                 ffn_dim, dropout}. num_decoder_layers is accepted
                                 but ignored (encoder-only design).
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.patch_size: int = m["patch_size"]
        d_model: int = m["d_model"]
        n_heads: int = m["num_heads"]
        enc_layers: int = m["num_encoder_layers"]
        ffn: int = m["ffn_dim"]
        dropout: float = m.get("dropout", 0.1)

        # Patch projection: (B, 1, T) -> (B, num_patches, d_model)
        self.patch_proj = nn.Linear(self.patch_size, d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # Output head: project each patch token back to patch_size samples
        self.head = nn.Linear(d_model, self.patch_size)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, T) -> (B, num_patches, patch_size)."""
        B, C, T = x.shape
        assert C == 1
        assert T % self.patch_size == 0, (
            f"Input length {T} must be divisible by patch_size {self.patch_size}"
        )
        return x.view(B, T // self.patch_size, self.patch_size)

    def _from_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, patch_size) -> (B, 1, T)."""
        B, P, S = x.shape
        return x.reshape(B, 1, P * S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        patches = self._to_patches(x)          # (B, P, patch_size)
        tokens = self.patch_proj(patches)      # (B, P, d_model)
        tokens = self.pos_enc(tokens)

        enc_out = self.encoder(tokens)         # (B, P, d_model)

        out_patches = self.head(enc_out)       # (B, P, patch_size)
        return self._from_patches(out_patches) # (B, 1, T)
