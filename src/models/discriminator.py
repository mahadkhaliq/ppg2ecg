"""1D PatchGAN discriminator for ECG signal authenticity.

Classifies overlapping patches of a 500-sample ECG window as real or generated.
Output is a 1D map of patch-level scores (not a single scalar), which gives the
generator per-region gradient feedback — identical to Isola et al. 2017 (pix2pix)
applied to 1D signals.

Uses LSGAN objective (MSE to {0,1} targets) rather than BCE for training stability
on small datasets. All Conv1d layers are spectrally normalised to bound the
discriminator's Lipschitz constant, preventing overconfident separation on small
datasets (Miyato et al. 2018).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator1D(nn.Module):
    """1D PatchGAN discriminator with spectral normalisation.

    Architecture: four strided Conv1d blocks, no final sigmoid.
    Receptive field per output element: ~70 samples (~0.56 s at 125 Hz).
    Spectral norm replaces BatchNorm — the two are incompatible.

    Args:
        n_filters: base channel count (doubles each block, capped at 4×).
    """

    def __init__(self, n_filters: int = 32):
        super().__init__()

        def sn_conv(in_c, out_c, stride=2):
            return spectral_norm(
                nn.Conv1d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=True)
            )

        nf = n_filters
        self.net = nn.Sequential(
            sn_conv(1,    nf),    nn.LeakyReLU(0.2, inplace=True),   # 250
            sn_conv(nf,   nf*2), nn.LeakyReLU(0.2, inplace=True),   # 125
            sn_conv(nf*2, nf*4), nn.LeakyReLU(0.2, inplace=True),   # 63
            sn_conv(nf*4, 1, stride=1),                               # 62
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)  →  (B, 1, T') patch-level logits
        return self.net(x)
