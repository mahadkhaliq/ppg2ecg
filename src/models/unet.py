"""1D U-Net for PPG-to-ECG reconstruction.

Standard encoder-decoder with skip connections, adapted to 1D signals.
Input shape: (B, 1, T), Output shape: (B, 1, T).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv1d -> BatchNorm -> ReLU blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool -> DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample -> concat skip -> DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if shapes mismatch (rare with even input lengths)
        diff = skip.size(-1) - x.size(-1)
        if diff != 0:
            x = F.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """1D U-Net.

    Args:
        config: dict with 'encoder_channels' key, e.g. [32, 64, 128, 256, 512].
    """

    def __init__(self, config: dict):
        super().__init__()
        chs = config["model"]["encoder_channels"]
        assert len(chs) == 5, "encoder_channels must have 5 entries"

        self.inc = DoubleConv(1, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])

        self.up1 = Up(chs[4], chs[3])
        self.up2 = Up(chs[3], chs[2])
        self.up3 = Up(chs[2], chs[1])
        self.up4 = Up(chs[1], chs[0])

        self.outc = nn.Conv1d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        x1 = self.inc(x)        # (B, c0, T)
        x2 = self.down1(x1)     # (B, c1, T/2)
        x3 = self.down2(x2)     # (B, c2, T/4)
        x4 = self.down3(x3)     # (B, c3, T/8)
        x5 = self.down4(x4)     # (B, c4, T/16)

        u = self.up1(x5, x4)
        u = self.up2(u, x3)
        u = self.up3(u, x2)
        u = self.up4(u, x1)
        return self.outc(u)
