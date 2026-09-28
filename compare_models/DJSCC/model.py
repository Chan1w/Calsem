# -*- coding: utf-8 -*-
"""Traditional single-stream DeepJSCC baseline.

This model uses the same dataset, channel module, spatial downsampling ratio,
and total transmitted feature budget as CalSem. Unlike CalSem, it directly
reconstructs the image and does not use SD/VAE, semantic tokens, or HARQ heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from calsem.channel import power_normalize
from calsem.config import Config


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
            nn.GroupNorm(1, cout),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(1, ch),
        )
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
            nn.GroupNorm(1, cout),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepJSCC(nn.Module):
    """Single-stream DJSCC image autoencoder.

    The transmitted feature tensor has shape (B, total_ch, H/8, W/8), where:
        total_ch = round((c_base + c_sem + c_fid) * 64)

    With the current CalSem config c_base=0.25, c_sem=0.125, c_fid=0.125,
    total_ch is 32, matching the sum of CalSem's 16+8+8 feature channels.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.power = cfg.power
        self.resolution = cfg.resolution
        self.total_ch = max(1, int(round((cfg.c_base + cfg.c_sem + cfg.c_fid) * 64)))
        self.width = int(cfg.get("djscc_width", cfg.get("jscc_width", 128)))
        self.n_res_blocks = int(cfg.get("djscc_res_blocks", cfg.get("jscc_res_blocks", 2)))

        stem = max(self.width // 2, 32)
        self.encoder = nn.Sequential(
            ConvBlock(3, stem, stride=2),          # H/2
            ConvBlock(stem, self.width, stride=2), # H/4
            ConvBlock(self.width, self.width, stride=2), # H/8
            *[ResBlock(self.width) for _ in range(self.n_res_blocks)],
            nn.Conv2d(self.width, self.total_ch, 1),
        )
        self.decoder = nn.Sequential(
            ConvBlock(self.total_ch, self.width),
            *[ResBlock(self.width) for _ in range(self.n_res_blocks)],
            UpBlock(self.width, self.width),
            UpBlock(self.width, stem),
            UpBlock(stem, stem),
            nn.Conv2d(stem, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return power_normalize(self.encoder(x), self.power)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(y)

    def forward(self, x: torch.Tensor, channel: nn.Module, snr: torch.Tensor | float) -> torch.Tensor:
        s = self.encode(x)
        y = channel(s, snr)
        return self.decode(y)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
