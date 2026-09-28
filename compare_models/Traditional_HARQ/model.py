# -*- coding: utf-8 -*-
"""Traditional Chase-combining HARQ image reconstruction baseline.

The encoder/decoder budget matches the DJSCC baseline:
    total_ch = round((c_base + c_sem + c_fid) * 64)

HARQ behavior is non-progressive and traditional: the same transmitted feature
tensor is sent repeatedly, noisy received copies are averaged, and the decoder
reconstructs from the combined feature.
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


class TraditionalHARQ(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.power = cfg.power
        self.total_ch = max(1, int(round((cfg.c_base + cfg.c_sem + cfg.c_fid) * 64)))
        self.width = int(cfg.get("harq_width", cfg.get("djscc_width", cfg.get("jscc_width", 128))))
        self.n_res_blocks = int(
            cfg.get("harq_res_blocks", cfg.get("djscc_res_blocks", cfg.get("jscc_res_blocks", 2)))
        )

        stem = max(self.width // 2, 32)
        self.encoder = nn.Sequential(
            ConvBlock(3, stem, stride=2),
            ConvBlock(stem, self.width, stride=2),
            ConvBlock(self.width, self.width, stride=2),
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

    def combine(self, s: torch.Tensor, channel: nn.Module, snr: torch.Tensor | float,
                rounds: int) -> torch.Tensor:
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")
        received = [channel(s, snr) for _ in range(rounds)]
        return torch.stack(received, dim=0).mean(dim=0)

    def decode(self, y_combined: torch.Tensor) -> torch.Tensor:
        return self.decoder(y_combined)

    def forward(self, x: torch.Tensor, channel: nn.Module, snr: torch.Tensor | float,
                rounds: int = 1) -> torch.Tensor:
        s = self.encode(x)
        y = self.combine(s, channel, snr, rounds)
        return self.decode(y)

    @torch.no_grad()
    def reconstruct_all_rounds(self, x: torch.Tensor, channel: nn.Module,
                               snr: torch.Tensor | float, max_rounds: int) -> list[torch.Tensor]:
        s = self.encode(x)
        outs = []
        running = None
        for r in range(1, max_rounds + 1):
            y = channel(s, snr)
            running = y if running is None else running + y
            outs.append(self.decode(running / r))
        return outs

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
