# -*- coding: utf-8 -*-
"""Progressive JSCC branches.

The public interface intentionally stays compatible with the old implementation:
encode(), decode_base(), decode_sem(), decode_fid(), and n_params().
"""
from __future__ import annotations

import torch
import torch.nn as nn

from calsem.channel import power_normalize
from calsem.config import Config


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 3, stride: int = 1,
                 norm: bool = True):
        super().__init__()
        pad = kernel // 2
        layers: list[nn.Module] = [nn.Conv2d(cin, cout, kernel, stride, pad)]
        if norm:
            layers.append(nn.GroupNorm(1, cout))
        layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBlock(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(1, ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class _SharedEncoder(nn.Module):
    def __init__(self, width: int, n_res_blocks: int):
        super().__init__()
        stem = max(width // 2, 32)
        self.net = nn.Sequential(
            _ConvBlock(3, stem, stride=2),
            _ConvBlock(stem, width, stride=2),
            _ConvBlock(width, width, stride=2),
            *[_ResidualBlock(width) for _ in range(n_res_blocks)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EncHead(nn.Module):
    def __init__(self, width: int, ch: int, n_res_blocks: int):
        super().__init__()
        self.head = nn.Sequential(
            *[_ResidualBlock(width) for _ in range(n_res_blocks)],
            _ConvBlock(width, width),
            nn.Conv2d(width, ch, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class _LatentDecoder(nn.Module):
    def __init__(self, cin: int, cout: int, width: int, n_res_blocks: int):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBlock(cin, width),
            *[_ResidualBlock(width) for _ in range(n_res_blocks)],
            _ConvBlock(width, width),
            nn.Conv2d(width, cout, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SemDecoder(nn.Module):
    def __init__(self, cin: int, sem_embed_dim: int, sem_embed_len: int,
                 width: int, n_res_blocks: int):
        super().__init__()
        self.sem_embed_len = sem_embed_len
        self.net = nn.Sequential(
            _ConvBlock(cin, width),
            *[_ResidualBlock(width) for _ in range(n_res_blocks)],
            nn.Conv2d(width, sem_embed_dim, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pos_emb = nn.Parameter(torch.randn(1, sem_embed_len, sem_embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.net(x)
        g = self.pool(g).flatten(1)
        g = g.unsqueeze(1).expand(-1, self.sem_embed_len, -1)
        return g + self.pos_emb


class ProgressiveJSCC(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.base_ch = max(1, int(round(cfg.c_base * 64)))
        self.sem_ch = max(1, int(round(cfg.c_sem * 64)))
        self.fid_ch = max(1, int(round(cfg.c_fid * 64)))
        self.latent_ch = cfg.latent_channels
        self.width = int(cfg.get("jscc_width", 128))
        self.n_res_blocks = int(cfg.get("jscc_res_blocks", 2))
        self.power = cfg.power
        self.resolution = cfg.resolution
        self.latent_h = cfg.resolution // 8
        self.latent_w = cfg.resolution // 8

        self.shared_encoder = _SharedEncoder(self.width, self.n_res_blocks)
        self.e_base = _EncHead(self.width, self.base_ch, self.n_res_blocks)
        self.e_sem = _EncHead(self.width, self.sem_ch, self.n_res_blocks)
        self.e_fid = _EncHead(self.width, self.fid_ch, self.n_res_blocks)

        self.d_base = _LatentDecoder(self.base_ch, self.latent_ch,
                                     self.width, self.n_res_blocks)
        self.d_sem = _SemDecoder(self.sem_ch, cfg.sem_embed_dim, cfg.sem_embed_len,
                                 self.width, self.n_res_blocks)
        self.d_fid = _LatentDecoder(self.fid_ch + self.latent_ch, self.latent_ch,
                                    self.width, self.n_res_blocks)

    def encode(self, x: torch.Tensor):
        feat = self.shared_encoder(x)
        s_base = self.e_base(feat)
        s_base = power_normalize(s_base, self.power)

        s_sem = self.e_sem(feat)
        s_sem = power_normalize(s_sem, self.power)

        s_fid = self.e_fid(feat)
        s_fid = power_normalize(s_fid, self.power)
        return s_base, s_sem, s_fid

    def decode_base(self, y_base: torch.Tensor) -> torch.Tensor:
        return self.d_base(y_base)

    def decode_sem(self, y_sem: torch.Tensor) -> torch.Tensor:
        return self.d_sem(y_sem)

    def decode_fid(self, y_fid: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return self.d_fid(torch.cat([y_fid, z1], dim=1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
