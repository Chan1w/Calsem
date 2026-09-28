# -*- coding: utf-8 -*-
"""信道模型：AWGN（主场景）+ Rayleigh（附录），analog JSCC。

功率契约：发送符号 s 满足 ‖s‖² / n ≤ P（每符号功率预算）。
接收 y = s + w，w ~ N(0, σ²)，σ² = P / 10^(SNR/10)（复信道实部，DeepJSCC 约定）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def power_normalize(z: torch.Tensor, P: float = 1.0) -> torch.Tensor:
    """将符号张量按 DeepJSCC 约定归一化到每符号功率 P。

    z: (B, C, H, W) 或 (B, L)。返回同形状、功率归一化后的符号。
    """
    if z.dim() not in (2, 4):
        raise ValueError(f"power_normalize expects 2D or 4D tensor, got {z.dim()}D")
    batch = z.size(0)
    k = z[0].numel()  # 每样本符号数 n
    z_flat = z.reshape(batch, -1)
    # ‖s‖² = sum(s^2)；归一化使 ‖s‖²/n = P
    norm = z_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (P * k) ** 0.5 / norm
    return (z_flat * scale).reshape_as(z)


class AWGNChannel(nn.Module):
    """AWGN 信道，SNR 可逐样本指定（训练随机采样，评估逐点）。"""

    def __init__(self, snr: float | None = None, P: float = 1.0):
        super().__init__()
        self.snr = snr
        self.P = P

    def forward(self, z: torch.Tensor, snr: torch.Tensor | float | None = None) -> torch.Tensor:
        """z: (B, C, H, W) 已功率归一化。snr: 标量或 (B,) 逐样本 SNR (dB)。"""
        if snr is None:
            snr = self.snr if self.snr is not None else 20.0
        if isinstance(snr, torch.Tensor):
            snr_b = snr.to(z.device, z.dtype).view(-1, 1, 1, 1)
        else:
            snr_b = float(snr)
        # σ² = P / 10^(SNR/10)；DeepJSCC 用 sqrt(σ²/2) 的实虚部噪声 → 直接 sqrt(σ²)
        sigma2 = self.P / (10 ** (torch.as_tensor(snr_b, dtype=z.dtype, device=z.device) / 10.0))
        noise = torch.randn_like(z) * torch.sqrt(sigma2)
        return z + noise


class RayleighChannel(nn.Module):
    """Block Rayleigh fading（附录用，简化版：对特征图前一半/后一半施加块衰落）。"""

    def __init__(self, snr: float | None = None, P: float = 1.0):
        super().__init__()
        self.snr = snr
        self.P = P

    def forward(self, z: torch.Tensor, snr: torch.Tensor | float | None = None) -> torch.Tensor:
        if snr is None:
            snr = self.snr if self.snr is not None else 20.0
        snr_b = torch.as_tensor(snr, dtype=z.dtype, device=z.device)
        sigma2 = self.P / (10 ** (snr_b / 10.0))
        h = torch.randn(z.size(0), 1, 1, 1, device=z.device, dtype=z.dtype)
        half = z.size(1) // 2
        out = z.clone()
        out[:, :half] = h * z[:, :half]
        # 另一块用独立 h2（近似两个块衰落）
        h2 = torch.randn(z.size(0), 1, 1, 1, device=z.device, dtype=z.dtype)
        out[:, half:] = h2 * z[:, half:]
        noise = torch.randn_like(z) * torch.sqrt(sigma2)
        return out + noise


def make_channel(channel_type: str, P: float = 1.0) -> nn.Module:
    if channel_type == "AWGN":
        return AWGNChannel(P=P)
    if channel_type == "Rayleigh":
        return RayleighChannel(P=P)
    raise ValueError(f"Unknown channel type: {channel_type}")
