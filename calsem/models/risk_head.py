# -*- coding: utf-8 -*-
"""风险编码器 + 三输出动作风险头（核心贡献，可训练 ~3-5M）。

输入（FINAL_PROPOSAL）:
  - K 个 DINOv2 embedding 的 mean pooling（语义中心）
  - K 个 DINOv2 embedding 的 variance pooling（语义 disagreement，核心特征）
  - representative embedding h_rep（保留 A₀ 输出 identity）
  - 信道统计（SNR/CSI/residual）

输出:
  三输出 logits → 单温度 scaling → p_a = P(Y_a=1 | u, a)，a ∈ {A₀, A_sem, A_fid}。
  三个动作失败概率彼此独立，用 sigmoid 而非 softmax。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from calsem.config import Config


class RiskHead(nn.Module):
    def __init__(self, cfg: Config, channel_stat_dim: int | None = None):
        super().__init__()
        d = cfg.dinov2_embed  # 384
        if channel_stat_dim is None:
            channel_stat_dim = cfg.get("channel_stat_dim", 2)
        in_dim = d * 3 + channel_stat_dim  # mean + var + h_rep + channel_stat
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),           # 三输出 logits（A₀/A_sem/A_fid）
        )
        # 单温度 T：单调变换、严格保序；stage-3 在 calibration split 上拟合
        self.log_T = nn.Parameter(torch.zeros(1))  # T = exp(log_T) > 0

    def forward(self, emb_mean: torch.Tensor, emb_var: torch.Tensor,
                h_rep: torch.Tensor, channel_stat: torch.Tensor) -> torch.Tensor:
        """返回三输出 logits (B, 3)。"""
        x = torch.cat([emb_mean, emb_var, h_rep, channel_stat], dim=-1)
        return self.mlp(x)

    def predict_proba(self, emb_mean: torch.Tensor, emb_var: torch.Tensor,
                      h_rep: torch.Tensor, channel_stat: torch.Tensor) -> torch.Tensor:
        """返回 p_a = sigmoid(logits / T) ∈ (0,1)，(B, 3)。"""
        logits = self.forward(emb_mean, emb_var, h_rep, channel_stat)
        T = torch.exp(self.log_T).clamp(min=1e-3)
        return torch.sigmoid(logits / T)

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_T).clamp(min=1e-3)


def mean_var_pooling(embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """embs: (B, K, D) → (mean, variance) 各 (B, D)。"""
    mean = embs.mean(dim=1)
    var = embs.var(dim=1, unbiased=False)
    return mean, var


def channel_stats(snr: torch.Tensor, y_base: torch.Tensor | None = None,
                  s_base: torch.Tensor | None = None) -> torch.Tensor:
    """构造信道统计特征。

    snr: (B,) dB。默认返回 (B,1)；若给定 y_base/s_base 则追加归一化残差功率（诊断用）。
    """
    snr = snr.float().view(-1, 1)
    feats = [snr]
    if y_base is not None and s_base is not None:
        resid = (y_base - s_base).pow(2).mean(dim=(1, 2, 3), keepdim=False).view(-1, 1)
        pwr = s_base.pow(2).mean(dim=(1, 2, 3), keepdim=False).view(-1, 1).clamp(min=1e-9)
        feats.append((resid / pwr).log1p())
    return torch.cat(feats, dim=-1)
