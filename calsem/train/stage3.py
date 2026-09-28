# -*- coding: utf-8 -*-
"""Stage-3：单温度校准（保序）。

在 calibration split 上用 NLL 拟合单温度 T：p_a = sigmoid(logits / T)。
单调变换严格保序，不损害排序（policy regret 验证）。Brier 仅作评估指标。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from calsem.config import Config, resolve_device
from calsem.models.risk_head import RiskHead
from calsem.train.stage2 import build_training_sample
from calsem.pipeline import CalSemPipeline
from calsem.utils.logging import get_logger

logger = get_logger("calsem.train.stage3")


def calibrate_temperature(cfg: Config, pipeline: CalSemPipeline, risk_head: RiskHead,
                          cal_loader: DataLoader, output_dir: str | None = None) -> float:
    """拟合单温度 T，返回 T 值。只优化 risk_head.log_T，其余冻结。"""
    device = resolve_device(cfg)
    pipeline.jscc.to(device).eval()
    pipeline.channel.to(device)
    for p in pipeline.jscc.parameters():
        p.requires_grad_(False)
    risk_head.to(device).eval()
    for p in risk_head.parameters():
        p.requires_grad_(False)
    risk_head.log_T.requires_grad_(True)

    opt = torch.optim.Adam([risk_head.log_T], lr=1e-2)
    n_steps = min(200, len(cal_loader) * 5)
    logger.info(f"stage-3 单温度校准，{n_steps} 步")

    cal_iter = iter(cal_loader)
    progress = tqdm(range(n_steps), desc="stage3 calibration", leave=False)
    for step in progress:
        try:
            x, _ = next(cal_iter)
        except StopIteration:
            cal_iter = iter(cal_loader)
            x, _ = next(cal_iter)
        x = x.to(device)
        snr = torch.full((x.size(0),), cfg.snr_train_max, device=device)
        emb_mean, emb_var, h_rep, ch_stat, y = build_training_sample(
            pipeline, x, snr, cfg, seed=cfg.seed + step)
        p = risk_head.predict_proba(emb_mean, emb_var, h_rep, ch_stat)
        loss = F.binary_cross_entropy(p, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        progress.set_postfix(loss=f"{loss.item():.4f}", T=f"{risk_head.temperature().item():.4f}")

    T = risk_head.temperature().item()
    logger.info(f"[stage3] 拟合温度 T={T:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(risk_head.state_dict(), os.path.join(output_dir, "stage3_risk_head_cal.pt"))
    return T
