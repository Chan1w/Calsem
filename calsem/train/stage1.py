# -*- coding: utf-8 -*-
"""Stage-1：渐进编码支路 joint 训练（可微，不经扩散反传）。

Loss（FINAL_PROPOSAL §Training Plan，latent/token 对齐项的可微实现）:
    L_base = λ_latent · ‖ẑ₁ − z_native‖²
    L_sem  = λ_sem_tok · ‖ĉ_sem − CLIP_text(class_names(x))‖²
    L_fid  = λ_latent_fid · ‖(ẑ₁.detach() + Δẑ) − z_native‖²
    L_stage1 = L_base + L_sem + L_fid

语义对齐项（DINO cos）在评估阶段计算；如需训练期语义对齐，可扩展为
「可微 VAE decode + 可微 DINO」的增强路径（见 README 说明）。
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from calsem.config import Config, resolve_device
from calsem.data.metrics import class_names_to_prompt
from calsem.models.jscc import ProgressiveJSCC
from calsem.pipeline import CalSemPipeline
from calsem.utils.logging import get_logger
from calsem.utils.seed import set_seed

logger = get_logger("calsem.train.stage1")


def _draw_loss_curve(history: dict[str, list[float]], path: str) -> None:
    width, height = 1000, 640
    margin_l, margin_r, margin_t, margin_b = 86, 32, 42, 78
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    colors = {
        "total": (220, 50, 47),
        "base": (38, 139, 210),
        "sem": (133, 153, 0),
        "sem_mse": (42, 161, 152),
        "sem_cos": (108, 113, 196),
        "fid": (181, 137, 0),
    }

    keys = [k for k in ("total", "base", "sem", "sem_mse", "sem_cos", "fid")
            if history.get(k)]
    values = [v for k in keys for v in history[k]]
    if not values:
        return

    y_min = min(values)
    y_max = max(values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    epochs = len(history[keys[0]])
    denom = max(epochs - 1, 1)

    def xy(epoch_idx: int, value: float) -> tuple[float, float]:
        x = margin_l + plot_w * epoch_idx / denom
        y = margin_t + plot_h * (y_max - value) / (y_max - y_min)
        return x, y

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((margin_l, margin_t, width - margin_r, height - margin_b),
                   outline=(180, 180, 180), width=1)
    draw.text((margin_l, 14), "Stage-1 loss convergence", fill=(0, 0, 0))
    draw.text((20, margin_t), f"{y_max:.4g}", fill=(0, 0, 0))
    draw.text((20, height - margin_b - 10), f"{y_min:.4g}", fill=(0, 0, 0))
    draw.text((margin_l, height - margin_b + 24), "epoch", fill=(0, 0, 0))
    draw.text((10, height // 2), "loss", fill=(0, 0, 0))

    for i in range(6):
        y = margin_t + plot_h * i / 5
        draw.line((margin_l, y, width - margin_r, y), fill=(235, 235, 235))

    for key in keys:
        points = [xy(i, value) for i, value in enumerate(history[key])]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[key])
        else:
            draw.line(points, fill=colors[key], width=3)
        legend_x = margin_l + 130 * keys.index(key)
        draw.line((legend_x, height - 34, legend_x + 32, height - 34),
                  fill=colors[key], width=3)
        draw.text((legend_x + 40, height - 42), key, fill=(0, 0, 0))

    img.save(path)


def mse_loss_float(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE in fp32 while preserving gradients to pred."""
    target = target.to(device=pred.device)
    return F.mse_loss(pred.float(), target.float())


def clip_token_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=pred.device)
    pred_flat = pred.float().flatten(1)
    target_flat = target.float().flatten(1)
    return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1).mean()


def sample_snr(batch: int, cfg: Config, device: str) -> torch.Tensor:
    """训练期随机采样 SNR ∈ [min, max]。"""
    return torch.empty(batch, device=device).uniform_(cfg.snr_train_min, cfg.snr_train_max)


def stage1_loss(pipeline: CalSemPipeline, jscc: ProgressiveJSCC, x: torch.Tensor,
                labels: torch.Tensor, snr: torch.Tensor, cfg: Config) -> dict[str, torch.Tensor]:
    """计算 stage-1 loss（各分量 + 总量）。"""
    z_native = pipeline.native_latent(x)          # 目标 native latent（no_grad）

    # base
    z1, _, _ = pipeline.transmit_base(x, snr)
    l_base = mse_loss_float(z1, z_native)

    # semantic
    c_sem = pipeline.transmit_sem(x, snr)
    prompts = class_names_to_prompt(labels)
    clip_t = pipeline.clip_target(prompts)
    l_sem_mse = mse_loss_float(c_sem, clip_t)
    l_sem_cos = clip_token_cosine_loss(c_sem, clip_t)
    l_sem = l_sem_mse + cfg.get("lambda_sem_cos", 1.0) * l_sem_cos

    # fidelity（Δẑ 应补偿 ẑ₁ 的残差，detach ẑ₁ 使该 loss 只优化 fid 支路）
    dz = pipeline.transmit_fid(x, snr, z1.detach())
    z_fid = z1.detach() + dz
    l_fid = mse_loss_float(z_fid, z_native)

    total = (cfg.lambda_latent * l_base + cfg.lambda_sem_tok * l_sem
             + cfg.lambda_latent_fid * l_fid)
    return {
        "base": l_base,
        "sem": l_sem,
        "sem_mse": l_sem_mse,
        "sem_cos": l_sem_cos,
        "fid": l_fid,
        "total": total,
    }


def train_stage1(cfg: Config, jscc: ProgressiveJSCC, pipeline: CalSemPipeline,
                 train_loader: DataLoader, output_dir: str | None = None,
                 val_loader: DataLoader | None = None) -> ProgressiveJSCC:
    device = resolve_device(cfg)
    set_seed(cfg.seed)
    jscc.to(device).train()
    optimizer = torch.optim.AdamW(jscc.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    history: dict[str, list[float]] = {
        "total": [], "base": [], "sem": [], "sem_mse": [], "sem_cos": [], "fid": []
    }
    logger.info(f"stage-1 开始，可训练参数 {jscc.n_params():,}，epochs={cfg.epochs}")
    for epoch in range(cfg.epochs):
        epoch_losses = {
            "total": 0.0, "base": 0.0, "sem": 0.0,
            "sem_mse": 0.0, "sem_cos": 0.0, "fid": 0.0,
        }
        n = 0
        progress = tqdm(train_loader, desc=f"stage1 epoch {epoch+1}/{cfg.epochs}", leave=False)
        for x, labels in progress:
            x = x.to(device)
            snr = sample_snr(x.size(0), cfg, device)
            losses = stage1_loss(pipeline, jscc, x, labels, snr, cfg)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item() * x.size(0)
            n += x.size(0)
            progress.set_postfix(
                loss=f"{losses['total'].item():.4f}",
                base=f"{losses['base'].item():.4f}",
                sem=f"{losses['sem'].item():.4f}",
                sem_cos=f"{losses['sem_cos'].item():.4f}",
                fid=f"{losses['fid'].item():.4f}",
            )
        avg_losses = {key: value / max(n, 1) for key, value in epoch_losses.items()}
        for key, value in avg_losses.items():
            history[key].append(value)
        logger.info(f"[stage1] epoch {epoch+1}/{cfg.epochs} loss={avg_losses['total']:.4f} "
                    f"(base={avg_losses['base']:.4f} sem={avg_losses['sem']:.4f} "
                    f"sem_cos={avg_losses['sem_cos']:.4f} "
                    f"fid={avg_losses['fid']:.4f})")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "stage1_jscc_test.pt")
        torch.save(jscc.state_dict(), path)
        logger.info(f"[stage1] 模型保存至 {path}")
        history_path = os.path.join(output_dir, "stage1_loss_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        curve_path = os.path.join(output_dir, "stage1_loss_curve.png")
        _draw_loss_curve(history, curve_path)
        logger.info(f"[stage1] loss history saved to {history_path}")
        logger.info(f"[stage1] loss curve saved to {curve_path}")
    return jscc
