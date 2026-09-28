# -*- coding: utf-8 -*-
"""Stage-2：条件动作结果头（软 MC 训练，区分随机性来源）。

对每个训练状态执行三个动作的 img2img（no_grad），构造：
  - 风险特征 u：K 个 DINOv2 embedding 的 mean/var pooling + representative h_rep + 信道统计；
  - 失败标签 Y：A₀ 用 realized representative 单次二值；A_sem/A_fid 用 S 次 MC 软频率。

Y_a 语义失败定义：DINO cos 代理（cos < threshold 判失败），正式评估替换为 VOC F1（冻结 detector）。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from calsem.config import Config, resolve_device
from calsem.models.frozen import cos_sim
from calsem.models.risk_head import RiskHead, channel_stats, mean_var_pooling
from calsem.pipeline import CalSemPipeline
from calsem.utils.logging import get_logger
from calsem.utils.seed import set_seed

logger = get_logger("calsem.train.stage2")


def representative(embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """embs: (B, K, D) → (h_rep, mean, var)。h_rep = nearest-to-centroid。"""
    mean, var = mean_var_pooling(embs)
    centroid = mean.unsqueeze(1)                       # (B,1,D)
    sims = (embs * centroid).sum(-1)                   # (B,K)
    idx = sims.argmax(dim=1)                           # (B,)
    h_rep = embs[torch.arange(embs.size(0)), idx]      # (B,D)
    return h_rep, mean, var
    # h_rep：代表性 embedding，nearest-to-centroid; mean:K 次 embedding 的均值; var：K 次 embedding 的方差

def _dino_failure(pipeline: CalSemPipeline, x: torch.Tensor, z_start: torch.Tensor,
                  prompt_embeds: torch.Tensor | None, seed: int, threshold: float,
                  mc_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """执行 mc_samples 次生成，返回 (失败频率 ŷ, mean DINO cos)。

    ŷ = (1/S)Σ 1[cos(DINO(x), DINO(x̂^(s))) < threshold]，DINO cos 为语义失败代理。
    """
    gt_emb = pipeline.dino.embed_tensor(x)             # (B,D)
    fails, coss = [], []
    for s in range(mc_samples):
        _, embs = pipeline.generate(z_start, prompt_embeds=prompt_embeds, K=1, seed=seed + s)
        emb = embs[:, 0, :]                            # (B,D)
        c = cos_sim(emb, gt_emb)
        fails.append((c < threshold).float())
        coss.append(c)
    fails = torch.stack(fails, dim=0).mean(dim=0)      # (B,)
    coss = torch.stack(coss, dim=0).mean(dim=0)
    return fails, coss


def build_training_sample(pipeline: CalSemPipeline, x: torch.Tensor, snr: torch.Tensor,
                          cfg: Config, seed: int):
    """构造一条 stage-2 训练样本 (u 特征, Y 标签)。

    返回 (emb_mean, emb_var, h_rep, ch_stat, y) 其中 y: (B,3)。
    """
    threshold = cfg.sem_fail_threshold  # DINO cos 代理阈值（评估用 VOC F1）

    # ---- A₀：K 次采样 → 风险特征 + realized 二值失败 ----
    z1, s_base, y_base = pipeline.transmit_base(x, snr)
    f, embs0 = pipeline.generate(z1, prompt_embeds=None, K=cfg.k_samples, seed=seed)
    h_rep, emb_mean, emb_var = representative(embs0)
    gt_emb = pipeline.dino.embed_tensor(x)
    y0 = (cos_sim(h_rep, gt_emb) < threshold).float().unsqueeze(1)   # (B,1) realized

    # ---- A_sem / A_fid：S 次软频率 ----
    c_sem = pipeline.transmit_sem(x, snr)
    y_sem, _ = _dino_failure(pipeline, x, z1, prompt_embeds=c_sem, seed=seed + 1000,
                             threshold=threshold, mc_samples=cfg.mc_samples)
    dz = pipeline.transmit_fid(x, snr, z1)
    y_fid, _ = _dino_failure(pipeline, x, z1 + dz, prompt_embeds=None, seed=seed + 2000,
                             threshold=threshold, mc_samples=cfg.mc_samples)

    y = torch.cat([y0, y_sem.unsqueeze(1), y_fid.unsqueeze(1)], dim=1)  # (B,3)
    ch_stat = channel_stats(snr, y_base, s_base)
    return emb_mean, emb_var, h_rep, ch_stat, y


def train_stage2(cfg: Config, pipeline: CalSemPipeline, risk_head: RiskHead,
                 train_loader: DataLoader, output_dir: str | None = None) -> RiskHead:
    device = resolve_device(cfg)
    set_seed(cfg.seed)
    # 冻结 jscc（pipeline.jscc），只训 risk_head
    pipeline.jscc.to(device).eval()
    pipeline.channel.to(device)
    for p in pipeline.jscc.parameters():
        p.requires_grad_(False)
    risk_head.to(device).train()
    optimizer = torch.optim.AdamW(risk_head.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    logger.info(f"stage-2 开始，mc_samples={cfg.mc_samples}，epochs={cfg.epochs}")
    for epoch in range(cfg.epochs):
        epoch_loss, n = 0.0, 0
        progress = tqdm(train_loader, desc=f"stage2 epoch {epoch+1}/{cfg.epochs}", leave=False)
        for x, _labels in progress:
            x = x.to(device)
            snr = torch.full((x.size(0),), cfg.snr_train_max, device=device)
            seed = cfg.seed + epoch * 10000 + n
            emb_mean, emb_var, h_rep, ch_stat, y = build_training_sample(
                pipeline, x, snr, cfg, seed)
            p = risk_head.predict_proba(emb_mean, emb_var, h_rep, ch_stat)
            loss = F.binary_cross_entropy(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        logger.info(f"[stage2] epoch {epoch+1}/{cfg.epochs} loss={epoch_loss/max(n,1):.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(risk_head.state_dict(), os.path.join(output_dir, "stage2_risk_head.pt"))
        logger.info(f"[stage2] 模型保存至 {output_dir}/stage2_risk_head.pt")
    return risk_head
