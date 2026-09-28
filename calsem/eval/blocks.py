# -*- coding: utf-8 -*-
"""评估主循环 + 实验块（B1-B5）。

evaluate() 对数据集执行三个动作（img2img），收集风险头预测 p_a 与真实失败标签 Y_a，
再计算决策、oracle 与各指标。Y_a 用 DINO cos 代理（评估期可替换为 VOC F1 detector）。
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from calsem.config import Config, resolve_device
from calsem.eval.decision import action_costs, decide, oracle_action
from calsem.eval.metrics import (ece, expected_cup, mean_latency, policy_regret,
                                 roc_auc, semantic_failure_rate)
from calsem.models.risk_head import RiskHead
from calsem.pipeline import CalSemPipeline
from calsem.train.stage2 import build_training_sample
from calsem.utils.logging import get_logger

logger = get_logger("calsem.eval")


def _progress_bar(iterable, desc: str):
    return tqdm(iterable, desc=desc, leave=False, dynamic_ncols=False, ncols=100,
                file=sys.stdout, position=0, mininterval=1.0)


def evaluate(cfg: Config, pipeline: CalSemPipeline, risk_head: RiskHead,
             loader: DataLoader, snr: float, lambda_R: float, lambda_T: float,
             seed: int = 0, progress_desc: str | None = None) -> dict:
    """执行评估，返回指标 dict。"""
    device = resolve_device(cfg)
    pipeline.jscc.to(device).eval()
    pipeline.channel.to(device)
    for p in pipeline.jscc.parameters():
        p.requires_grad_(False)
    risk_head.to(device).eval()

    preds, ys, logits = [], [], []
    progress = _progress_bar(loader, progress_desc or f"eval snr={snr:g}dB")
    for x, _ in progress:
        x = x.to(device)
        snr_t = torch.full((x.size(0),), snr, device=device)
        emb_mean, emb_var, h_rep, ch_stat, y = build_training_sample(
            pipeline, x, snr_t, cfg, seed)
        with torch.no_grad():
            p = risk_head.predict_proba(emb_mean, emb_var, h_rep, ch_stat)
            raw = risk_head.forward(emb_mean, emb_var, h_rep, ch_stat)
        preds.append(p.cpu())
        ys.append(y.cpu())
        logits.append(raw.cpu())

    p = torch.cat(preds, dim=0)          # (N,3)
    y = torch.cat(ys, dim=0)             # (N,3)
    logits = torch.cat(logits, dim=0)

    dR, dT = action_costs(cfg, p.device)
    pred_actions = decide(p, lambda_R, lambda_T, dR, dT)
    oracle_actions = oracle_action(y, lambda_R, lambda_T, dR, dT)

    base_cup = cfg.c_base
    result = {
        "snr": snr,
        "lambda_R": lambda_R,
        "lambda_T": lambda_T,
        "avg_cup": expected_cup(pred_actions, dR, base_cup),
        "sem_fail_rate": semantic_failure_rate(y, pred_actions),
        "mean_latency": mean_latency(pred_actions, dT),
        "policy_regret": policy_regret(pred_actions, oracle_actions, y,
                                       lambda_R, lambda_T, dR, dT),
        "oracle_cup": expected_cup(oracle_actions, dR, base_cup),
        "oracle_fail_rate": semantic_failure_rate(y, oracle_actions),
        # 逐动作 AUC / ECE
        "per_action_auc": [roc_auc(p[:, i].numpy(), y[:, i].numpy()) for i in range(3)],
        "per_action_ece": [ece(p[:, i].numpy(), y[:, i].numpy()) for i in range(3)],
    }
    return result


def run_block1(cfg, pipeline, risk_head, loader, snrs, lambda_R, lambda_T, seed=0) -> list[dict]:
    """B1: 主表——率-可靠-时延 Pareto + policy regret（逐 SNR）。"""
    rows = []
    for snr in snrs:
        r = evaluate(cfg, pipeline, risk_head, loader, snr, lambda_R, lambda_T, seed,
                     progress_desc=f"B1 eval snr={snr:g}dB")
        logger.info(f"[B1] snr={snr}dB CUP={r['avg_cup']:.4f} "
                    f"fail={r['sem_fail_rate']:.4f} regret={r['policy_regret']:.4f}")
        rows.append(r)
    return rows


def run_block2(cfg, pipeline, risk_head, loader, snr, lambda_R, lambda_T, seed=0) -> dict:
    """B2: novelty isolation——三输出 vs 标量 vs 信道驱动（此处返回三输出基线，标量/信道驱动由脚本扩展）。"""
    return evaluate(cfg, pipeline, risk_head, loader, snr, lambda_R, lambda_T, seed,
                    progress_desc=f"B2 eval snr={snr:g}dB")


def run_block3(cfg, pipeline, risk_head, loader, snr, seed=0) -> dict:
    """B3: 校准——raw logits vs 单温度校准的 ECE/AUC。"""
    device = resolve_device(cfg)
    pipeline.jscc.to(device).eval()
    pipeline.channel.to(device)
    for p in pipeline.jscc.parameters():
        p.requires_grad_(False)
    risk_head.to(device).eval()
    ys, cal_p, raw_p = [], [], []
    progress = _progress_bar(loader, f"B3 eval snr={snr:g}dB")
    for x, _ in progress:
        x = x.to(device)
        snr_t = torch.full((x.size(0),), snr, device=device)
        emb_mean, emb_var, h_rep, ch_stat, y = build_training_sample(pipeline, x, snr_t, cfg, seed)
        with torch.no_grad():
            cal_p.append(torch.sigmoid(risk_head.forward(emb_mean, emb_var, h_rep, ch_stat)
                                       / risk_head.temperature()).cpu())
            raw_p.append(torch.sigmoid(risk_head.forward(emb_mean, emb_var, h_rep, ch_stat)).cpu())
        ys.append(y.cpu())
    ys = torch.cat(ys, 0)
    cal = torch.cat(cal_p, 0)
    raw = torch.cat(raw_p, 0)
    return {
        "raw_ece": [ece(raw[:, i].numpy(), ys[:, i].numpy()) for i in range(3)],
        "cal_ece": [ece(cal[:, i].numpy(), ys[:, i].numpy()) for i in range(3)],
        "cal_auc": [roc_auc(cal[:, i].numpy(), ys[:, i].numpy()) for i in range(3)],
    }


def run_block4(cfg, pipeline, risk_head, loader, snr, lambda_R, lambda_T,
               k_samples: int | None = None, seed=0) -> dict:
    """B4: 消融（K 采样次数等，通过临时覆盖 cfg.k_samples 实现）。"""
    if k_samples is not None:
        old = cfg.k_samples
        cfg.k_samples = k_samples
    r = evaluate(cfg, pipeline, risk_head, loader, snr, lambda_R, lambda_T, seed)
    if k_samples is not None:
        cfg.k_samples = old
    return r
