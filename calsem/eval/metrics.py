# -*- coding: utf-8 -*-
"""评估指标：CUP / 语义失败率 / 时延 / policy regret / AUC / ECE。"""
from __future__ import annotations

import numpy as np
import torch


def expected_cup(actions: torch.Tensor, dR: torch.Tensor, base_cup: float) -> float:
    """平均 CUP = 基础码率 + 选中动作的额外码率期望。"""
    return base_cup + dR[actions].mean().item()


def semantic_failure_rate(y: torch.Tensor, actions: torch.Tensor) -> float:
    """选中动作后仍语义失败的比例。y: (N,3)，actions: (N,)。"""
    return y[torch.arange(y.size(0)), actions].mean().item()


def mean_latency(actions: torch.Tensor, dT: torch.Tensor) -> float:
    return dT[actions].mean().item()


def policy_regret(pred_actions: torch.Tensor, oracle_actions: torch.Tensor,
                  y: torch.Tensor, lambda_R: float, lambda_T: float,
                  dR: torch.Tensor, dT: torch.Tensor) -> float:
    """policy regret = oracle 期望效用 − 策略期望效用。

    U_a = E[Y_A0 − Y_a] − λ_R ΔR_a − λ_T ΔT_a（用真实 y 计算期望）。
    """
    y0 = y[:, 0:1]
    E = y0 - y - lambda_R * dR.unsqueeze(0) - lambda_T * dT.unsqueeze(0)
    E[:, 0] = 0.0
    U_oracle = E[torch.arange(y.size(0)), oracle_actions]
    U_policy = E[torch.arange(y.size(0)), pred_actions]
    return (U_oracle - U_policy).mean().item()


def ece(pred: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error。pred/target: (N,) 概率与 {0,1} 标签。"""
    pred = np.asarray(pred)
    target = np.asarray(target)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (pred > bins[i]) & (pred <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(pred)) * abs(pred[mask].mean() - target[mask].mean())
    return float(ece)


def roc_auc(pred: np.ndarray, target: np.ndarray) -> float:
    """逐动作 ROC AUC。"""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    pred = np.asarray(pred)
    target = np.asarray(target)
    if len(np.unique(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, pred))
