# -*- coding: utf-8 -*-
"""HARQ 效用最大化决策 + oracle。

U_a = p_{A₀} − p_a − λ_R·ΔR_a − λ_T·ΔT_a
  - a=A₀（接受）：ΔR=0, ΔT=0（U 恒为 0）
  - a=A_sem（语义细化）：ΔR=c_sem, ΔT=T_gen
  - a=A_fid（保真细化）：ΔR=c_fid, ΔT=T_gen
选最高正效用动作，否则接受。
"""
from __future__ import annotations

import torch

from calsem.config import Config

ACTION_NAMES = ["A0", "Asem", "Afid"]


def action_costs(cfg: Config, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (ΔR, ΔT)，各 (3,)。ΔR 单位为 CUP，ΔT 为一次生成时延（相对单位）。"""
    dR = torch.tensor([0.0, cfg.c_sem, cfg.c_fid], device=device)
    # 一次 img2img 生成时延（以「单次生成」为单位，A_sem/A_fid 各需 1 次额外生成）
    dT = torch.tensor([0.0, 1.0, 1.0], device=device)
    return dR, dT


def utilities(p: torch.Tensor, lambda_R: float, lambda_T: float,
              dR: torch.Tensor, dT: torch.Tensor) -> torch.Tensor:
    """p: (B,3) 失败概率 p_a（p[:,0]=p_{A₀}）。返回 U: (B,3)，U[:,0]=0。"""
    p0 = p[:, 0:1]
    U = p0 - p - lambda_R * dR.unsqueeze(0) - lambda_T * dT.unsqueeze(0)
    U = U.clone()
    U[:, 0] = 0.0  # 接受动作效用恒 0
    return U


def decide(p: torch.Tensor, lambda_R: float, lambda_T: float,
           dR: torch.Tensor, dT: torch.Tensor) -> torch.Tensor:
    """返回动作索引 (B,)：0=A0, 1=Asem, 2=Afid。"""
    U = utilities(p, lambda_R, lambda_T, dR, dT)
    return U.argmax(dim=1)


def oracle_action(y: torch.Tensor, lambda_R: float, lambda_T: float,
                  dR: torch.Tensor, dT: torch.Tensor) -> torch.Tensor:
    """Oracle：用真实失败标签 y（(B,3)，y[:,0]=Y_{A₀}）求 a* = argmax E[Y_A0−Y_a−λ_RΔR−λ_TΔT]。

    注意：oracle 已知「每个动作后的真实结果」，是期望效用上界。
    """
    y0 = y[:, 0:1]
    U = y0 - y - lambda_R * dR.unsqueeze(0) - lambda_T * dT.unsqueeze(0)
    U = U.clone()
    U[:, 0] = 0.0
    return U.argmax(dim=1)


def chosen_action_cup(actions: torch.Tensor, dR: torch.Tensor) -> torch.Tensor:
    """每个样本选中动作的额外码率（含 base 支路基础码率）。"""
    return dR[actions]
