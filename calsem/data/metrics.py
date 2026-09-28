    # -*- coding: utf-8 -*-
"""语义失败评估指标（VOC class-set）。

Y_a = 1[F1(x, x̂_a) < 0.5]，即重建图与原图的多标签 F1 低于阈值判为语义失败。

训练/评估解耦（FINAL_PROPOSAL）:
  - stage-1 训练用 GT VOC 标签（presence_GT）作 soft 监督；
  - 评估期用独立冻结 detector 算 F1。此处提供基于 presence 向量的 F1 核心，
    以及冻结 detector 的接入点（评估阶段替换）。
"""
from __future__ import annotations

import numpy as np
import torch

from calsem.data.voc import VOC_CLASSES


VOC_CLASS_PROMPTS = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "sofa",
    "train": "train",
    "tvmonitor": "television monitor",
}


def class_name_to_prompt(name: str) -> str:
    """Return a natural-language SD-style prompt for one VOC class."""
    return f"a photo of a {VOC_CLASS_PROMPTS.get(name, name)}"


def class_names_to_natural_prompt(names: list[str]) -> str:
    """Return one natural-language prompt for a multi-label VOC class set."""
    if not names:
        return ""
    pretty = [VOC_CLASS_PROMPTS.get(name, name) for name in names]
    if len(pretty) == 1:
        return f"a photo of a {pretty[0]}"
    if len(pretty) == 2:
        joined = f"{pretty[0]} and {pretty[1]}"
    else:
        joined = ", ".join(pretty[:-1]) + f", and {pretty[-1]}"
    return f"a photo containing {joined}"


def multilabel_f1(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """多标签 F1（samples 平均）。

    pred: (N, C) ∈ {0,1} 或 [0,1] 概率（阈值 0.5 二值化）
    gt:   (N, C) ∈ {0,1}
    """
    if pred.is_floating_point():
        pred = (pred >= 0.5).float()
    gt = gt.float()
    tp = (pred * gt).sum(dim=1)
    fp = (pred * (1 - gt)).sum(dim=1)
    fn = ((1 - pred) * gt).sum(dim=1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1


def semantic_failure(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Y_a = 1[F1 < threshold]，返回 (N,) {0,1}。"""
    f1 = multilabel_f1(pred, gt)
    return (f1 < threshold).float()


def class_names_to_prompt(presence: torch.Tensor, topk: int = 10) -> list[str]:
    """把 presence 向量转成类别名 prompt（供 A_sem 文本条件用）。

    presence: (N, C)。返回 list[str]（每样本一个 "cls1, cls2, ..." 字符串）。
    """
    prompts = []
    presence = presence.detach().cpu().numpy()
    for row in presence:
        idx = np.argsort(-row)[:topk]
        names = [VOC_CLASSES[i] for i in idx if row[i] > 0.5]
        prompts.append(class_names_to_natural_prompt(names))
    return prompts
