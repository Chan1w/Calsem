# -*- coding: utf-8 -*-
"""Pascal VOC 2012 数据集加载。

目录结构（标准 VOCdevkit 解压后）:
    VOC2012/
      JPEGImages/                 # 图像 (*.jpg)
      ImageSets/Main/train.txt    # 图像 ID（每行一个）
      ImageSets/Main/val.txt
      ImageSets/Main/{class}_train.txt   # 每行 "ID 1/-1/0"（多标签分类标注）

返回: image ∈ [-1,1] (3,H,W) + 20 类 GT presence 多标签 {0,1}。
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# VOC 2012 20 类（固定顺序）
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

CLASS2IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


def _read_image_ids(root: str, image_set: str) -> list[str]:
    """读取 ImageSets/Main/{image_set}.txt 中的图像 ID 列表。"""
    path = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _read_class_labels(root: str, image_set: str) -> dict[str, list[int]]:
    """读取每个类别的 {class}_{image_set}.txt，返回 {image_id: [20 维 0/1]}。

    VOC 标注：+1 正例，-1 难例/未标注（视为负），0 负例。
    cal 是 train 的子集，其类别标注复用 train 的 {class}_train.txt。
    """
    label_set = "train" if image_set == "cal" else image_set
    id2labels: dict[str, list[int]] = {}
    for c in VOC_CLASSES:
        path = os.path.join(root, "ImageSets", "Main", f"{c}_{label_set}.txt")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                img_id, lab = parts[0], int(parts[1])
                if img_id not in id2labels:
                    id2labels[img_id] = [0] * len(VOC_CLASSES)
                if lab == 1:
                    id2labels[img_id][CLASS2IDX[c]] = 1
    return id2labels


def make_cal_split(root: str, num_cal: int = 800, seed: int = 42) -> list[str]:
    """从 train.txt 用固定 seed 划出 cal split，写入 ImageSets/Main/cal.txt（幂等）。

    返回 cal 的图像 ID 列表。用于 stage-3 校准，与 stage-2 训练不重叠
    （stage-2 训练时通过 exclude_ids 排除这批 id）。
    """
    cal_path = os.path.join(root, "ImageSets", "Main", "cal.txt")
    if os.path.exists(cal_path):
        with open(cal_path, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        if ids:
            return ids
    train_ids = _read_image_ids(root, "train")
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_ids))
    cal_ids = [train_ids[i] for i in idx[:num_cal]]
    os.makedirs(os.path.dirname(cal_path), exist_ok=True)
    with open(cal_path, "w") as f:
        f.write("\n".join(cal_ids) + "\n")
    return cal_ids


class VOCDataset(Dataset):
    def __init__(self, root: str, image_set: str = "train", resolution: int = 384,
                 return_id: bool = False, exclude_ids=None):
        self.root = root
        self.image_set = image_set
        self.resolution = resolution
        self.return_id = return_id
        self.ids = _read_image_ids(root, image_set)
        if exclude_ids is not None:
            excl = set(exclude_ids)
            self.ids = [i for i in self.ids if i not in excl]
        self.id2labels = _read_class_labels(root, image_set)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img_path = os.path.join(self.root, "JPEGImages", f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        # 保持宽高比 resize 到 resolution（中心裁剪会丢失语义，这里直接 resize）
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        x = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1)          # (3,H,W) ∈ [0,1]
        x = x * 2.0 - 1.0               # → [-1,1]（VAE/SD 期望）
        labels = torch.tensor(self.id2labels.get(img_id, [0] * len(VOC_CLASSES)),
                              dtype=torch.float32)
        if self.return_id:
            return x, labels, img_id
        return x, labels


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys
