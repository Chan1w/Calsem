# -*- coding: utf-8 -*-
"""配置加载：YAML 配置 + 默认值合并。

用法:
    from calsem.config import Config
    cfg = Config.from_yaml("configs/stage1.yaml")
    print(cfg.resolution, cfg.snrs)
"""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml


# ----------------------------------------------------------------------------
# 默认配置（与 FINAL_PROPOSAL / EXPERIMENT_PLAN 对齐）
# ----------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    # ---- 数据 ----
    "dataset": "voc2012",
    "data_root": "/home/wchan/projects/dataset/VOC2012",          # VOC 2012 根目录（含 JPEGImages/ 与 ImageSets/）
    "image_set": "train",
    "resolution": 384,                       # pilot 实测 384px；512px 需远程 GPU
    "num_workers": 4,
    # ---- 模型 / 码率 (CUP = channel uses per pixel) ----
    # 符号以 (C, H/8, W/8) 空间结构传输，符号通道数 C = c * 64。
    "c_base": 1.0 / 4.0,                     # base branch: 16 symbol channels
    "c_sem": 1.0 / 8.0,                      # semantic branch: 8 symbol channels
    "c_fid": 1.0 / 8.0,                      # fidelity branch: 8 symbol channels
    "jscc_width": 128,
    "jscc_res_blocks": 2,
    "latent_channels": 4,                    # SD1.5 VAE latent 通道
    "sem_embed_dim": 768,                    # CLIP-ViT-L/14 text encoder 隐藏维
    "sem_embed_len": 16,                     # CLIP text encoder 序列长度
    "dinov2_embed": 384,                     # DINOv2-small CLS 维
    # ---- 信道 ----
    "channel_type": "AWGN",
    "snrs": [0, 5, 10, 15, 20],              # 训练随机采样范围（eval 逐点）
    "snr_train_min": 0.0,
    "snr_train_max": 20.0,
    "power": 1.0,                            # 每符号功率预算 P
    # ---- 生成 / 扩散 ----
    "sd_ckpt": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "dinov2_name": "facebook/dinov2-small",
    "strength": 0.8,                        # lower strength keeps reconstructions closer to JSCC latent
    "num_inference_steps": 25,
    "guidance_scale": 1.0,                   # null prompt
    "k_samples": 3,                          # K 次采样
    "fp16": True,
    # ---- 训练 ----
    "seed": 0,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 30,
    "grad_accum": 1,
    "gradient_checkpointing": True,
    "lambda_align": 1.0,
    "lambda_latent": 1.0,
    "lambda_sem_tok": 1.0,
    "lambda_sem_cos": 1.0,
    "lambda_latent_fid": 1.0,
    "lambda_soft": 0.5,
    "mc_samples": 8,                         # stage-2 软 MC 采样数 S
    "calibration": {"method": "single_temperature", "init_T": 1.0},
    # ---- HARQ 决策 ----
    "lambda_R": 0.1,                         # 码率代价系数（val split 外层选）
    "lambda_T": 0.1,                         # 时延代价系数
    "sem_fail_threshold": 0.5,               # Y_a = 1[F1 < 0.5]
    # ---- 路径 / 设备 ----
    "output_dir": "./runs",
    "device": "auto",
    # ---- 冻结权重本地路径（可选，避免服务器 HF 下载）----
    "sd_cache_dir": None,                    # 若 set，加载 SD 时用本地缓存
    "dinov2_cache_dir": None,
}


class Config:
    """支持属性访问的配置对象（dict 子类）。"""

    def __init__(self, data: dict[str, Any] | None = None):
        object.__setattr__(self, "_data", copy.deepcopy(data or {}))

    @classmethod
    def from_yaml(cls, path: str | None) -> "Config":
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        if path is not None and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            _deep_update(cfg, user)
        return cls(cfg)

    @classmethod
    def defaults(cls) -> "Config":
        return cls(copy.deepcopy(DEFAULT_CONFIG))

    # ---- 属性访问 ----
    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(f"Config has no key '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _deep_update(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = _coerce_config_value(base.get(k), v)


def _coerce_config_value(default: Any, value: Any) -> Any:
    if isinstance(value, str):
        if isinstance(default, bool):
            return value
        if isinstance(default, int) and not isinstance(default, bool):
            try:
                return int(value)
            except ValueError:
                return value
        if isinstance(default, float):
            try:
                return float(value)
            except ValueError:
                return value
    return value


def resolve_device(cfg: Config) -> str:
    import torch
    if cfg.device != "auto":
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"
