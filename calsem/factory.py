# -*- coding: utf-8 -*-
"""工厂函数：从配置构建完整模型栈。"""
from __future__ import annotations

from calsem.channel import make_channel
from calsem.config import Config, resolve_device
from calsem.models.frozen import FrozenDINO, FrozenSD
from calsem.models.jscc import ProgressiveJSCC
from calsem.models.risk_head import RiskHead
from calsem.pipeline import CalSemPipeline


def build_models(cfg: Config):
    """构建 (jscc, sd, dino, channel, pipeline, risk_head)。"""
    device = resolve_device(cfg)
    jscc = ProgressiveJSCC(cfg)
    sd = FrozenSD(cfg.sd_ckpt, device=device, fp16=cfg.fp16,
                  cache_dir=cfg.sd_cache_dir)
    dino = FrozenDINO(cfg.dinov2_name, device=device,
                      cache_dir=cfg.dinov2_cache_dir)
    channel = make_channel(cfg.channel_type, cfg.power)
    pipeline = CalSemPipeline(cfg, jscc, sd, dino, channel)
    risk_head = RiskHead(cfg)
    return jscc, sd, dino, channel, pipeline, risk_head
