# -*- coding: utf-8 -*-
"""核心前向流程封装：编码 → 信道 → 解码 → 生成 → 语义 embedding。

供训练（stage-1/2/3）与评估（B1-B5）复用。区分两种模式：
  - 训练：latent/token 对齐（可微，不经过扩散反传）；
  - 推理：img2img K 次采样（no_grad，最终生成图像）。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from calsem.config import Config
from calsem.models.frozen import FrozenDINO, FrozenSD
from calsem.models.jscc import ProgressiveJSCC


class CalSemPipeline:
    def __init__(self, cfg: Config, jscc: ProgressiveJSCC, sd: FrozenSD,
                 dino: FrozenDINO, channel: nn.Module):
        self.cfg = cfg
        self.jscc = jscc
        self.sd = sd
        self.dino = dino
        self.channel = channel

    # ---- 发送/接收（可微，训练用）----
    def transmit_base(self, x: torch.Tensor, snr) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A₀：x → s_base → AWGN → y_base → ẑ₁。返回 (ẑ₁, s_base, y_base)。"""
        s_base, _, _ = self.jscc.encode(x)
        y_base = self.channel(s_base, snr)
        z1 = self.jscc.decode_base(y_base)
        return z1, s_base, y_base

    def transmit_sem(self, x: torch.Tensor, snr) -> torch.Tensor:
        """A_sem：x → s_sem → AWGN → ĉ_sem。返回 ĉ_sem (B,77,768)。"""
        _, s_sem, _ = self.jscc.encode(x)
        y_sem = self.channel(s_sem, snr)
        return self.jscc.decode_sem(y_sem)

    def transmit_fid(self, x: torch.Tensor, snr, z1: torch.Tensor) -> torch.Tensor:
        """A_fid：x → s_fid → AWGN → Δẑ（依赖 ẑ₁）。返回 Δẑ (B,4,h,w)。"""
        _, _, s_fid = self.jscc.encode(x)
        y_fid = self.channel(s_fid, snr)
        return self.jscc.decode_fid(y_fid, z1)

    def native_latent(self, x: torch.Tensor) -> torch.Tensor:
        """x → z_native（冻结 VAE encoder，no_grad 目标）。"""
        return self.sd.vae_encode(x)

    def clip_target(self, prompts: Sequence[str]) -> torch.Tensor:
        """类别名 prompt → CLIP 文本嵌入（no_grad 目标）。"""
        return self.sd.clip_text_encode(prompts, max_length=self.cfg.sem_embed_len)

    # ---- 生成（推理，no_grad）----
    @torch.no_grad()
    def generate(self, z_start: torch.Tensor, prompt_embeds: torch.Tensor | None = None,
                 K: int | None = None, seed: int = 0) -> tuple[list[torch.Tensor], torch.Tensor]:
        """从 latent 起点做 K 次 img2img，返回 (imgs, embs)。

        imgs: list[K] of (B,3,H,W)；embs: (B,K,D) L2-normalized DINOv2 embedding。
        """
        K = K or self.cfg.k_samples
        imgs, embs = [], []
        for k in range(K):
            g = torch.Generator(device=z_start.device).manual_seed(seed + k)
            gen = self.sd.img2img_from_latent(
                z_start, strength=self.cfg.strength,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                prompt_embeds=prompt_embeds, generator=g)
            imgs.append(gen)
            embs.append(self.dino.embed_tensor(gen))
        embs = torch.stack(embs, dim=1)  # (B, K, D)
        return imgs, embs

    @torch.no_grad()
    def run_action(self, x: torch.Tensor, snr, action: str, z1: torch.Tensor | None = None,
                   c_sem: torch.Tensor | None = None, dz: torch.Tensor | None = None,
                   K: int | None = None, seed: int = 0):
        """执行一个动作并返回生成结果与 embedding。

        action ∈ {'A0', 'Asem', 'Afid'}。A₀ 用 null prompt；A_sem 用 ĉ_sem 条件；A_fid 用 ẑ₁+Δẑ。
        返回 (z_start, imgs, embs)。
        """
        if action == "A0":
            if z1 is None:
                z1, _, _ = self.transmit_base(x, snr)
            z_start = z1
            prompt_embeds = None
        elif action == "Asem":
            if z1 is None:
                z1, _, _ = self.transmit_base(x, snr)
            if c_sem is None:
                c_sem = self.transmit_sem(x, snr)
            z_start = z1
            prompt_embeds = c_sem
        elif action == "Afid":
            if z1 is None:
                z1, _, _ = self.transmit_base(x, snr)
            if dz is None:
                dz = self.transmit_fid(x, snr, z1)
            z_start = z1 + dz
            prompt_embeds = None
        else:
            raise ValueError(f"Unknown action: {action}")
        imgs, embs = self.generate(z_start, prompt_embeds=prompt_embeds, K=K, seed=seed)
        return z_start, imgs, embs
