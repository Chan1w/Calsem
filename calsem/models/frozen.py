# -*- coding: utf-8 -*-
"""冻结组件封装：Stable Diffusion v1.5（VAE/UNet/text-encoder）+ DINOv2。

全部冻结复用（FINAL_PROPOSAL 冻结底座）：
  - SD1.5 VAE      : z_native = VAE_enc(x)，scaling_factor=0.18215
  - SD1.5 UNet     : img2img 随机解码（K 次采样）
  - SD1.5 text-enc : CLIP-ViT-L/14，77×768 文本嵌入空间（A_sem 条件注入）
  - DINOv2-small   : 唯一 runtime 语义 embedding 源（384-d CLS）
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from calsem.utils.logging import get_logger

logger = get_logger("calsem.frozen")


# ----------------------------------------------------------------------------
# Frozen SD 1.5
# ----------------------------------------------------------------------------
class FrozenSD:
    """冻结 Stable Diffusion v1.5，提供 latent 编解码 + img2img（可注入文本条件）。"""

    def __init__(self, ckpt: str, device: str = "cuda", fp16: bool = True,
                 cache_dir: str | None = None):
        from diffusers import StableDiffusionImg2ImgPipeline
        dtype = torch.float16 if fp16 else torch.float32
        load_kwargs = {"torch_dtype": dtype, "safety_checker": None,
                       "requires_safety_checker": False, "local_files_only": True}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        try:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(ckpt, **load_kwargs)
        except Exception:
            # 社区镜像回退
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", **load_kwargs)
        self.pipe = self.pipe.to(device)
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass
        self.device = device
        self.dtype = dtype
        self.scaling_factor = self.pipe.vae.config.scaling_factor

        # 冻结
        for m in (self.pipe.vae, self.pipe.unet, self.pipe.text_encoder):
            m.requires_grad_(False)
            m.eval()

    # ---- VAE 编解码（latent 空间）----
    @torch.no_grad()
    def vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) ∈ [-1,1] → z_native (B,4,h,w)（已乘 scaling_factor）。"""
        z = self.pipe.vae.encode(x.to(self.device, self.dtype)).latent_dist.sample()
        return z * self.scaling_factor

    @torch.no_grad()
    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B,4,h,w) scaled latent → x (B,3,H,W) ∈ [-1,1]。"""
        z = z.to(self.device, self.dtype) / self.scaling_factor
        x = self.pipe.vae.decode(z).sample
        return x

    # ---- 文本条件 ----
    @torch.no_grad()
    def clip_text_encode(self, prompts: Sequence[str], max_length: int | None = None) -> torch.Tensor:
        """prompts: list[str] → (B,77,768) last hidden state（CLIP-ViT-L/14）。"""
        max_length = max_length or self.pipe.tokenizer.model_max_length
        tok = self.pipe.tokenizer(
            list(prompts), padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.pipe.text_encoder(**tok).last_hidden_state
        return out

    @torch.no_grad()
    def null_prompt_embeds(self, batch: int, max_length: int | None = None) -> torch.Tensor:
        """空文本（null prompt）的 (B,77,768) embedding。"""
        return self.clip_text_encode([""] * batch, max_length=max_length)

    # ---- img2img（从 latent 起点，可注入文本条件）----
    @torch.no_grad()
    def img2img_from_latent(self, z_start: torch.Tensor, strength: float,
                            num_inference_steps: int = 10, guidance_scale: float = 1.0,
                            prompt_embeds: torch.Tensor | None = None,
                            generator: torch.Generator | None = None) -> torch.Tensor:
        """从 latent 起点（z_start，scaled）做 img2img，返回 (B,3,H,W) ∈ [-1,1]。

        prompt_embeds=None → 用 null prompt（A₀/A_fid 独占）；否则注入条件（A_sem）。
        """
        z_start = z_start.to(self.device, self.dtype)
        batch = z_start.size(0)
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        # 根据 strength 决定起始 timestep
        init_step = min(int(num_inference_steps * strength), num_inference_steps - 1)

        latents = z_start / self.scaling_factor  # 转回 unscaled
        noise = torch.randn(latents.shape, generator=generator,
                            device=self.device, dtype=self.dtype)
        latents = scheduler.add_noise(latents, noise, timesteps[init_step])

        if prompt_embeds is None:
            prompt_embeds = self.null_prompt_embeds(batch)
        prompt_embeds = prompt_embeds.to(self.device, self.dtype)

        for t in timesteps[init_step:]:
            latent_model_input = latents  # guidance_scale=1.0 无需 CFG concat
            noise_pred = self.pipe.unet(latent_model_input, t,
                                        encoder_hidden_states=prompt_embeds).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        image = self.pipe.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def img2img_from_image(self, init_image: Image.Image, strength: float,
                           num_inference_steps: int = 25, guidance_scale: float = 1.0,
                           prompt: str = "", generator: torch.Generator | None = None) -> Image.Image:
        """标准 img2img（PIL 图像起点），供定性/调试用。"""
        out = self.pipe(prompt=prompt, image=init_image, strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps, generator=generator)
        return out.images[0]


# ----------------------------------------------------------------------------
# Frozen DINOv2
# ----------------------------------------------------------------------------
class FrozenDINO(nn.Module):
    """冻结 DINOv2-small，语义 embedding 源。"""

    def __init__(self, name: str = "facebook/dinov2-small", device: str = "cuda",
                 cache_dir: str | None = None):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
        kw = {"local_files_only": True}
        if cache_dir:
            kw["cache_dir"] = cache_dir
        self.model = AutoModel.from_pretrained(name, **kw).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def embed_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) ∈ [-1,1] → (B,384) L2-normalized CLS embedding。"""
        # DINOv2 期望 [0,1] 图像，processor 会做归一化；这里用 processor 处理 PIL，
        # 对 tensor 则手动标准化到 ImageNet mean/std。
        imgs = [(im * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() for im in x]
        imgs_pil = [Image.fromarray((im * 255).astype(np.uint8)) for im in imgs]
        return self.embed_pil(imgs_pil)

    @torch.no_grad()
    def embed_pil(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """images: list[PIL] → (B,384) L2-normalized CLS embedding。"""
        inp = self.processor(images=list(images), return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp)
        v = out.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(v, dim=-1)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """两批 L2-normalized embedding 的 cosine 相似度。a:(B,D), b:(B,D) → (B,)。"""
    return (a * b).sum(-1)
