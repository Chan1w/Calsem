# -*- coding: utf-8 -*-
"""Check whether z_native itself can reconstruct images.

This script bypasses JSCC and the channel:
    x -> Stable Diffusion VAE encode -> z_native
    z_native -> VAE decode
    z_native -> img2img_from_latent

The VAE decode column checks the direct autoencoder target. The img2img column
checks the exact stage-1 target latent when used by the current generation path.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from calsem.config import Config, resolve_device
from calsem.data.metrics import class_names_to_prompt
from calsem.data.voc import VOCDataset
from calsem.models.frozen import FrozenSD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("VAE", "Img2Img", "Img2Img+GTPrompt")


def resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    p2 = PROJECT_ROOT / p
    if p2.exists():
        return str(p2)
    return path


def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.float() * 0.5 + 0.5).clamp(0.0, 1.0)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(to_01(pred), to_01(target)).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = to_01(pred).unsqueeze(0)
    target = to_01(target).unsqueeze(0)
    channels = pred.size(1)
    window = torch.ones((channels, 1, 11, 11), device=pred.device, dtype=pred.dtype) / 121.0

    mu_x = F.conv2d(pred, window, padding=5, groups=channels)
    mu_y = F.conv2d(target, window, padding=5, groups=channels)
    sigma_x = F.conv2d(pred * pred, window, padding=5, groups=channels) - mu_x.pow(2)
    sigma_y = F.conv2d(target * target, window, padding=5, groups=channels) - mu_y.pow(2)
    sigma_xy = F.conv2d(pred * target, window, padding=5, groups=channels) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
             / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)))
    return score.mean().item()


def build_lpips(device: str):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("LPIPS requires `lpips`. Install it with: pip install lpips") from exc

    model = lpips.LPIPS(net="alex").to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def lpips_score(model, pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(model(pred.float().unsqueeze(0), target.float().unsqueeze(0)).item())


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = to_01(x.detach().cpu())
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def label_for(name: str, sample: dict) -> str:
    m = sample["metrics"][name]
    return f"{name}\nPSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f}\nLPIPS {m['lpips']:.3f}"


def save_comparison(sample: dict, out_path: str) -> None:
    tiles = [
        ("Original", sample["original"]),
        (label_for("VAE", sample), sample["recon"]["VAE"]),
        (label_for("Img2Img", sample), sample["recon"]["Img2Img"]),
        (label_for("Img2Img+GTPrompt", sample), sample["recon"]["Img2Img+GTPrompt"]),
    ]
    pil_images = [tensor_to_pil(tensor) for _, tensor in tiles]
    w, h = pil_images[0].size
    label_h = 64
    canvas = Image.new("RGB", (w * len(tiles), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, ((label, _), img) in enumerate(zip(tiles, pil_images)):
        x0 = i * w
        canvas.paste(img, (x0, label_h))
        draw.text((x0 + 8, 8), label, fill=(0, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


@torch.no_grad()
def infer_one(dataset, index: int, cfg: Config, sd: FrozenSD, lpips_model,
              seed: int, device: str) -> dict:
    x, labels, image_id = dataset[index]
    x = x.unsqueeze(0).to(device)
    labels = labels.to(device)

    z_native = sd.vae_encode(x)
    vae_recon = sd.vae_decode(z_native)
    generator = torch.Generator(device=device).manual_seed(seed)
    img2img_recon = sd.img2img_from_latent(
        z_native,
        strength=cfg.strength,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        prompt_embeds=None,
        generator=generator,
    )
    prompt = class_names_to_prompt(labels.unsqueeze(0))[0]
    prompt_embeds = sd.clip_text_encode([prompt], max_length=cfg.sem_embed_len)
    prompt_generator = torch.Generator(device=device).manual_seed(seed)
    prompted_recon = sd.img2img_from_latent(
        z_native,
        strength=cfg.strength,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        prompt_embeds=prompt_embeds,
        generator=prompt_generator,
    )

    original = x[0].detach()
    recon = {
        "VAE": vae_recon[0].detach(),
        "Img2Img": img2img_recon[0].detach(),
        "Img2Img+GTPrompt": prompted_recon[0].detach(),
    }
    metrics = {}
    for name, img in recon.items():
        metrics[name] = {
            "psnr": psnr(img, original),
            "ssim": ssim(img, original),
            "lpips": lpips_score(lpips_model, img, original),
        }

    return {
        "index": index,
        "image_id": image_id,
        "prompt": prompt,
        "original": original,
        "z_shape": tuple(z_native.shape),
        "recon": recon,
        "metrics": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--image-set", default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="output/vae_native_9-27_16length")
    args = ap.parse_args()

    cfg = Config.from_yaml(resolve_path(args.config))
    if args.data_root:
        cfg.data_root = args.data_root

    device = resolve_device(cfg)
    dataset = VOCDataset(cfg.data_root, args.image_set, cfg.resolution, return_id=True)
    sd = FrozenSD(cfg.sd_ckpt, device=device, fp16=cfg.fp16, cache_dir=cfg.sd_cache_dir)
    lpips_model = build_lpips(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(args.num_samples):
        index = args.index + i * args.stride
        if index >= len(dataset):
            break
        sample = infer_one(dataset, index, cfg, sd, lpips_model,
                           args.seed + i * 10000, device)
        samples.append(sample)
        out_path = out_dir / f"vae_native_{i:02d}_idx{index}.png"
        save_comparison(sample, str(out_path))

        print(f"[sample] image_id={sample['image_id']} index={index} z_shape={sample['z_shape']}")
        print(f"[prompt] {sample['prompt']!r}")
        for name in METHODS:
            m = sample["metrics"][name]
            print(f"[metric] {name:7s} PSNR={m['psnr']:.3f} "
                  f"SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f}")
        print(f"[image] saved {out_path}")

    if not samples:
        raise ValueError(f"No samples selected. index={args.index}, dataset_size={len(dataset)}")

    metrics_path = out_dir / "vae_native_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "image_id", "prompt", "method", "psnr", "ssim", "lpips"])
        writer.writeheader()
        for sample in samples:
            for name in METHODS:
                writer.writerow({
                    "index": sample["index"],
                    "image_id": sample["image_id"],
                    "prompt": sample["prompt"],
                    "method": name,
                    **sample["metrics"][name],
                })

    for name in METHODS:
        avg_psnr = sum(s["metrics"][name]["psnr"] for s in samples) / len(samples)
        avg_ssim = sum(s["metrics"][name]["ssim"] for s in samples) / len(samples)
        avg_lpips = sum(s["metrics"][name]["lpips"] for s in samples) / len(samples)
        print(f"[mean] {name:7s} PSNR={avg_psnr:.3f} "
              f"SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}")
    print(f"[done] saved {len(samples)} images and metrics: {metrics_path}")


if __name__ == "__main__":
    main()
