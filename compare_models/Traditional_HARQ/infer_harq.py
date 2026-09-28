# -*- coding: utf-8 -*-
"""Visualize traditional HARQ reconstruction after each retransmission round."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calsem.channel import make_channel
from calsem.config import Config, resolve_device
from calsem.data.voc import VOCDataset
from compare_models.Traditional_HARQ.model import TraditionalHARQ


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


def save_rounds(original: torch.Tensor, recons: list[torch.Tensor],
                metrics: list[dict], out_path: str) -> None:
    tiles = [("Original", original)]
    for i, (recon, m) in enumerate(zip(recons, metrics), start=1):
        label = f"Round {i}\nPSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f}\nLPIPS {m['lpips']:.3f}"
        tiles.append((label, recon))

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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", default="checkpoints/traditional_harq.pt")
    ap.add_argument("--image-set", default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    cfg = Config.from_yaml(resolve_path(args.config))
    if args.data_root:
        cfg.data_root = args.data_root

    device = resolve_device(cfg)
    dataset = VOCDataset(cfg.data_root, args.image_set, cfg.resolution, return_id=True)
    model = TraditionalHARQ(cfg).to(device).eval()
    ckpt = resolve_path(args.ckpt)
    if ckpt is None or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Traditional HARQ checkpoint not found: {args.ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
        ckpt_rounds = int(payload.get("max_rounds", 3))
    else:
        model.load_state_dict(payload)
        ckpt_rounds = 3
    max_rounds = args.max_rounds or ckpt_rounds

    channel = make_channel(cfg.channel_type, cfg.power).to(device)
    lpips_model = build_lpips(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.num_samples):
        index = args.index + i * args.stride
        if index >= len(dataset):
            break
        x, _labels, image_id = dataset[index]
        x = x.unsqueeze(0).to(device)
        snr = torch.full((1,), args.snr, device=device)
        recons = model.reconstruct_all_rounds(x, channel, snr, max_rounds)
        original = x[0].detach()
        metrics = []
        for round_idx, recon in enumerate(recons, start=1):
            img = recon[0].detach()
            m = {
                "psnr": psnr(img, original),
                "ssim": ssim(img, original),
                "lpips": lpips_score(lpips_model, img, original),
            }
            metrics.append(m)
            rows.append({
                "index": index,
                "image_id": image_id,
                "snr": args.snr,
                "round": round_idx,
                **m,
            })
            print(f"[sample] image_id={image_id} index={index} round={round_idx} "
                  f"snr={args.snr:g}dB PSNR={m['psnr']:.3f} "
                  f"SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f}")

        out_path = out_dir / f"traditional_harq_{i:02d}_idx{index}.png"
        save_rounds(original, [r[0].detach() for r in recons], metrics, str(out_path))
        print(f"[image] saved {out_path}")

    if not rows:
        raise ValueError(f"No samples selected. index={args.index}, dataset_size={len(dataset)}")

    metrics_path = out_dir / "traditional_harq_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "image_id", "snr", "round", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    for round_idx in range(1, max_rounds + 1):
        subset = [row for row in rows if row["round"] == round_idx]
        print(f"[mean] round={round_idx} PSNR={sum(r['psnr'] for r in subset) / len(subset):.3f} "
              f"SSIM={sum(r['ssim'] for r in subset) / len(subset):.4f} "
              f"LPIPS={sum(r['lpips'] for r in subset) / len(subset):.4f}")
    print(f"[done] saved images and metrics: {metrics_path}")


if __name__ == "__main__":
    main()
