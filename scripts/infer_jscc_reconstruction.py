# -*- coding: utf-8 -*-
"""Visualize JSCC-only reconstruction quality for base / semantic / fidelity branches.

Example:
    python scripts/infer_jscc_reconstruction.py \
        --config configs/eval.yaml \
        --data-root data/VOC2012 \
        --jscc-ckpt scripts/runs/stage1/stage1_jscc_test.pt \
        --num-samples 5 \
        --out-dir scripts/runs/jscc_recon
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from calsem.config import Config, resolve_device
from calsem.data.metrics import class_name_to_prompt, class_names_to_prompt
from calsem.data.voc import VOC_CLASSES, VOCDataset
from calsem.factory import build_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("Base", "Sem", "Fid")


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


def save_comparison(sample: dict, out_path: str) -> None:
    tiles = [
        ("Original", sample["original"]),
        (label_for("Base", sample), sample["recon"]["Base"]),
        (label_for("Sem", sample), sample["recon"]["Sem"]),
        (label_for("Fid", sample), sample["recon"]["Fid"]),
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


def label_for(branch: str, sample: dict) -> str:
    m = sample["metrics"][branch]
    return f"{branch}\nPSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f}\nLPIPS {m['lpips']:.3f}"


@torch.no_grad()
def prompt_nearest_neighbors(pipeline, prompt_embeds: torch.Tensor, labels: torch.Tensor,
                             topk: int = 5) -> dict:
    gt_prompt = class_names_to_prompt(labels.unsqueeze(0))[0]
    candidates = [""] + [class_name_to_prompt(name) for name in VOC_CLASSES]
    if gt_prompt and gt_prompt not in candidates:
        candidates.append(gt_prompt)

    candidate_embeds = pipeline.sd.clip_text_encode(
        candidates, max_length=prompt_embeds.size(1))
    query = F.normalize(prompt_embeds.reshape(prompt_embeds.size(0), -1).float(), dim=-1)
    keys = F.normalize(candidate_embeds.reshape(candidate_embeds.size(0), -1).float(), dim=-1)
    sims = (query @ keys.t())[0]
    vals, idxs = torch.topk(sims, k=min(topk, len(candidates)))
    nearest = [
        {"prompt": candidates[int(i)], "score": float(v)}
        for v, i in zip(vals.detach().cpu(), idxs.detach().cpu())
    ]

    gt_score = None
    if gt_prompt:
        gt_idx = candidates.index(gt_prompt)
        gt_score = float(sims[gt_idx].detach().cpu())

    return {
        "gt_prompt": gt_prompt,
        "gt_score": gt_score,
        "nearest": nearest,
    }


@torch.no_grad()
def infer_one(dataset, index: int, cfg: Config, pipeline, lpips_model,
              snr_value: float, seed: int, device: str) -> dict:
    x, labels, image_id = dataset[index]
    x = x.unsqueeze(0).to(device)
    labels = labels.to(device)
    snr = torch.full((1,), snr_value, device=device)

    z1, _s_base, _y_base = pipeline.transmit_base(x, snr)
    c_sem = pipeline.transmit_sem(x, snr)
    dz = pipeline.transmit_fid(x, snr, z1)
    sem_prompt = prompt_nearest_neighbors(pipeline, c_sem, labels)

    base_imgs, _ = pipeline.generate(z1, prompt_embeds=None, K=1, seed=seed)
    sem_imgs, _ = pipeline.generate(z1, prompt_embeds=c_sem, K=1, seed=seed + 1000)
    fid_imgs, _ = pipeline.generate(z1 + dz, prompt_embeds=None, K=1, seed=seed + 2000)

    original = x[0].detach()
    recon = {
        "Base": base_imgs[0][0].detach(),
        "Sem": sem_imgs[0][0].detach(),
        "Fid": fid_imgs[0][0].detach(),
    }
    metrics = {}
    for branch, img in recon.items():
        metrics[branch] = {
            "psnr": psnr(img, original),
            "ssim": ssim(img, original),
            "lpips": lpips_score(lpips_model, img, original),
        }

    return {
        "index": index,
        "image_id": image_id,
        "original": original,
        "recon": recon,
        "metrics": metrics,
        "sem_prompt": sem_prompt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--image-set", default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--snr", type=float, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jscc-ckpt", default="/home/wchan/projects/CalSem/scripts/runs/stage1/test/stage1_jscc_test.pt")
    ap.add_argument("--out-dir", default="scripts/runs/jscc_recon_9_27_10snr")
    args = ap.parse_args()

    cfg = Config.from_yaml(resolve_path(args.config))
    if args.data_root:
        cfg.data_root = args.data_root

    device = resolve_device(cfg)
    dataset = VOCDataset(cfg.data_root, args.image_set, cfg.resolution, return_id=True)
    jscc, _sd, _dino, _channel, pipeline, _risk_head = build_models(cfg)

    jscc_ckpt = resolve_path(args.jscc_ckpt)
    if jscc_ckpt is None or not os.path.exists(jscc_ckpt):
        raise FileNotFoundError(f"JSCC checkpoint not found: {args.jscc_ckpt}")
    jscc.load_state_dict(torch.load(jscc_ckpt, map_location="cpu"))
    print(f"[model] loaded JSCC: {jscc_ckpt}")

    pipeline.jscc.to(device).eval()
    pipeline.channel.to(device)
    lpips_model = build_lpips(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(args.num_samples):
        index = args.index + i * args.stride
        if index >= len(dataset):
            break
        sample = infer_one(dataset, index, cfg, pipeline, lpips_model,
                           args.snr, args.seed + i * 10000, device)
        samples.append(sample)
        out_path = out_dir / f"jscc_recon_{i:02d}_idx{index}.png"
        save_comparison(sample, str(out_path))

        print(f"[sample] image_id={sample['image_id']} index={index} snr={args.snr:g}dB")
        sem_prompt = sample["sem_prompt"]
        print(f"[sem target] {sem_prompt['gt_prompt']!r} score={sem_prompt['gt_score']}")
        print("[sem nearest] " + " | ".join(
            f"{item['prompt']!r}:{item['score']:.4f}" for item in sem_prompt["nearest"]))
        for branch in BRANCHES:
            m = sample["metrics"][branch]
            print(f"[metric] {branch:4s} PSNR={m['psnr']:.3f} "
                  f"SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f}")
        print(f"[image] saved {out_path}")

    if not samples:
        raise ValueError(f"No samples selected. index={args.index}, dataset_size={len(dataset)}")

    metrics_path = out_dir / "jscc_recon_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "image_id", "branch", "psnr", "ssim", "lpips",
                           "sem_target_prompt", "sem_target_score",
                           "sem_nearest_prompts"])
        writer.writeheader()
        for sample in samples:
            sem_prompt = sample["sem_prompt"]
            nearest = "; ".join(
                f"{item['prompt']}:{item['score']:.4f}" for item in sem_prompt["nearest"])
            for branch in BRANCHES:
                row = {
                    "index": sample["index"],
                    "image_id": sample["image_id"],
                    "branch": branch,
                    "sem_target_prompt": sem_prompt["gt_prompt"],
                    "sem_target_score": sem_prompt["gt_score"],
                    "sem_nearest_prompts": nearest,
                    **sample["metrics"][branch],
                }
                writer.writerow(row)

    for branch in BRANCHES:
        avg_psnr = sum(s["metrics"][branch]["psnr"] for s in samples) / len(samples)
        avg_ssim = sum(s["metrics"][branch]["ssim"] for s in samples) / len(samples)
        avg_lpips = sum(s["metrics"][branch]["lpips"] for s in samples) / len(samples)
        print(f"[mean] {branch:4s} PSNR={avg_psnr:.3f} "
              f"SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}")
    print(f"[done] saved {len(samples)} images and metrics: {metrics_path}")


if __name__ == "__main__":
    main()
