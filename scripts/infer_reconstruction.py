# -*- coding: utf-8 -*-
"""Run one end-to-end CalSem inference sample and save reconstruction images.

Example:
    python scripts/infer_reconstruction.py \
        --config configs/eval.yaml \
        --data-root data/VOC2012 \
        --index 0 \
        --num-samples 4 \
        --snr 10 \
        --out scripts/runs/infer_sample.png
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from calsem.config import Config, resolve_device
from calsem.data.voc import VOCDataset
from calsem.eval.decision import ACTION_NAMES, action_costs, decide
from calsem.factory import build_models
from calsem.models.risk_head import channel_stats
from calsem.train.stage2 import representative


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    p2 = PROJECT_ROOT / p
    if p2.exists():
        return str(p2)
    return path


def _first_existing(paths: list[str]) -> str:
    for path in paths:
        resolved = _resolve_path(path)
        if resolved is not None and os.path.exists(resolved):
            return resolved
    return _resolve_path(paths[0]) or paths[0]


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.float() * 0.5 + 0.5).clamp(0.0, 1.0)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = _to_01(pred)
    target = _to_01(target)
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = _to_01(pred).unsqueeze(0)
    target = _to_01(target).unsqueeze(0)
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
        raise ImportError(
            "LPIPS metric requires the `lpips` package. Install it with: pip install lpips"
        ) from exc

    model = lpips.LPIPS(net="alex").to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def lpips_score(model, pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float().unsqueeze(0)
    target = target.float().unsqueeze(0)
    return float(model(pred, target).item())


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = _to_01(x.detach().cpu())
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def make_comparison(rows: list[list[tuple[str, torch.Tensor]]], out_path: str) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    first = tensor_to_pil(rows[0][0][1])
    w, h = first.size
    label_h = 58
    cols = len(rows[0])
    canvas = Image.new("RGB", (w * cols, (h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, row in enumerate(rows):
        y0 = row_idx * (h + label_h)
        for col_idx, (label, tensor) in enumerate(row):
            img = tensor_to_pil(tensor)
            x0 = col_idx * w
            canvas.paste(img, (x0, y0 + label_h))
            draw.text((x0 + 8, y0 + 8), label, fill=(0, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _tile_label(action_name: str, sample: dict) -> str:
    metrics = sample["metrics"][action_name]
    marker = "*" if action_name == sample["action_name"] else ""
    return (f"{action_name}{marker}\n"
            f"P {metrics['psnr']:.2f} S {metrics['ssim']:.3f}\n"
            f"L {metrics['lpips']:.3f}")


@torch.no_grad()
def infer_one(dataset, index: int, cfg: Config, pipeline, risk_head, lpips_model,
              snr_value: float, seed: int, device: str):
    x, _labels, image_id = dataset[index]
    x = x.unsqueeze(0).to(device)
    snr = torch.full((1,), snr_value, device=device)

    z1, s_base, y_base = pipeline.transmit_base(x, snr)
    imgs0, embs0 = pipeline.generate(z1, prompt_embeds=None, K=cfg.k_samples, seed=seed)
    h_rep, emb_mean, emb_var = representative(embs0)
    ch_stat = channel_stats(snr, y_base, s_base)
    probs = risk_head.predict_proba(emb_mean, emb_var, h_rep, ch_stat)

    dR, dT = action_costs(cfg, probs.device)
    action_idx = int(decide(probs, cfg.lambda_R, cfg.lambda_T, dR, dT).item())
    action_name = ACTION_NAMES[action_idx]

    c_sem = pipeline.transmit_sem(x, snr)
    dz = pipeline.transmit_fid(x, snr, z1)
    imgs_sem, _ = pipeline.generate(z1, prompt_embeds=c_sem, K=1, seed=seed + 1000)
    imgs_fid, _ = pipeline.generate(z1 + dz, prompt_embeds=None, K=1, seed=seed + 2000)

    original = x[0].detach()
    recon = {
        "A0": imgs0[0][0].detach(),
        "Asem": imgs_sem[0][0].detach(),
        "Afid": imgs_fid[0][0].detach(),
    }

    metrics = {}
    for name, img in recon.items():
        metrics[name] = {
            "psnr": psnr(img, original),
            "ssim": ssim(img, original),
            "lpips": lpips_score(lpips_model, img, original),
            "risk": float(probs[0, ACTION_NAMES.index(name)].item()),
        }

    return {
        "index": index,
        "image_id": image_id,
        "original": original,
        "recon": recon,
        "action_name": action_name,
        "metrics": metrics,
        "probs": probs[0].detach().cpu(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--image-set", default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jscc-ckpt", default=None)
    ap.add_argument("--stage2-risk-ckpt", default=None)
    ap.add_argument("--stage3-risk-ckpt", default=None)
    ap.add_argument("--out", default="output/infer_reconstruction.png")
    args = ap.parse_args()

    cfg = Config.from_yaml(_resolve_path(args.config))
    if args.data_root:
        cfg.data_root = args.data_root

    jscc_ckpt = args.jscc_ckpt or _first_existing([
        "scripts/runs/stage1/stage1_jscc.pt",
        "scripts/runs/stage1/stage1_jscc_test.pt",
        "runs/stage1/stage1_jscc.pt",
        "runs/stage1/stage1_jscc_test.pt",
    ])
    stage2_ckpt = args.stage2_risk_ckpt or _first_existing([
        "scripts/runs/stage2/stage2_risk_head.pt",
        "runs/stage2/stage2_risk_head.pt",
    ])
    stage3_ckpt = args.stage3_risk_ckpt or _first_existing([
        "scripts/runs/stage3/stage3_risk_head_cal.pt",
        "runs/stage3/stage3_risk_head_cal.pt",
    ])

    device = resolve_device(cfg)
    dataset = VOCDataset(cfg.data_root, args.image_set, cfg.resolution, return_id=True)

    jscc, _sd, _dino, channel, pipeline, risk_head = build_models(cfg)
    jscc.load_state_dict(torch.load(jscc_ckpt, map_location="cpu"))
    print(f"[model] loaded JSCC: {jscc_ckpt}")

    if os.path.exists(stage2_ckpt):
        risk_head.load_state_dict(torch.load(stage2_ckpt, map_location="cpu"))
        print(f"[model] loaded stage-2 risk head: {stage2_ckpt}")
    if os.path.exists(stage3_ckpt):
        risk_head.load_state_dict(torch.load(stage3_ckpt, map_location="cpu"))
        print(f"[model] loaded stage-3 calibrated risk head: {stage3_ckpt}")
    elif not os.path.exists(stage2_ckpt):
        raise FileNotFoundError(f"No risk-head checkpoint found: {stage2_ckpt} or {stage3_ckpt}")

    pipeline.jscc.to(device).eval()
    pipeline.channel.to(device)
    risk_head.to(device).eval()
    lpips_model = build_lpips(device)

    samples = []
    for i in range(args.num_samples):
        index = args.index + i * args.stride
        if index >= len(dataset):
            break
        sample = infer_one(dataset, index, cfg, pipeline, risk_head, lpips_model,
                           args.snr, args.seed + i * 10000, device)
        samples.append(sample)

        print(f"[sample] image_id={sample['image_id']} index={index} snr={args.snr:g}dB")
        print("[risk] " + ", ".join(
            f"{name}={sample['probs'][j].item():.4f}" for j, name in enumerate(ACTION_NAMES)))
        print(f"[decision] selected={sample['action_name']}")
        for name in ACTION_NAMES:
            m = sample["metrics"][name]
            marker = " <- selected" if name == sample["action_name"] else ""
            print(f"[metric] {name:4s} PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
                  f"LPIPS={m['lpips']:.4f} risk={m['risk']:.4f}{marker}")

    if not samples:
        raise ValueError(f"No samples selected. index={args.index}, dataset_size={len(dataset)}")

    for name in ACTION_NAMES:
        avg_psnr = sum(s["metrics"][name]["psnr"] for s in samples) / len(samples)
        avg_ssim = sum(s["metrics"][name]["ssim"] for s in samples) / len(samples)
        avg_lpips = sum(s["metrics"][name]["lpips"] for s in samples) / len(samples)
        print(f"[mean] {name:4s} PSNR={avg_psnr:.3f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}")

    image_rows = []
    for sample in samples:
        selected = sample["action_name"]
        image_rows.append([
            (f"Original\nidx={sample['index']} id={sample['image_id']}", sample["original"]),
            (_tile_label("A0", sample), sample["recon"]["A0"]),
            (_tile_label("Asem", sample), sample["recon"]["Asem"]),
            (_tile_label("Afid", sample), sample["recon"]["Afid"]),
            (f"Selected\n{_tile_label(selected, sample)}", sample["recon"][selected]),
        ])

    make_comparison(image_rows, args.out)
    print(f"[done] saved comparison image: {args.out}")


if __name__ == "__main__":
    main()
