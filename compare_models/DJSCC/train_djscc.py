# -*- coding: utf-8 -*-
"""Train the traditional DJSCC baseline on the same data/config as CalSem."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calsem.channel import make_channel
from calsem.config import Config, resolve_device
from calsem.data.voc import VOCDataset, collate_fn
from calsem.utils.seed import set_seed
from compare_models.DJSCC.model import DeepJSCC


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


def sample_snr(batch: int, cfg: Config, device: str) -> torch.Tensor:
    return torch.empty(batch, device=device).uniform_(cfg.snr_train_min, cfg.snr_train_max)


def draw_loss_curve(losses: list[float], path: str) -> None:
    if not losses:
        return
    width, height = 900, 560
    margin_l, margin_r, margin_t, margin_b = 78, 32, 42, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_min, y_max = min(losses), max(losses)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad
    denom = max(len(losses) - 1, 1)

    def xy(i: int, value: float) -> tuple[float, float]:
        x = margin_l + plot_w * i / denom
        y = margin_t + plot_h * (y_max - value) / (y_max - y_min)
        return x, y

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((margin_l, margin_t, width - margin_r, height - margin_b),
                   outline=(180, 180, 180), width=1)
    draw.text((margin_l, 14), "DJSCC train loss", fill=(0, 0, 0))
    draw.text((16, margin_t), f"{y_max:.4g}", fill=(0, 0, 0))
    draw.text((16, height - margin_b - 10), f"{y_min:.4g}", fill=(0, 0, 0))
    points = [xy(i, value) for i, value in enumerate(losses)]
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(38, 139, 210))
    else:
        draw.line(points, fill=(38, 139, 210), width=3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default="compare_models/DJSCC/runs")
    args = ap.parse_args()

    cfg = Config.from_yaml(resolve_path(args.config))
    if args.data_root:
        cfg.data_root = args.data_root

    device = resolve_device(cfg)
    set_seed(cfg.seed)
    train_ds = VOCDataset(cfg.data_root, "train", cfg.resolution)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn)
    print(f"[data] train samples: {len(train_ds)}")

    model = DeepJSCC(cfg).to(device).train()
    channel = make_channel(cfg.channel_type, cfg.power).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    print(f"[model] DJSCC params={model.n_params():,} tx_channels={model.total_ch}")
    print(f"[budget] c_total={cfg.c_base + cfg.c_sem + cfg.c_fid:.4f} "
          f"channels={model.total_ch} spatial={cfg.resolution // 8}x{cfg.resolution // 8}")

    history = []
    for epoch in range(cfg.epochs):
        epoch_loss, n = 0.0, 0
        progress = tqdm(train_loader, desc=f"DJSCC epoch {epoch+1}/{cfg.epochs}", leave=False)
        for x, _labels in progress:
            x = x.to(device)
            snr = sample_snr(x.size(0), cfg, device)
            recon = model(x, channel, snr)
            loss = F.mse_loss(recon.float(), x.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
            progress.set_postfix(loss=f"{loss.item():.5f}")

        avg = epoch_loss / max(n, 1)
        history.append(avg)
        print(f"[DJSCC] epoch {epoch+1}/{cfg.epochs} loss={avg:.6f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "djscc.pt"
    torch.save(model.state_dict(), ckpt_path)
    with open(out_dir / "djscc_loss_history.json", "w", encoding="utf-8") as f:
        json.dump({"mse": history}, f, indent=2)
    draw_loss_curve(history, str(out_dir / "djscc_loss_curve.png"))
    print(f"[done] saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
