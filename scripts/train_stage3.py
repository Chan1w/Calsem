# -*- coding: utf-8 -*-
"""Stage-3 单温度校准入口。"""
import argparse

from torch.utils.data import DataLoader

from calsem.config import Config
from calsem.data.voc import VOCDataset, collate_fn
from calsem.factory import build_models
from calsem.train.stage3 import calibrate_temperature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/wchan/projects/CalSem/configs/stage3.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--jscc-ckpt", default=None)
    ap.add_argument("--risk-head-ckpt", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.data_root = args.data_root
    output_dir = args.output_dir or cfg.output_dir

    cal_ds = VOCDataset(cfg.data_root, "val", cfg.resolution)  # cal split 用 val
    cal_loader = DataLoader(cal_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn)
    print(f"[data] cal 样本数: {len(cal_ds)}")

    jscc, sd, dino, channel, pipeline, risk_head = build_models(cfg)
    import torch
    if args.jscc_ckpt:
        jscc.load_state_dict(torch.load(args.jscc_ckpt, map_location="cpu"))
    if args.risk_head_ckpt:
        risk_head.load_state_dict(torch.load(args.risk_head_ckpt, map_location="cpu"))

    calibrate_temperature(cfg, pipeline, risk_head, cal_loader, output_dir=output_dir)


if __name__ == "__main__":
    main()
