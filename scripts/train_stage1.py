# -*- coding: utf-8 -*-
"""Stage-1 渐进编码支路训练入口。

用法:
    python scripts/train_stage1.py --config configs/stage1.yaml
"""
import argparse

from torch.utils.data import DataLoader

from calsem.config import Config
from calsem.data.voc import VOCDataset, collate_fn
from calsem.factory import build_models
from calsem.train.stage1 import train_stage1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--data-root", default=None, help="VOC2012 根目录")
    ap.add_argument("--output-dir", default="./checkpoint/stage1/9-28-16length")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.data_root = args.data_root
    output_dir = args.output_dir or cfg.output_dir

    train_ds = VOCDataset(cfg.data_root, "train", cfg.resolution)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn)
    print(f"[data] train 样本数: {len(train_ds)}")

    jscc, sd, dino, channel, pipeline, risk_head = build_models(cfg)
    print(f"[model] ProgressiveJSCC 可训练参数: {jscc.n_params():,}")

    train_stage1(cfg, jscc, pipeline, train_loader, output_dir=output_dir)


if __name__ == "__main__":
    main()
