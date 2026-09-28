# -*- coding: utf-8 -*-
"""Stage-2 动作结果头训练入口。"""
import argparse

from torch.utils.data import DataLoader

from calsem.config import Config
from calsem.data.voc import VOCDataset, collate_fn
from calsem.factory import build_models
from calsem.train.stage2 import train_stage2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/wchan/projects/CalSem/configs/stage2.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--jscc-ckpt", default=None, help="stage-1 权重；优先级高于 config")
    ap.add_argument("--output-dir", default=None)
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
    jscc_ckpt = args.jscc_ckpt or cfg.get("jscc_ckpt", None)
    if jscc_ckpt:
        import torch
        jscc.load_state_dict(torch.load(jscc_ckpt, map_location="cpu"))
        print(f"[model] 加载 stage-1 权重 {jscc_ckpt}")
    else:
        print("[model] 未指定 stage-1 JSCC 权重，将使用随机初始化 JSCC")

    train_stage2(cfg, pipeline, risk_head, train_loader, output_dir=output_dir)


if __name__ == "__main__":
    main()
