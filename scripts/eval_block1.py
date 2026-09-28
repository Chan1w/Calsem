# -*- coding: utf-8 -*-
"""B1 主表评估入口：率-可靠-时延 Pareto + policy regret（逐 SNR）。

用法:
    python scripts/eval_block1.py --config configs/eval.yaml \
        --jscc-ckpt runs/stage1_jscc.pt --risk-head-ckpt runs/stage3_risk_head_cal.pt \
        --data-root data/VOC2012 --snrs 0 5 10 15 20
"""
import argparse
import json

from torch.utils.data import DataLoader

from calsem.config import Config
from calsem.data.voc import VOCDataset, collate_fn
from calsem.eval.blocks import run_block1
from calsem.factory import build_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/wchan/projects/CalSem/configs/eval.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--jscc-ckpt", default=None)
    ap.add_argument("--risk-head-ckpt", default=None)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.data_root = args.data_root

    test_ds = VOCDataset(cfg.data_root, "val", cfg.resolution)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate_fn)
    print(f"[data] test 样本数: {len(test_ds)}")

    jscc, sd, dino, channel, pipeline, risk_head = build_models(cfg)
    import torch
    if args.jscc_ckpt:
        jscc.load_state_dict(torch.load(args.jscc_ckpt, map_location="cpu"))
    if args.risk_head_ckpt:
        risk_head.load_state_dict(torch.load(args.risk_head_ckpt, map_location="cpu"))
    risk_head.to("cuda" if torch.cuda.is_available() else "cpu")

    rows = run_block1(cfg, pipeline, risk_head, test_loader, args.snrs,
                      cfg.lambda_R, cfg.lambda_T)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"[done] 结果写入 {args.out}")
    else:
        print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
