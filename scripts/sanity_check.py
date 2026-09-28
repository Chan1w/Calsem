# -*- coding: utf-8 -*-
"""M0 sanity check：验证代码结构可运行（无需下载权重即可通过 Layer-1）。

分层：
  Layer-1（无权重）：jscc 编解码形状 / 信道 / 风险头 / 决策 / 指标；
  Layer-2（需要 SD+DINO 权重 + GPU）：冻结组件 + img2img 推理。

用法:
    python scripts/sanity_check.py            # 默认 Layer-1
    python scripts/sanity_check.py --full     # Layer-1 + Layer-2（需权重）
"""
import argparse

import torch

from calsem.channel import AWGNChannel, power_normalize
from calsem.config import Config
from calsem.eval.decision import action_costs, decide, oracle_action
from calsem.eval.metrics import ece, policy_regret, roc_auc
from calsem.models.jscc import ProgressiveJSCC
from calsem.models.risk_head import RiskHead, channel_stats, mean_var_pooling


def layer1(cfg: Config) -> None:
    print("=== Layer-1: 无权重结构检查 ===")
    device = "cpu"
    jscc = ProgressiveJSCC(cfg)
    x = torch.randn(2, 3, cfg.resolution, cfg.resolution)
    s_base, s_sem, s_fid = jscc.encode(x)
    print(f"[jscc] s_base={tuple(s_base.shape)} s_sem={tuple(s_sem.shape)} "
          f"s_fid={tuple(s_fid.shape)}")
    assert s_base.shape[1] == jscc.base_ch and s_sem.shape[1] == jscc.sem_ch

    ch = AWGNChannel(P=cfg.power)
    y_base = ch(s_base, 10.0)
    z1 = jscc.decode_base(y_base)
    c_sem = jscc.decode_sem(ch(s_sem, 10.0))
    dz = jscc.decode_fid(ch(s_fid, 10.0), z1)
    print(f"[jscc] z1={tuple(z1.shape)} c_sem={tuple(c_sem.shape)} dz={tuple(dz.shape)}")
    assert z1.shape[1] == cfg.latent_channels
    assert c_sem.shape[1:] == (cfg.sem_embed_len, cfg.sem_embed_dim)
    assert dz.shape[1] == cfg.latent_channels
    print(f"[jscc] 可训练参数: {jscc.n_params():,}")

    # 功率归一化检查
    p = (s_base ** 2).mean()
    print(f"[channel] 归一化后平均功率={p.item():.4f} (期望≈{cfg.power})")

    # 风险头
    head = RiskHead(cfg)
    embs = torch.randn(2, 3, cfg.dinov2_embed)
    m, v = mean_var_pooling(embs)
    h_rep = m
    cs = channel_stats(torch.tensor([10.0, 20.0]))
    p_a = head.predict_proba(m, v, h_rep, cs)
    print(f"[risk_head] p_a={tuple(p_a.shape)} ∈ [0,1]: {p_a.min().item():.3f}~{p_a.max().item():.3f}")

    # 决策 + 指标
    dR, dT = action_costs(cfg)
    pred_a = decide(p_a, 0.1, 0.1, dR, dT)
    y = (torch.rand(2, 3) > 0.5).float()
    oracle_a = oracle_action(y, 0.1, 0.1, dR, dT)
    reg = policy_regret(pred_a, oracle_a, y, 0.1, 0.1, dR, dT)
    print(f"[decision] pred={pred_a.tolist()} oracle={oracle_a.tolist()} regret={reg:.4f}")
    p0 = p_a[:, 0].detach().cpu().numpy()
    y0 = y[:, 0].detach().cpu().numpy()
    print(f"[metrics] ece={ece(p0, y0):.4f} auc={roc_auc(p0, y0)}")
    print("Layer-1 PASS")


def layer2(cfg: Config) -> None:
    print("=== Layer-2: 冻结组件 + img2img（需权重 + GPU）===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from calsem.models.frozen import FrozenDINO, FrozenSD
    sd = FrozenSD(cfg.sd_ckpt, device=device, fp16=cfg.fp16,
                  cache_dir=cfg.sd_cache_dir)
    dino = FrozenDINO(cfg.dinov2_name, device=device, cache_dir=cfg.dinov2_cache_dir)
    print(f"[sd] scaling_factor={sd.scaling_factor}, device={device}")

    x = torch.randn(1, 3, cfg.resolution, cfg.resolution).to(device)
    if cfg.fp16 and device == "cuda":
        x = x.half()
    z = sd.vae_encode(x)
    print(f"[sd] z_native={tuple(z.shape)}")
    rec = sd.img2img_from_latent(z, strength=0.3, num_inference_steps=5)
    print(f"[sd] img2img 输出={tuple(rec.shape)}")
    emb = dino.embed_tensor(rec)
    print(f"[dino] embedding={tuple(emb.shape)}")
    print("Layer-2 PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--full", action="store_true", help="含 Layer-2（需权重）")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    layer1(cfg)
    if args.full:
        layer2(cfg)


if __name__ == "__main__":
    main()
