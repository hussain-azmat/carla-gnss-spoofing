"""
Extra learned baselines for the detection benchmark (T-ITS reviewer ask: "is a
different architecture stronger than your GRU?").

Same causal per-timestep contract, same 39 features, same splits, same val-frozen
thresholds as train_monitor.py. Architectures:
  * lstm    -- 2x96 causal LSTM (direct GRU swap)
  * bigru   -- 2x96 BI-directional GRU: NON-causal offline upper bound (how much the
               online/causal constraint costs)
  * rawcnn  -- 1D dilated CNN on GNSS-ONLY kinematic + context channels (no cross-sensor
               residual features): the "no consistency engineering" lower bound

Usage:
  python arch_baselines.py --features ../features --out ../results/arch_baselines \
      --archs lstm bigru rawcnn --epochs 25
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from train_monitor import load_split, windows
from metrics_detect import evaluate

SPLITS_TEST = ["val", "test_town", "test_attack", "test_both"]

# GNSS-only kinematic feature names produced by build_features.py (no cross-sensor
# residuals, no un-spoofable reference). Used by the rawcnn baseline.
# GNSS-only / context channels present in feature_names.json — NO cross-sensor
# residual, NO un-spoofable reference. The "no consistency engineering" lower bound.
RAW_GNSS_FEATS = ["speed", "yaw_rate_abs", "accel_lon", "accel_lat",
                  "gnss_disp_1s", "gnss_disp_3s", "gnss_disp_10s",
                  "dist_to_junction", "is_canyon", "is_tunnel", "is_rain", "is_night"]


class LSTMMonitor(nn.Module):
    def __init__(self, f_in, hidden=96, layers=2, p=0.1):
        super().__init__()
        self.rnn = nn.LSTM(f_in, hidden, layers, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x, h=None):
        z, h = self.rnn(x, h)
        return self.head(z).squeeze(-1), h


class BiGRUMonitor(nn.Module):
    """NON-causal (bidirectional) GRU -- offline upper bound, not deployable online."""
    def __init__(self, f_in, hidden=96, layers=2, p=0.1):
        super().__init__()
        self.rnn = nn.GRU(f_in, hidden, layers, batch_first=True, dropout=p,
                          bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x, h=None):
        z, h = self.rnn(x, h)
        return self.head(z).squeeze(-1), h


class _CausalConv(nn.Module):
    def __init__(self, c, k, d, p):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c, c, k, dilation=d)
        self.c2 = nn.Conv1d(c, c, k, dilation=d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        y = torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0))))
        y = self.drop(y)
        y = torch.relu(self.c2(nn.functional.pad(y, (self.pad, 0))))
        return x + self.drop(y)


class RawCNN(nn.Module):
    """1D dilated causal CNN; used on GNSS-only kinematic channels."""
    def __init__(self, f_in, hidden=96, layers=6, k=3, p=0.1):
        super().__init__()
        self.inp = nn.Conv1d(f_in, hidden, 1)
        self.blocks = nn.ModuleList([_CausalConv(hidden, k, 2 ** i, p) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x, h=None):
        z = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            z = b(z)
        return self.head(z.transpose(1, 2)).squeeze(-1), None


def build(arch, f_in):
    if arch == "lstm":
        return LSTMMonitor(f_in)
    if arch == "bigru":
        return BiGRUMonitor(f_in)
    if arch == "rawcnn":
        return RawCNN(f_in, hidden=64, layers=4)   # light: CPU conv1d is slow
    raise ValueError(arch)


def run_arch(arch, feats, out, mean, std, names, args, P):
    dev = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if arch == "rawcnn":
        keep_idx = [i for i, nm in enumerate(names) if nm in RAW_GNSS_FEATS]
        if not keep_idx:
            P(f"  rawcnn: none of {RAW_GNSS_FEATS} in feature_names -> using first 12 cols")
            keep_idx = list(range(min(12, len(names))))
        sub = lambda A: A[:, keep_idx]
        m_, s_ = mean[keep_idx], std[keep_idx]
        f_in = len(keep_idx)
        P(f"  rawcnn uses {f_in} GNSS-only channels: {[names[i] for i in keep_idx]}")
    else:
        sub = lambda A: A
        m_, s_ = mean, std
        f_in = len(names)

    tr_seqs, _ = load_split(feats, "train")
    tr_seqs = [(sub(X), y) for X, y in tr_seqs]
    Xtr, Ytr = windows(tr_seqs, args.window, args.stride, m_, s_)
    pos_w = min(float((Ytr == 0).sum() / max((Ytr == 1).sum(), 1)), 30.0)
    P(f"[{arch}] train windows {Xtr.shape} pos {(Ytr==1).sum()/(Ytr>=0).sum():.3f} pos_w {pos_w:.1f}")

    lossf = nn.BCEWithLogitsLoss(reduction="none")
    model = build(arch, f_in).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    N = len(Xt)

    causal = arch != "bigru"

    @torch.no_grad()
    def score_split(split, bs=24):
        model.eval()
        seqs, meta = load_split(feats, split)
        seqs = [(((sub(X) - m_) / s_).astype(np.float32), y) for X, y in seqs]
        S = [None] * len(seqs)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
        # bigru (non-causal): smaller batches, padding-zeros bleed slightly into
        # the tail of each seq via the backward pass -- acceptable for an
        # offline upper-bound reference. causal nets: padded tail is exact.
        bs_eff = bs if causal else 6
        for b0 in range(0, len(order), bs_eff):
            idx = order[b0:b0 + bs_eff]
            L = max(len(seqs[i][0]) for i in idx)
            xb = torch.zeros(len(idx), L, f_in, device=dev)
            for k, i in enumerate(idx):
                xb[k, :len(seqs[i][0])] = torch.from_numpy(seqs[i][0])
            logit, _ = model(xb)
            prob = torch.sigmoid(logit).cpu().numpy()
            for k, i in enumerate(idx):
                S[i] = prob[k, :len(seqs[i][0])]
        Y = [seqs[i][1] for i in range(len(seqs))]
        G = [np.full(len(seqs[i][1]), i, np.int64) for i in range(len(seqs))]
        return np.concatenate(S), np.concatenate(Y), np.concatenate(G), meta

    best = (-1.0, None)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
            m = (yb >= 0).float()
            w = torch.where(yb == 1, pos_w, 1.0) * m
            logit, _ = model(xb)
            loss = (lossf(logit, yb.clamp(min=0)) * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); tot += loss.item() * len(idx)
        sv, yv, gv, mv = score_split("val")
        r = evaluate(arch, yv, sv, gv, mv, yv, sv)
        P(f"  ep{ep:2d} loss {tot/N:.4f} val AUC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} "
          f"TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f}")
        if r["pr_auc"] > best[0]:
            best = (r["pr_auc"], {k: v.cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, f"{arch}_monitor.pt"))
    sv, yv, gv, mv = score_split("val")
    rows = []
    for split in SPLITS_TEST:
        s, y, g, meta = score_split(split)
        r = evaluate(f"{arch}_monitor", y, s, g, meta, yv, sv)
        r["split"] = split
        rows.append(r)
        P(f"  {split:11s} AUC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} "
          f"TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f} miss {r['miss_rate']:.2f} "
          f"HN-FAR {r['hardneg_far_per_hr']:.1f}/h")
        np.savez_compressed(os.path.join(out, f"scores_{arch}_{split}.npz"), scores=s, y=y, groups=g)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../features")
    ap.add_argument("--out", default="../results/arch_baselines")
    ap.add_argument("--archs", nargs="+", default=["lstm", "bigru", "rawcnn"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--stride", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    P = lambda *a: print(*a, flush=True)

    st = json.load(open(os.path.join(args.features, "norm_stats.json")))
    mean = np.array(st["mean"], np.float32); std = np.array(st["std"], np.float32)
    names = json.load(open(os.path.join(args.features, "feature_names.json")))

    csv_path = os.path.join(args.out, "arch_baselines.csv")
    prev = []
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        keep_m = {f"{a}_monitor" for a in args.archs}
        prev = old[~old["method"].isin(keep_m)].to_dict("records")
        if prev:
            P(f"keeping {len(prev)} rows from prior archs: "
              f"{sorted(set(r['method'] for r in prev))}")

    all_rows = list(prev)
    for arch in args.archs:
        all_rows += run_arch(arch, args.features, args.out, mean, std, names, args, P)
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(all_rows)
    with open(os.path.join(args.out, "SUMMARY.md"), "w") as f:
        f.write("# Extra learned baselines (architecture comparison)\n\n")
        f.write("Same 39 features / splits / val-frozen thresholds as the GRU monitor "
                "(rawcnn: GNSS-only kinematic channels, no cross-sensor consistency).\n\n")
        for metric in ["roc_auc", "pr_auc", "tpr_at_1pct_fpr"]:
            piv = df.pivot(index="split", columns="method", values=metric).round(3).reindex(SPLITS_TEST)
            f.write(f"## {metric}\n\n{piv.to_markdown()}\n\n")
    P("\n" + open(os.path.join(args.out, "SUMMARY.md")).read())


if __name__ == "__main__":
    main()
