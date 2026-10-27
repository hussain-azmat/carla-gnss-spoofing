"""
Phase 3 — learned recurrent GNSS-spoofing consistency monitor.

A causal (unidirectional) GRU over the 39-feature 10 Hz sequence emits a per-timestep
spoof probability. Trained on `train`, threshold frozen on `val`, scored on every test
split with the same protocol as the classical baselines (code/metrics_detect.py).

Train on fixed windows for batching; evaluate on full sequences causally.

Usage:
  python train_monitor.py --features ../features --out ../results/monitor \
      --epochs 25 --hidden 96 --window 600 --stride 300
Runs on CPU (small model); set --device cuda on a GPU box / Kaggle notebook.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from metrics_detect import evaluate

SPLITS_TEST = ["val", "test_town", "test_attack", "test_both"]


def load_split(fdir, split):
    z = np.load(os.path.join(fdir, f"{split}.npz"))
    meta = pd.read_parquet(os.path.join(fdir, f"{split}_index.parquet")).reset_index(drop=True)
    X, y, g = z["X"].astype(np.float32), z["y"].astype(np.float32), z["groups"].astype(np.int64)
    seqs = []
    for gi in range(len(meta)):
        m = g == gi
        seqs.append((X[m], y[m]))
    return seqs, meta


def windows(seqs, W, S, mean, std):
    xs, ys = [], []
    for X, y in seqs:
        Xn = (X - mean) / std
        n = len(Xn)
        if n <= W:
            pad = W - n
            xs.append(np.pad(Xn, ((0, pad), (0, 0))))
            ys.append(np.pad(y, (0, pad), constant_values=-1))
            continue
        for a in range(0, n - W + 1, S):
            xs.append(Xn[a:a + W]); ys.append(y[a:a + W])
        if (n - W) % S:
            xs.append(Xn[-W:]); ys.append(y[-W:])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


class GRUMonitor(nn.Module):
    def __init__(self, f_in, hidden=96, layers=2, p=0.1):
        super().__init__()
        self.gru = nn.GRU(f_in, hidden, layers, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x, h=None):
        z, h = self.gru(x, h)
        return self.head(z).squeeze(-1), h


class _CausalConv(nn.Module):
    def __init__(self, c, k, d, p):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c, c, k, dilation=d)
        self.c2 = nn.Conv1d(c, c, k, dilation=d)
        self.drop = nn.Dropout(p)

    def forward(self, x):                       # x: [B, C, T]
        y = torch.relu(self.c1(torch.nn.functional.pad(x, (self.pad, 0))))
        y = self.drop(y)
        y = torch.relu(self.c2(torch.nn.functional.pad(y, (self.pad, 0))))
        return x + self.drop(y)


class TCNMonitor(nn.Module):
    """causal temporal-conv net; same [B,T]->[B,T] per-timestep logit contract."""
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


class TransformerMonitor(nn.Module):
    """causal Transformer encoder; same [B,T]->[B,T] per-timestep logit contract.

    A left-to-right (causal) self-attention mask keeps it deployable online: the
    logit at t depends only on inputs <= t, exactly like the GRU/TCN.
    """
    def __init__(self, f_in, hidden=96, layers=3, heads=4, p=0.1, span=900):
        super().__init__()
        self.span = span                      # bounded causal attention context
        self.inp = nn.Linear(f_in, hidden)
        self.pos = nn.Parameter(torch.zeros(1, 2 * span, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, p,
                                         batch_first=True, activation="gelu",
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, 1))

    def _block(self, z):                       # z: [B, t, H], t <= 2*span
        t = z.size(1)
        z = z + self.pos[:, :t]
        mask = torch.triu(torch.ones(t, t, device=z.device, dtype=torch.bool), 1)
        return self.enc(z, mask=mask)

    def forward(self, x, h=None):
        z = self.inp(x)
        T = z.size(1)
        if T <= 2 * self.span:
            return self.head(self._block(z)).squeeze(-1), None
        # long sequence: emit in chunks of `span`, each attending to itself plus
        # the preceding `span` steps of context (bounded-context online transformer).
        outs, s = [], self.span
        for a in range(0, T, s):
            lo = max(0, a - s)
            zc = self._block(z[:, lo:a + s])
            outs.append(zc[:, a - lo:])        # keep only the fresh `span` steps
        return self.head(torch.cat(outs, 1)).squeeze(-1), None


def build_model(arch, f_in, hidden):
    if arch == "tcn":
        return TCNMonitor(f_in, hidden)
    if arch == "transformer":
        return TransformerMonitor(f_in, hidden)
    return GRUMonitor(f_in, hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../features")
    ap.add_argument("--out", default="../results/monitor")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--stride", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="gru", choices=["gru", "tcn", "transformer"])
    ap.add_argument("--lofo", type=int, default=0,
                    help="1 = leave-one-attack-family-out sweep instead of the normal run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    P = lambda *a: print(*a, flush=True)

    st = json.load(open(os.path.join(args.features, "norm_stats.json")))
    mean, std = np.array(st["mean"], np.float32), np.array(st["std"], np.float32)
    names = json.load(open(os.path.join(args.features, "feature_names.json")))

    if args.lofo:
        return run_lofo(args, mean, std, names, dev, P)

    tr_seqs, _ = load_split(args.features, "train")
    Xtr, Ytr = windows(tr_seqs, args.window, args.stride, mean, std)
    P(f"[{args.arch}] train windows {Xtr.shape}  pos {float((Ytr == 1).sum() / (Ytr >= 0).sum()):.3f}")

    pos_w = float((Ytr == 0).sum() / max((Ytr == 1).sum(), 1))
    pos_w = min(pos_w, 30.0)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    model = build_model(args.arch, len(names), args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    Xtr_t = torch.from_numpy(Xtr); Ytr_t = torch.from_numpy(Ytr)
    N = len(Xtr_t)

    @torch.no_grad()
    def score_split(split):
        model.eval()
        seqs, meta = load_split(args.features, split)
        S, Y, G = [], [], []
        for gi, (X, y) in enumerate(seqs):
            Xn = torch.from_numpy(((X - mean) / std).astype(np.float32))[None].to(dev)
            logit, _ = model(Xn)
            S.append(torch.sigmoid(logit)[0].cpu().numpy())
            Y.append(y); G.append(np.full(len(y), gi, np.int64))
        return np.concatenate(S), np.concatenate(Y), np.concatenate(G), meta

    best = (-1, None)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = Xtr_t[idx].to(dev), Ytr_t[idx].to(dev)
            m = (yb >= 0).float()
            w = torch.where(yb == 1, pos_w, 1.0) * m
            logit, _ = model(xb)
            loss = (lossf(logit, yb.clamp(min=0)) * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); tot += loss.item() * len(idx)
        sv, yv, gv, mv = score_split("val")
        r = evaluate(args.arch, yv, sv, gv, mv, yv, sv)
        P(f"ep{ep:2d} loss {tot/N:.4f}  val AUC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} "
          f"TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f} HN-FAR {r['hardneg_far_per_hr']:.1f}/h")
        if r["pr_auc"] > best[0]:
            best = (r["pr_auc"], {k: v.cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    torch.save(model.state_dict(), os.path.join(args.out, f"{args.arch}_monitor.pt"))

    sv, yv, gv, mv = score_split("val")
    rows = []
    for split in SPLITS_TEST:
        try:
            s, y, g, meta = score_split(split)
            r = evaluate(f"{args.arch}_monitor", y, s, g, meta, yv, sv)
            r["split"] = split
            rows.append(r)
            P(f"{split:11s} AUC {r['roc_auc']:.3f}  PR {r['pr_auc']:.3f}  "
              f"TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f}  medTTA {r['median_tta_s']:.1f}s  "
              f"miss {r['miss_rate']:.2f}  HN-FAR {r['hardneg_far_per_hr']:.1f}/h")
            np.savez_compressed(os.path.join(args.out, f"scores_{split}.npz"), scores=s, y=y, groups=g)
        except Exception as e:
            P(f"{split}: eval failed: {e}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "monitor.csv"), index=False)
    P(f"\nwrote {args.out}/monitor.csv")


def run_lofo(args, mean, std, names, dev, P):
    """leave-one-attack-family-out: exclude one family from training, test on it."""
    FAMS = ["gradual_pulloff", "jump_and_hold", "random_walk", "zv_freeze", "replay", "target_bias"]
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    tr_seqs, tr_meta = load_split(args.features, "train")
    rows = []
    for held in FAMS:
        keep = [i for i, fam in enumerate(tr_meta["family"]) if fam != held]
        Xtr, Ytr = windows([tr_seqs[i] for i in keep], args.window, args.stride, mean, std)
        pos_w = min(float((Ytr == 0).sum() / max((Ytr == 1).sum(), 1)), 30.0)
        model = build_model(args.arch, len(names), args.hidden).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), args.batch):
                idx = perm[i:i + args.batch]
                xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
                w = torch.where(yb == 1, pos_w, 1.0) * (yb >= 0).float()
                logit, _ = model(xb)
                loss = (lossf(logit, yb.clamp(min=0)) * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()

        @torch.no_grad()
        def score(split, fam_only):
            model.eval()
            seqs, meta = load_split(args.features, split)
            keep = [i for i in range(len(meta))
                    if meta["family"].iloc[i] in (fam_only, "clean")]
            S, Y, G = [], [], []
            for j, gi in enumerate(keep):
                X, y = seqs[gi]
                Xn = torch.from_numpy(((X - mean) / std).astype(np.float32))[None].to(dev)
                logit, _ = model(Xn)
                S.append(torch.sigmoid(logit)[0].cpu().numpy())
                Y.append(y); G.append(np.full(len(y), j, np.int64))
            return np.concatenate(S), np.concatenate(Y), np.concatenate(G), meta.iloc[keep].reset_index(drop=True)

        sv, yv, gv, mv = score("val", held)
        for split in ["test_town", "test_attack", "test_both"]:
            try:
                s, y, g, m = score(split, held)
                if y.sum() == 0:
                    continue
                r = evaluate(f"lofo_{held}", y, s, g, m, yv, sv)
                r["held_out_family"] = held; r["split"] = split
                rows.append(r)
                P(f"held {held:16s} {split:11s} AUC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} "
                  f"TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f} miss {r['miss_rate']:.2f}")
            except Exception as e:
                P(f"held {held} {split}: {e}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "lofo_results.csv"), index=False)
    P(f"wrote {args.out}/lofo_results.csv")


if __name__ == "__main__":
    main()
