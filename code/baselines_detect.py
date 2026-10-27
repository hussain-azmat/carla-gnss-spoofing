"""
Phase 3 — classical / non-temporal detection baselines (docs/metrics.md).

Each baseline maps the per-timestep feature vector to a spoof score; thresholds
are frozen on `val` and applied unchanged to every test split.

  raim_chi2        single-epoch chi-square on the 1 s GNSS-vs-LiDAR-odom residual,
                   normalised by the per-condition nominal sigma.
  pairwise_z       max |z| across GNSS-vs-{LiDAR-odom, wheel} 3 s residuals.
  cusum            the running two-sided CUSUM feature (change detection on the
                   GNSS-vs-DR residual).
  huber_resid      robust (Huber) score of the 3 s GNSS-vs-LO residual — a filter
                   that down-weights but has no explicit detector.
  logreg_1epoch    logistic regression on the full feature vector, no temporal
                   context (isolates the value of recurrence for the learned model).

Usage:  python baselines_detect.py --features ../features --out ../results/baselines
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from metrics_detect import evaluate

SPLITS = ["val", "test_town", "test_attack", "test_both"]


def _load(fdir, split):
    z = np.load(os.path.join(fdir, f"{split}.npz"))
    meta = pd.read_parquet(os.path.join(fdir, f"{split}_index.parquet")).reset_index(drop=True)
    return z["X"].astype(np.float32), z["y"].astype(np.uint8), z["groups"].astype(np.int32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../features")
    ap.add_argument("--out", default="../results/baselines")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = json.load(open(os.path.join(args.features, "feature_names.json")))
    ix = {n: i for i, n in enumerate(names)}
    st = json.load(open(os.path.join(args.features, "norm_stats.json")))
    mean, std = np.array(st["mean"], np.float32), np.array(st["std"], np.float32)

    Xtr, ytr, gtr, mtr = _load(args.features, "train")
    data = {s: _load(args.features, s) for s in SPLITS}

    # nominal sigma of the 1 s GNSS-vs-LO residual, from clean train timesteps
    r1 = Xtr[:, ix["resid_gnss_lo_1s"]]
    sig = float(np.std(r1[ytr == 0]) + 1e-3)

    def scores(X):
        r1 = X[:, ix["resid_gnss_lo_1s"]]
        r3_lo = X[:, ix["resid_gnss_lo_3s"]]
        r3_wh = X[:, ix["resid_gnss_wheel_3s"]]
        out = {}
        out["raim_chi2"] = (r1 / sig) ** 2
        z_lo = np.abs(r3_lo) / (np.std(Xtr[:, ix["resid_gnss_lo_3s"]][ytr == 0]) + 1e-3)
        z_wh = np.abs(r3_wh) / (np.std(Xtr[:, ix["resid_gnss_wheel_3s"]][ytr == 0]) + 1e-3)
        out["pairwise_z"] = np.maximum(z_lo, z_wh)
        out["cusum"] = np.maximum(X[:, ix["cusum_pos"]], X[:, ix["cusum_neg"]])
        d = 1.345
        a = np.abs(r3_lo) / sig
        out["huber_resid"] = np.where(a <= d, 0.5 * a * a, d * (a - 0.5 * d))
        return out

    # logistic regression on the full normalised feature vector (no time context)
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=300, class_weight="balanced", C=1.0)
        clf.fit((Xtr - mean) / std, ytr)
        have_lr = True
    except Exception as e:
        print("sklearn unavailable, skipping logreg_1epoch:", e)
        have_lr = False

    sc_val = scores(data["val"][0])
    if have_lr:
        sc_val["logreg_1epoch"] = clf.predict_proba((data["val"][0] - mean) / std)[:, 1]
    yv = data["val"][1]

    rows = []
    for split in SPLITS:
        X, y, g, meta = data[split]
        sc = scores(X)
        if have_lr:
            sc["logreg_1epoch"] = clf.predict_proba((X - mean) / std)[:, 1]
        for name, s in sc.items():
            r = evaluate(name, y, s, g, meta, yv, sc_val[name])
            r["split"] = split
            rows.append(r)
            print(f"{split:11s} {name:14s} AUC {r['roc_auc']:.3f}  PR {r['pr_auc']:.3f}  "
                  f"TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f}  medTTA {r['median_tta_s']:.1f}s  "
                  f"miss {r['miss_rate']:.2f}  HN-FAR {r['hardneg_far_per_hr']:.1f}/h")

    df = pd.DataFrame(rows)[["split", "method", "roc_auc", "pr_auc", "tpr_at_1pct_fpr",
                             "tpr_at_0p1pct_fpr", "median_tta_s", "p90_tta_s", "miss_rate",
                             "hardneg_far_per_hr", "realized_fpr_1pct"]]
    df.to_csv(os.path.join(args.out, "baselines.csv"), index=False)
    df.to_parquet(os.path.join(args.out, "baselines.parquet"))
    print(f"\nwrote {args.out}/baselines.csv")


if __name__ == "__main__":
    main()
