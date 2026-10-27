"""
Phase 5 — detectability-limits analysis (contribution #4).

For every attack variant: is the GRU monitor's (smoothed) alarm sustained within the
attack window, at a threshold frozen to 1% FPR on val? Plot detection vs the attack's
severity (max offset) and stealth (ramp rate). The region {offset > 3.5 m AND missed}
is the exploitable stealth zone — an attack that causes a wrong-lane fault yet evades
the monitor.

Usage:  python detectability.py --scores ../results/monitor_gru96 --feat ../features --out ../results/detectability
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

FAULT_LAT_M = 3.5   # wrong-lane lateral error
EMA = 0.3
HOLD = 10           # samples (1 s) of sustained alarm


def _ema(x, a=EMA):
    y = np.empty_like(x, dtype=float)
    acc = 0.0
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc
        y[i] = acc
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="../results/monitor_gru96")
    ap.add_argument("--feat", default="../features")
    ap.add_argument("--out", default="../results/detectability")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # val 1%-FPR threshold on smoothed scores
    zv = np.load(os.path.join(args.scores, "scores_val.npz"))
    sv, yv, gv = zv["scores"], zv["y"], zv["groups"]
    sm_neg = []
    for g in np.unique(gv):
        m = gv == g
        s = _ema(sv[m])
        sm_neg.append(s[yv[m] == 0])
    thr = float(np.percentile(np.concatenate(sm_neg), 99))
    print(f"val 1%-FPR threshold (smoothed) = {thr:.3f}", flush=True)

    rows = []
    for split in ["test_town", "test_attack", "test_both"]:
        z = np.load(os.path.join(args.scores, f"scores_{split}.npz"))
        s_all, y_all, g_all = z["scores"], z["y"], z["groups"]
        meta = pd.read_parquet(os.path.join(args.feat, f"{split}_index.parquet")).reset_index(drop=True)
        for gi in range(len(meta)):
            r = meta.iloc[gi]
            if r["is_attack"] != 1:
                continue
            m = g_all == gi
            s = _ema(s_all[m])
            y = y_all[m]
            atk = np.where(y == 1)[0]
            if len(atk) == 0:
                continue
            onset = atk[0]
            alarm = s > thr
            det, lat = False, np.nan
            run = 0
            for i in range(onset, len(alarm)):
                run = run + 1 if alarm[i] else 0
                if run >= HOLD:
                    det = True
                    lat = (i - HOLD + 1 - onset) / 10.0
                    break
            dur = max(float(r["duration_s"]), 1.0)
            rows.append(dict(split=split, family=r["family"],
                             max_offset_m=float(r["max_offset_m"]),
                             ramp_rate=float(r["max_offset_m"]) / dur,
                             duration_s=dur, detected=int(det), latency_s=lat))

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(args.out, "detectability.parquet"))

    overall = df["detected"].mean()
    dangerous = df[df["max_offset_m"] > FAULT_LAT_M]
    stealth = dangerous[dangerous["detected"] == 0]
    by_fam = df.groupby("family").agg(n=("detected", "size"),
                                      det_rate=("detected", "mean"),
                                      med_latency=("latency_s", "median")).round(3)
    # ramp-rate ceiling: highest ramp_rate at which detection is still < 50% for the
    # steerable families
    steer = df[df.family.isin(["gradual_pulloff", "target_bias"])].copy()
    steer["rr_bin"] = pd.cut(steer["ramp_rate"], bins=[0, 0.1, 0.25, 0.5, 1, 2, 10])
    rr = steer.groupby("rr_bin", observed=True)["detected"].mean().round(2)

    with open(os.path.join(args.out, "SUMMARY.md"), "w") as f:
        f.write("# Detectability-limits analysis\n\n")
        f.write(f"Threshold frozen to 1% FPR on val (smoothed score > {thr:.3f}), "
                f"sustained {HOLD/10:.0f} s within the attack window.\n\n")
        f.write(f"- Overall attack detection rate: **{overall*100:.0f}%** ({len(df)} variants)\n")
        f.write(f"- Attacks large enough to cause a wrong-lane fault (> {FAULT_LAT_M} m): "
                f"{len(dangerous)}; of these **{len(stealth)} ({len(stealth)/max(len(dangerous),1)*100:.0f}%) "
                f"go undetected** — the exploitable stealth region.\n\n")
        f.write("## Detection rate by family\n\n")
        f.write(by_fam.to_markdown() + "\n\n")
        f.write("## Detection rate vs ramp rate (gradual_pulloff + target_bias)\n\n")
        f.write(rr.to_markdown() + "\n")
    print(open(os.path.join(args.out, "SUMMARY.md")).read())


if __name__ == "__main__":
    main()
