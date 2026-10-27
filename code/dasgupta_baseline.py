"""
Dasgupta-style learned dead-reckoning-vs-GNSS spoofing detector (baseline).

Re-implementation of the idea in Dasgupta, Rahman, Islam, Chowdhury, "A Sensor Fusion-Based
GNSS Spoofing Attack Detection Framework for Autonomous Vehicles", IEEE T-ITS 23(12), 2022
(DOI 10.1109/TITS.2022.3197817), on OUR benchmark and protocol:

  * an LSTM is trained on CLEAN driving only to predict, from low-cost in-vehicle signals
    (accelerometer, steering angle, wheel speed), how far the vehicle moved in each 0.1 s step;
  * the predicted per-step displacements are placed in the world frame with the compass heading
    and summed over a sliding window of w seconds (dead reckoning);
  * the score is the distance between that predicted shift and the GNSS-reported shift over
    the same window (higher = more likely spoofed).

Not reproduced: the paper's kNN+DTW turn-classification module (turn-by-turn / overshoot
attack logic) and its own driving data. The steering angle is the ego `steer` channel stored in
gt_pose.parquet (the dataset logs no separate steering sensor). No LiDAR odometry is used, i.e.
this baseline has NO un-spoofable exteroceptive reference -- that is the point of the comparison.

The window w is the only tuned quantity; it is picked on val ROC-AUC from {1,2,3,5,10,20,30,45,60} s.
Scored with the same splits, features index, val-frozen 1%-FPR threshold and metrics as the
GRU monitor (metrics_detect.evaluate).

Usage:  python dasgupta_baseline.py --in .. --feat ../features --out ../results/dasgupta_baseline
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from metrics_detect import evaluate

DT = 0.1
WINDOWS_S = (1, 2, 3, 5, 10, 20, 30, 45, 60)
SPLITS = ["val", "test_town", "test_attack", "test_both"]


def _interp(t_dst, t_src, v):
    return np.interp(t_dst, t_src, v, left=v[0], right=v[-1])


def session_arrays(sdir, t_grid):
    """inputs [T,5] (ax, ay, gz, wheel speed, steer), gt per-step displacement [T], compass [T], gt e/n."""
    imu = pd.read_parquet(os.path.join(sdir, "imu.parquet"))
    whl = pd.read_parquet(os.path.join(sdir, "wheel.parquet"))
    gt = pd.read_parquet(os.path.join(sdir, "gt_pose.parquet"))
    X = np.c_[_interp(t_grid, imu["t"].values, imu["ax"].values),
              _interp(t_grid, imu["t"].values, imu["ay"].values),
              _interp(t_grid, imu["t"].values, imu["gz"].values),
              _interp(t_grid, whl["t"].values, whl["speed_mps"].values),
              _interp(t_grid, gt["t"].values, gt["steer"].values)].astype(np.float32)
    e = _interp(t_grid, gt["t"].values, gt["e"].values)
    n = _interp(t_grid, gt["t"].values, gt["n"].values)
    d = np.r_[0.0, np.hypot(np.diff(e), np.diff(n))].astype(np.float32)   # m moved in the step ending at t_k
    ok = np.isfinite(imu["compass"].values)              # a few sessions have isolated NaN compass samples
    ct, cc = imu["t"].values[ok], imu["compass"].values[ok]
    compass = np.arctan2(_interp(t_grid, ct, np.sin(cc)), _interp(t_grid, ct, np.cos(cc)))
    return X, d, compass, e, n


class DRNet(nn.Module):
    def __init__(self, f_in=5, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(f_in, hidden, 2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fit_heading(comp_list, e_list, n_list):
    """theta_enu = s*compass + c : pick s in {+1,-1}, c = circular mean offset, on moving samples."""
    best = None
    for s in (1.0, -1.0):
        cs, sn, cnt = 0.0, 0.0, 0
        for comp, e, n in zip(comp_list, e_list, n_list):
            de, dn = np.gradient(e), np.gradient(n)
            mov = np.hypot(de, dn) > 0.3          # > 3 m/s
            th = np.arctan2(dn, de)[mov]
            off = th - s * comp[mov]
            cs += np.cos(off).sum(); sn += np.sin(off).sum(); cnt += mov.sum()
        c = np.arctan2(sn, cs)
        R = np.hypot(cs, sn) / max(cnt, 1)      # mean resultant length: 1 = perfect
        if best is None or R > best[2]:
            best = (s, c, R)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="..")
    ap.add_argument("--feat", default="../features")
    ap.add_argument("--out", default="../results/dasgupta_baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ddir = os.path.join(args.inp, "data")
    log = open(os.path.join(args.out, "run.log"), "w")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); log.write(s + "\n"); log.flush()

    idx = {sp: pd.read_parquet(os.path.join(args.feat, f"{sp}_index.parquet")).reset_index(drop=True)
           for sp in ["train"] + SPLITS}
    # session time grid = GNSS 10 Hz grid of that session
    cache = {}

    def get_session(sess):
        if sess not in cache:
            sdir = os.path.join(ddir, sess)
            tg = pd.read_parquet(os.path.join(sdir, "gnss_clean.parquet"))["t"].values
            cache[sess] = (tg,) + session_arrays(sdir, tg)
        return cache[sess]

    train_sess = sorted(idx["train"]["session"].unique())
    P(f"train sessions: {len(train_sess)}")
    tr = [get_session(s) for s in train_sess]

    # heading calibration on training sessions only
    s_sign, c_off, R = fit_heading([x[3] for x in tr], [x[4] for x in tr], [x[5] for x in tr])
    P(f"heading calibration: theta_enu = {s_sign:+.0f}*compass + {c_off:.3f} rad  (mean resultant length {R:.3f})")

    allX = np.concatenate([x[1] for x in tr]); mean, std = allX.mean(0), allX.std(0) + 1e-6
    W, S = 300, 150
    xs, ys = [], []
    for tg, X, d, comp, e, n in tr:
        Xn = (X - mean) / std
        for a in range(0, len(Xn) - W + 1, S):
            xs.append(Xn[a:a + W]); ys.append(d[a:a + W] / DT)     # target: speed-equivalent m/s
    Xt, Yt = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
    P(f"training windows: {tuple(Xt.shape)}")
    net = DRNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for i in range(0, len(Xt), 32):
            b = perm[i:i + 32]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == 1:
            P(f"  ep{ep:2d} train MSE {tot / len(Xt):.4f} (m/s)^2   {time.time() - t0:.0f}s")
    net.eval()

    pred_cache = {}

    def predicted_step(sess):
        """per-step predicted displacement (m) for a session, causal LSTM over the whole session."""
        if sess not in pred_cache:
            tg, X, d, comp, e, n = get_session(sess)
            with torch.no_grad():
                p = net(torch.from_numpy(((X - mean) / std)[None]))[0].numpy() * DT
            pred_cache[sess] = np.clip(p, 0, None)
        return pred_cache[sess]

    # held-out accuracy of the dead-reckoning predictor itself (val sessions, clean)
    errs = []
    for sess in sorted(idx["val"]["session"].unique()):
        tg, X, d, comp, e, n = get_session(sess)
        errs.append(np.abs(np.cumsum(predicted_step(sess)) - np.cumsum(d))[-1] / max(np.cumsum(d)[-1], 1e-6))
    P(f"val sessions: total-path error of the predictor {100*np.mean(errs):.2f}% (mean)")

    def score_split(sp, windows):
        """returns {w: (S, Y, G)} over every variant of the split, in index order."""
        meta = idx[sp]
        out = {w: ([], [], []) for w in windows}
        for gi, row in meta.iterrows():
            vdf = pd.read_parquet(os.path.join(args.inp, "dataset", "variants",
                                               row["variant"].replace("/", os.sep) + ".parquet"))
            t = vdf["t"].values
            tg, X, d, comp, e, n = get_session(row["session"])
            assert len(t) == len(tg), (row["variant"], len(t), len(tg))
            step = predicted_step(row["session"])
            th = s_sign * comp + c_off
            dx, dy = step * np.cos(th), step * np.sin(th)
            cx, cy = np.cumsum(np.r_[0.0, dx]), np.cumsum(np.r_[0.0, dy])      # cumulative predicted position
            gx, gy = vdf["e"].values, vdf["n"].values
            y = vdf["label"].values.astype(np.uint8)
            for w in windows:
                k = int(round(w / DT)); ii = np.arange(len(t)); jj = np.maximum(ii - k, 0)
                px = cx[ii + 1] - cx[jj + 1]; py = cy[ii + 1] - cy[jj + 1]    # sum of steps (jj, ii]
                r = np.hypot(px - (gx[ii] - gx[jj]), py - (gy[ii] - gy[jj]))
                out[w][0].append(r); out[w][1].append(y); out[w][2].append(np.full(len(t), gi, np.int64))
        return {w: tuple(np.concatenate(v) for v in out[w]) for w in windows}

    P("scoring val for all windows ...")
    val_all = score_split("val", WINDOWS_S)
    from metrics_detect import roc_auc
    val_auc = {w: roc_auc(val_all[w][1], val_all[w][0]) for w in WINDOWS_S}
    P("val ROC-AUC by window (s):", {w: round(a, 3) for w, a in val_auc.items()})
    w_best = max(val_auc, key=val_auc.get)
    P(f"chosen window: {w_best} s")

    rows, win_rows = [], []
    sv, yv, gv = val_all[w_best]
    for sp in SPLITS:
        res = score_split(sp, WINDOWS_S) if sp != "val" else val_all
        for w in WINDOWS_S:
            s, y, g = res[w]
            win_rows.append(dict(split=sp, window_s=w, roc_auc=roc_auc(y, s)))
        s, y, g = res[w_best]
        r = evaluate(f"dasgupta_style_w{w_best}s", y, s, g, idx[sp], yv, sv)
        r["split"] = sp; r["window_s"] = w_best
        rows.append(r)
        P(f"  {sp:11s} AUC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f} "
          f"miss {r['miss_rate']:.2f} HN-FAR {r['hardneg_far_per_hr']:.1f}/h")
        np.savez_compressed(os.path.join(args.out, f"scores_{sp}.npz"), scores=s, y=y, groups=g)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "dasgupta_baseline.csv"), index=False)
    pd.DataFrame(win_rows).to_csv(os.path.join(args.out, "window_sweep.csv"), index=False)
    json.dump(dict(heading_sign=s_sign, heading_offset=float(c_off), resultant=float(R), window_s=int(w_best)),
              open(os.path.join(args.out, "config.json"), "w"))
    P("done")


if __name__ == "__main__":
    main()
