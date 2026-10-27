"""
Phase 4 — attack-resilient adaptive fusion.

A 6-state EKF (e, n, heading, speed, gyro-bias, wheel-scale) fuses wheel + gyro
(process) with GNSS position and Kabsch-aligned LiDAR-odometry position. The
learned monitor's per-timestep spoof probability drives how much the filter
trusts GNSS. Six variants:

  A  fixed R_gnss + chi-square innovation gate            (standard INS/GNSS)
  B  RAIM: tighter single-epoch chi-square reject         (classical integrity)
  C  hard detector gate: drop GNSS while s_t > tau        (learned, binary)
  D  soft covariance adaptation: R_gnss = R0 * exp(k*s_t) (learned, proposed)
  E  D + LiDAR-odometry consistency guard                 (full system)
  F  oracle gate: drop GNSS on the true attack label      (upper bound)

LiDAR-odometry is available to every method; only the GNSS handling changes.

Scored on the attack variants of the test splits, against GT pose:
  ate_attack   RMS position error over the attack window
  maxerr_attack
  recovery_s   time after attack end until err < 1 m sustained 3 s
  nees_attack  mean normalised estimation-error-squared (2-DOF; ~1 = consistent)

Usage:  python adaptive_fusion.py --in .. --scores ../results/monitor_gru96 \
            --out ../results/fusion --splits test_town,test_attack,test_both
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

RATE = 10.0
DT = 1.0 / RATE
ALIGN_S = 15.0
METHODS = ["A0_ins_gnss", "A_fixed_gate", "B_raim", "C_hard_gate",
           "D_soft_adapt", "E_hybrid", "F_oracle"]
NO_LIDAR = {"A0_ins_gnss"}   # traditional INS/GNSS, no LiDAR-odometry cross-check
# E_hybrid = the proposed full system: soft covariance adaptation + a hard-gate
# override on abrupt onsets + a LiDAR-odometry consistency guard + the classical
# chi-square innovation backstop (every learned method keeps the backstop now).

CHI2 = 13.8       # 99.9% 2-DOF innovation gate -- backstop for ALL methods except oracle
CHI2_B = 5.99     # 95% 2-DOF -- RAIM's more aggressive single-epoch reject
BETA_E = 3.0      # consistency-guard innovation clip (sigmas)
R0 = 4.0          # nominal GNSS var (m^2) ~ 2 m 1-sigma
Q_DIAG = np.array([0.02, 0.02, 1e-4, 0.25, 1e-8, 1e-8])   # process noise / s
R_WHEEL = 0.25
LO_DRIFT = 0.02   # R_LO ~ (LO_DRIFT * dist_since_align)^2, floored
# tuned on the val split 2026-09-07 (grid search, min E_hybrid median ATE); --tune re-runs it
PARAMS = dict(kappa=4.6, ema=0.5, tau_hi=0.7, tau_lo=0.35)


def _inv2(M):
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]]) / det


def _resample(t_dst, t_src, v):
    return np.interp(t_dst, t_src, v, left=v[0], right=v[-1])


def _align_lo(gnss, lo_xy, label, m):
    a = gnss[:m][label[:m] == 0] - gnss[0]
    b = lo_xy[:m][label[:m] == 0] - lo_xy[0]
    if len(a) >= 5 and np.linalg.norm(b) > 1.0:
        H = b.T @ a
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
    else:
        R = np.eye(2)
    return (lo_xy - lo_xy[0]) @ R.T + gnss[0]


def run_ekf(method, gnss, lo_w, gt, omega, wheel_v, spoof_s, label):
    n = len(gnss)
    x = np.zeros(6)
    x[:2] = gnss[0]
    # init heading from first GT (a real system uses moving-alignment; fine here)
    x[2] = gt["yaw"][0]
    x[3] = wheel_v[0]
    x[5] = 1.0
    P = np.diag([R0, R0, 0.2, 1.0, 1e-4, 1e-3])
    Q = np.diag(Q_DIAG) * DT
    est = np.zeros((n, 2))
    Ppos = np.zeros((n, 2, 2))
    s_ema = 0.0
    dist_align = 0.0
    for k in range(n):
        # --- predict ---
        e, nn, psi, v, bg, sw = x
        psi_n = psi + (omega[k] - bg) * DT
        e_n = e + v * np.cos(psi) * DT
        n_n = nn + v * np.sin(psi) * DT
        F = np.eye(6)
        F[0, 2] = -v * np.sin(psi) * DT; F[0, 3] = np.cos(psi) * DT
        F[1, 2] = v * np.cos(psi) * DT;  F[1, 3] = np.sin(psi) * DT
        F[2, 4] = -DT
        x = np.array([e_n, n_n, psi_n, v, bg, sw])
        P = F @ P @ F.T + Q
        dist_align += abs(v) * DT

        # --- wheel speed update (h = v / sw) ---
        h = x[3] / x[5]
        H = np.zeros(6); H[3] = 1 / x[5]; H[5] = -x[3] / x[5] ** 2
        y = wheel_v[k] - h
        S = H @ P @ H + R_WHEEL
        K = P @ H / S
        x = x + K * y
        P = (np.eye(6) - np.outer(K, H)) @ P

        # --- LiDAR-odometry position update (spoof-independent) ---
        R_lo = max((LO_DRIFT * dist_align) ** 2, 1.0)
        if method not in NO_LIDAR:
            ylo = lo_w[k] - x[:2]
            Slo = P[:2, :2] + np.eye(2) * R_lo
            Klo = P[:, :2] @ _inv2(Slo)
            x = x + Klo @ ylo
            P = P - Klo @ P[:2, :]

        # --- GNSS position update (attack enters here) ---
        p = PARAMS
        s_ema = p["ema"] * spoof_s[k] + (1 - p["ema"]) * s_ema
        use_gnss, Rg = True, R0
        if method in ("A0_ins_gnss", "A_fixed_gate", "B_raim"):
            Rg = R0
        elif method == "C_hard_gate":
            use_gnss = s_ema < p["tau_hi"]
        elif method == "D_soft_adapt":
            Rg = R0 * np.exp(p["kappa"] * s_ema)
        elif method == "E_hybrid":
            Rg = R0 * np.exp(p["kappa"] * s_ema)
            # hard override uses the RAW probability (not the smoothed s_ema) so it
            # reacts immediately to abrupt onsets; s_ema still drives the soft R scale.
            use_gnss = max(spoof_s[k], s_ema) < p["tau_hi"]
        elif method == "F_oracle":
            use_gnss = label[k] == 0

        if use_gnss:
            yg = gnss[k] - x[:2]
            if method == "E_hybrid":
                # consistency guard: clip the GNSS innovation to what the
                # LiDAR-odometry-propagated state considers plausible.
                lo_pred = lo_w[k] - x[:2]
                lim = BETA_E * np.sqrt(R_lo) + np.linalg.norm(lo_pred)
                if np.linalg.norm(yg) > lim:
                    yg = yg * (lim / np.linalg.norm(yg))
            Sg = P[:2, :2] + np.eye(2) * Rg
            Sg_i = _inv2(Sg)
            nis = yg @ Sg_i @ yg
            gate = CHI2_B if method == "B_raim" else (1e18 if method == "F_oracle" else CHI2)
            if nis <= gate:
                Kg = P[:, :2] @ Sg_i
                x = x + Kg @ yg
                P = P - Kg @ P[:2, :]

        x[2] = (x[2] + np.pi) % (2 * np.pi) - np.pi
        x[5] = np.clip(x[5], 0.8, 1.2)
        est[k] = x[:2]
        Ppos[k] = P[:2, :2]
    return est, Ppos


def metrics(est, Ppos, gt_en, label):
    err = np.linalg.norm(est - gt_en, axis=1)
    atk = label == 1
    out = dict(ate_full=float(np.sqrt(np.mean(err ** 2))))
    if not atk.any():
        out.update(ate_attack=np.nan, maxerr_attack=np.nan, recovery_s=np.nan, nees_attack=np.nan)
        return out
    out["ate_attack"] = float(np.sqrt(np.mean(err[atk] ** 2)))
    out["maxerr_attack"] = float(err[atk].max())
    # NEES over attack window
    d = est[atk] - gt_en[atk]
    nees = []
    for i, P in enumerate(Ppos[atk]):
        try:
            nees.append(d[i] @ np.linalg.solve(P + np.eye(2) * 1e-6, d[i]))
        except Exception:
            pass
    out["nees_attack"] = float(np.mean(nees) / 2.0) if nees else np.nan
    # recovery: first index after attack end with err<1 sustained 3 s
    end = np.where(atk)[0][-1]
    hold = int(3 * RATE)
    rec = np.nan
    run = 0
    for i in range(end + 1, len(err)):
        run = run + 1 if err[i] < 1.0 else 0
        if run >= hold:
            rec = (i - hold + 1 - end) * DT
            break
    out["recovery_s"] = float(rec) if not np.isnan(rec) else float((len(err) - end) * DT)
    return out


def load_variant(inp, scores, split, gi, idx, S, G):
    ddir, sroot = os.path.join(inp, "dataset"), os.path.join(inp, "data")
    r = idx.loc[gi]
    spoof_s = S[G == gi]
    sess, vshort = r["variant"].split("/", 1)
    v = pd.read_parquet(os.path.join(ddir, "variants", sess, vshort + ".parquet"))
    t = v["t"].to_numpy(float)
    gnss = np.c_[v["e"], v["n"]].astype(float)
    label = v["label"].to_numpy(np.uint8)
    n = len(t)
    if len(spoof_s) != n:
        spoof_s = np.interp(np.arange(n), np.linspace(0, n - 1, len(spoof_s)), spoof_s)
    sdir = os.path.join(sroot, r["session"])
    gp = pd.read_parquet(os.path.join(sdir, "gt_pose.parquet"))
    imu = pd.read_parquet(os.path.join(sdir, "imu.parquet"))
    wh = pd.read_parquet(os.path.join(sdir, "wheel.parquet"))
    lo = pd.read_parquet(os.path.join(sdir, "lidar_odom.parquet"))
    gt = {"yaw": -_resample(t, gp["t"].to_numpy(float), gp["yaw"].to_numpy(float))}
    gt_en = np.c_[_resample(t, gp["t"].to_numpy(float), gp["e"].to_numpy(float)),
                  _resample(t, gp["t"].to_numpy(float), gp["n"].to_numpy(float))]
    omega = -_resample(t, imu["t"].to_numpy(float), imu["gz"].to_numpy(float))
    wheel_v = _resample(t, wh["t"].to_numpy(float), wh["speed_mps"].to_numpy(float))
    lo_xy = np.c_[_resample(t, lo["t"].to_numpy(float), lo["lo_x"].to_numpy(float)),
                  _resample(t, lo["t"].to_numpy(float), lo["lo_y"].to_numpy(float))]
    lo_w = _align_lo(gnss, lo_xy, label, min(int(ALIGN_S * RATE), n - 1))
    return r, dict(gnss=gnss, lo_w=lo_w, gt=gt, gt_en=gt_en, omega=omega,
                   wheel_v=wheel_v, spoof_s=spoof_s, label=label)


def tune_on_val(inp, scores, n_var=48):
    """grid-search kappa/ema/tau_hi on val attack variants; minimise median E_hybrid ATE."""
    idx = pd.read_parquet(os.path.join(inp, "features", "val_index.parquet")).reset_index(drop=True)
    sc = np.load(os.path.join(scores, "scores_val.npz"))
    S, G = sc["scores"], sc["groups"]
    rows_i = idx[idx.is_attack == 1].index.tolist()
    rows_i = rows_i[:: max(1, len(rows_i) // n_var)][:n_var]
    data = [load_variant(inp, scores, "val", gi, idx, S, G) for gi in rows_i]
    best, best_p = 1e9, dict(PARAMS)
    for kappa in (3.5, 4.6, 6.0):
        for ema in (0.2, 0.35, 0.5):
            for tau_hi in (0.4, 0.55, 0.7):
                PARAMS.update(kappa=kappa, ema=ema, tau_hi=tau_hi, tau_lo=tau_hi * 0.5)
                ates = []
                for _, d in data:
                    est, Pp = run_ekf("E_hybrid", d["gnss"], d["lo_w"], d["gt"], d["omega"],
                                      d["wheel_v"], d["spoof_s"], d["label"])
                    ates.append(metrics(est, Pp, d["gt_en"], d["label"])["ate_attack"])
                m = float(np.nanmedian(ates))
                if m < best:
                    best, best_p = m, dict(PARAMS)
    PARAMS.update(best_p)
    print(f"tuned PARAMS = {best_p}  (val E_hybrid median ATE {best:.2f} m)", flush=True)
    return best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="..")
    ap.add_argument("--scores", default="../results/monitor_gru96")
    ap.add_argument("--out", default="../results/fusion")
    ap.add_argument("--splits", default="test_town,test_attack,test_both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tune", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.tune:
        p = tune_on_val(args.inp, args.scores)
        json.dump(p, open(os.path.join(args.out, "tuned_params.json"), "w"))

    rows = []
    for split in args.splits.split(","):
        idx = pd.read_parquet(os.path.join(args.inp, "features", f"{split}_index.parquet")).reset_index(drop=True)
        sc = np.load(os.path.join(args.scores, f"scores_{split}.npz"))
        S, G = sc["scores"], sc["groups"]
        atk_rows = idx[idx.is_attack == 1].index.tolist()
        if args.limit:
            step = max(1, len(atk_rows) // args.limit)
            atk_rows = atk_rows[::step][:args.limit]
        for ci, gi in enumerate(atk_rows):
            if ci % 40 == 0:
                print(f"  {split} {ci}/{len(atk_rows)}", flush=True)
            r, d = load_variant(args.inp, args.scores, split, gi, idx, S, G)
            for mth in METHODS:
                est, Ppos = run_ekf(mth, d["gnss"], d["lo_w"], d["gt"], d["omega"],
                                    d["wheel_v"], d["spoof_s"], d["label"])
                mrow = metrics(est, Ppos, d["gt_en"], d["label"])
                mrow.update(method=mth, split=split, family=r["family"],
                            variant=r["variant"], onset_s=r["onset_s"], duration_s=r["duration_s"],
                            max_offset_m=r["max_offset_m"])
                rows.append(mrow)
        print(f"{split}: {len(atk_rows)} attack variants x {len(METHODS)} methods done", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(args.out, "fusion_raw.parquet"))
    agg = (df.groupby(["split", "method"])
             .agg(ate_attack=("ate_attack", "median"),
                  maxerr_attack=("maxerr_attack", "median"),
                  recovery_s=("recovery_s", "median"),
                  nees_attack=("nees_attack", "median"),
                  ate_full=("ate_full", "median"),
                  n=("variant", "count")).reset_index())
    agg.to_csv(os.path.join(args.out, "fusion_summary.csv"), index=False)
    print(agg.to_string(index=False))
    fam = (df.groupby(["split", "family", "method"])["ate_attack"].median().reset_index())
    fam.to_csv(os.path.join(args.out, "fusion_by_family.csv"), index=False)
    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
