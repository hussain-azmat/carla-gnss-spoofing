"""
Phase 2 — consistency-feature extraction for the CARLA GNSS-spoofing benchmark.

For every labelled variant in dataset/dataset_index.parquet:
  - load the (possibly spoofed) GNSS stream + per-timestep attack label,
  - load the parent session's raw IMU / wheel / LiDAR-odometry streams,
  - resample everything to the GNSS 10 Hz grid,
  - align the LiDAR-odometry frame to world ENU on the first clean seconds,
  - compute GNSS-vs-{LiDAR-odom, wheel, IMU} consistency features over
    1 / 3 / 10 s sliding windows plus rolling residual statistics and a CUSUM,
  - write one float32 feature sequence + per-timestep labels per variant.

Splits (see docs/metrics.md), by town x attack-family:
  train       train towns  x {clean, train families}   (minus held-out val sessions)
  val         held-out train-town sessions x same
  test_town   unseen towns x {clean, train families}
  test_attack train towns  x {replay, target_bias}
  test_both   unseen towns x {replay, target_bias}

Output:  features/<split>.npz          X[SumT,F] f32, y[SumT] u8, groups[SumT] i32
         features/<split>_index.parquet   one row per variant (meta + n_steps)
         features/feature_names.json
         features/norm_stats.json         per-feature mean/std computed on TRAIN only

Usage:  python build_features.py --in .. --out ../features
"""
from __future__ import annotations
import argparse, json, os, hashlib
import numpy as np
import pandas as pd

RATE_HZ = 10.0
DT = 1.0 / RATE_HZ
WINDOWS_S = (1.0, 3.0, 10.0)
ALIGN_S = 20.0          # seconds of leading clean data used to align the LO frame
VAL_FRACTION = 0.15     # of train-town sessions held out for val
TRAIN_FAMILIES = {"gradual_pulloff", "jump_and_hold", "random_walk", "zv_freeze"}
TEST_FAMILIES = {"replay", "target_bias"}


def _resample_to(t_dst, t_src, cols):
    """Linear-interpolate each column of `cols` (dict name->array) onto t_dst."""
    out = {}
    for name, v in cols.items():
        out[name] = np.interp(t_dst, t_src, v, left=v[0], right=v[-1])
    return out


def _win_steps(w_s):
    return max(1, int(round(w_s * RATE_HZ)))


def _disp(p_xy, k):
    """||p(t) - p(t-k)|| per timestep, with the first k samples measured from t=0."""
    n = len(p_xy)
    prev = np.empty_like(p_xy)
    prev[:k] = p_xy[0]
    prev[k:] = p_xy[:-k]
    d = p_xy - prev
    return np.hypot(d[:, 0], d[:, 1]), d


def _bearing(dvec):
    return np.arctan2(dvec[:, 1], dvec[:, 0])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rolling(x, k):
    """causal rolling mean & std over k samples (min_periods=1)."""
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - k + 1)
    cnt = (idx - lo + 1).astype(float)
    s = c1[idx + 1] - c1[lo]
    ss = c2[idx + 1] - c2[lo]
    mean = s / cnt
    var = np.maximum(0.0, ss / cnt - mean * mean)
    return mean, np.sqrt(var)


def _features_for_variant(vdf: pd.DataFrame, sdir: str):
    t = vdf["t"].to_numpy(float)
    n = len(t)
    gnss = np.c_[vdf["e"].to_numpy(float), vdf["n"].to_numpy(float)]
    label = vdf["label"].to_numpy(np.uint8)

    imu = pd.read_parquet(os.path.join(sdir, "imu.parquet"))
    whl = pd.read_parquet(os.path.join(sdir, "wheel.parquet"))
    lo = pd.read_parquet(os.path.join(sdir, "lidar_odom.parquet"))

    r = _resample_to(t, imu["t"].to_numpy(float),
                     {"ax": imu["ax"].to_numpy(float), "ay": imu["ay"].to_numpy(float),
                      "gz": imu["gz"].to_numpy(float)})
    ax, ay, gz = r["ax"], r["ay"], r["gz"]
    rw = _resample_to(t, whl["t"].to_numpy(float),
                      {"spd": whl["speed_mps"].to_numpy(float),
                       "dist": whl["dist_m"].to_numpy(float)})
    speed, wheel_dist = rw["spd"], rw["dist"]
    rl = _resample_to(t, lo["t"].to_numpy(float),
                      {"x": lo["lo_x"].to_numpy(float), "y": lo["lo_y"].to_numpy(float),
                       "d2j": lo["dist_to_junction"].to_numpy(float)})
    lo_xy = np.c_[rl["x"], rl["y"]]
    d2j = np.clip(rl["d2j"], 0, 60)

    # --- align the LiDAR-odom frame to world ENU on leading clean data -----------
    m = int(ALIGN_S * RATE_HZ)
    m = min(m, n - 1)
    clean_lead = (label[:m] == 0)
    a = gnss[:m][clean_lead] - gnss[0]
    b = lo_xy[:m][clean_lead] - lo_xy[0]
    if len(a) >= 5 and np.linalg.norm(b) > 1.0:
        # least-squares rotation (Kabsch, 2-D) mapping LO displacements -> world
        H = b.T @ a
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
    else:
        R = np.eye(2)
    lo_w = (lo_xy - lo_xy[0]) @ R.T + gnss[0]

    feats = {}
    feats["speed"] = speed
    feats["yaw_rate_abs"] = np.abs(gz)
    feats["accel_lon"] = ax
    feats["accel_lat"] = ay
    feats["dist_to_junction"] = d2j

    ev = json.load(open(os.path.join(sdir, "events.json")))
    cond = ev.get("gnss_condition", "nominal")
    wx = ev.get("weather", "")
    feats["is_canyon"] = np.full(n, float(cond == "canyon"))
    feats["is_tunnel"] = np.full(n, float(cond == "tunnel"))
    feats["is_rain"] = np.full(n, float("rain" in wx or "wet" in wx))
    feats["is_night"] = np.full(n, float("night" in wx))

    resid_gl_1s = None
    for w in WINDOWS_S:
        k = _win_steps(w)
        dg, dgv = _disp(gnss, k)
        dl, dlv = _disp(lo_w, k)
        dw = wheel_dist - np.r_[np.full(k, wheel_dist[0]), wheel_dist[:-k]]
        tag = f"{int(w)}s"
        feats[f"gnss_disp_{tag}"] = dg
        feats[f"lo_disp_{tag}"] = dl
        feats[f"resid_gnss_lo_{tag}"] = dg - dl
        feats[f"resid_gnss_wheel_{tag}"] = dg - dw
        feats[f"resid_lo_wheel_{tag}"] = dl - dw
        feats[f"ratio_gnss_lo_{tag}"] = dg / (dl + 0.05)
        bres = _wrap(_bearing(dgv) - _bearing(dlv))
        moving = (dg > 0.3) & (dl > 0.3)
        bres = np.where(moving, bres, 0.0)
        feats[f"bearing_resid_{tag}"] = bres
        dres = np.gradient(dg - dl) * RATE_HZ
        feats[f"d_resid_gnss_lo_{tag}"] = dres
        if w == 1.0:
            resid_gl_1s = dg - dl

    for k_s in (5.0, 15.0):
        mean, std = _rolling(resid_gl_1s, int(k_s * RATE_HZ))
        feats[f"resid_gl_mean_{int(k_s)}s"] = mean
        feats[f"resid_gl_std_{int(k_s)}s"] = std

    # standard two-sided CUSUM on the 1 s GNSS-vs-LO residual
    kcs, hcs = 0.10, 5.0
    sp = np.zeros(n); sm = np.zeros(n)
    for i in range(1, n):
        sp[i] = max(0.0, sp[i - 1] + resid_gl_1s[i] - kcs)
        sm[i] = max(0.0, sm[i - 1] - resid_gl_1s[i] - kcs)
    feats["cusum_pos"] = np.minimum(sp, 3 * hcs)
    feats["cusum_neg"] = np.minimum(sm, 3 * hcs)

    names = list(feats.keys())
    X = np.stack([feats[k].astype(np.float32) for k in names], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if ABLATE == "no_lidar":          # drop only the GNSS-vs-LiDAR-odometry features
        drop = ("lo_disp", "resid_gnss_lo", "resid_lo_wheel", "bearing_resid",
                "d_resid_gnss_lo", "ratio_gnss_lo", "resid_gl", "cusum")
    elif ABLATE == "no_ref":          # drop EVERY un-spoofable-reference cross-check
        drop = ("lo_disp", "resid_gnss_lo", "resid_lo_wheel", "bearing_resid",
                "d_resid_gnss_lo", "ratio_gnss_lo", "resid_gl", "cusum",
                "resid_gnss_wheel")   # only GNSS-vs-IMU / raw GNSS dynamics remain
    else:
        drop = ()
    for j, nm in enumerate(names):
        if any(k in nm for k in drop):
            X[:, j] = 0.0
    return names, X, label


ABLATE = None


def _split_of(row, val_sessions):
    tt, fam = row["town_split"], row["family"]
    fs = row["attack_split"]
    is_test_fam = fam in TEST_FAMILIES
    if tt == "train":
        if is_test_fam:
            return "test_attack"
        return "val" if row["session"] in val_sessions else "train"
    else:  # test_unseen town
        return "test_both" if is_test_fam else "test_town"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="..")
    ap.add_argument("--out", default="../features")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ablate", choices=["no_lidar", "no_ref"], default=None,
                    help="no_lidar = zero GNSS-vs-LiDAR-odom feats; "
                         "no_ref = zero every un-spoofable-reference cross-check")
    args = ap.parse_args()
    global ABLATE
    ABLATE = args.ablate
    if ABLATE:
        print(f"ABLATION: {ABLATE}", flush=True)

    ddir = os.path.join(args.inp, "dataset")
    sroot = os.path.join(args.inp, "data")
    idx = pd.read_parquet(os.path.join(ddir, "dataset_index.parquet"))
    os.makedirs(args.out, exist_ok=True)

    train_sessions = sorted(idx[idx.town_split == "train"].session.unique())
    hold = {s for s in train_sessions
            if int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 < int(VAL_FRACTION * 100)}
    print(f"val sessions ({len(hold)}/{len(train_sessions)}): {sorted(hold)}")

    if args.limit:
        idx = idx.groupby("family", group_keys=False).head(max(1, args.limit // 6))

    buckets = {k: {"X": [], "y": [], "g": [], "meta": []}
               for k in ["train", "val", "test_town", "test_attack", "test_both"]}
    feat_names = None
    for i, row in idx.reset_index(drop=True).iterrows():
        vpath = os.path.join(ddir, row["path"].replace("\\", "/"))
        vdf = pd.read_parquet(vpath)
        sdir = os.path.join(sroot, row["session"])
        names, X, y = _features_for_variant(vdf, sdir)
        feat_names = feat_names or names
        sp = _split_of(row, hold)
        b = buckets[sp]
        gid = len(b["meta"])
        b["X"].append(X); b["y"].append(y)
        b["g"].append(np.full(len(y), gid, np.int32))
        b["meta"].append(dict(variant=f'{row["session"]}/{row["variant"]}',
                              session=row["session"], town=row["town"], weather=row["weather"],
                              family=row["family"], is_attack=int(row["is_attack"]),
                              hard_negative=int(row["hard_negative"]),
                              onset_s=row["onset_s"], duration_s=row["duration_s"],
                              max_offset_m=row["max_offset_m"], n_steps=len(y), split=sp))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(idx)}")

    stats = None
    for sp, b in buckets.items():
        if not b["meta"]:
            continue
        X = np.concatenate(b["X"]).astype(np.float32)
        y = np.concatenate(b["y"]).astype(np.uint8)
        g = np.concatenate(b["g"]).astype(np.int32)
        np.savez_compressed(os.path.join(args.out, f"{sp}.npz"), X=X, y=y, groups=g)
        pd.DataFrame(b["meta"]).to_parquet(os.path.join(args.out, f"{sp}_index.parquet"))
        pos = float(y.mean())
        print(f"{sp:11s} {len(b['meta']):4d} seq  {len(y):8d} steps  {pos*100:.2f}% attack")
        if sp == "train":
            stats = dict(mean=X.mean(0).tolist(), std=(X.std(0) + 1e-6).tolist())

    json.dump(feat_names, open(os.path.join(args.out, "feature_names.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(args.out, "norm_stats.json"), "w"))
    print(f"\n{len(feat_names)} features -> {args.out}")


if __name__ == "__main__":
    main()
