"""
Sim-to-real transfer test (the decisive T-ITS experiment).

Take the GRU spoofing monitor trained ONLY on CARLA, inject the same 6 attack
families on real KITTI drives, build the same 39 consistency features, and score
with the CARLA-frozen normalisation + val threshold. No retraining, no fine-tuning.

Reports, per family and overall on real data:
  ROC-AUC, PR-AUC, TPR @ (CARLA-val) 1% FPR, median time-to-alarm, miss rate,
  and the clean-drive false-alarm rate (the KITTI clean sequences are the
  real-world hard negatives).

Usage:
  python kitti_ingest.py       --kitti ../kitti_raw --out ../data_kitti
  python kitti_lidar_odom.py   --kitti ../kitti_raw --out ../data_kitti
  python kitti_eval.py --data_kitti ../data_kitti --carla_feats ../features \
      --ckpt ../results/monitor_v11/gru_monitor.pt --out ../results/kitti
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
import torch

from sensors import enu_to_geo
from attacks import CleanGnss, apply, FAMILIES
import build_features as BF
from metrics_detect import roc_auc, pr_auc, threshold_at_fpr, evaluate
from train_monitor import GRUMonitor

N_PER_FAMILY = 6
MIN_MOVING_SPEED = 3.0


def load_drive(ddir):
    gt = pd.read_parquet(os.path.join(ddir, "gt_pose.parquet"))
    ev = json.load(open(os.path.join(ddir, "events.json")))
    lat0, lon0, alt0 = ev["origin"]
    t = gt["t"].to_numpy(float)
    e = gt["e"].to_numpy(float); n = gt["n"].to_numpy(float)
    yaw = gt["yaw"].to_numpy(float)
    lat = np.empty(len(t)); lon = np.empty(len(t))
    for i in range(len(t)):
        la, lo, _ = enu_to_geo(e[i], n[i], 0.0, lat0, lon0, alt0)
        lat[i], lon[i] = la, lo
    clean = CleanGnss(t=t, lat=lat, lon=lon, alt=np.full(len(t), alt0),
                      origin=(lat0, lon0, alt0),
                      gt_en=np.c_[e, n], gt_heading=yaw)
    return clean, np.c_[e, n], t


def make_variants(clean, en, t, seed0):
    """clean + N spoofed variants per family; each -> (vdf, meta)."""
    rng = np.random.default_rng(seed0)
    speed = np.hypot(np.gradient(en[:, 0]) , np.gradient(en[:, 1])) / np.median(np.diff(t))
    moving = t[speed > MIN_MOVING_SPEED]
    onset_events = list(rng.choice(moving, size=min(12, len(moving)), replace=False)) if len(moving) else None

    out = [(pd.DataFrame(dict(t=t, e=en[:, 0], n=en[:, 1],
                              label=np.zeros(len(t), np.uint8))),
            dict(family="clean", is_attack=0, hard_negative=1, onset_s=np.nan,
                 duration_s=np.nan, max_offset_m=0.0))]
    for fi, fam in enumerate(FAMILIES):
        for j in range(N_PER_FAMILY):
            r = apply(clean, fam, seed=int(seed0 + 1000 * fi + j), onset_events=onset_events)
            off = r["offset_en"]
            vdf = pd.DataFrame(dict(t=t, e=en[:, 0] + off[:, 0], n=en[:, 1] + off[:, 1],
                                   label=r["label"]))
            out.append((vdf, dict(family=fam, is_attack=1, hard_negative=0,
                                  onset_s=r["meta"]["onset_s"], duration_s=r["meta"]["duration_s"],
                                  max_offset_m=r["meta"]["max_applied_offset_m"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_kitti", default="../data_kitti")
    ap.add_argument("--carla_feats", default="../features")
    ap.add_argument("--ckpt", default="../results/monitor_v11/gru_monitor.pt")
    ap.add_argument("--carla_val_scores", default="../results/monitor_v11/scores_val.npz")
    ap.add_argument("--out", default="../results/kitti")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    P = lambda *a: print(*a, flush=True)

    st = json.load(open(os.path.join(args.carla_feats, "norm_stats.json")))
    mean = np.array(st["mean"], np.float32); std = np.array(st["std"], np.float32)
    names = json.load(open(os.path.join(args.carla_feats, "feature_names.json")))
    model = GRUMonitor(len(names), 96)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    zv = np.load(args.carla_val_scores)
    thr1 = threshold_at_fpr(zv["y"], zv["scores"], 0.01)
    thr01 = threshold_at_fpr(zv["y"], zv["scores"], 0.001)
    P(f"CARLA-val thresholds: 1%FPR={thr1:.3f}  0.1%FPR={thr01:.3f}")

    drives = sorted(d for d in glob.glob(os.path.join(args.data_kitti, "*"))
                    if os.path.isdir(d) and os.path.exists(os.path.join(d, "lidar_odom.parquet")))
    if not drives:
        raise SystemExit("no ingested KITTI drives with lidar_odom.parquet; "
                         "run kitti_ingest.py then kitti_lidar_odom.py")
    P(f"{len(drives)} KITTI drive(s): {[os.path.basename(d) for d in drives]}")

    S, Y, G, META = [], [], [], []
    gid = 0
    for ddir in drives:
        name = os.path.basename(ddir)
        clean, en, t = load_drive(ddir)
        variants = make_variants(clean, en, t, args.seed)
        P(f"\n{name}: {len(variants)} variants ({len(t)} steps, {t[-1]:.0f}s)")
        for vdf, vmeta in variants:
            fn, X, y = BF._features_for_variant(vdf, ddir)
            X = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            with torch.no_grad():
                logit, _ = model(torch.from_numpy(X)[None])
                s = torch.sigmoid(logit)[0].numpy()
            S.append(s); Y.append(y.astype(np.uint8))
            G.append(np.full(len(y), gid, np.int64))
            META.append(dict(drive=name, **vmeta))
            gid += 1

    S = np.concatenate(S); Y = np.concatenate(Y); G = np.concatenate(G)
    meta = pd.DataFrame(META); meta["hard_negative"] = meta["hard_negative"].astype(int)
    np.savez_compressed(os.path.join(args.out, "kitti_scores.npz"), scores=S, y=Y, groups=G)
    meta.to_parquet(os.path.join(args.out, "kitti_index.parquet"))

    # overall + per-family metrics, thresholds frozen from CARLA val
    rows = []
    def block(tag, mask_groups):
        gsel = np.isin(G, mask_groups)
        yb, sb, gb = Y[gsel], S[gsel], G[gsel]
        if yb.sum() == 0 or (yb == 0).sum() == 0:
            return
        sub = meta.iloc[sorted(set(gb.tolist()))].reset_index(drop=True)
        remap = {g: i for i, g in enumerate(sorted(set(gb.tolist())))}
        gb2 = np.array([remap[g] for g in gb])
        r = evaluate(tag, yb, sb, gb2, sub, zv["y"], zv["scores"])
        r["tag"] = tag; r["n_seq"] = len(set(gb.tolist()))
        rows.append(r)

    all_g = meta.index.to_numpy()
    atk_g = meta.index[meta.is_attack == 1].to_numpy()
    cln_g = meta.index[meta.is_attack == 0].to_numpy()
    block("ALL", all_g)
    block("attacks_only", np.r_[atk_g, cln_g])
    for fam in FAMILIES:
        fg = meta.index[meta.family == fam].to_numpy()
        block(fam, np.r_[fg, cln_g])

    def far_per_hr(groups_to_check, thr):
        fa = hrs = 0.0
        for g in groups_to_check:
            det = S[G == g] >= thr
            if len(det) == 0:
                continue
            fa += int(np.sum(det[1:] & ~det[:-1]) + det[0])
            hrs += len(det) * 0.1 / 3600.0
        return fa / hrs if hrs else float("nan")

    # clean-drive false alarm rate at the CARLA operating point (real-world hard negs)
    clean_far = far_per_hr(cln_g, thr1)

    # ---- domain adaptation: align the CLEAN score distribution (logit space) to the
    #      target domain, leave-one-drive-out, using real clean data only (no labels) --
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # CARLA clean-score stats (negatives of the val split) in logit space
    Lv = _logit(zv["scores"]); yv = zv["y"]
    mu_c, sd_c = Lv[yv == 0].mean(), Lv[yv == 0].std() + 1e-6
    z_val = (Lv - mu_c) / sd_c
    thr_z = threshold_at_fpr(yv, z_val, 0.01)          # frozen on CARLA, in aligned units

    drive_names = sorted(meta["drive"].unique())
    lodo = []
    if len(drive_names) >= 2:
        for dn in drive_names:
            cal_clean = meta.index[(meta.drive != dn) & (meta.is_attack == 0)]
            s_cal = np.concatenate([S[G == g] for g in cal_clean])
            Lc = _logit(s_cal)
            mu_t, sd_t = Lc.mean(), Lc.std() + 1e-6
            thr_pctl = float(np.percentile(s_cal, 99))   # naive percentile threshold

            te = meta.index[meta.drive == dn]
            te_cln = [g for g in te if meta.loc[g, "is_attack"] == 0]
            gsel = np.isin(G, list(te))
            yb, sb = Y[gsel], S[gsel]
            zb = (_logit(sb) - mu_t) / sd_t              # aligned score on held-out drive
            rec_pctl = float((sb[yb == 1] >= thr_pctl).mean())
            rec_z = float((zb[yb == 1] >= thr_z).mean())
            # FAR under aligned threshold
            fa = hrs = 0.0
            for g in te_cln:
                zc = (_logit(S[G == g]) - mu_t) / sd_t
                det = zc >= thr_z
                fa += int(np.sum(det[1:] & ~det[:-1]) + det[0]); hrs += len(det) * 0.1 / 3600.0
            far_z = fa / hrs if hrs else float("nan")
            lodo.append(dict(drive=dn, auc=roc_auc(yb, sb),
                             recall_pctl_thr=rec_pctl, far_pctl_thr=far_per_hr(te_cln, thr_pctl),
                             recall_aligned=rec_z, far_aligned=far_z,
                             far_carla_thr=far_per_hr(te_cln, thr1)))
    lodo_df = pd.DataFrame(lodo)
    if len(lodo_df):
        lodo_df.to_csv(os.path.join(args.out, "kitti_domain_adapt.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "kitti_metrics.csv"), index=False)
    with open(os.path.join(args.out, "SUMMARY.md"), "w") as f:
        f.write("# Sim-to-real: CARLA-trained monitor on real KITTI drives\n\n")
        f.write(f"Model: GRU (2x96) trained only on CARLA. Normalisation + operating "
                f"threshold frozen from CARLA. {len(drives)} KITTI drive(s), "
                f"{N_PER_FAMILY} spoof variants/family/drive + clean.\n\n")
        f.write(f"- Clean-drive false-alarm rate at CARLA-val 1% FPR: "
                f"**{clean_far:.1f} / hour**\n")
        f.write(f"- CARLA reference (test_town): ROC-AUC 0.964, PR-AUC 0.771, "
                f"recall@1%FPR 0.761\n\n")
        cols = ["tag", "n_seq", "roc_auc", "pr_auc", "tpr_at_1pct_fpr",
                "median_tta_s", "miss_rate"]
        f.write(df[cols].round(3).to_markdown(index=False) + "\n")
        if len(lodo_df):
            f.write("\n## Domain adaptation (leave-one-drive-out, real clean data only)\n\n")
            f.write("`pctl_thr` = threshold at 99th percentile of real clean scores. "
                    "`aligned` = CARLA score distribution matched to the target domain in "
                    "logit space (mean+std of clean scores), CARLA-frozen 1% FPR threshold "
                    "applied in the aligned units. Both use unlabelled real clean data.\n\n")
            f.write(lodo_df.round(3).to_markdown(index=False) + "\n\n")
            f.write(f"- CARLA-threshold (no adaptation): recall "
                    f"{df.loc[df.tag=='ALL','tpr_at_1pct_fpr'].iloc[0]:.2f}, "
                    f"FAR {lodo_df['far_carla_thr'].mean():.0f}/hr\n")
            f.write(f"- percentile recalibration:        recall "
                    f"{lodo_df['recall_pctl_thr'].mean():.2f}, "
                    f"FAR {lodo_df['far_pctl_thr'].mean():.0f}/hr\n")
            f.write(f"- **distribution alignment:        recall "
                    f"{lodo_df['recall_aligned'].mean():.2f}, "
                    f"FAR {lodo_df['far_aligned'].mean():.0f}/hr**\n")
    P("\n" + open(os.path.join(args.out, "SUMMARY.md")).read())


if __name__ == "__main__":
    main()
