#!/usr/bin/env python3
"""
qa_report.py -- check every session in a wave against scenario_matrix.yaml:qa_gates
(see COLLECTION_PLAN.md §6). Runs in the **carla** env (py3.10). Exit code 0 iff all pass.

    conda activate carla
    python qa_report.py --in ~/out --wave 1a_core_train
    python qa_report.py --in ~/out                 # all sessions
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml


def _rate(df):
    t = df["t"].to_numpy()
    return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0.0


def _rigid_align_2d(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    return (R @ (A - ca).T).T + cb


def check(sdir, G):
    ev = json.load(open(os.path.join(sdir, "events.json")))
    cond = ev.get("gnss_condition", "nominal")
    imu = pd.read_parquet(os.path.join(sdir, "imu.parquet"))
    gnss = pd.read_parquet(os.path.join(sdir, "gnss_clean.parquet"))
    gt = pd.read_parquet(os.path.join(sdir, "gt_pose.parquet"))
    lf = pd.read_parquet(os.path.join(sdir, "lidar_frames.parquet"))
    res, ok = {}, True

    r_imu, r_gn, r_li = _rate(imu), _rate(gnss), _rate(lf)
    lo_imu, hi_imu = G["sensor_rate_hz"]["imu"]
    lo_gn, hi_gn = G["sensor_rate_hz"]["gnss"]
    lo_li, hi_li = G["sensor_rate_hz"]["lidar"]
    p1 = lo_imu <= r_imu <= hi_imu and lo_gn <= r_gn <= hi_gn and lo_li <= r_li <= hi_li
    res["rates"] = f"imu {r_imu:.1f} gnss {r_gn:.1f} lidar {r_li:.1f}  {'OK' if p1 else 'FAIL'}"
    ok &= p1

    ge = np.interp(gnss["t"], gt["t"], gt["e"]); gn_ = np.interp(gnss["t"], gt["t"], gt["n"])
    err = np.hypot(gnss["e"] - ge, gnss["n"] - gn_)
    lo, hi = G["gnss_vs_gt_median_m"].get(cond, [0, 99])
    p2 = lo <= np.median(err) <= hi
    res["gnss_vs_gt"] = f"median {np.median(err):.2f} m (band {lo}-{hi})  {'OK' if p2 else 'FAIL'}"
    ok &= p2

    lo_path = os.path.join(sdir, "lidar_odom.parquet")
    if os.path.exists(lo_path):
        lo_df = pd.read_parquet(lo_path)
        P = lo_df[["lo_x", "lo_y"]].to_numpy()
        Gt = np.stack([np.interp(lo_df["t"], gt["t"], gt["e"]),
                       np.interp(lo_df["t"], gt["t"], gt["n"])], 1)
        Pa = _rigid_align_2d(P, Gt)
        e = np.linalg.norm(Pa - Gt, axis=1)
        gp = np.linalg.norm(np.diff(Gt, axis=0), axis=1).sum()
        op = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        drift = 100 * e[-1] / max(gp, 1e-6)
        prel = abs(op - gp) / max(gp, 1e-6)
        p3 = drift <= G["lidar_odom_drift_pct_max"] and prel <= G["lidar_odom_pathlen_rel_err_max"]
        res["lidar_odom"] = f"drift {drift:.1f}%  path-relerr {prel*100:.1f}%  {'OK' if p3 else 'FAIL'}"
        ok &= p3
    else:
        res["lidar_odom"] = "no lidar_odom.parquet  FAIL"
        ok = False

    dur_min = (gt["t"].iloc[-1] - gt["t"].iloc[0]) / 60.0
    n_ev = len(ev.get("junction_onset_events", []))
    per8 = n_ev / max(dur_min / 8.0, 1e-6)
    lo_j, hi_j = G["junction_events_per_8min"]
    p4 = lo_j <= per8 <= hi_j
    res["junction_events"] = f"{n_ev} ({per8:.1f}/8min)  {'OK' if p4 else 'FAIL'}"
    ok &= p4

    stat = float((gt["speed"] < 0.5).mean())
    smax = G.get("stationary_fraction_max", 0.35)
    p5 = stat <= smax
    res["stationary_frac"] = f"{stat*100:.0f}%  (max {smax*100:.0f}%)  {'OK' if p5 else 'FAIL'}"
    ok &= p5

    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.expanduser("~/out"))
    ap.add_argument("--wave", default=None)
    ap.add_argument("--matrix", default="scenario_matrix.yaml")
    args = ap.parse_args()
    G = yaml.safe_load(open(args.matrix))["qa_gates"]

    sessions = sorted(d for d in os.listdir(args.inp)
                      if os.path.exists(os.path.join(args.inp, d, "manifest.json")))
    all_ok = True
    n = 0
    for s in sessions:
        sdir = os.path.join(args.inp, s)
        ev = json.load(open(os.path.join(sdir, "events.json")))
        if args.wave and ev.get("wave") != args.wave:
            continue
        n += 1
        try:
            ok, res = check(sdir, G)
        except Exception as e:
            ok, res = False, {"error": repr(e)}
        all_ok &= ok
        print(f"\n{'PASS' if ok else 'FAIL'}  {s}")
        for k, v in res.items():
            print(f"    {k:16s} {v}")

    print(f"\n{n} sessions checked, {'ALL PASS' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
