#!/usr/bin/env python3
"""
run_lidar_odom.py -- offline LiDAR odometry over collect.py sessions (RESEARCH_PLAN §12).

Runs in the **carla** env (Python 3.10, has kiss-icp / open3d). collect.py (py3.7) can't
import kiss-icp, so it just dumps downsampled clouds; this turns them into per-frame poses
and then -- unless the session is flagged keep_clouds -- deletes the clouds to reclaim disk.

    conda activate carla
    python run_lidar_odom.py --in ./out                # process every un-done session
    python run_lidar_odom.py --in ./out --session <id>  # just one

Idempotent: skips sessions whose events.json has lidar_odom_done == true.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from lidar_odom import LidarOdometry


def _load_cloud(path: str) -> np.ndarray:
    d = np.load(path)
    key = "xyzi" if "xyzi" in d else ("xyz" if "xyz" in d else d.files[0])
    return np.asarray(d[key], dtype=np.float64)


def process_session(sdir: str, force: bool = False) -> str:
    ev_path = os.path.join(sdir, "events.json")
    ev = json.load(open(ev_path))
    if ev.get("lidar_odom_done") and not force:
        return "skip (done)"
    lf_path = os.path.join(sdir, "lidar_frames.parquet")
    cloud_dir = os.path.join(sdir, "lidar_clouds")
    if not (os.path.exists(lf_path) and os.path.isdir(cloud_dir)):
        return "skip (no clouds)"

    lf = pd.read_parquet(lf_path).sort_values("frame").reset_index(drop=True)
    odom = LidarOdometry(max_range=100.0, min_range=1.0, deskew=False, voxel_size=1.0)
    if not odom.ok:
        return f"FAIL kiss-icp unusable: {odom.err}"

    rows = []
    t0 = time.time()
    for _, r in lf.iterrows():
        fp = os.path.join(cloud_dir, "%08d.npz" % int(r["frame"]))
        if not os.path.exists(fp):
            continue
        pts = _load_cloud(fp)
        pose = odom.register(pts)
        row = dict(t=float(r["t"]), frame=int(r["frame"]),
                   n_points=int(r["n_points"]), n_points_ds=int(r.get("n_points_ds", len(pts))),
                   dist_to_junction=float(r.get("dist_to_junction", 30.0)))
        row.update(LidarOdometry.pose_to_row(pose))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(sdir, "lidar_odom.parquet"), index=False)

    # trajectory length sanity
    if len(out) > 1:
        d = np.diff(out[["lo_x", "lo_y", "lo_z"]].to_numpy(), axis=0)
        pathlen = float(np.linalg.norm(d, axis=1).sum())
    else:
        pathlen = 0.0

    ev["lidar_odom_done"] = True
    ev["lidar_odom_path_m"] = round(pathlen, 1)
    json.dump(ev, open(ev_path, "w"), indent=2)

    kept = ev.get("keep_clouds", False)
    if not kept:
        shutil.rmtree(cloud_dir, ignore_errors=True)

    man_path = os.path.join(sdir, "manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        man["files"] = sorted(f for f in os.listdir(sdir))
        json.dump(man, open(man_path, "w"), indent=2)

    return (f"ok {len(out)} frames, odom path {pathlen:6.0f} m, "
            f"{time.time()-t0:.0f}s, clouds {'kept' if kept else 'pruned'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="./out")
    ap.add_argument("--session", default=None, help="process only this session folder")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(d for d in os.listdir(args.inp)
                          if os.path.exists(os.path.join(args.inp, d, "manifest.json")))
    for s in sessions:
        print(f"{s} : {process_session(os.path.join(args.inp, s), args.force)}", flush=True)


if __name__ == "__main__":
    main()
