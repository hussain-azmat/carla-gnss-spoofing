"""
LiDAR odometry (KISS-ICP) on KITTI raw Velodyne scans -> the GNSS-independent,
drift-bounded reference the spoofing monitor needs.

Writes / overwrites  data_kitti/<drive>/lidar_odom.parquet  with
    t, lo_x, lo_y, dist_to_junction(=60 const; unknown for KITTI)
t is taken frame-for-frame from the drive's OXTS timeline (KITTI sync = 1:1 across
sensors at ~10 Hz), so it lines up with gnss_clean / imu / wheel.

Usage:  python kitti_lidar_odom.py --kitti ../kitti_raw --out ../data_kitti
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import load_config
from kiss_icp.pybind import kiss_icp_pybind as _kp
from kitti_ingest import _find_drives, _read_timestamps


def _correct_kitti_scan(pts):
    # KITTI HDL-64E has a known vertical-angle calibration error; every KITTI
    # odometry method applies this correction (KISS-ICP's own kitti dataloader does).
    return np.asarray(_kp._correct_kitti_scan(_kp._Vector3dVector(pts)))


def _scan_timestamps(pts):
    # KITTI convention: per-point time from azimuth, normalised to [0,1)
    yaw = -np.arctan2(pts[:, 1], pts[:, 0])
    return 0.5 * (yaw / np.pi + 1.0)


def lidar_odom_for_drive(drive_dir, out_root):
    name = os.path.basename(drive_dir)
    d = os.path.join(out_root, name)
    if not os.path.isdir(d):
        print(f"skip {name}: run kitti_ingest.py first"); return
    bins = sorted(glob.glob(os.path.join(drive_dir, "velodyne_points", "data", "*.bin")))
    if not bins:
        print(f"skip {name}: no velodyne_points/data/*.bin"); return
    t = _read_timestamps(os.path.join(drive_dir, "oxts", "timestamps.txt"))
    n = min(len(bins), len(t))
    bins, t = bins[:n], t[:n]

    ki = KissICP(load_config(None, max_range=100.0))
    xy = np.zeros((n, 2))
    for i, fp in enumerate(bins):
        pts = np.fromfile(fp, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        pts = _correct_kitti_scan(pts)
        ki.register_frame(pts, _scan_timestamps(pts))
        p = np.asarray(ki.last_pose)
        xy[i] = p[0, 3], p[1, 3]
        if (i + 1) % 200 == 0:
            print(f"  {name}: {i+1}/{n}", flush=True)

    pd.DataFrame(dict(t=t, lo_x=xy[:, 0], lo_y=xy[:, 1],
                      dist_to_junction=np.full(n, 60.0))).to_parquet(
        os.path.join(d, "lidar_odom.parquet"))
    # drift sanity: rigid-align LO to the OXTS track (Kabsch, 2-D) then ATE
    gt = pd.read_parquet(os.path.join(d, "gt_pose.parquet"))
    G = np.c_[gt["e"].to_numpy() - gt["e"].iloc[0], gt["n"].to_numpy() - gt["n"].iloc[0]][:n]
    Lc, Gc = xy - xy.mean(0), G - G.mean(0)
    U, _, Vt = np.linalg.svd(Lc.T @ Gc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    err = np.hypot(*((Lc @ R.T + G.mean(0)) - G).T)
    glen = float(np.hypot(*np.diff(G, axis=0).T).sum())
    print(f"{name}: {n} scans, path {glen:.0f} m, LiDAR-odom ATE {err.mean():.2f} m "
          f"mean / {err.max():.2f} m max ({100*err.mean()/max(glen,1):.2f}% / "
          f"{100*err.max()/max(glen,1):.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti", default="../kitti_raw")
    ap.add_argument("--out", default="../data_kitti")
    args = ap.parse_args()
    drives = _find_drives(args.kitti)
    if not drives:
        raise SystemExit(f"no *_sync drives under {args.kitti}")
    for dd in drives:
        lidar_odom_for_drive(dd, args.out)


if __name__ == "__main__":
    main()
