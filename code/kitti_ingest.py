"""
KITTI raw (sync+rectified) -> the same 6-stream layout as a CARLA session, so the
CARLA-trained spoofing monitor can be evaluated on real drives with zero retraining
(sim-to-real transfer -- the decisive T-ITS experiment).

Per KITTI drive it writes  data_kitti/<drive>/:
    gnss_clean.parquet   t,e,n           (OXTS lat/lon -> local ENU, origin = frame 0)
    gt_pose.parquet      t,e,n,yaw       (OXTS RTK/INS fused pose = ground truth)
    imu.parquet          t,ax,ay,gz      (OXTS forward/left accel, yaw rate)
    wheel.parquet        t,speed_mps,dist_m   (OXTS forward velocity + its integral)
    lidar_odom.parquet   t,lo_x,lo_y,dist_to_junction   (from KISS-ICP -- see kitti_lidar_odom.py;
                                                         a straight-through GT copy is written as
                                                         a placeholder if --lo_from_gt)
    events.json          {gnss_condition:"nominal", weather:"clear_day", source:"kitti"}

OXTS field order: dataformat.txt from the KITTI raw devkit.

Usage:
  python kitti_ingest.py --kitti ../kitti_raw --out ../data_kitti
  # then LiDAR odometry:  python kitti_lidar_odom.py --kitti ../kitti_raw --out ../data_kitti
"""
from __future__ import annotations
import argparse, glob, json, os
from datetime import datetime
import numpy as np
import pandas as pd
from sensors import geo_to_enu

# OXTS columns we use (0-indexed, per KITTI dataformat.txt)
LAT, LON, ALT, ROLL, PITCH, YAW = 0, 1, 2, 3, 4, 5
VF = 8                       # forward velocity  [m/s]
AF, AL = 14, 15              # forward / leftward acceleration [m/s^2]
WZ, WU = 19, 22              # yaw rate around z / around upward axis [rad/s]


def _read_timestamps(path):
    ts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "2011-09-30 12:34:56.123456789"  -> truncate ns to us for strptime
            head, frac = line.split(".")
            dt = datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
            ts.append(dt.timestamp() + float("0." + frac))
    t = np.asarray(ts, float)
    return t - t[0]


def _load_oxts(drive_dir):
    files = sorted(glob.glob(os.path.join(drive_dir, "oxts", "data", "*.txt")))
    if not files:
        raise FileNotFoundError(f"no oxts/data/*.txt under {drive_dir}")
    rows = [np.fromstring(open(fp).read().strip(), sep=" ") for fp in files]
    O = np.stack(rows)
    t = _read_timestamps(os.path.join(drive_dir, "oxts", "timestamps.txt"))
    n = min(len(t), len(O))
    return t[:n], O[:n]


def _find_drives(kitti_root):
    """any dir matching *_sync that has an oxts/ subdir."""
    out = []
    for d in glob.glob(os.path.join(kitti_root, "**", "*_sync"), recursive=True):
        if os.path.isdir(os.path.join(d, "oxts")):
            out.append(d)
    return sorted(out)


def ingest_drive(drive_dir, out_root, lo_from_gt=False):
    name = os.path.basename(drive_dir)
    t, O = _load_oxts(drive_dir)
    lat0, lon0, alt0 = O[0, LAT], O[0, LON], O[0, ALT]
    e = np.empty(len(t)); n = np.empty(len(t))
    for i in range(len(t)):
        e[i], n[i], _ = geo_to_enu(O[i, LAT], O[i, LON], O[i, ALT], lat0, lon0, alt0)
    yaw = O[:, YAW]                                   # ENU yaw (rad), right-handed, CCW from East
    vf = O[:, VF]
    gz = O[:, WU] if np.any(O[:, WU]) else O[:, WZ]

    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(dict(t=t, e=e, n=n)).to_parquet(os.path.join(d, "gnss_clean.parquet"))
    pd.DataFrame(dict(t=t, e=e, n=n, yaw=yaw)).to_parquet(os.path.join(d, "gt_pose.parquet"))
    pd.DataFrame(dict(t=t, ax=O[:, AF], ay=O[:, AL], gz=gz)).to_parquet(os.path.join(d, "imu.parquet"))
    dist = np.concatenate([[0.0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * np.diff(t))])
    pd.DataFrame(dict(t=t, speed_mps=vf, dist_m=dist)).to_parquet(os.path.join(d, "wheel.parquet"))

    if lo_from_gt:
        # placeholder ONLY for a plumbing smoke-test -- NOT a valid experiment
        # (the reference must be GNSS-independent). Real run uses kitti_lidar_odom.py.
        pd.DataFrame(dict(t=t, lo_x=e, lo_y=n,
                          dist_to_junction=np.full(len(t), 60.0))).to_parquet(
            os.path.join(d, "lidar_odom.parquet"))

    json.dump(dict(gnss_condition="nominal", weather="clear_day", source="kitti",
                   drive=name, n_frames=int(len(t)), duration_s=float(t[-1]),
                   origin=[float(lat0), float(lon0), float(alt0)]),
              open(os.path.join(d, "events.json"), "w"), indent=1)

    speed_kmh = float(np.median(vf) * 3.6)
    print(f"{name}: {len(t)} frames, {t[-1]:.0f}s, {dist[-1]:.0f} m, "
          f"median {speed_kmh:.0f} km/h"
          + ("  [lidar_odom = GT placeholder]" if lo_from_gt else ""), flush=True)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti", default="../kitti_raw")
    ap.add_argument("--out", default="../data_kitti")
    ap.add_argument("--lo_from_gt", action="store_true",
                    help="write a GT-copy lidar_odom placeholder (plumbing smoke only)")
    args = ap.parse_args()
    drives = _find_drives(args.kitti)
    if not drives:
        raise SystemExit(f"no *_sync drives with oxts/ found under {args.kitti}")
    print(f"found {len(drives)} drive(s):")
    names = [ingest_drive(dd, args.out, args.lo_from_gt) for dd in drives]
    print("\ningested:", ", ".join(names))
    print(f"next: python kitti_lidar_odom.py --kitti {args.kitti} --out {args.out}")


if __name__ == "__main__":
    main()
