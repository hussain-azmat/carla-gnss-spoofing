#!/usr/bin/env python3
"""
lidar_odom.py -- thin wrapper over KISS-ICP so run_lidar_odom.py can turn collect.py's
downsampled clouds into per-frame poses (RESEARCH_PLAN §12).

Runs in the **carla** env (py3.10). Targets kiss-icp 1.3.x:
  pose            = KissICP.last_pose  (4x4)
  register_frame(frame_xyz, timestamps) -> (frame, source)
Config is built with kiss_icp.config.load_config(None, max_range=...) so every derived
field (voxel_size, adaptive-threshold seeds, ...) is populated -- hand-setting only
voxel_size on a bare KISSConfig() leaves other fields None and the filter diverges.
"""
from __future__ import annotations
from math import asin, atan2

import numpy as np


class LidarOdometry:
    def __init__(self, max_range: float = 100.0, min_range: float = 1.0,
                 deskew: bool = False, voxel_size: float | None = None):
        self.ok = False
        self.err = None
        self._impl = None
        self._deskew = deskew
        self.last_pose = np.eye(4)
        try:
            from kiss_icp.kiss_icp import KissICP
            try:
                from kiss_icp.config import load_config
                cfg = load_config(None, max_range=max_range)
                cfg.data.min_range = min_range
                cfg.data.deskew = deskew
                if voxel_size:
                    cfg.mapping.voxel_size = voxel_size
            except Exception:
                from kiss_icp.config import KISSConfig
                cfg = KISSConfig()
                cfg.data.max_range = max_range
                cfg.data.min_range = min_range
                cfg.data.deskew = deskew
                cfg.mapping.voxel_size = voxel_size or (max_range / 100.0)
            self._impl = KissICP(cfg)
            self.ok = True
        except Exception as e:  # pragma: no cover
            self.err = repr(e)

    def register(self, points_xyzi: np.ndarray, timestamps: np.ndarray | None = None) -> np.ndarray:
        if not self.ok:
            return self.last_pose
        xyz = np.ascontiguousarray(points_xyzi[:, :3], dtype=np.float64)
        if timestamps is None or not self._deskew:
            ts = np.zeros(len(xyz), dtype=np.float64)
        else:
            ts = np.ascontiguousarray(timestamps, dtype=np.float64)
        try:
            self._impl.register_frame(xyz, ts)
            self.last_pose = np.asarray(self._impl.last_pose, dtype=float)
        except Exception as e:  # pragma: no cover
            self.err = repr(e)
        return self.last_pose

    @staticmethod
    def pose_to_row(pose: np.ndarray) -> dict:
        R = pose[:3, :3]; t = pose[:3, 3]
        yaw = atan2(R[1, 0], R[0, 0])
        pitch = asin(max(-1.0, min(1.0, -R[2, 0])))
        roll = atan2(R[2, 1], R[2, 2])
        return dict(lo_x=float(t[0]), lo_y=float(t[1]), lo_z=float(t[2]),
                    lo_roll=float(roll), lo_pitch=float(pitch), lo_yaw=float(yaw))
