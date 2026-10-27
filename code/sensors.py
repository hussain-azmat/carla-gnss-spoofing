#!/usr/bin/env python3
"""
sensors.py -- sensor suite for the GNSS-spoofing benchmark (RESEARCH_PLAN.md §12).

One place that defines every sensor blueprint + noise model + mounting, plus small
helpers to spawn them onto a vehicle and register ring-buffer callbacks. Imported by
collect.py; also runnable standalone for a quick attach/detach check.

CARLA 0.9.15 API. World tick = 100 Hz (fixed_delta_seconds = 0.01).

Sensor rates (sensor_tick):
    IMU     100 Hz  (every tick)          consumer-MEMS noise
    GNSS     10 Hz                        base + optional canyon multipath; SPOOF applied post-hoc
    LiDAR    10 Hz   64 ch ~130k pts/rev  ray-cast (Vulkan path)
    camera   10 Hz   960x540              validation subset only (off by default)

Wheel odometry: CARLA has no wheel-encoder sensor. We derive it in collect.py from
vehicle.get_velocity() projected on heading + scale error + white noise (see WHEEL_*).
"""
from __future__ import annotations
import argparse
import collections
import dataclasses
import math
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

try:
    import carla
except ImportError:  # allow docs/lint without the egg
    carla = None  # type: ignore


# --------------------------------------------------------------------------------------
# noise models  (locked Phase-0 values -- see docs/threat_model.md)
# --------------------------------------------------------------------------------------

# Consumer-grade MEMS IMU (same class as the units the ICIICE 2025 dataset assumed).
IMU_NOISE = dict(
    noise_accel_stddev_x="0.10", noise_accel_stddev_y="0.10", noise_accel_stddev_z="0.10",   # m/s^2
    noise_gyro_stddev_x="0.010", noise_gyro_stddev_y="0.010", noise_gyro_stddev_z="0.010",   # rad/s
    noise_gyro_bias_x="0.0012", noise_gyro_bias_y="0.0012", noise_gyro_bias_z="0.0012",
)

# GNSS receiver base noise (open-sky, ~consumer L1). Canyon multipath added on top per-scenario.
GNSS_BASE_NOISE = dict(
    noise_lat_stddev="0.30e-5", noise_lon_stddev="0.30e-5", noise_alt_stddev="0.60",  # deg / deg / m
    noise_lat_bias="0.0", noise_lon_bias="0.0", noise_alt_bias="0.0",
)
# ~1e-5 deg latitude ~= 1.11 m.  0.30e-5 deg ~= 0.33 m 1-sigma horizontal.
GNSS_CANYON_MULTIPATH_STDDEV_M = 2.5   # extra horizontal 1-sigma in "urban canyon" hard-negative runs
GNSS_TUNNEL_OUTAGE = True             # in tunnels: drop GNSS fixes entirely (hard negative)

# Wheel odometry synthesis (used in collect.py, kept here so all sensor params live together)
WHEEL_SCALE_ERROR = 0.02      # +/- 2% systematic scale (tire wear / pressure), drawn once per session
WHEEL_SPEED_NOISE_STD = 0.05  # m/s white noise on wheel speed
WHEEL_RATE_HZ = 50

# 64-beam LiDAR.
# CARLA sync-mode gotcha (verified on the VM): a full 360deg sweep in one callback ONLY
# happens when rotation_frequency == world tick rate (1/fixed_delta_seconds) AND
# sensor_tick == 0 (fire every tick). Any other combo delivers a rotating partial wedge,
# which makes LiDAR odometry diverge. So we run the LiDAR at the world rate (100 Hz full
# sweeps) and collect.py keeps every 10th frame -> 10 Hz, full 360deg, ~70k pts.
# The LiDAR fires a ~36deg wedge every world tick (rotation_frequency == 10, so one full
# turn per 10 ticks). collect.py accumulates 10 consecutive wedges, de-skews each by its
# own sensor->world pose, and stores one full 360deg sweep at 10 Hz. This costs ~5x less
# ray-casting than forcing rotation_frequency == 100 for an instantaneous full sweep, and
# the de-skewed result is closer to how a real spinning LiDAR behaves.
LIDAR_ROT_HZ = 10
LIDAR_KEEP_EVERY = 10                  # wedges per stored sweep (100 Hz -> 10 Hz)
LIDAR_PPS = 1_300_000                  # ~13k pts/wedge -> ~130k pts/sweep
LIDAR_ATTR = dict(
    channels="64", range="90.0", points_per_second=str(LIDAR_PPS),
    rotation_frequency=str(LIDAR_ROT_HZ), upper_fov="10.0", lower_fov="-30.0",
    dropoff_general_rate="0.10", dropoff_intensity_limit="0.8",
    dropoff_zero_intensity="0.4", noise_stddev="0.01", sensor_tick="0.0",
)

# mounting transforms (metres, CARLA left-handed; x fwd, y right, z up)
MOUNT = dict(
    imu=(0.0, 0.0, 0.0),
    gnss=(0.0, 0.0, 0.0),
    lidar=(0.0, 0.0, 1.8),
    camera=(1.5, 0.0, 1.6),
)


@dataclasses.dataclass
class SensorSpec:
    name: str
    bp_id: str
    attrs: Dict[str, str]
    location: tuple
    rotation: tuple = (0.0, 0.0, 0.0)          # pitch, yaw, roll
    enabled: bool = True


def gnss_attrs_for_condition(condition: str = "nominal") -> Dict[str, str]:
    """GNSS blueprint attrs for a scenario_matrix gnss_condition.

    nominal -> base open-sky noise
    canyon  -> base noise inflated to include ~GNSS_CANYON_MULTIPATH_STDDEV_M horizontal
    tunnel  -> same inflated noise (geometry-based outage windows are applied in collect.py
               when it can detect a tunnel segment; otherwise this is the graceful fallback)
    """
    attrs = {**GNSS_BASE_NOISE, "sensor_tick": "0.1"}
    if condition in ("canyon", "tunnel"):
        # deg/m: 1e-5 deg lat ~= 1.11 m -> convert the extra metres to a deg stddev
        extra_deg = GNSS_CANYON_MULTIPATH_STDDEV_M / 1.11e5
        base_deg = float(GNSS_BASE_NOISE["noise_lat_stddev"])
        infl = f"{(base_deg**2 + extra_deg**2) ** 0.5:.3e}"
        attrs["noise_lat_stddev"] = infl
        attrs["noise_lon_stddev"] = infl
        attrs["noise_alt_stddev"] = f"{float(GNSS_BASE_NOISE['noise_alt_stddev']) + 3.0:.2f}"
    return attrs


def sensor_specs(with_camera: bool = False, gnss_condition: str = "nominal") -> List[SensorSpec]:
    specs = [
        SensorSpec("imu", "sensor.other.imu",
                   {**IMU_NOISE, "sensor_tick": "0.01"}, MOUNT["imu"]),
        SensorSpec("gnss", "sensor.other.gnss",
                   gnss_attrs_for_condition(gnss_condition), MOUNT["gnss"]),
        SensorSpec("lidar", "sensor.lidar.ray_cast",
                   dict(LIDAR_ATTR), MOUNT["lidar"]),
    ]
    if with_camera:
        specs.append(SensorSpec("camera", "sensor.camera.rgb",
                                {"image_size_x": "960", "image_size_y": "540",
                                 "fov": "90", "sensor_tick": "0.1"}, MOUNT["camera"]))
    return specs


# --------------------------------------------------------------------------------------
# spawn / callback plumbing
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class SensorHandle:
    spec: SensorSpec
    actor: "carla.Actor"
    buf: Deque                                  # bounded deque of (frame, sim_time, payload)


def _transform(spec: SensorSpec) -> "carla.Transform":
    x, y, z = spec.location
    p, yaw, r = spec.rotation
    return carla.Transform(carla.Location(x=x, y=y, z=z),
                           carla.Rotation(pitch=p, yaw=yaw, roll=r))


def _decode(name: str, data) -> dict:
    """CARLA sensor measurement -> plain dict of numpy/py scalars (pickle-safe)."""
    if name == "imu":
        a, g = data.accelerometer, data.gyroscope
        return dict(accel=[a.x, a.y, a.z], gyro=[g.x, g.y, g.z], compass=float(data.compass))
    if name == "gnss":
        return dict(lat=float(data.latitude), lon=float(data.longitude), alt=float(data.altitude))
    if name == "lidar":
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        # sensor->world 4x4 (CARLA left-handed) at capture time, for wedge de-skewing
        m = np.array(data.transform.get_matrix(), dtype=np.float64)
        return dict(points=pts, horizontal_angle=float(data.horizontal_angle), s2w=m)
    if name == "camera":
        a = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(data.height, data.width, 4)
        return dict(bgr=a[:, :, :3].copy())
    return dict(raw=bytes(data.raw_data))


def spawn_suite(world, vehicle, specs: List[SensorSpec], bufsize: int = 400) -> Dict[str, SensorHandle]:
    bp_lib = world.get_blueprint_library()
    handles: Dict[str, SensorHandle] = {}
    for spec in specs:
        if not spec.enabled:
            continue
        bp = bp_lib.find(spec.bp_id)
        for k, v in spec.attrs.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        actor = world.spawn_actor(bp, _transform(spec), attach_to=vehicle)
        buf: Deque = collections.deque(maxlen=bufsize)
        name = spec.name

        def _cb(data, _name=name, _buf=buf):
            _buf.append((data.frame, data.timestamp, _decode(_name, data)))

        actor.listen(_cb)
        handles[name] = SensorHandle(spec, actor, buf)
    return handles


def destroy_suite(handles: Dict[str, SensorHandle]) -> None:
    for h in handles.values():
        try:
            h.actor.stop()
            h.actor.destroy()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# geodetic helpers -- CARLA uses an equirectangular projection around the map origin
# --------------------------------------------------------------------------------------

_EARTH_R = 6_378_137.0


def geo_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Small-area equirectangular: matches CARLA's own lat/lon<->xy to <1 mm over a town."""
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    east = _EARTH_R * dlon * math.cos(math.radians(lat0))
    north = _EARTH_R * dlat
    return east, north, (alt - alt0)


def enu_to_geo(east, north, up, lat0, lon0, alt0):
    dlat = north / _EARTH_R
    dlon = east / (_EARTH_R * math.cos(math.radians(lat0)))
    return lat0 + math.degrees(dlat), lon0 + math.degrees(dlon), alt0 + up


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="attach the sensor suite for 3 s and print rates")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--camera", action="store_true")
    a = ap.parse_args()

    client = carla.Client(a.host, a.port); client.set_timeout(30.0)
    world = client.get_world()
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.01
    world.apply_settings(s)
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    sp = np.random.choice(world.get_map().get_spawn_points())
    veh = world.spawn_actor(bp, sp); veh.set_autopilot(True)
    H = spawn_suite(world, veh, sensor_specs(a.camera))
    try:
        for _ in range(300):
            world.tick()
        for name, h in H.items():
            print(f"{name:8s} {len(h.buf):4d} msgs buffered  last={h.buf[-1][2].keys() if h.buf else None}")
    finally:
        destroy_suite(H); veh.destroy()
        s.synchronous_mode = False; world.apply_settings(s)
