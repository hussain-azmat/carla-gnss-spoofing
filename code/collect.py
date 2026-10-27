#!/usr/bin/env python3
"""
collect.py -- one clean CARLA session for the GNSS-spoofing benchmark (RESEARCH_PLAN §12, Phase 1).

Drives an ego vehicle on autopilot through live traffic and logs the full sensor suite +
ground truth + lightly-downsampled LiDAR clouds, and marks the sim-times where the ego is
approaching a junction so the attack injector (attacks.py) can place adversarially-timed
spoof onsets.

Runs in the **carla37** conda env (Python 3.7, CARLA's bundled wheel -- the PyPI 3.10
wheel's Traffic Manager segfaults on this box). LiDAR odometry is therefore NOT computed
here; run_lidar_odom.py does it offline in the py3.10 env and can then prune the clouds.

Attacks are added later, offline, by make_dataset.py -- one clean session -> many labelled
attack variants. This script only produces clean data + metadata.

Run (server must be up -- use ./carla_server.sh start):
    conda activate carla37
    python collect.py --town Town03 --weather clear_noon --minutes 8 \
                      --traffic 80 --walkers 40 --seed 1 --out ./out

A session folder is valid only once manifest.json exists (atomic-ish); a run killed by
spot eviction leaves no manifest and is simply re-run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import carla

from sensors import (sensor_specs, spawn_suite, destroy_suite, geo_to_enu,
                     WHEEL_SCALE_ERROR, WHEEL_SPEED_NOISE_STD, WHEEL_RATE_HZ,
                     LIDAR_KEEP_EVERY)

TICK_HZ = 100
DT = 1.0 / TICK_HZ
JUNCTION_PROBE_M = 30.0       # look this far ahead for the next junction
JUNCTION_NEAR_M = 15.0       # "approaching" threshold -> adversarial onset candidate
CLOUD_VOXEL_M = 0.10          # near-lossless for LiDAR odometry; ~130k -> ~70-90k pts


def assemble_sweep(wedges):
    """wedges: list of (pts_Nx4_leftHanded, s2w_4x4). De-skew each wedge into the LAST
    wedge's sensor frame, concat, then flip Y -> right-handed (x fwd, y left, z up).
    Returns (M,4) float32."""
    ref_inv = np.linalg.inv(wedges[-1][1])
    outs = []
    for pts, s2w in wedges:
        if len(pts) == 0:
            continue
        rel = ref_inv @ s2w                       # wedge sensor -> ref sensor (left-handed)
        xyz1 = np.c_[pts[:, :3].astype(np.float64), np.ones(len(pts))]
        xyz = (xyz1 @ rel.T)[:, :3]
        w = np.c_[xyz, pts[:, 3:4]]
        outs.append(w)
    p = np.concatenate(outs, axis=0)
    p[:, 1] *= -1.0                               # left-handed -> right-handed
    return p.astype(np.float32)


def voxel_downsample(pts: np.ndarray, voxel: float = CLOUD_VOXEL_M) -> np.ndarray:
    """numpy-only voxel grid downsample (open3d isn't available in the py3.7 env).
    Keeps the first point seen in each occupied voxel. pts: (N,4) xyz+intensity."""
    if len(pts) == 0:
        return pts
    q = np.floor(pts[:, :3] / voxel).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return pts[np.sort(idx)]

WEATHER = {
    "clear_noon": carla.WeatherParameters.ClearNoon,
    "cloudy_noon": carla.WeatherParameters.CloudyNoon,
    "wet_noon": carla.WeatherParameters.WetNoon,
    "hard_rain_noon": carla.WeatherParameters.HardRainNoon,
    "clear_sunset": carla.WeatherParameters.ClearSunset,
    "clear_night": carla.WeatherParameters.ClearNight,
    "wet_cloudy_sunset": carla.WeatherParameters.WetCloudySunset,
}


def _spawn_traffic(world, client, tm, n_vehicles, n_walkers, rng, spawns=None):
    bp_lib = world.get_blueprint_library()
    v_bps = [b for b in bp_lib.filter("vehicle.*")
             if int(b.get_attribute("number_of_wheels")) == 4]
    spawns = list(spawns) if spawns is not None else world.get_map().get_spawn_points()
    rng.shuffle(spawns)
    batch, vehicles = [], []
    for i in range(min(n_vehicles, len(spawns))):
        bp = v_bps[int(rng.integers(len(v_bps)))]
        if bp.has_attribute("color"):
            vals = bp.get_attribute("color").recommended_values
            bp.set_attribute("color", str(vals[int(rng.integers(len(vals)))]))
        bp.set_attribute("role_name", "autopilot")
        batch.append(carla.command.SpawnActor(bp, spawns[i])
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            vehicles.append(r.actor_id)

    walkers, controllers = [], []
    w_bps = bp_lib.filter("walker.pedestrian.*")
    wbatch, locs = [], []
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        locs.append(loc)
        bp = w_bps[int(rng.integers(len(w_bps)))]
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        wbatch.append(carla.command.SpawnActor(bp, carla.Transform(loc)))
    for r in client.apply_batch_sync(wbatch, True):
        if not r.error:
            walkers.append(r.actor_id)
    cbp = bp_lib.find("controller.ai.walker")
    cbatch = [carla.command.SpawnActor(cbp, carla.Transform(), w) for w in walkers]
    for r in client.apply_batch_sync(cbatch, True):
        if not r.error:
            controllers.append(r.actor_id)
    world.tick()
    for cid in controllers:
        c = world.get_actor(cid)
        try:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(1.0 + rng.random() * 0.8)
        except Exception:
            pass
    return vehicles, walkers, controllers


def _dist_to_junction(carla_map, location) -> float:
    """Metres along the road to the next junction ahead, capped at JUNCTION_PROBE_M.
    0.0 means the ego is inside a junction right now."""
    wp = carla_map.get_waypoint(location, project_to_road=True)
    if wp is None:
        return JUNCTION_PROBE_M
    if wp.is_junction:
        return 0.0
    step = 2.0
    d = step
    while d <= JUNCTION_PROBE_M:
        nxts = wp.next(d)
        if not nxts:
            break
        if any(n.is_junction for n in nxts):
            return d
        d += step
    return JUNCTION_PROBE_M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default="Town10HD_Opt")
    ap.add_argument("--weather", default="clear_noon", choices=list(WEATHER))
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--traffic", type=int, default=35)
    ap.add_argument("--walkers", type=int, default=18)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--camera", action="store_true", help="also record a 960x540 front camera")
    ap.add_argument("--keep-clouds", action="store_true",
                    help="mark this session so run_lidar_odom.py keeps the raw clouds (QA subset)")
    ap.add_argument("--ego-fast", action="store_true",
                    help="raise ego autopilot speed (highway high-dynamics regime)")
    ap.add_argument("--out", default="./out")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gnss_condition = os.environ.get("GNSS_CONDITION", "nominal")
    session_id = f"{args.town}__{args.weather}__s{args.seed}__{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    sdir = os.path.join(args.out, session_id)
    os.makedirs(sdir, exist_ok=True)
    cloud_dir = os.path.join(sdir, "lidar_clouds")
    os.makedirs(cloud_dir, exist_ok=True)   # always written; run_lidar_odom.py prunes later

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.town)
    world.set_weather(WEATHER[args.weather])
    cmap = world.get_map()
    origin_geo = cmap.transform_to_geolocation(carla.Location(0, 0, 0))
    lat0, lon0, alt0 = origin_geo.latitude, origin_geo.longitude, origin_geo.altitude

    original = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = DT
    world.apply_settings(st)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    # hybrid physics: only vehicles near the ego get full physics -> faster sim + far fewer
    # TM gridlocks in dense maps (Town10HD/Town02).
    try:
        tm.set_hybrid_physics_mode(True)
        tm.set_hybrid_physics_radius(80.0)
    except Exception:
        pass

    actors = []
    try:
        v_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        v_bp.set_attribute("role_name", "hero")   # REQUIRED for TM hybrid-physics to keep
                                                  # the ego (and nearby traffic) fully simulated
        spawns = cmap.get_spawn_points()

        def _cfg_ego(e):
            # the ego is a data-collection / survey vehicle -> it should keep FLOWING, not
            # sit at every light (dense CARLA maps otherwise leave it stationary 40-60% of
            # the time, which was a core flaw of the old dataset).
            e.set_autopilot(True, tm.get_port())
            tm.ignore_lights_percentage(e, rng.uniform(90, 100))
            tm.ignore_signs_percentage(e, rng.uniform(90, 100))
            try:
                tm.ignore_vehicles_percentage(e, rng.uniform(0, 12))  # ease through gridlock
                tm.keep_right_rule_percentage(e, 0)                   # use all lanes
            except Exception:
                pass
            tm.vehicle_percentage_speed_difference(
                e, rng.uniform(-45, -25) if args.ego_fast else rng.uniform(-15, 5))
            tm.distance_to_leading_vehicle(e, 2.0)
            tm.random_left_lanechange_percentage(e, 25)
            tm.random_right_lanechange_percentage(e, 25)

        # spawn the ego and confirm it actually drives; bad spawn points (dead ends,
        # facing a wall) get retried at a different location.
        ego = None
        tried = [int(i) for i in rng.permutation(len(spawns))]
        for attempt, ego_sp_idx in enumerate(tried[:5]):
            try:
                ego = world.spawn_actor(v_bp, spawns[ego_sp_idx])
            except RuntimeError:
                ego = None; continue
            _cfg_ego(ego)
            p0 = ego.get_location()
            for _ in range(int(8 * TICK_HZ)):     # 8 s to get rolling
                world.tick()
            moved = ego.get_location().distance(p0)
            if moved > 6.0:
                break
            print("  ego didn't move (%.1fm) from spawn %d -- retrying" % (moved, ego_sp_idx),
                  flush=True)
            ego.destroy(); ego = None
            for _ in range(5):
                world.tick()
        if ego is None:
            raise SystemExit(3)
        actors.append(ego)

        traffic_spawns = [sp for i, sp in enumerate(spawns) if i != ego_sp_idx
                          and sp.location.distance(spawns[ego_sp_idx].location) > 12.0]
        # dense/compact maps gridlock -> scale traffic down
        n_veh = args.traffic
        if args.town.startswith(("Town10HD", "Town02", "Town01")):
            n_veh = min(args.traffic, 18)
        elif args.town.startswith("Town07"):
            # rural map: narrow single-lane roads, stop-sign-heavy junctions -> even
            # moderate traffic gridlocks the ego (measured 57% stationary at traffic=35).
            n_veh = min(args.traffic, 12)
        veh_ids, walk_ids, ctrl_ids = _spawn_traffic(
            world, client, tm, n_veh, args.walkers, rng, traffic_spawns)

        suite = spawn_suite(world, ego, sensor_specs(args.camera, gnss_condition))

        wheel_scale = 1.0 + rng.uniform(-WHEEL_SCALE_ERROR, WHEEL_SCALE_ERROR)
        wheel_every = int(round(TICK_HZ / WHEEL_RATE_HZ))

        for _ in range(int(3 * TICK_HZ)):  # 3 s warm-up
            world.tick()
        for h in suite.values():
            h.buf.clear()

        rows_imu, rows_gnss, rows_wheel, rows_gt, rows_lf = [], [], [], [], []
        onset_events = []
        wheel_dist = 0.0
        n_ticks = int(args.minutes * 60 * TICK_HZ)
        lidar_seen = set()
        wedge_buf = []
        djun = prev_djun = JUNCTION_PROBE_M
        stuck_since = None
        t_wall0 = time.time()

        ended_early = None
        for k in range(n_ticks):
            snap = world.get_snapshot()
            world.tick()
            snap = world.get_snapshot()
            sim_t = snap.timestamp.elapsed_seconds

            if not ego.is_alive:
                ended_early = "ego_destroyed"
                print("  !! ego lost at t=%.1fs -- ending session early" % (k * DT), flush=True)
                break
            try:
                tf = ego.get_transform()
                vel = ego.get_velocity()
                av = ego.get_angular_velocity()
                ctl = ego.get_control()
            except RuntimeError:
                ended_early = "ego_destroyed"
                print("  !! ego access failed at t=%.1fs -- ending early" % (k * DT), flush=True)
                break
            loc, rot = tf.location, tf.rotation
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            if speed < 0.5:
                stuck_since = sim_t if stuck_since is None else stuck_since
                stuck_for = sim_t - stuck_since
                if stuck_for > 25.0 and int(stuck_for) % 8 == 0:
                    try:                                   # force a way around the blockage
                        tm.random_left_lanechange_percentage(ego, 100)
                        tm.random_right_lanechange_percentage(ego, 100)
                        tm.ignore_vehicles_percentage(ego, 60)
                    except Exception:
                        pass
                if stuck_for > 90.0:
                    ended_early = "ego_stuck"
                    print("  !! ego stuck >90s at t=%.1fs -- ending early" % (k * DT), flush=True)
                    break
            else:
                if stuck_since is not None:               # just un-stuck -> restore normal
                    try:
                        tm.random_left_lanechange_percentage(ego, 25)
                        tm.random_right_lanechange_percentage(ego, 25)
                        tm.ignore_vehicles_percentage(ego, 8)
                    except Exception:
                        pass
                stuck_since = None
            g = cmap.transform_to_geolocation(loc)
            e, n, u = geo_to_enu(g.latitude, g.longitude, g.altitude, lat0, lon0, alt0)
            djun = _dist_to_junction(cmap, loc) if k % 10 == 0 else djun  # 10 Hz is plenty
            # adversarial onset candidate: crossing into the "approaching" band while moving
            if (djun <= JUNCTION_NEAR_M and prev_djun > JUNCTION_NEAR_M and speed > 2.0):
                onset_events.append(round(sim_t, 3))
            prev_djun = djun
            rows_gt.append(dict(t=sim_t, x=loc.x, y=loc.y, z=loc.z, e=e, n=n, u=u,
                                roll=np.radians(rot.roll), pitch=np.radians(rot.pitch),
                                yaw=np.radians(rot.yaw), vx=vel.x, vy=vel.y, vz=vel.z,
                                wz=np.radians(av.z), speed=speed,
                                throttle=ctl.throttle, steer=ctl.steer, brake=ctl.brake,
                                dist_to_junction=float(djun)))

            if k % wheel_every == 0:
                ws = speed * wheel_scale + rng.normal(0, WHEEL_SPEED_NOISE_STD)
                wheel_dist += max(ws, 0.0) * (wheel_every * DT)
                rows_wheel.append(dict(t=sim_t, speed_mps=ws, dist_m=wheel_dist))

            for (fr, ts, p) in list(suite["imu"].buf):
                rows_imu.append(dict(t=ts, ax=p["accel"][0], ay=p["accel"][1], az=p["accel"][2],
                                     gx=p["gyro"][0], gy=p["gyro"][1], gz=p["gyro"][2],
                                     compass=p["compass"]))
            suite["imu"].buf.clear()

            for (fr, ts, p) in list(suite["gnss"].buf):
                ge, gn, gu = geo_to_enu(p["lat"], p["lon"], p["alt"], lat0, lon0, alt0)
                rows_gnss.append(dict(t=ts, lat=p["lat"], lon=p["lon"], alt=p["alt"],
                                      e=ge, n=gn, u=gu))
            suite["gnss"].buf.clear()

            # LiDAR fires a 36deg wedge every tick; accumulate LIDAR_KEEP_EVERY wedges,
            # de-skew, store one full sweep at 10 Hz.
            for (fr, ts, p) in list(suite["lidar"].buf):
                if fr in lidar_seen:
                    continue
                lidar_seen.add(fr)
                wedge_buf.append((p["points"], p["s2w"], ts))
            suite["lidar"].buf.clear()
            if len(wedge_buf) >= LIDAR_KEEP_EVERY:
                grp = wedge_buf[:LIDAR_KEEP_EVERY]
                del wedge_buf[:LIDAR_KEEP_EVERY]
                sweep = assemble_sweep([(w[0], w[1]) for w in grp])
                ds = voxel_downsample(sweep)
                sweep_ts = grp[-1][2]
                np.savez_compressed(os.path.join(cloud_dir, "%08d.npz" % k),
                                    xyzi=ds.astype(np.float32), t=np.float64(sweep_ts))
                rows_lf.append(dict(t=sweep_ts, frame=k, n_points=int(sum(len(w[0]) for w in grp)),
                                    n_points_ds=int(len(ds)),
                                    dist_to_junction=float(djun)))


            if args.camera and "camera" in suite:
                for (fr, ts, p) in list(suite["camera"].buf):
                    d = os.path.join(sdir, "camera"); os.makedirs(d, exist_ok=True)
                    try:
                        import cv2
                        cv2.imwrite(os.path.join(d, f"{fr:08d}.jpg"), p["bgr"])
                    except Exception:
                        pass
                suite["camera"].buf.clear()

            if k % (5 * TICK_HZ) == 0 and k > 0:
                wall = time.time() - t_wall0
                rtf = (k * DT) / max(wall, 1e-6)
                total_s = args.minutes * 60
                eta = (total_s - k * DT) / max(rtf, 1e-3)
                print("  t=%6.1fs/%.0fs  %4.1f%%  rtf=%4.2fx  wall=%4.0fs eta=%4.0fs  "
                      "imu=%d gnss=%d lidar=%d speed=%.1f junc=%d"
                      % (k * DT, total_s, 100.0 * k * DT / total_s, rtf, wall, eta,
                         len(rows_imu), len(rows_gnss), len(rows_lf),
                         speed, len(onset_events)), flush=True)

        got_s = len(rows_gt) * DT
        if ended_early and got_s < 90.0:
            print("  session ended early with only %.0fs -- discarding (no manifest)" % got_s)
            raise SystemExit(2)

        def _wp(rows, name):
            pd.DataFrame(rows).to_parquet(os.path.join(sdir, name), index=False)

        _wp(rows_imu, "imu.parquet")
        _wp(rows_gnss, "gnss_clean.parquet")
        _wp(rows_wheel, "wheel.parquet")
        _wp(rows_gt, "gt_pose.parquet")
        _wp(rows_lf, "lidar_frames.parquet")

        spec_blob = json.dumps([s.__dict__ for s in sensor_specs(args.camera, gnss_condition)],
                               sort_keys=True)
        events = dict(
            session_id=session_id, town=args.town, weather=args.weather,
            gnss_condition=gnss_condition, wave=os.environ.get("COLLECT_WAVE"),
            ego_fast=bool(args.ego_fast), camera=bool(args.camera),
            minutes=args.minutes, seed=args.seed, dt=DT,
            origin_latlonalt=[lat0, lon0, alt0],
            traffic=len(veh_ids), walkers=len(walk_ids),
            wheel_scale=wheel_scale,
            junction_onset_events=onset_events,
            cloud_voxel_m=CLOUD_VOXEL_M,
            ended_early=ended_early, collected_seconds=round(got_s, 1),
            keep_clouds=bool(args.keep_clouds),   # run_lidar_odom.py: keep raw clouds if True
            lidar_odom_done=False,                # run_lidar_odom.py flips this
            sensor_spec_sha1=hashlib.sha1(spec_blob.encode()).hexdigest(),
            counts=dict(imu=len(rows_imu), gnss=len(rows_gnss), wheel=len(rows_wheel),
                        gt=len(rows_gt), lidar=len(rows_lf)),
        )
        with open(os.path.join(sdir, "events.json"), "w") as f:
            json.dump(events, f, indent=2)

        manifest = dict(session_id=session_id, created_utc=datetime.now(timezone.utc).isoformat(),
                        carla_server=client.get_server_version(),
                        files=sorted(os.listdir(sdir)), ok=True)
        tmp = os.path.join(sdir, ".manifest.tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, os.path.join(sdir, "manifest.json"))
        print(f"OK  {session_id}  -> {sdir}")

    finally:
        # teardown order matters: stop sensor callbacks, take the ego off the TM, let the
        # sim settle, THEN destroy -- otherwise CARLA aborts with "destroyed actor".
        try:
            for h in suite.values():          # type: ignore
                try:
                    h.actor.stop()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            ego.set_autopilot(False, tm.get_port())   # type: ignore
        except Exception:
            pass
        time.sleep(0.3)
        try:
            destroy_suite(suite)              # type: ignore
        except Exception:
            pass
        try:
            client.apply_batch_sync(
                [carla.command.DestroyActor(x) for x in
                 (veh_ids + walk_ids + ctrl_ids)], True)   # type: ignore
        except Exception:
            pass
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original)
        except Exception:
            pass


if __name__ == "__main__":
    main()
