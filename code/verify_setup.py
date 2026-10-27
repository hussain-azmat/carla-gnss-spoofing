#!/usr/bin/env python3
"""
verify_setup.py -- sanity-check a CARLA VM before Phase 1 collection.

Connects, loads a town, spawns a vehicle on autopilot with the full sensor suite
(IMU / GNSS / 64-beam LiDAR / RGB camera), runs 10 s in synchronous mode, and
reports per-sensor callback rates. Writes verify_out/ with a LiDAR frame + a
camera PNG + a JSON summary.

Usage (CARLA server must already be running with -RenderOffScreen -carla-rpc-port=2000):
    python verify_setup.py --host localhost --port 2000 --town Town10HD_Opt
"""
import argparse, json, os, time
from collections import Counter
import numpy as np

try:
    import carla
except ImportError:
    raise SystemExit("`import carla` failed -- run inside the `carla` conda env.")

OUT = "verify_out"
DT = 0.025          # 40 Hz world tick
SECONDS = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town10HD_Opt")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print("server version:", client.get_server_version())
    world = client.load_world(args.town)

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    bp = world.get_blueprint_library()
    actors = []
    counts = Counter()
    last_lidar = {}
    last_cam = {}

    try:
        veh_bp = bp.filter("vehicle.tesla.model3")[0]
        sp = np.random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(veh_bp, sp)
        vehicle.set_autopilot(True, tm.get_port())
        actors.append(vehicle)

        # ---- IMU @ 100 Hz ----
        imu_bp = bp.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", "0.01")
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(lambda d: counts.update(["imu"]))
        actors.append(imu)

        # ---- GNSS @ 10 Hz ----
        g_bp = bp.find("sensor.other.gnss")
        g_bp.set_attribute("sensor_tick", "0.1")
        gnss = world.spawn_actor(g_bp, carla.Transform(), attach_to=vehicle)
        gnss.listen(lambda d: counts.update(["gnss"]))
        actors.append(gnss)

        # ---- 64-beam LiDAR @ 10 Hz, ~130k pts/rev ----
        l_bp = bp.find("sensor.lidar.ray_cast")
        for k, v in {
            "channels": "64", "range": "100.0", "points_per_second": "1300000",
            "rotation_frequency": "10", "upper_fov": "10.0", "lower_fov": "-30.0",
            "dropoff_general_rate": "0.10", "dropoff_intensity_limit": "0.8",
            "dropoff_zero_intensity": "0.4", "noise_stddev": "0.01",
            "sensor_tick": "0.1",
        }.items():
            l_bp.set_attribute(k, v)
        lidar = world.spawn_actor(
            l_bp, carla.Transform(carla.Location(z=1.8)), attach_to=vehicle)

        def on_lidar(d):
            counts.update(["lidar"])
            pts = np.frombuffer(d.raw_data, dtype=np.float32).reshape(-1, 4)
            last_lidar["n"] = len(pts)
            last_lidar["pts"] = pts
        lidar.listen(on_lidar)
        actors.append(lidar)

        # ---- RGB camera 640x360 @ 10 Hz ----
        c_bp = bp.find("sensor.camera.rgb")
        c_bp.set_attribute("image_size_x", "640")
        c_bp.set_attribute("image_size_y", "360")
        c_bp.set_attribute("sensor_tick", "0.1")
        cam = world.spawn_actor(
            c_bp, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=vehicle)

        def on_cam(img):
            counts.update(["camera"])
            a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            last_cam["bgr"] = a[:, :, :3].copy()
        cam.listen(on_cam)
        actors.append(cam)

        print(f"warming up...")
        for _ in range(20):
            world.tick()
        counts.clear()

        n_ticks = int(SECONDS / DT)
        print(f"running {SECONDS}s ({n_ticks} ticks)...")
        t0 = time.time()
        for i in range(n_ticks):
            world.tick()
        wall = time.time() - t0

        speeds = vehicle.get_velocity()
        summary = {
            "server_version": client.get_server_version(),
            "town": args.town,
            "world_dt": DT,
            "sim_seconds": SECONDS,
            "wall_seconds": round(wall, 1),
            "realtime_factor": round(SECONDS / wall, 2),
            "sensor_hz": {k: round(v / SECONDS, 1) for k, v in counts.items()},
            "lidar_points_last_frame": last_lidar.get("n"),
            "vehicle_speed_mps": round((speeds.x**2 + speeds.y**2) ** 0.5, 2),
        }
        print(json.dumps(summary, indent=2))
        with open(f"{OUT}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if "pts" in last_lidar:
            np.save(f"{OUT}/lidar_frame.npy", last_lidar["pts"])
        if "bgr" in last_cam:
            try:
                import cv2
                cv2.imwrite(f"{OUT}/camera.png", last_cam["bgr"])
            except Exception:
                pass

        ok = (
            summary["sensor_hz"].get("imu", 0) > 80 and
            summary["sensor_hz"].get("gnss", 0) > 7 and
            summary["sensor_hz"].get("lidar", 0) > 7 and
            (last_lidar.get("n") or 0) > 60000
        )
        print("\nRESULT:", "PASS -- box is ready for Phase 1" if ok else
              "CHECK -- rates or LiDAR density below expectation, inspect summary.json")

    finally:
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        world.apply_settings(original)
        tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
