#!/usr/bin/env python3
"""
run_collection.py -- drive one WAVE of Phase-1 collection from scenario_matrix.yaml
(see COLLECTION_PLAN.md). Runs in the **carla37** env; shells out to the **carla** env
for run_lidar_odom.py after each session.

Idempotent + resumable: skips rows whose session folder already has manifest.json, so a
spot eviction just means "re-run the wave".

    conda activate carla37
    python run_collection.py --wave 1a_core_train --out ~/out
    python run_collection.py --list                       # show waves + counts
    python run_collection.py --wave 0_pilot --dry-run
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time

import yaml


def _port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


class CarlaServer:
    def __init__(self, carla_root, port=2000, quality="Epic"):
        self.sh = os.path.join(carla_root, "carla_server.sh")
        self.carla_root, self.port, self.quality = carla_root, port, quality

    def _script(self, *a):
        s = os.path.expanduser("~/carla_server.sh")
        return subprocess.call(["bash", s, *a])

    def restart(self):
        self._script("stop")
        rc = self._script("start", self.quality)
        if rc != 0 or not _port_open("localhost", self.port):
            raise RuntimeError("CARLA server failed to start")


def _rows_for_wave(M, wave):
    d = M["defaults"]
    out = []
    for r in M["waves"][wave]:
        row = dict(d)
        row.update(r)
        row["wave"] = wave
        out.append(row)
    return out


def _exists(out_dir, row):
    pre = f'{row["town"]}__{row["weather"]}__s{row["seed"]}__'
    if not os.path.isdir(out_dir):
        return False
    for dd in os.listdir(out_dir):
        if dd.startswith(pre) and os.path.exists(os.path.join(out_dir, dd, "manifest.json")):
            try:
                ev = json.load(open(os.path.join(out_dir, dd, "events.json")))
                if ev.get("gnss_condition", "nominal") == row.get("gnss", "nominal"):
                    return True
            except Exception:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="scenario_matrix.yaml")
    ap.add_argument("--wave", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/out"))
    ap.add_argument("--carla", default=os.path.expanduser("~/CARLA_0.9.15"))
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--py310", default=os.path.expanduser("~/miniconda3/envs/carla/bin/python"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-odom", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    M = yaml.safe_load(open(args.matrix))

    if args.list:
        for w, rows in M["waves"].items():
            mins = sum(r.get("minutes", M["defaults"]["minutes"]) for r in rows)
            print(f"  {w:22s} {len(rows):2d} sessions  {mins:3d} sim-min")
        return
    if not args.wave:
        ap.error("--wave required (or --list)")

    os.makedirs(args.out, exist_ok=True)
    rows = _rows_for_wave(M, args.wave)
    todo = [r for r in rows if not _exists(args.out, r)]
    print(f"wave {args.wave}: {len(rows)} rows, {len(todo)} to collect")
    if args.dry_run:
        for r in todo:
            print("  ", {k: r[k] for k in ("town", "weather", "gnss", "seed", "minutes") if k in r})
        return

    server = CarlaServer(args.carla, args.port, M.get("server_quality", "Epic"))
    done = 0
    cur_town = None
    for r in todo:
        if r["town"] != cur_town:
            server.restart()
            cur_town = r["town"]
        cmd = [sys.executable, "collect.py",
               "--host", "localhost", "--port", str(args.port),
               "--town", r["town"], "--weather", r["weather"],
               "--minutes", str(r.get("minutes", 8)),
               "--traffic", str(r.get("traffic", 70)),
               "--walkers", str(r.get("walkers", 35)),
               "--seed", str(r["seed"]), "--out", args.out]
        if r.get("camera"):
            cmd.append("--camera")
        if r.get("keep_clouds"):
            cmd.append("--keep-clouds")
        if r.get("ego_fast"):
            cmd.append("--ego-fast")
        env = dict(os.environ, GNSS_CONDITION=r.get("gnss", "nominal"),
                   COLLECT_WAVE=args.wave)
        print(f"\n=== [{done+1}/{len(todo)}] {r['town']} {r['weather']} "
              f"{r.get('gnss','nominal')} s{r['seed']} {r.get('minutes',8)}min ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, env=env)
        print(f"    collect rc={rc}  {time.time()-t0:.0f}s", flush=True)
        if rc == 0:
            done += 1
            if not args.skip_odom and os.path.exists(args.py310):
                t1 = time.time()
                orc = subprocess.call([args.py310, "run_lidar_odom.py", "--in", args.out])
                print(f"    lidar-odom rc={orc}  {time.time()-t1:.0f}s", flush=True)
        if args.limit and done >= args.limit:
            print("hit --limit"); break

    server._script("stop")
    print(f"\nwave {args.wave}: collected {done} new sessions -> {args.out}")
    print("next: python qa_report.py --in %s --wave %s" % (args.out, args.wave))


if __name__ == "__main__":
    main()
