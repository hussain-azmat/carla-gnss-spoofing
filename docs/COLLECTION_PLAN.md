# Phase-1 data collection plan — GNSS-spoofing multi-sensor benchmark

Companion to `RESEARCH_PLAN.md §12` and the machine-readable `scenario_matrix.yaml`.
This document is the **professional collection protocol**: what we collect, why, in what
order, and the acceptance gates. Collection is split into **waves** — each wave is one
self-contained sitting (start VM → collect → QA → odometry → back up → deallocate VM),
so the work is checkpointed and the cloud credit is spent in controlled increments.

---

## 1. What this dataset is for

| Use | Requirement it drives |
|-----|-----------------------|
| **Publishable spoofing-detection benchmark** (T-IV / T-IFS / IoT-J) | 6 attack families, per-timestep labels, controlled parameter sweeps, hard-negative set, held-out splits on 4 axes |
| **Attack-resilient fusion research** (thesis Ch. + follow-on) | full raw sensor streams + 100 Hz ground truth + covariance-usable LiDAR odometry |
| **Detectability-limits analysis** | attack sweeps that approach the plausibility boundary (slow pull-off, sub-lane offset) |
| **General multi-sensor fusion / sim-to-real** | consistent frames, documented noise models, a camera-validation subset, reproducibility manifest |

The dataset must let a reviewer believe three claims: (a) the detector generalises to
**unseen towns**, (b) it generalises to **unseen attack families**, (c) it does **not
false-alarm on natural GNSS degradation** (urban canyon, tunnels).

---

## 2. Sensor suite & signal design (locked)

| Stream | Rate | Notes |
|--------|------|-------|
| IMU (`sensor.other.imu`) | 100 Hz | consumer-MEMS noise (`sensors.py:IMU_NOISE`) |
| GNSS (`sensor.other.gnss`) | 10 Hz | base ~0.42 m horiz; `canyon` +2.5 m; `tunnel` outage windows |
| Wheel odometry | 50 Hz | synthesised from GT speed + 2 % scale error + white noise |
| LiDAR (`sensor.lidar.ray_cast`, 64 ch) | 10 Hz | ~115 k pts/sweep. Fires 36° wedges at 100 Hz; collect.py accumulates 10, **de-skews each by its own sensor→world pose**, stores one full 360° sweep. Right-handed (x-fwd, y-left, z-up). |
| Ground-truth pose | 100 Hz | position (world + local ENU), orientation, velocity, ang-vel, controls, `dist_to_junction` |
| LiDAR odometry (KISS-ICP) | 10 Hz | computed offline (`run_lidar_odom.py`, py3.10). **~1.2 % drift** verified; stored as poses, raw clouds pruned except the pilot QA sessions |
| Front camera (`sensor.camera.rgb` 960×540) | 10 Hz | **wave 5 + a few wave-4 sessions only** — sim-to-real / qualitative figures |

World runs synchronous at 100 Hz (`fixed_delta_seconds = 0.01`) — fixes the variable-rate
problem that hurt the ICIICE-2025 dataset. Collection runs at **~0.42× real-time** on the
NC16as_T4_v3 (LiDAR ray-casting bound).

Two conda envs on the VM:
- **`carla37`** (Python 3.7, CARLA's bundled wheel) — runs `collect.py` / `run_collection.py`.
  The PyPI py3.10 wheel's Traffic Manager segfaults on this box; the bundled 3.7 wheel is fine.
- **`carla`** (Python 3.10) — runs `run_lidar_odom.py`, `make_dataset.py`, all analysis/training.

---

## 3. Coverage design

### 3.1 Towns (generalisation axis 1)
| Split | Towns | Regime |
|-------|-------|--------|
| **train** | Town03, Town05, Town10HD | large urban + tunnel/overpass, squared grid, dense downtown canyon |
| **test (unseen)** | Town02, Town04, Town06, Town07 | small town, highway loop, highway network, rural/open-sky |

### 3.2 Weather / lighting (axis 2)
train `{clear_noon, cloudy_noon, wet_noon, clear_sunset}` · test `{hard_rain_noon, clear_night}`.
Affects camera strongly, LiDAR mildly (rain dropoff), and adds realism diversity.

### 3.3 GNSS condition (the hard-negative axis)
`nominal` (open sky) · `canyon` (+2.5 m building-biased multipath) · `tunnel` (outage windows).
Canyon/tunnel sessions are collected **clean** and used as **hard negatives** — the detector
must not raise alarms on them. Sampled at 3× weight in evaluation.

### 3.4 Attacks (axis 3 — injected offline, `make_dataset.py`)
6 families (`attacks.py`): `gradual_pulloff`, `jump_and_hold`, `random_walk`, `zv_freeze`
(**train**) · `replay`, `target_bias` (**held out — test only**).
Per clean session: 3 variants/family + the clean stream as a negative ≈ **19–22 labelled
sequences/session**. Onset: 50 % adversarial (aligned to logged junction-approach times),
50 % uniform-random. Duration 8–40 s. Parameter ranges in `attacks.py:PARAM_RANGES`.

### 3.5 Traffic / dynamics
Default 70 vehicles + 35 walkers. Heavy variants (110–140) for LiDAR-occlusion stress.
`ego_fast` sessions raise the autopilot speed for the highway high-dynamics regime.

### 3.6 Route diversity
Random ego spawn + Traffic-Manager wandering, distinct **TM seed per session**, so no two
sessions retrace a route. (A route-planner upgrade for guaranteed coverage is a noted
Phase-1.5 improvement, not required for v1.)

---

## 4. The waves

Run one wave per sitting. `scenario_matrix.yaml:waves` is the exact row list.

| Wave | Purpose | Sessions | Sim min | ~Wall |
|------|---------|----------|---------|-------|
| **0 · pilot** | validate full pipeline end-to-end; keep raw clouds for odometry QA | 2 | 8 | ~1 h (incl. setup) |
| **1a · core train** | backbone training data — Town03/05, core weather, nominal+canyon+tunnel | 10 | 80 | ~3.5 h |
| **1b · core train** | backbone training data — Town10HD + fill-ins | 10 | 80 | ~3.5 h |
| **2 · hard negatives** | dedicated 10-min clean canyon/tunnel/rural runs | 8 | 80 | ~3.5 h |
| **3 · unseen-town test** | Town02/04/06/07 + unseen weather (test-both cell) | 12 | 96 | ~4.5 h |
| **4 · stress** | highway high-speed, heavy-traffic rain, night, extreme-attack seeds | 8 | 68 | ~3.5 h |
| **5 · camera validation** | front camera on; sim-to-real + figures | 4 | 32 | ~2 h |
| | **total** | **54** | **~460** | **~26 h** |

Estimated GPU spend ≈ **$35** (NC16as_T4_v3 on-demand ≈ $1.30/h incl. disk + IP), leaving
the bulk of the $200 credit for model training. **Deallocate the VM after every wave**
(`az vm deallocate` — disk-only cost) — and note `/mnt` (the 352 GB temp disk) is wiped on
deallocate, so nothing important lives there.

### Recommended calendar
7 sittings over ~1.5–2 weeks: pilot; 1a; 1b; 2; 3; 4; 5. Each sitting is independent —
if a wave fails QA, re-run only its failed rows before moving on.

---

## 5. Per-wave runbook

```bash
# 1. bring the VM up (from your machine)
az vm start -g carla-gdn -n carla-datacollection
IP=$(az vm show -d -g carla-gdn -n carla-datacollection --query publicIps -o tsv)

# 2. sync code up
rsync -avz -e "ssh -i <ssh-key>.pem" ~/carla_collect/ <user>@$IP:~/carla_collect/

# 3. on the VM: start server, run the wave (collect + inline odometry), in tmux
ssh -i <ssh-key>.pem <user>@$IP
  ~/carla_server.sh start Epic
  conda activate carla37
  cd ~/carla_collect
  tmux new -s wave
  python run_collection.py --wave 1a_core_train --out ~/out
  # -> per session: collect.py (py3.7) then run_lidar_odom.py (py3.10), clouds pruned

# 4. QA (see §6). If all pass:
  python qa_report.py --in ~/out --wave 1a_core_train        # prints the gate table

# 5. back up (from your machine)
rsync -avz -e "ssh -i <ssh-key>.pem" <user>@$IP:~/out/ ./out/
#    then push the increment to Kaggle (same flow as the existing dataset)

# 6. stop billing
az vm deallocate -g carla-gdn -n carla-datacollection
```

Spot-eviction / crash safety: `run_collection.py` skips any `(town,weather,seed,gnss)`
row that already has a `manifest.json`, so re-running the wave resumes where it stopped.

---

## 6. QA gates (a wave is not done until every session passes)

From `scenario_matrix.yaml:qa_gates`, checked by `qa_report.py`:

1. **Sensor rates** — IMU 98–102 Hz, GNSS 9.5–10.5 Hz, LiDAR 9.5–10.5 Hz.
2. **GNSS-vs-GT horizontal error (median)** — nominal 0–0.7 m, canyon 1–4.5 m, tunnel 1–6 m.
3. **LiDAR-odometry drift** — final position error < 3 % of path length; odom path length
   within 5 % of GT path length.
4. **Junction events** — 2–20 per 8-min session (0 ⇒ detector loses adversarial-onset signal;
   >20 ⇒ `_dist_to_junction` mis-tuned for that town).
5. **Integrity** — a session folder without `manifest.json` is invalid and re-run.

Pilot wave additionally: eyeball a LiDAR sweep (Open3D) and an odometry-vs-GT plot; confirm
attack injection produces the expected offset trajectories for all 6 families
(`python attacks.py --plot`).

---

## 7. Dataset assembly (after all waves — offline, local, minutes)

```bash
conda activate carla
python make_dataset.py --in ./out --out ./dataset --matrix scenario_matrix.yaml
```

Produces `dataset/`:
- `variants/<session>/<family>__v<k>.parquet` — spoofed GNSS stream + per-timestep label +
  applied ENU offset, one per attack variant, plus `clean__v0` negative.
- `dataset_index.parquet` — one row per (session, variant): town, weather, `town_split`,
  `gnss_condition`, `family`, `attack_split`, `is_attack`, `hard_negative`,
  `onset_adversarial`, `onset_s`, `duration_s`, `max_offset_m`, `path`.

Expected scale: **~54 sessions × ~22 ≈ 1150 labelled sequences** + 54 clean negatives + the
hard-negative subset at 3× eval weight.

### Evaluation splits (frozen — see `docs/metrics.md`)
| split | towns | attack families |
|-------|-------|-----------------|
| train | Town03/05/10HD | pull-off, jump-hold, random-walk, zv-freeze |
| val | held-out sessions, train towns/families | " |
| test-town | Town02/04/06/07 | train families |
| test-attack | train towns | replay, target_bias |
| test-both | Town02/04/06/07 | replay, target_bias |

---

## 8. Reproducibility manifest (recorded per session in `events.json`)

CARLA 0.9.15 · Ubuntu 24.04 · driver 595 · `world_tick_hz=100` · TM seed = session seed ·
`sensor_spec_sha1` · `origin_latlonalt` · `cloud_voxel_m` · `junction_onset_events` ·
`lidar_odom_path_m`. Plus: git SHA of `carla_collect/` at collection time, and this plan +
`scenario_matrix.yaml` versioned alongside.

---

## 9. Known limitations / noted improvements (not blockers for v1)

- **LiDAR odometry ~1.2 % drift** (KISS-ICP defaults). Fine as an independent reference over
  10–30 s detection windows; tighten with tuned params / loop closure in Phase 2 if needed.
- **Tunnel outage** currently falls back to canyon-level noise where per-town tunnel-segment
  geometry isn't wired in; Town03 tunnel segment coords are a small Phase-1.5 addition.
- **Route coverage** relies on TM wandering, not a planner — acceptable for v1, matches
  common CARLA-dataset practice.
- **Sim-only.** Sim-to-real transfer (KITTI raw + injected attacks) is Phase 5, and wave 5's
  camera subset exists to support that discussion.
