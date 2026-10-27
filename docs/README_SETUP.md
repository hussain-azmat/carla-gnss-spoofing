# CARLA collection VM — setup (AS BUILT)

Headless Azure GPU VM running CARLA 0.9.15, driven over SSH, producing the
**GNSS-spoofing multi-sensor benchmark** (`RESEARCH_PLAN.md §12`).
Collection protocol + wave schedule: **`COLLECTION_PLAN.md`**.

--------------------------------------------------------------------------------
## VM (already provisioned)
--------------------------------------------------------------------------------
| | |
|-|-|
| Resource group | `carla-gdn` (East US) |
| VM | `carla-datacollection` — **Standard_NC16as_T4_v3** (16 vCPU, 110 GiB, Tesla T4 16 GB) |
| OS | Ubuntu 24.04 LTS, kernel 6.17-azure |
| Disk | 128 GB OS (Premium SSD) + 352 GB ephemeral `/mnt` (wiped on deallocate!) |
| Login | `<user>@<public-ip>` with `<ssh-key>.pem` (keep the key outside the repository) |
| Cost | on-demand ≈ **$1.30/h** running incl. disk + IP; disk-only when deallocated |

```bash
az vm start      -g carla-gdn -n carla-datacollection
az vm show -d    -g carla-gdn -n carla-datacollection --query publicIps -o tsv
az vm deallocate -g carla-gdn -n carla-datacollection     # STOP BILLING after every wave
```

--------------------------------------------------------------------------------
## What's installed on the VM
--------------------------------------------------------------------------------
- **NVIDIA driver 595** (not 550 — kernel 6.17 is too new for 550), Vulkan ICD OK.
- `~/CARLA_0.9.15/` — CARLA 0.9.15 (UE4 4.26). Start/stop with **`~/carla_server.sh start|stop|status [Epic]`**
  (`-quality-level=Low` segfaults on this build — use Epic).
- **conda env `carla37`** — Python 3.7 + CARLA's *bundled* wheel. Runs `collect.py` /
  `run_collection.py`. (The PyPI py3.10 `carla` wheel's Traffic Manager segfaults here.)
- **conda env `carla`** — Python 3.10 + kiss-icp, open3d, pyarrow, torch-ready. Runs
  `run_lidar_odom.py`, `qa_report.py`, `make_dataset.py`, training.
- `apt` is pinned to IPv4 (`/etc/apt/apt.conf.d/99force-ipv4`) — without it it crawls at 15 KB/s.

--------------------------------------------------------------------------------
## Per-wave workflow (from your machine)
--------------------------------------------------------------------------------
```bash
IP=$(az vm show -d -g carla-gdn -n carla-datacollection --query publicIps -o tsv)
rsync -avz -e "ssh -i <ssh-key>.pem" ./carla_collect/ <user>@$IP:~/carla_collect/

ssh -i <ssh-key>.pem <user>@$IP
  conda activate carla37 && cd ~/carla_collect
  tmux new -s wave
  python run_collection.py --wave 1a_core_train --out ~/out      # collect + inline odometry
  conda activate carla
  python qa_report.py --in ~/out --wave 1a_core_train             # must be ALL PASS
  exit

rsync -avz -e "ssh -i <ssh-key>.pem" <user>@$IP:~/out/ ./out/   # back up
az vm deallocate -g carla-gdn -n carla-datacollection
```

`run_collection.py` is idempotent — re-running a wave skips sessions that already have a
`manifest.json`, so a spot eviction / crash just means "run the wave again".

--------------------------------------------------------------------------------
## Files in this folder
--------------------------------------------------------------------------------
| file | env | role |
|------|-----|------|
| `sensors.py` | 3.7 | sensor suite, noise models, LiDAR config, geodetic helpers |
| `attacks.py` | any | 6 GNSS-spoof families, offline injection (`--plot` self-test) |
| `collect.py` | 3.7 | one clean session → parquet streams + GT + de-skewed LiDAR sweeps |
| `lidar_odom.py` | 3.10 | KISS-ICP wrapper |
| `run_lidar_odom.py` | 3.10 | offline LiDAR odometry over sessions, prunes raw clouds |
| `run_collection.py` | 3.7 | wave driver (server restart, resume, inline odometry) |
| `qa_report.py` | 3.10 | QA-gate check per wave |
| `make_dataset.py` | 3.10 | clean sessions → labelled attack dataset + `dataset_index.parquet` |
| `verify_setup.py` | 3.7 | quick sensor-rate smoke test |
| `scenario_matrix.yaml` | — | machine-readable plan (splits, waves, QA gates) |
| `COLLECTION_PLAN.md` | — | the protocol + rationale + runbook |
| `docs/threat_model.md`, `docs/metrics.md` | — | locked Phase-0 design |

--------------------------------------------------------------------------------
## Key gotchas (all resolved, keep in mind if rebuilding)
--------------------------------------------------------------------------------
- **Traffic Manager**: only works from the py3.7 bundled wheel + `load_world()` first.
  py3.10 PyPI wheel → `InMemoryMap::Load` segfault.
- **LiDAR density**: CARLA sync mode emits `points_per_second × fixed_delta_seconds` per
  tick and won't accumulate a full sweep. We run the LiDAR at 10 Hz rotation firing a
  36° wedge every tick, then `collect.py` accumulates 10 wedges, de-skews each by its
  `sensor→world` pose, and stores one full 360° sweep (~115 k pts) at 10 Hz.
- **Handedness**: CARLA sensor frame is left-handed; `collect.py` flips Y so clouds are
  right-handed (x-fwd, y-left, z-up). Without it KISS-ICP mirrors → heading drift blows up
  (ATE 10 m vs 0.05–1 m over 200 m).
- **`/mnt` is ephemeral** — wiped on deallocate. Keep data on the OS disk; `rsync` it off
  before deallocating.
- Server `-quality-level=Low` segfaults; use `Epic`. Collection runs ~0.42× real-time.
