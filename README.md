# CARLA GNSS-Spoofing Detection Benchmark: code

Code, trained monitor and result tables for **"Learned Multi-Sensor GNSS Spoofing Detection and
Attack-Resilient Fusion for Autonomous Vehicles"** (Hussain and Khurram, NED University of
Engineering and Technology; preprint).

- **Dataset (CC-BY-4.0):** https://www.kaggle.com/datasets/hussainazmat/carla-gnss-spoofing-benchmark
  (57 clean CARLA 0.9.15 sessions, 2,793 labelled attack sequences, KISS-ICP LiDAR-odometry reference).
- **This repository (MIT):** collection, attack injection, feature extraction, baselines, the causal GRU/TCN/Transformer
  monitor, the fusion filter, detectability analysis, quantization study, KITTI transfer, and the result tables used in the paper.

## Headline results (2,793-sequence benchmark)

| split | RAIM chi-square | best classical baseline | GRU monitor |
|-------|----------------:|------------------------:|------------:|
| val | 0.64 | 0.72 | 0.963 |
| test_town (unseen towns) | 0.68 | 0.77 | 0.964 |
| test_attack (unseen attack families) | 0.58 | 0.66 | 0.917 |
| test_both | 0.68 | 0.76 | 0.888 |

ROC-AUC. Zero-shot transfer to five real KITTI drives: 0.862. Clean-drive false alarms on
canyon/tunnel validation drives: 23 per hour for the monitor, 670 per hour for RAIM. The monitor is 413 KB and
runs 108x faster than the 10 Hz real-time budget on one CPU core. Full tables: `results/`.

## Layout

| path | contents |
|------|----------|
| `code/` | pipeline scripts (see below) |
| `docs/` | threat model, metric definitions, collection plan (infrastructure details removed) |
| `results/` | `SUMMARY.md` / CSV tables behind every table and figure in the paper |
| `models/` | trained monitor (`gru_monitor_v11.pt`, GRU 2x96, 39 features) and the train-only feature normalization |

Pipeline order: `collect.py` / `run_collection.py` (CARLA) -> `run_lidar_odom.py` (KISS-ICP) ->
`qa_report.py` -> `make_dataset.py` (attack injection, uses `attacks.py`, `scenario_matrix.yaml`) ->
`build_features.py` (39 consistency features) -> `baselines_detect.py`, `train_monitor.py`
(`--arch gru|tcn|transformer`, `--lofo 1`), `arch_baselines.py`, `dasgupta_baseline.py` ->
`adaptive_fusion.py`, `detectability.py`, `efficiency.py`, `kitti_ingest.py` / `kitti_lidar_odom.py` / `kitti_eval.py`.

## Quick start

1. Download the dataset from Kaggle and place `data/` and `dataset/` next to `code/` (the scripts expect `..`).
2. `pip install numpy pandas pyarrow scipy scikit-learn torch matplotlib` (KISS-ICP only for odometry and KITTI scripts).
3. `cd code && python build_features.py --in .. --out ../features`, then
   `python train_monitor.py --features ../features --out ../results/monitor`.

Model-ready features (about 2 GB) are not shipped; rebuild them with step 3. On some Windows/conda installs numpy
crashes on linear algebra unless `MKL_THREADING_LAYER=SEQUENTIAL` is set.

## Scope and limitations

Simulation data with offline attack injection; the attacker controls GNSS only. Replayed tracks and a few slow target-directed
biases remain undetected (see the paper). Resilient fusion restores filter consistency but not point accuracy.
The threshold does not transfer to real data as cleanly as the ranking does.

## Citation

Please cite the arXiv preprint (link to be added when posted). Prior benchmark: Hussain and Khurram, ICIICE 2026,
DOI 10.1109/ICIICE69672.2026.11565150.

## Contact

Azmat Hussain, hussainazmat025@gmail.com
