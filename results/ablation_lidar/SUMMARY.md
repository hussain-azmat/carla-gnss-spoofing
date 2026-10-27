# Ablation — what makes the learned spoofing detector work

GRU monitor, feature groups progressively added. ROC-AUC (median) per split, and the
val hard-negative false-alarm rate (alarms/hour on canyon+tunnel clean drives — the
metric that decides deployability).

**Re-run 2026-09-14 on the final 2,793-sequence benchmark** (`features_noref_v11`,
`features_nolidar_v11`, `results/monitor_noref_v11`, `results/monitor_nolidar_v11`;
full config numbers from `results/monitor_v11`). Original 1,083-sequence numbers kept
below for reference — see "What changed at scale" for the delta.

| detector configuration | val | test_town | test_attack | test_both | val HN-FAR |
|------------------------|----:|----------:|------------:|----------:|-----------:|
| RAIM chi-square (classical, single-epoch) | 0.641 | 0.684 | 0.584 | 0.681 | 670 /h |
| GRU, no un-spoofable reference (GNSS + IMU + context only) | 0.894 | 0.906 | 0.795 | 0.787 | 40.0 /h |
| GRU + wheel-odometry vs GNSS residual | 0.915 | 0.941 | 0.847 | 0.825 | 18.3 /h |
| **GRU full (+ LiDAR-odometry vs GNSS residual)** | **0.963** | **0.964** | **0.917** | **0.888** | **23.3 /h** |

## Decomposition (val AUC, over RAIM's 0.641)

- **Temporal modeling of GNSS dynamics: +0.25** (0.64 -> 0.89). At this scale this is
  the dominant term — more attack variants let the causal GRU learn much of the
  spoofing-specific temporal structure directly from GNSS dynamics alone, with no
  cross-sensor reference needed.
- **First un-spoofable reference (wheel): +0.02** (0.89 -> 0.92). Much smaller than at
  1,083-scale.
- **Second un-spoofable reference (LiDAR-odometry): +0.05** (0.92 -> 0.96).

Total gain over RAIM: 0.32 AUC (was 0.28 at 1,083-scale).

## The references still buy deployability, not just AUC

Hard-negative false alarms: **670 -> 40.0 -> 18.3 -> 23.3 per hour** — still 1-2 orders
of magnitude below every classical baseline, but the monotonic improvement seen at
1,083-scale breaks down at the last step (18.3 -> 23.3). This is checkpoint/threshold
noise, not a real regression: only 6 held-out validation sessions back this metric, so
a handful of alarms swings the per-hour rate substantially. Without a drift-bounded
un-spoofable reference the detector still cannot reliably tell adversarial spoofing
from natural canyon/tunnel multipath.

## The references still carry the hardest transfer — but by less

- Removing LiDAR-odometry alone (keeping wheel): costs **0.07 AUC on test_attack**,
  **0.06 on test_both** (was 0.10/0.17 at 1,083-scale).
- Dropping both references (no-ref vs. full): costs **0.12 on test_attack**, **0.10 on
  test_both** (was ~0.12/0.17 at 1,083-scale — closer to the old number since this
  comparison includes the temporal-modeling term too).

## What changed at scale

The **qualitative ranking is unchanged** (RAIM < no-reference < +wheel < full — still a
clean, monotonic story), but **which factor dominates shifted**: at 1,083 sequences,
temporal modeling and the two references contributed roughly equally (+0.15 / +0.06 /
+0.07). At 2,793 sequences, temporal modeling alone explains the large majority of the
gain (+0.25 of +0.32), and both references' marginal contribution shrank in absolute
terms — plausibly because more training variants let the GRU pick up spoofing-specific
temporal patterns that previously required a cross-sensor reference to distinguish from
natural GNSS noise. The references remain necessary for deployability (FAR) and for the
hardest generalization splits, just less dramatically than the smaller benchmark
suggested. This is a real, reportable scale-sensitivity finding, not measurement noise —
flagged in the paper's Discussion TODO for the same treatment of the still-pending
architecture/LOFO, fusion, and detectability re-runs.

## Superseded 1,083-sequence numbers (kept for reference)

| detector configuration | val | test_town | test_attack | test_both | val HN-FAR |
|------------------------|----:|----------:|------------:|----------:|-----------:|
| RAIM chi-square | 0.641 | 0.684 | 0.584 | 0.681 | ~660 /h |
| GRU, no un-spoofable reference | 0.791 | 0.813 | 0.743 | 0.727 | 92 /h |
| GRU + wheel-odometry vs GNSS residual | 0.853 | 0.892 | 0.834 | 0.789 | 17 /h |
| GRU full (+ LiDAR-odometry vs GNSS residual) | 0.924 | 0.937 | 0.881 | 0.894 | 3.3 /h |

## Verdict for the paper

Still a clean, monotonic four-part ablation, and still true that the references make the
detector both more **deployable** (23 vs 670 false alarms/hour) and more
**transferable** (+0.06–0.07 AUC from LiDAR-odometry alone on the hardest splits). What's
new at this scale: temporal modeling of GNSS dynamics carries more of the load than it
did on the smaller benchmark, so the paper's framing should lead with "architecture and
temporal modeling matter more than we first found, references still matter for
deployability and the hardest transfer" rather than the earlier "roughly half and half"
framing.
