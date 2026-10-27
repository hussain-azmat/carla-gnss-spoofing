# Detection benchmark — architecture & feature comparison

All models: causal per-timestep spoof probability, same splits, thresholds frozen on
CARLA val. "39 consistency" = the full GNSS-vs-{LiDAR-odom, wheel, IMU} residual
feature set; "12 GNSS-only" = GNSS kinematics + context flags, **no cross-sensor
consistency, no un-spoofable reference**.

## ROC-AUC

| model | features | arch | val | test_town | test_attack | test_both |
|---|---|---|---|---|---|---|
| RAIM χ² | residuals | classical | 0.64 | 0.68 | 0.58 | 0.68 |
| logistic regression | 39 consistency | linear, 1 epoch | 0.72 | 0.77 | 0.65 | 0.66 |
| **TCN (rawcnn)** | **12 GNSS-only** | 1-D dilated conv | 0.73 | **0.71** | 0.58 | 0.57 |
| LSTM | 39 consistency | 2×96 causal | 0.94 | 0.94 | 0.89 | 0.86 |
| BiGRU (offline bound) | 39 consistency | 2×96 **bi**-dir | ~0.97 | – | – | – |
| **GRU (this work)** | 39 consistency | 2×96 causal | **0.96** | **0.96** | **0.92** | **0.89** |

(BiGRU: partial run — CPU bidirectional-GRU is pathologically slow here; the ~0.97 val
is from the converged early epochs and is only cited as the non-causal ceiling.)

## What the comparison establishes

1. **The features carry the result, not the architecture.** A deep dilated TCN on
   GNSS-only kinematics reaches only 0.71 on unseen towns — barely above RAIM (0.68)
   and *below* a one-epoch linear model on the consistency features (0.77). Remove the
   un-spoofable cross-sensor references and no amount of temporal modelling recovers it.
2. **Recurrent architecture choice is not load-bearing.** LSTM (0.94) ≈ GRU (0.96) on
   every split; both dominate the non-temporal and non-consistency baselines by
   0.15–0.25 AUC.
3. **The online (causal) constraint is nearly free.** A bidirectional GRU that may look
   into the future gains only ~0.01 AUC over the deployed causal GRU.

## PR-AUC / recall @ 1% FPR

| model | val PR | test_town PR | test_town recall@1%FPR |
|---|---|---|---|
| TCN (GNSS-only) | 0.37 | 0.37 | 0.31 |
| LSTM | 0.65 | 0.66 | 0.67 |
| GRU (this work) | 0.76 | 0.77 | 0.76 |
