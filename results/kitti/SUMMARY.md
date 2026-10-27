# Sim-to-real: CARLA-trained monitor on real KITTI drives

Model: GRU (2x96) trained only on CARLA. Normalisation + operating threshold frozen from CARLA. 5 KITTI drive(s), 6 spoof variants/family/drive + clean.

- Clean-drive false-alarm rate at CARLA-val 1% FPR: **51.4 / hour**
- CARLA reference (test_town): ROC-AUC 0.964, PR-AUC 0.771, recall@1%FPR 0.761

| tag             |   n_seq |   roc_auc |   pr_auc |   tpr_at_1pct_fpr |   median_tta_s |   miss_rate |
|:----------------|--------:|----------:|---------:|------------------:|---------------:|------------:|
| ALL             |     185 |     0.862 |    0.512 |             0.576 |           0.7  |       0.05  |
| attacks_only    |     185 |     0.862 |    0.512 |             0.576 |           0.7  |       0.05  |
| gradual_pulloff |      35 |     0.886 |    0.427 |             0.684 |           3.5  |       0     |
| jump_and_hold   |      35 |     0.873 |    0.27  |             0.528 |           0.2  |       0.067 |
| replay          |      35 |     0.823 |    0.426 |             0.409 |           0.35 |       0.133 |
| random_walk     |      35 |     0.851 |    0.321 |             0.55  |           1.05 |       0     |
| zv_freeze       |      35 |     0.988 |    0.937 |             0.95  |           0.3  |       0.033 |
| target_bias     |      35 |     0.686 |    0.158 |             0.303 |          13.15 |       0.067 |

## Domain adaptation (leave-one-drive-out, real clean data only)

`pctl_thr` = threshold at 99th percentile of real clean scores. `aligned` = CARLA score distribution matched to the target domain in logit space (mean+std of clean scores), CARLA-frozen 1% FPR threshold applied in the aligned units. Both use unlabelled real clean data.

| drive                      |   auc |   recall_pctl_thr |   far_pctl_thr |   recall_aligned |   far_aligned |   far_carla_thr |
|:---------------------------|------:|------------------:|---------------:|-----------------:|--------------:|----------------:|
| 2011_09_26_drive_0009_sync | 0.704 |             0.167 |          0     |            0.169 |         0     |         161.074 |
| 2011_09_26_drive_0117_sync | 0.869 |             0.281 |         54.545 |            0.255 |        54.545 |         109.091 |
| 2011_09_30_drive_0028_sync | 0.809 |             0.157 |         27.815 |            0.19  |        34.769 |          48.677 |
| 2011_09_30_drive_0033_sync | 0.893 |             0.181 |          0     |            0.093 |         0     |          22.585 |
| 2011_09_30_drive_0034_sync | 0.833 |             0.101 |          0     |            0.085 |         0     |          29.412 |

- CARLA-threshold (no adaptation): recall 0.58, FAR 74/hr
- percentile recalibration:        recall 0.18, FAR 16/hr
- **distribution alignment:        recall 0.16, FAR 18/hr**

## Reading

**Features transfer; the operating threshold does not.** The CARLA-trained monitor,
run zero-shot on real KITTI drives, holds a threshold-free ROC-AUC of **0.862** across
5 drives / 185 attack sequences — down only ~0.10 from the CARLA test-town reference
(0.964) despite zero real-world training, zero retraining, and CARLA-frozen
normalisation. The result is stable: 2 drives → 0.82, 4 → 0.869, 5 → 0.862, where the
5th drive (0028) carries 3× the data of any other. This is the paper's core
sim-to-real claim.

**Scene dependence is real and reportable.** Residential / low-density streets (0033
0.89, 0034 0.83, 0028 0.81) and static city (0117 0.87) transfer cleanly; dense,
dynamic urban traffic (0009 0.70) degrades — the consistency residuals pick up genuine
multipath / NLOS GNSS error and aggressive real driving dynamics that the CARLA
training distribution under-represents. State this as a limitation, not hide it.

**Per-family: physically-impossible attacks transfer best.** zv_freeze (frozen
position while moving) 0.99 — the wheel/IMU contradiction is domain-invariant. Slow,
small-magnitude attacks (target_bias 0.69) are the hard case in both CARLA and the
real world; nothing about sim-to-real makes that worse.

**The threshold shift is the honest catch.** Real clean scores pile near 1.0, so the
CARLA operating point fires ~51 false alarms/hr on clean real drives. Unsupervised
recalibration on real clean data (leave-one-drive-out percentile or logit-space
distribution alignment) cuts FAR to ~16–18/hr but trades recall down to ~0.17.
Neither cleanly solves it. AUC (threshold-free) is the quantity that transfers;
picking a real-world operating point needs a modest amount of target-domain clean
data, which is cheap to collect.
