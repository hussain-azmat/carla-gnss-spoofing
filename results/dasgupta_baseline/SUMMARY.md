# Dasgupta-style learned dead-reckoning-vs-GNSS baseline (2,793-sequence benchmark)

Re-implementation of the idea of Dasgupta et al., IEEE T-ITS 23(12), 2022 (DOI 10.1109/TITS.2022.3197817) on our benchmark and protocol. An LSTM (2x64) trained on CLEAN driving only (33 train sessions) predicts per-step displacement from accelerometer (x,y), yaw rate, wheel speed and steering angle (`steer` channel of gt_pose.parquet; no separate steering sensor is logged). Predicted steps are placed in the world frame with the calibrated compass heading, summed over a w-second window and compared with the GNSS shift; score = distance between the two shifts. Window w is the only tuned quantity (val ROC-AUC, grid 1-60 s). Same splits, val-frozen 1%-FPR threshold and metrics as the GRU monitor.

Not reproduced: the paper's kNN+DTW turn-classification module, its own driving data. No LiDAR odometry is used (no un-spoofable exteroceptive reference).

Predictor sanity: total-path error 1.04% on held-out clean val sessions; heading calibration mean resultant length 1.000.

## Result (chosen w = 20 s)

| split       |   roc_auc |   pr_auc |   tpr_at_1pct_fpr |   miss_rate |   hardneg_far_per_hr |
|:------------|----------:|---------:|------------------:|------------:|---------------------:|
| val         |     0.722 |    0.177 |             0.217 |       0.75  |                  315 |
| test_town   |     0.742 |    0.137 |             0.263 |       0.606 |                  nan |
| test_attack |     0.776 |    0.344 |             0.472 |       0.502 |                  nan |
| test_both   |     0.796 |    0.281 |             0.511 |       0.413 |                  nan |

## Reference: GRU monitor (full features, 2,793 seq)

| split       |   roc_auc |   pr_auc |   tpr_at_1pct_fpr |   miss_rate |   hardneg_far_per_hr |
|:------------|----------:|---------:|------------------:|------------:|---------------------:|
| val         |     0.963 |    0.756 |             0.706 |       0.031 |               23.333 |
| test_town   |     0.964 |    0.771 |             0.761 |       0.007 |              nan     |
| test_attack |     0.917 |    0.657 |             0.505 |       0.155 |              nan     |
| test_both   |     0.888 |    0.52  |             0.511 |       0.056 |              nan     |

## Window sweep (ROC-AUC)

|   window_s |   val |   test_town |   test_attack |   test_both |
|-----------:|------:|------------:|--------------:|------------:|
|          1 | 0.619 |       0.694 |         0.661 |       0.734 |
|          2 | 0.652 |       0.728 |         0.708 |       0.768 |
|          3 | 0.669 |       0.744 |         0.729 |       0.788 |
|          5 | 0.692 |       0.754 |         0.752 |       0.805 |
|         10 | 0.72  |       0.758 |         0.773 |       0.812 |
|         20 | 0.722 |       0.742 |         0.776 |       0.796 |
|         30 | 0.71  |       0.724 |         0.763 |       0.774 |
|         45 | 0.7   |       0.704 |         0.753 |       0.747 |
|         60 | 0.714 |       0.695 |         0.753 |       0.74  |
