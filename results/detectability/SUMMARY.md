# Detectability-limits analysis

Threshold frozen to 1% FPR on val (smoothed score > 0.928), sustained 1 s within the attack window.

- Overall attack detection rate: **93%** (1488 variants)
- Attacks large enough to cause a wrong-lane fault (> 3.5 m): 1133; of these **29 (3%) go undetected** — the exploitable stealth region.

## Detection rate by family

| family          |   n |   det_rate |   med_latency |
|:----------------|----:|-----------:|--------------:|
| gradual_pulloff | 144 |      0.993 |          3.3  |
| jump_and_hold   | 144 |      0.979 |          0.8  |
| random_walk     | 144 |      1     |          2.1  |
| replay          | 456 |      0.956 |          0.9  |
| target_bias     | 456 |      0.811 |          5.25 |
| zv_freeze       | 144 |      1     |          0.9  |

## Detection rate vs ramp rate (gradual_pulloff + target_bias)

| rr_bin      |   detected |
|:------------|-----------:|
| (0.0, 0.1]  |       0.8  |
| (0.1, 0.25] |       0.83 |
| (0.25, 0.5] |       0.92 |
| (0.5, 1.0]  |       0.98 |
