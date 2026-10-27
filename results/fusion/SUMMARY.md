# Fusion re-run on the final 2,793-sequence benchmark (2026-10-19)

Same code (code/adaptive_fusion.py), scores from results/monitor_v11, parameters re-tuned on val
(kappa 6.0, ema 0.2, tau_hi 0.55, tau_lo 0.275), 120 attack variants per split, medians.
Previous (1,083-seq) results are in ../fusion_v1_backup/.

test_town: ATE / max err / recovery s / NEES
A0 6.01/8.41/6.7/42.7 | A 5.03/7.55/6.85/42.5 | B 5.08/8.12/9.1/31.4 | C 6.79/10.4/14.2/5.6 |
D 5.89/10.1/11.85/5.6 | E 6.55/10.9/13.4/5.1 | F(oracle) 5.98/9.84/0.4/2.6

Read: standard gating (A) is best on point accuracy; detector-driven C/D/E are 0.9-1.8 m WORSE (E mean
15.5 m vs 11.6 m for A, p90 43 vs 15 m; E better than A on 48% of variants) and even the oracle does not
beat A. The consistency win holds (NEES 42.5 -> 5.1-5.6; on test_attack/test_both 9.4/12.3 -> 3.8/4.2).
By family E is worse than A on gradual_pulloff (8.1 vs 5.0) and random_walk (7.3 vs 3.7), about equal
on jump_and_hold and zv_freeze. The earlier "~25% better on gradual_pulloff" no longer holds.
See fusion_summary.csv, fusion_by_family.csv, fusion_raw.parquet.
