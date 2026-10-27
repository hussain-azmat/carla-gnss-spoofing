# Classical detection baselines — results

Phase 3 baselines from `code/baselines_detect.py`, scored per `code/docs/metrics.md`.
Thresholds frozen on **val**, applied unchanged to every test split.
Feature set: 39 consistency features / 10 Hz timestep (`code/build_features.py`).

Attack prevalence ~3–4 % of timesteps. Splits:
train 429 seq / 2.23 M steps · val 78 / 0.56 M · test_town 234 / 1.38 M ·
test_attack 234 / 1.29 M · test_both 108 / 0.64 M.

## Headline (ROC-AUC)

| method | val | test_town | test_attack | test_both |
|--------|-----|-----------|-------------|-----------|
| RAIM χ² (single-epoch) | 0.64 | 0.68 | 0.58 | 0.68 |
| pairwise z-check | 0.67 | 0.71 | 0.65 | 0.72 |
| CUSUM | 0.59 | 0.68 | 0.53 | 0.61 |
| Huber residual | 0.69 | 0.75 | 0.66 | 0.76 |
| **logreg (39-feat, no time context)** | **0.72** | **0.77** | 0.65 | 0.66 |

## What the numbers say (this is the gap the learned monitor must close)

1. **Single-epoch detection tops out around AUC 0.7.** No classical method exceeds
   0.77 on any split; RAIM χ² — the standard onboard integrity check — never beats 0.68.

2. **RAIM / pairwise / Huber are unusable on hard negatives.** On val canyon/tunnel
   clean sequences they fire **≈ 610–670 false alarms per hour**. This is the metric
   `metrics.md` calls "the one that kills naive detectors" — confirmed.

3. **CUSUM trades it for zero precision.** Hard-neg FAR drops to 5/h but PR-AUC is
   0.04–0.06 — it is essentially always alarming (TPR@1%FPR 0.97 is meaningless).

4. **Best non-temporal baseline = logreg on the full feature vector**: val AUC 0.72,
   test-town 0.77, hard-neg FAR 148/h, but still **54–60 % misses at the 1 %-FPR
   operating point**. This is the "single-epoch MLP" bar — the recurrent model must
   beat it to justify temporal modelling.

5. **Replay ≈ undetectable by residual methods.** On test_attack (replay + target_bias)
   every residual method sits at exactly ~50 % miss rate: replay is internally
   consistent by construction so the GNSS-vs-LiDAR-odom residual stays small, while
   target_bias grows a residual and is often caught. A temporal model watching the
   *dynamics* (replay freezes then lags) is the intended answer — and a good
   detectability-limits story for Phase 5.

## Known limitation — hard-negative coverage

All canyon/tunnel clean sessions are in **train towns** (Town03/Town05/Town10HD); the
unseen test towns (Town02/04/06/07) were collected with nominal GNSS only. So
hard-negative false-alarm rate is measurable on **val only** (3 sequences) — we cannot
yet test "does the detector false-alarm on canyon in an *unseen* town". Cheap fix: a
~4-session collection wave of canyon/tunnel clean drives in Town02/04. Not blocking.

Full numbers: `baselines.csv`.
