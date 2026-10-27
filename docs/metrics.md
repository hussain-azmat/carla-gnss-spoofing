# Metrics & evaluation protocol (LOCKED, Phase 0)

Companion to RESEARCH_PLAN.md §12. Every method (learned monitor + baselines) is scored
the same way on the same splits.

## Splits (generalisation is the whole point)
| split | towns | attack families | purpose |
|-------|-------|-----------------|---------|
| **train** | Town03, Town05, Town10HD | gradual_pulloff, jump_and_hold, random_walk, zv_freeze | fit |
| **val** | held-out sessions from train towns/families | same | model selection, threshold pick |
| **test-town** | Town02, Town04, Town07 | train families | unseen environment |
| **test-attack** | train towns | replay, target_bias | unseen attack family |
| **test-both** | Town02/04/07 | replay, target_bias | hardest |

Thresholds are frozen on **val** and never re-tuned per test split.

## Detection metrics (per-timestep, attack vs no-attack)
- **ROC-AUC** and **PR-AUC** over all test timesteps (attack-active = positive).
- **TPR @ FPR = 1 %** and **@ FPR = 0.1 %** — operating points a real integrity monitor
  would use.
- **Time-to-alarm (TTA)**: seconds from attack onset to first sustained detection
  (detection held ≥ 1 s). Report median + 90th percentile per family. Undetected attacks
  → TTA = ∞, counted separately as **miss rate**.
- **False-alarm rate on hard negatives**: alarms per hour on canyon/tunnel clean sessions
  (natural GNSS degradation, no attack). This is the metric that kills naive detectors.

## Hard-negative set (no attack — must NOT alarm)
- **Urban canyon**: Town03 / Town10HD sessions with `gnss_condition: canyon` —
  +2.5 m horizontal multipath 1-σ, biased toward the building side.
- **Tunnel**: Town03 sessions with `gnss_condition: tunnel` — GNSS outage windows under
  tunnel segments, then re-acquisition jump.
- **Aggressive normal driving**: hard braking, fast lane changes in clean nominal
  sessions (already present via traffic-manager behaviour + speed offsets).

Hard negatives are sampled at 3× weight in evaluation reporting.

## Resilience metrics (attack-aware fusion, Phase 4)
Measured on test splits, attack sessions only:
- **ATE-under-attack**: RMSE of fused position vs GT over the attack window.
- **Max position error** during the attack window.
- **Recovery time**: seconds after attack ends until fused error returns below 1.0 m and
  stays there ≥ 3 s.
- **NEES** (normalised estimation error squared) vs the filter's own covariance — is the
  filter still consistent under attack + mitigation?
- Baseline to beat: EKF with standard innovation gating (no learned detector).

## Baselines (Phase 3)
1. **RAIM** — single-epoch χ² on GNSS residual vs predicted position, standard threshold.
2. **Pairwise innovation cross-check** — GNSS-vs-inertial and GNSS-vs-wheel innovation
   z-scores with a fixed gate.
3. **CUSUM** — change detection on the GNSS-vs-dead-reckoning residual.
4. **Huber / robust EKF** — down-weights large residuals but has no explicit detector.
5. **Single-epoch MLP** — same features as the learned monitor but no temporal context
   (isolates the value of the recurrence).

## Reporting
Per split: table of ROC-AUC / PR-AUC / TPR@1%FPR / median TTA / hard-neg FAR, broken out
by attack family. Plus the detectability-limits plot: detection rate vs (ramp rate ×
max offset) for `gradual_pulloff` and `target_bias`.
