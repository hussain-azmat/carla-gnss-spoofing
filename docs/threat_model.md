# Threat model — learned multi-sensor GNSS spoofing detection (LOCKED, Phase 0)

Companion to RESEARCH_PLAN.md §12. This is the contract the benchmark, detector, and
baselines are all built against.

## System under attack
Autonomous ground vehicle with: consumer-grade MEMS IMU (100 Hz), wheel odometry
(~50 Hz, 2 % scale error), 64-beam LiDAR (10 Hz, ≥130 k pts/rev), single-antenna GNSS
receiver (10 Hz, L1). Onboard state estimator = tuned classical EKF (the §7b baseline)
fusing IMU + wheel + GNSS, with LiDAR odometry (KISS-ICP) available as an independent
pose stream. Optional forward camera (validation subset only).

## Attacker capability
- **Controls the GNSS position/velocity solution only.** The attacker has receiver lock
  (spoofer overpowers the live signal) and can set the reported lat/lon/alt (and the
  velocity derived from it) to any value, subject to the per-family plausibility limits
  below. Cannot touch IMU, wheel, LiDAR, or camera.
- **Knows the approximate vehicle trajectory** (road network, current lane, speed) — e.g.
  from tailing the vehicle or map priors. Does **not** have the onboard filter's internal
  state or covariance.
- **Can time the attack.** Onset is either uniform-random or adversarially aligned to a
  manoeuvre point (approaching an intersection, lane change, highway merge). The benchmark
  logs junction-approach times per session for the adversarial case.
- Single attacker, single vehicle. No collusion, no replay of *other* vehicles' signals
  beyond meaconing this vehicle's own earlier signal.

## Attacker goal
Induce a bounded **lateral or longitudinal position error** in the fused estimate that is
large enough to cause a driving fault (wrong lane ≈ 3.5 m lateral, stop-line overrun ≈
5–10 m longitudinal, wrong turn at a junction) **while evading onboard integrity
monitoring** (RAIM χ², innovation gating). Not trying to crash the filter — trying to
steer it.

## Six attack families (parameter ranges in `attacks.py:PARAM_RANGES`)
| # | family | idea | plausibility constraint |
|---|--------|------|-------------------------|
| 1 | `gradual_pulloff` | ramp the offset in slowly | ramp rate ≤ 0.8 m/s; max offset ≤ 25 m |
| 2 | `jump_and_hold` | single step, then hold | step 2–12 m (small enough to pass a loose gate) |
| 3 | `replay` (meaconing) | re-emit this run's own GNSS, lag 3–20 s | internally consistent by construction |
| 4 | `random_walk` | Brownian offset overlay | σ 0.15–0.6 m/fix, norm-clipped ≤ 20 m |
| 5 | `zv_freeze` | freeze reported position at onset | none (crude; detectable — included as an easy case) |
| 6 | `target_bias` | closed-loop push toward true-path-shifted-3.5 m | rate ≤ 0.7 m/s (the “wrong lane” attack) |

Onset: random **or** adversarially-timed. Duration 8–40 s. Attacks are injected post-hoc
on clean runs, so every clean session yields many labelled variants.

## Explicitly out of scope (v1)
- **Jamming / full GNSS denial** — that is pure outage, already covered by the §10
  dead-reckoning baselines; not a detection problem.
- Sensor-level attacks on LiDAR / IMU / camera (spoofed returns, acoustic IMU attacks).
- Multi-vehicle collusion; attacks on V2X or map data.
- Attacks on the time base / PPS.

## What makes this detectable (the thesis bet)
LiDAR odometry drift is bounded to well under 1 m over a 10–30 s window and **cannot be
corrupted by a GNSS spoofer**. A learned recurrent monitor watching the *temporal*
structure of GNSS-vs-{inertial, wheel, LiDAR-odometry} residuals sees a signature that
(a) single-epoch RAIM cannot, and (b) transfers across towns and to held-out attack
families. `target_bias` and slow `gradual_pulloff` near the rate limit are the hard cases;
the detectability-limits analysis (Phase 5) characterises where detection fails.
