#!/usr/bin/env python3
"""
attacks.py -- GNSS spoofing injection for the benchmark (RESEARCH_PLAN.md §12, contribution 1).

Attacks are applied POST-HOC to a clean recorded GNSS stream, so one clean session yields
many labelled attack variants at zero extra simulation cost. Everything works in a local
ENU frame (metres) anchored at the map origin; convert back to lat/lon on the way out.

Six families (locked Phase-0 set):

  1 gradual_pulloff   ramped offset, rate-limited so the fix stays physically plausible;
                      the canonical "keeps you on a plausible path while steering you wrong"
  2 jump_and_hold     single step at onset, then constant -- crude but common; RAIM-visible
                      only at the step edge
  3 replay            meaconing: re-emit this run's own earlier GNSS, time-shifted by lag_s;
                      internally consistent, wrong in world frame
  4 random_walk       Brownian offset overlay (sigma_step per fix), norm-clipped
  5 zv_freeze         spoofed position frozen at onset location (truth keeps moving)
  6 target_bias       closed-loop: push the fix toward a target path = true path shifted
                      lateral_m (e.g. 3.5 m -> adjacent lane), rate-limited

Onset: random by default; pass onset_events (list of sim-times when the vehicle is
~approaching an intersection / lane change) to get adversarially-timed onsets.

Output of apply(): dict with
  lat/lon/alt      spoofed streams (same length/timestamps as input)
  label            per-sample uint8, 1 while attack active
  offset_en        (N,2) applied ENU offset, for analysis
  meta             family + sampled parameters + onset/offset times
"""
from __future__ import annotations
import argparse
import dataclasses
import json
from typing import Dict, List, Optional, Sequence

import numpy as np

from sensors import geo_to_enu, enu_to_geo

FAMILIES = ["gradual_pulloff", "jump_and_hold", "replay", "random_walk", "zv_freeze", "target_bias"]

# per-family parameter ranges -- sampled uniformly unless a fixed value is passed
PARAM_RANGES = {
    "gradual_pulloff": dict(max_offset_m=(3.0, 25.0), ramp_rate_mps=(0.2, 0.8),
                            bearing_mode=["lateral", "fixed"], hold_after_s=(2.0, 15.0)),
    "jump_and_hold":   dict(step_m=(2.0, 12.0), bearing_mode=["lateral", "fixed"]),
    "replay":          dict(lag_s=(3.0, 20.0)),
    "random_walk":     dict(sigma_step_m=(0.15, 0.6), clip_m=(4.0, 20.0)),
    "zv_freeze":       dict(),                       # position freeze has no free params
    "target_bias":     dict(lateral_m=(2.5, 4.5), ramp_rate_mps=(0.2, 0.7)),
}

DEFAULT_DURATION_S = (8.0, 40.0)     # how long an attack lasts once it starts


@dataclasses.dataclass
class CleanGnss:
    t: np.ndarray                    # (N,) sim seconds, monotonically increasing
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    origin: tuple                    # (lat0, lon0, alt0) map origin for ENU
    gt_en: Optional[np.ndarray] = None      # (N,2) ground-truth ENU, needed for target_bias
    gt_heading: Optional[np.ndarray] = None  # (N,) yaw rad, needed for "lateral" bearing

    @property
    def en(self) -> np.ndarray:
        e, n = np.vectorize(lambda a, b: geo_to_enu(a, b, 0, *self.origin)[:2])(self.lat, self.lon)
        return np.stack([e, n], axis=1)


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _sample_params(family: str, rng: np.random.Generator, overrides: dict) -> dict:
    out = {}
    for k, spec in PARAM_RANGES[family].items():
        if k in overrides:
            out[k] = overrides[k]
        elif isinstance(spec, list):
            out[k] = spec[int(rng.integers(len(spec)))]
        else:
            out[k] = _u(rng, spec)
    return out


def _pick_onset(t: np.ndarray, dur: float, rng: np.random.Generator,
                onset_events: Optional[Sequence[float]]) -> float:
    t0, t1 = float(t[0]), float(t[-1])
    latest = t1 - dur - 1.0
    if latest <= t0 + 2.0:
        return t0 + 1.0
    if onset_events:
        cand = [e for e in onset_events if t0 + 2.0 <= e <= latest]
        if cand:
            return float(rng.choice(cand))
    return float(rng.uniform(t0 + 2.0, latest))


def _bearing_unit(params, clean: CleanGnss, active: np.ndarray, rng) -> np.ndarray:
    """(N,2) unit vector for the offset direction."""
    N = len(clean.t)
    if params.get("bearing_mode") == "lateral" and clean.gt_heading is not None:
        # left-normal of heading (CARLA yaw: x=cos,y=sin in a left-handed frame -> use +90deg)
        h = clean.gt_heading
        u = np.stack([-np.sin(h), np.cos(h)], axis=1)
        # random sign so it's not always the same side
        u *= (1.0 if rng.random() < 0.5 else -1.0)
        return u
    # fixed random compass bearing
    ang = rng.uniform(0, 2 * np.pi)
    return np.tile([np.cos(ang), np.sin(ang)], (N, 1))


def apply(clean: CleanGnss, family: str, *, seed: int = 0,
          duration_s: Optional[float] = None,
          onset_events: Optional[Sequence[float]] = None,
          **overrides) -> Dict:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; pick from {FAMILIES}")
    rng = np.random.default_rng(seed)
    t = np.asarray(clean.t, float)
    N = len(t)
    dt = np.gradient(t)
    en = clean.en.astype(float)                       # (N,2) clean measured ENU

    dur = duration_s if duration_s is not None else _u(rng, DEFAULT_DURATION_S)
    onset = _pick_onset(t, dur, rng, onset_events)
    end = onset + dur
    active = (t >= onset) & (t <= end)
    tau = np.clip(t - onset, 0, None)                 # seconds since onset

    params = _sample_params(family, rng, overrides)
    offset = np.zeros((N, 2))

    if family == "gradual_pulloff":
        u = _bearing_unit(params, clean, active, rng)
        mag = np.minimum(params["ramp_rate_mps"] * tau, params["max_offset_m"])
        # optional shrink-back after hold_after_s past reaching max (kept simple: just hold)
        offset = (mag[:, None] * u) * active[:, None]

    elif family == "jump_and_hold":
        u = _bearing_unit(params, clean, active, rng)
        offset = (params["step_m"] * u) * active[:, None]

    elif family == "replay":
        lag = params["lag_s"]
        src_idx = np.clip(np.searchsorted(t, t - lag), 0, N - 1)
        replayed = en[src_idx]
        offset = (replayed - en) * active[:, None]

    elif family == "random_walk":
        steps = rng.normal(0, params["sigma_step_m"], size=(N, 2))
        walk = np.cumsum(steps * active[:, None], axis=0)
        nrm = np.linalg.norm(walk, axis=1, keepdims=True) + 1e-9
        walk = walk * np.minimum(1.0, params["clip_m"] / nrm)
        offset = walk * active[:, None]

    elif family == "zv_freeze":
        k0 = int(np.argmax(active)) if active.any() else 0
        frozen = en[k0]
        offset = (frozen - en) * active[:, None]

    elif family == "target_bias":
        if clean.gt_en is None or clean.gt_heading is None:
            raise ValueError("target_bias needs clean.gt_en and clean.gt_heading")
        h = clean.gt_heading
        lat_n = np.stack([-np.sin(h), np.cos(h)], axis=1)
        side = 1.0 if rng.random() < 0.5 else -1.0
        target = clean.gt_en + side * params["lateral_m"] * lat_n
        desired = target - en                          # full offset that would place fix on target
        # rate-limit the realised offset toward `desired`
        realised = np.zeros((N, 2))
        for k in range(1, N):
            if not active[k]:
                realised[k] = realised[k - 1] * (0.0 if not active[k - 1] else 1.0)
                continue
            step_cap = params["ramp_rate_mps"] * max(dt[k], 1e-3)
            delta = desired[k] - realised[k - 1]
            dn = np.linalg.norm(delta) + 1e-9
            realised[k] = realised[k - 1] + delta * min(1.0, step_cap / dn)
        offset = realised

    spoof_en = en + offset
    lat_s, lon_s, alt_s = [], [], []
    for k in range(N):
        la, lo, al = enu_to_geo(spoof_en[k, 0], spoof_en[k, 1],
                                clean.alt[k] - clean.origin[2], *clean.origin)
        lat_s.append(la); lon_s.append(lo); alt_s.append(clean.alt[k])

    return dict(
        lat=np.array(lat_s), lon=np.array(lon_s), alt=np.array(alt_s),
        label=active.astype(np.uint8),
        offset_en=offset,
        meta=dict(family=family, seed=seed, onset_s=onset, end_s=end, duration_s=dur,
                  params=params, max_applied_offset_m=float(np.linalg.norm(offset, axis=1).max())),
    )


def make_variants(clean: CleanGnss, n_per_family: int = 3, base_seed: int = 0,
                  families: Optional[List[str]] = None,
                  onset_events: Optional[Sequence[float]] = None) -> List[Dict]:
    fams = families or FAMILIES
    out = []
    for fi, fam in enumerate(fams):
        for j in range(n_per_family):
            if fam == "target_bias" and (clean.gt_en is None or clean.gt_heading is None):
                continue
            out.append(apply(clean, fam, seed=base_seed + 1000 * fi + j,
                             onset_events=onset_events))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="self-test the 6 attack families on a synthetic drive")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    # synthetic 60 s straight-then-turn drive at 10 Hz
    t = np.arange(0, 60, 0.1)
    speed = 8.0
    heading = np.where(t < 30, 0.0, np.clip((t - 30) * 0.05, 0, np.pi / 2))
    e = np.cumsum(np.cos(heading) * speed * 0.1)
    n = np.cumsum(np.sin(heading) * speed * 0.1)
    lat0, lon0, alt0 = 0.0, 0.0, 0.0
    lat, lon, _ = np.vectorize(lambda x, y: enu_to_geo(x, y, 0, lat0, lon0, alt0))(e, n)
    clean = CleanGnss(t=t, lat=np.array(lat), lon=np.array(lon), alt=np.full_like(t, alt0),
                      origin=(lat0, lon0, alt0),
                      gt_en=np.stack([e, n], 1), gt_heading=heading)

    for fam in FAMILIES:
        r = apply(clean, fam, seed=1)
        print(f"{fam:16s} active={int(r['label'].sum()):4d}/{len(t)}  "
              f"max_offset={r['meta']['max_applied_offset_m']:5.1f} m  params={r['meta']['params']}")

    if a.plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        for ax, fam in zip(axs.ravel(), FAMILIES):
            r = apply(clean, fam, seed=1)
            se, sn = np.vectorize(lambda la, lo: geo_to_enu(la, lo, 0, lat0, lon0, alt0)[:2])(r["lat"], r["lon"])
            ax.plot(e, n, "k-", lw=1, label="true")
            ax.plot(se, sn, "r-", lw=1, label="spoofed")
            ax.set_title(fam); ax.axis("equal")
        axs[0, 0].legend()
        plt.tight_layout(); plt.savefig("attacks_selftest.png", dpi=110)
        print("wrote attacks_selftest.png")
