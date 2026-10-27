#!/usr/bin/env python3
"""
make_dataset.py -- turn CLEAN collect.py sessions into the labelled spoofing dataset
(RESEARCH_PLAN §12, contribution 1). Runs offline, no CARLA, no GPU.

For every session under --in that has a manifest.json:
  * load gnss_clean.parquet + gt_pose.parquet
  * build attacks.CleanGnss (GNSS stream + GT ENU + GT heading, GT resampled to GNSS times)
  * for each attack family in the split, emit `variants_per_family` labelled variants:
      variants/<session>/<family>__v<k>.parquet   (t, lat, lon, alt, e, n, u, label, offset_e, offset_n)
  * also emit the clean stream itself as a negative
Writes dataset_index.parquet: one row per (session, variant) with split + metadata.

    python make_dataset.py --in ./out --out ./dataset --matrix scenario_matrix.yaml
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from attacks import CleanGnss, apply, FAMILIES


def _resample(t_src, arr_src, t_dst):
    return np.interp(t_dst, t_src, arr_src)


def _clean_from_session(sdir: str):
    ev = json.load(open(os.path.join(sdir, "events.json")))
    gnss = pd.read_parquet(os.path.join(sdir, "gnss_clean.parquet"))
    gt = pd.read_parquet(os.path.join(sdir, "gt_pose.parquet"))
    t = gnss["t"].to_numpy()
    gt_e = _resample(gt["t"].to_numpy(), gt["e"].to_numpy(), t)
    gt_n = _resample(gt["t"].to_numpy(), gt["n"].to_numpy(), t)
    yaw = np.unwrap(_resample(gt["t"].to_numpy(), np.unwrap(gt["yaw"].to_numpy()), t))
    spd = _resample(gt["t"].to_numpy(), gt["speed"].to_numpy(), t)
    clean = CleanGnss(
        t=t, lat=gnss["lat"].to_numpy(), lon=gnss["lon"].to_numpy(), alt=gnss["alt"].to_numpy(),
        origin=tuple(ev["origin_latlonalt"]),
        gt_en=np.stack([gt_e, gt_n], axis=1), gt_heading=yaw,
    )
    # attacks are only meaningful while the vehicle is MOVING -> restrict every onset
    # (adversarial junction times AND uniform-random) to moving instants with runway.
    moving_t = t[spd > 3.0]
    jev = [e for e in ev.get("junction_onset_events", [])
           if np.any(np.abs(moving_t - e) < 0.5)]
    return clean, jev, list(moving_t), ev


def _variant_df(clean: CleanGnss, r: dict) -> pd.DataFrame:
    from sensors import geo_to_enu
    e, n = np.vectorize(lambda a, b: geo_to_enu(a, b, 0, *clean.origin)[:2])(r["lat"], r["lon"])
    return pd.DataFrame(dict(
        t=clean.t, lat=r["lat"], lon=r["lon"], alt=r["alt"],
        e=e, n=n, u=clean.alt - clean.origin[2],
        label=r["label"], offset_e=r["offset_en"][:, 0], offset_n=r["offset_en"][:, 1],
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="./out")
    ap.add_argument("--out", default="./dataset")
    ap.add_argument("--matrix", default="scenario_matrix.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    M = yaml.safe_load(open(args.matrix))
    A = M["attacks"]
    fam_train = A["families_train"]
    fam_test = A["families_test_unseen"]
    nper = int(A["variants_per_family_per_session"])
    _norm = lambda t: t.replace("_Opt", "")
    train_towns = {_norm(t) for t in M["towns"]["train"]}
    dur = tuple(A.get("duration_s", [8, 40]))

    os.makedirs(args.out, exist_ok=True)
    index = []
    rng = np.random.default_rng(args.seed)

    sessions = sorted(d for d in os.listdir(args.inp)
                      if os.path.exists(os.path.join(args.inp, d, "manifest.json")))
    print(f"{len(sessions)} valid sessions")

    for sess in sessions:
        sdir = os.path.join(args.inp, sess)
        clean, jev, moving_t, ev = _clean_from_session(sdir)
        town = ev["town"]
        town_split = "train" if _norm(town) in train_towns else "test_unseen"
        vdir = os.path.join(args.out, "variants", sess)
        os.makedirs(vdir, exist_ok=True)

        # clean negative
        neg = dict(lat=clean.lat, lon=clean.lon, alt=clean.alt,
                   label=np.zeros(len(clean.t), np.uint8),
                   offset_en=np.zeros((len(clean.t), 2)))
        _variant_df(clean, neg).to_parquet(os.path.join(vdir, "clean__v0.parquet"), index=False)
        gnss_cond = ev.get("gnss_condition", "nominal")
        index.append(dict(session=sess, town=town, weather=ev["weather"], town_split=town_split,
                          gnss_condition=gnss_cond, variant="clean__v0", family="clean",
                          is_attack=0, hard_negative=int(gnss_cond in ("canyon", "tunnel")),
                          onset_s=np.nan, duration_s=np.nan, max_offset_m=0.0,
                          path=os.path.relpath(os.path.join(vdir, "clean__v0.parquet"), args.out)))

        for fam in fam_train + fam_test:
            fam_split = "train" if fam in fam_train else "test_unseen"
            for kv in range(nper):
                if fam == "target_bias" and clean.gt_en is None:
                    continue
                use_adv = (M["attacks"].get("onset_policy") == "mix" and rng.random() < 0.5)
                seed = args.seed + hash((sess, fam, kv)) % 1_000_000
                # adversarial -> junction-approach times; random -> any moving instant.
                cand = jev if (use_adv and jev) else moving_t
                r = apply(clean, fam, seed=seed,
                          duration_s=float(rng.uniform(*dur)),
                          onset_events=cand if len(cand) else None)
                vp = os.path.join(vdir, f"{fam}__v{kv}.parquet")
                _variant_df(clean, r).to_parquet(vp, index=False)
                index.append(dict(
                    session=sess, town=town, weather=ev["weather"], town_split=town_split,
                    gnss_condition=gnss_cond,
                    variant=f"{fam}__v{kv}", family=fam,
                    is_attack=1, hard_negative=0,
                    attack_split=fam_split, onset_adversarial=int(use_adv and bool(jev)),
                    onset_s=r["meta"]["onset_s"], duration_s=r["meta"]["duration_s"],
                    max_offset_m=r["meta"]["max_applied_offset_m"],
                    path=os.path.relpath(vp, args.out)))

    idx = pd.DataFrame(index)
    idx.to_parquet(os.path.join(args.out, "dataset_index.parquet"), index=False)
    print(idx.groupby(["town_split", "family"]).size())
    print(f"\nwrote {len(idx)} variants -> {args.out}/dataset_index.parquet")


if __name__ == "__main__":
    main()
