"""
Phase 6 — efficiency / deployability of the GNSS-spoofing monitor (T-ITS hardening).

Software-only (no hardware board). For the trained causal GRU monitor we report, for
FP32 / FP16 / dynamic-INT8 / simulated-INT4-weight-only:
  * model size on disk, parameter count
  * streaming (per-timestep) inference latency on CPU  -- the deployment mode: one
    10 Hz GNSS epoch in, one spoof probability out, hidden state carried
  * full-sequence throughput (timesteps / s)
  * detection accuracy on every split (ROC-AUC, PR-AUC, TPR@1%FPR) vs the FP32 model

Real-time budget: GNSS epochs arrive at 10 Hz => 100 ms/epoch. Anything well under
that runs online on a vehicle ECU-class CPU with no accelerator.

Usage:
  python efficiency.py --features ../features --ckpt ../results/monitor_v11/gru_monitor.pt \
      --out ../results/efficiency
"""
from __future__ import annotations
import argparse, copy, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from train_monitor import GRUMonitor
from metrics_detect import evaluate

SPLITS = ["val", "test_town", "test_attack", "test_both"]


def load_split(fdir, split, mean, std):
    z = np.load(os.path.join(fdir, f"{split}.npz"))
    meta = pd.read_parquet(os.path.join(fdir, f"{split}_index.parquet")).reset_index(drop=True)
    X, y, g = z["X"].astype(np.float32), z["y"], z["groups"]
    seqs = []
    for gi in range(len(meta)):
        m = g == gi
        seqs.append((((X[m] - mean) / std).astype(np.float32), y[m]))
    return seqs, meta


@torch.no_grad()
def score_all(model, feats, mean, std, bs=24):
    """returns dict split -> (scores, y, groups, meta). Batched, length-padded."""
    model.eval()
    out = {}
    for sp in SPLITS:
        seqs, meta = load_split(feats, sp, mean, std)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))  # bucket by len
        S = [None] * len(seqs)
        for b0 in range(0, len(order), bs):
            idx = order[b0:b0 + bs]
            L = max(len(seqs[i][0]) for i in idx)
            xb = torch.zeros(len(idx), L, seqs[idx[0]][0].shape[1])
            for k, i in enumerate(idx):
                xb[k, :len(seqs[i][0])] = torch.from_numpy(seqs[i][0])
            logit, _ = model(xb)
            prob = torch.sigmoid(logit).numpy()
            for k, i in enumerate(idx):
                S[i] = prob[k, :len(seqs[i][0])]
        Y = [seqs[i][1] for i in range(len(seqs))]
        G = [np.full(len(seqs[i][1]), i, np.int64) for i in range(len(seqs))]
        out[sp] = (np.concatenate(S), np.concatenate(Y), np.concatenate(G), meta)
    return out


def acc_rows(tag, scored):
    sv, yv, gv, mv = scored["val"]
    rows = []
    for sp in SPLITS:
        s, y, g, meta = scored[sp]
        r = evaluate(tag, y, s, g, meta, yv, sv)
        r["variant"] = tag; r["split"] = sp
        rows.append(r)
    return rows


@torch.no_grad()
def bench_streaming(model, x_seq, warmup=50, iters=400):
    """per-timestep latency with hidden state carried (deployment mode)."""
    model.eval()
    h = None
    frames = [torch.from_numpy(x_seq[i:i + 1])[None] for i in range(len(x_seq))]
    j = 0
    for _ in range(warmup):
        _, h = model(frames[j % len(frames)], h)
        j += 1
    ts = np.empty(iters)
    for k in range(iters):
        t0 = time.perf_counter()
        _, h = model(frames[j % len(frames)], h)
        ts[k] = (time.perf_counter() - t0) * 1e3
        j += 1
    return dict(lat_ms_median=float(np.median(ts)),
                lat_ms_p99=float(np.percentile(ts, 99)),
                lat_ms_mean=float(ts.mean()))


@torch.no_grad()
def bench_throughput(model, x_seq, reps=12):
    model.eval()
    xb = torch.from_numpy(x_seq)[None]
    for _ in range(3):
        model(xb)
    t0 = time.perf_counter()
    for _ in range(reps):
        model(xb)
    dt = time.perf_counter() - t0
    return float(len(x_seq) * reps / dt)   # timesteps / s


def disk_kb(sd):
    p = os.path.join(os.path.dirname(__file__), "_tmp_sd.pt")
    torch.save(sd, p)
    kb = os.path.getsize(p) / 1024.0
    os.remove(p)
    return kb


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def fake_quant_int4_(model):
    """symmetric per-tensor 4-bit weight-only quantization, in place (accuracy proxy).
    No INT4 CPU kernel exists in stock PyTorch, so latency is unchanged -- this row
    isolates the accuracy cost of 4-bit weights."""
    qmax = 7.0
    for _, p in model.named_parameters():
        if p.dim() < 2:
            continue
        w = p.data
        scale = w.abs().max() / qmax
        if scale > 0:
            p.data = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../features")
    ap.add_argument("--ckpt", default="../results/monitor_v11/gru_monitor.pt")
    ap.add_argument("--out", default="../results/efficiency")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    P = lambda *a: print(*a, flush=True)

    st = json.load(open(os.path.join(args.features, "norm_stats.json")))
    mean = np.array(st["mean"], np.float32); std = np.array(st["std"], np.float32)
    names = json.load(open(os.path.join(args.features, "feature_names.json")))
    f_in = len(names)

    base = GRUMonitor(f_in, 96)
    base.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    base.eval()

    # a representative attack sequence for latency (longest test_town attack)
    seqs, meta = load_split(args.features, "test_town", mean, std)
    rep = max((s for s in seqs), key=lambda t: len(t[0]))[0][:3000]  # 5 min, representative
    P(f"rep seq len {len(rep)} steps ({len(rep)/10:.0f} s), threads={args.threads}, "
      f"torch {torch.__version__}, cpus {os.cpu_count()}")

    variants = {}

    # ---- FP32 baseline ----
    variants["fp32"] = base

    # ---- FP16 (weights cast to half; CPU has no fast fp16 GEMM, expect slower) ----
    m16 = copy.deepcopy(base).half()
    # keep a float wrapper for scoring: cast input inside
    class _Half(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, h=None):
            o, h = self.m(x.half(), h)
            return o.float(), h
    variants["fp16"] = _Half(m16)

    # ---- dynamic INT8 (GRU + Linear weights int8, activations dynamic) ----
    q8 = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(base), {nn.GRU, nn.Linear}, dtype=torch.qint8)
    q8.eval()
    variants["int8_dynamic"] = q8

    # ---- simulated INT4 weight-only (accuracy proxy; latency == fp32) ----
    m4 = copy.deepcopy(base)
    fake_quant_int4_(m4)
    m4.eval()
    variants["int4_wo_sim"] = m4

    rows_acc, rows_perf = [], []
    ref_acc = None
    for tag, model in variants.items():
        P(f"\n=== {tag} ===  (scoring, all threads)")
        torch.set_num_threads(os.cpu_count() or 4)
        t0 = time.perf_counter()
        scored = score_all(model, args.features, mean, std)
        P(f"  scored 4 splits in {time.perf_counter()-t0:.0f}s")
        ra = acc_rows(tag, scored)
        rows_acc += ra
        if tag == "fp32":
            ref_acc = {r["split"]: r for r in ra}

        # size / params
        if tag == "int8_dynamic":
            size = disk_kb(model.state_dict())
            npar = "-"
        elif tag == "fp16":
            size = disk_kb(model.m.state_dict())
            npar = n_params(model.m)
        else:
            size = disk_kb(model.state_dict())
            npar = n_params(model)

        # latency/throughput pinned to `--threads` (deployment-realistic single core)
        torch.set_num_threads(args.threads)
        P(f"  benchmarking latency @ {args.threads} thread(s)")
        lat = bench_streaming(model, rep)
        thr = bench_throughput(model, rep)
        perf = dict(variant=tag, size_kb=round(size, 1), params=npar,
                    **{k: round(v, 4) for k, v in lat.items()},
                    throughput_steps_s=round(thr, 0),
                    realtime_margin_x=round(100.0 / lat["lat_ms_p99"], 1))
        rows_perf.append(perf)
        P(perf)
        for r in ra:
            d = ref_acc[r["split"]]
            P(f"  {r['split']:11s} AUC {r['roc_auc']:.3f} ({r['roc_auc']-d['roc_auc']:+.3f})  "
              f"PR {r['pr_auc']:.3f} ({r['pr_auc']-d['pr_auc']:+.3f})  "
              f"TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f} ({r['tpr_at_1pct_fpr']-d['tpr_at_1pct_fpr']:+.3f})")

    dfa = pd.DataFrame(rows_acc)
    dfp = pd.DataFrame(rows_perf)
    dfa.to_csv(os.path.join(args.out, "efficiency_accuracy.csv"), index=False)
    dfp.to_csv(os.path.join(args.out, "efficiency_perf.csv"), index=False)

    fp32_size = dfp.loc[dfp.variant == "fp32", "size_kb"].iloc[0]
    with open(os.path.join(args.out, "SUMMARY.md"), "w") as f:
        f.write("# Efficiency / deployability — GRU spoofing monitor\n\n")
        f.write(f"Causal GRU (2x96), {f_in} features @ 10 Hz. CPU, {args.threads} thread(s), "
                f"torch {torch.__version__}. Streaming latency = one 10 Hz GNSS epoch in, "
                f"one spoof probability out, hidden state carried. Real-time budget 100 ms/epoch.\n\n")
        f.write("## Cost / speed\n\n")
        pt = dfp.copy()
        pt["size_kb"] = pt["size_kb"].map(lambda v: f"{v:.0f}") + pt["variant"].map(
            lambda v: f" ({fp32_size/dict(zip(dfp.variant,dfp.size_kb))[v]:.1f}x smaller)"
            if dict(zip(dfp.variant, dfp.size_kb))[v] < fp32_size else "")
        f.write(pt[["variant", "size_kb", "lat_ms_median", "lat_ms_p99",
                    "throughput_steps_s", "realtime_margin_x"]].to_markdown(index=False))
        f.write("\n\n`realtime_margin_x` = 100 ms / p99 streaming latency.\n\n")
        f.write("## Accuracy vs FP32\n\n")
        piv = dfa.pivot(index="split", columns="variant", values="roc_auc").round(3)
        piv = piv.reindex(SPLITS)
        f.write("ROC-AUC:\n\n" + piv.to_markdown() + "\n\n")
        pivp = dfa.pivot(index="split", columns="variant", values="pr_auc").round(3).reindex(SPLITS)
        f.write("PR-AUC:\n\n" + pivp.to_markdown() + "\n\n")
        pivt = dfa.pivot(index="split", columns="variant", values="tpr_at_1pct_fpr").round(3).reindex(SPLITS)
        f.write("TPR @ 1% FPR:\n\n" + pivt.to_markdown() + "\n\n")
        f.write("## Notes\n\n"
                "- **int8_dynamic**: PyTorch dynamic quantization on GRU + Linear weights; "
                "the deployable compressed model.\n"
                "- **fp16**: weights cast to half. Stock x86 CPUs have no fast FP16 GEMM, so "
                "latency does not improve on CPU; listed for completeness (it is the useful "
                "path on a GPU/embedded-NPU target).\n"
                "- **int4_wo_sim**: simulated symmetric 4-bit weight-only quantization "
                "(accuracy proxy only — no INT4 CPU kernel in stock PyTorch, so latency is "
                "unchanged). Isolates the accuracy cost of 4-bit weights.\n")
    P("\n" + open(os.path.join(args.out, "SUMMARY.md")).read())


if __name__ == "__main__":
    main()
