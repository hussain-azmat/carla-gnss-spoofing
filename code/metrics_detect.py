"""
Shared detection-metric functions (docs/metrics.md). Used by baselines_detect.py
and the learned monitor. Every method is scored the same way on the same splits.

A "method" produces a per-timestep score in [0, inf) (higher = more likely spoofed).
Thresholds are frozen on val and never re-tuned per test split.
"""
from __future__ import annotations
import numpy as np


def roc_auc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    n_pos, n_neg = float(y.sum()), float((~y).sum())   # float: int32 overflow on Windows numpy<2
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - (cnt - 1) / 2.0)
    ranks = avg[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / y.sum()
    rec = np.r_[0, rec]
    prec = np.r_[1, prec]
    return float(np.trapz(prec, rec))


def threshold_at_fpr(y_val, s_val, fpr_target):
    """smallest threshold on val giving FPR <= target (on negatives)."""
    s_neg = np.sort(s_val[~np.asarray(y_val).astype(bool)])
    if len(s_neg) == 0:
        return float("inf")
    k = int(np.ceil((1 - fpr_target) * len(s_neg))) - 1
    k = min(max(k, 0), len(s_neg) - 1)
    return float(s_neg[k])


def tpr_at_threshold(y, s, thr):
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    return float((np.asarray(s, float)[y] >= thr).mean())


def fpr_at_threshold(y, s, thr):
    y = np.asarray(y).astype(bool)
    if (~y).sum() == 0:
        return float("nan")
    return float((np.asarray(s, float)[~y] >= thr).mean())


def time_to_alarm(groups, y, s, thr, onset_s, dt=0.1, hold_s=1.0):
    """
    per attack sequence: seconds from the first attacked timestep to the first
    detection sustained >= hold_s. Undetected -> inf (miss).
    Returns (median_TTA_over_detected, miss_rate, n_attack_seqs).
    """
    groups = np.asarray(groups)
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    hold = max(1, int(round(hold_s / dt)))
    ttas, misses, n = [], 0, 0
    for g in np.unique(groups):
        m = groups == g
        yg, sg = y[m], s[m]
        if not yg.any():
            continue
        n += 1
        onset_i = int(np.argmax(yg))
        det = sg >= thr
        # first index i>=onset with `hold` consecutive detections
        found = -1
        run = 0
        for i in range(onset_i, len(det)):
            run = run + 1 if det[i] else 0
            if run >= hold:
                found = i - hold + 1
                break
        if found < 0:
            misses += 1
        else:
            ttas.append((found - onset_i) * dt)
    med = float(np.median(ttas)) if ttas else float("inf")
    p90 = float(np.percentile(ttas, 90)) if ttas else float("inf")
    return dict(median_tta_s=med, p90_tta_s=p90,
                miss_rate=misses / n if n else float("nan"), n_attack_seqs=n)


def hard_negative_far(groups, y, s, thr, meta, dt=0.1):
    """false alarms per hour on hard-negative sequences (clean canyon/tunnel)."""
    groups = np.asarray(groups)
    s = np.asarray(s, float)
    hn = set(meta.index[meta["hard_negative"] == 1])
    gid_to_row = {i: i for i in range(len(meta))}
    alarms = 0.0
    hours = 0.0
    for g in np.unique(groups):
        if g not in hn:
            continue
        m = groups == g
        det = s[m] >= thr
        # count rising edges as distinct alarms
        alarms += int(np.sum(det[1:] & ~det[:-1]) + (det[0] if len(det) else 0))
        hours += len(det) * dt / 3600.0
    return float(alarms / hours) if hours > 0 else float("nan")


def evaluate(name, y, s, groups, meta, y_val, s_val, dt=0.1):
    """full metric row for one method on one split. Thresholds come from val."""
    thr1 = threshold_at_fpr(y_val, s_val, 0.01)
    thr01 = threshold_at_fpr(y_val, s_val, 0.001)
    tta = time_to_alarm(groups, y, s, thr1, None, dt=dt)
    return dict(
        method=name,
        roc_auc=roc_auc(y, s),
        pr_auc=pr_auc(y, s),
        tpr_at_1pct_fpr=tpr_at_threshold(y, s, thr1),
        tpr_at_0p1pct_fpr=tpr_at_threshold(y, s, thr01),
        realized_fpr_1pct=fpr_at_threshold(y, s, thr1),
        median_tta_s=tta["median_tta_s"],
        p90_tta_s=tta["p90_tta_s"],
        miss_rate=tta["miss_rate"],
        hardneg_far_per_hr=hard_negative_far(groups, y, s, thr1, meta, dt=dt),
    )
