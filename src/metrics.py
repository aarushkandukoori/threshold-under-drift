"""Metrics. Labels are used here only to grade the rules, never to set thresholds."""
import numpy as np
from sklearn.metrics import roc_auc_score

from thresholds import BUDGET


def auroc(s, y):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, s)


def false_alarm_rate(alarms, y, sl=slice(None)):
    a, yy = alarms[sl], y[sl]
    return float(a[yy == 0].mean()) if (yy == 0).any() else np.nan


def recall(alarms, y, sl=slice(None)):
    a, yy = alarms[sl], y[sl]
    return float(a[yy == 1].mean()) if (yy == 1).any() else np.nan


def trailing_fa(alarms, y, n_normal=200):
    """False-alarm rate over the last `n_normal` normal points, at every time step."""
    normal_idx = np.flatnonzero(y == 0)
    hits = alarms[normal_idx].astype(float)
    c = np.concatenate([[0.0], np.cumsum(hits)])
    out = np.full(len(y), np.nan)
    pos = np.searchsorted(normal_idx, np.arange(len(y)), side="right")  # normals seen so far
    ok = pos >= n_normal
    out[ok] = (c[pos[ok]] - c[pos[ok] - n_normal]) / n_normal
    return out


def time_to_recover(alarms, y, start, n_normal=200, limit=2 * BUDGET):
    """Points after `start` until the false-alarm rate over the last 200 normal points
    first falls to 2x the budget (5%) or lower, counting only windows that lie fully after
    `start`. Returns NaN if it never happens before the stream ends."""
    tr = trailing_fa(alarms, y, n_normal)
    normal_idx = np.flatnonzero(y == 0)
    after = normal_idx[normal_idx >= start]
    if len(after) < n_normal:
        return np.nan
    first_full = after[n_normal - 1]
    cand = np.flatnonzero((np.arange(len(y)) >= first_full) & (tr <= limit))
    return float(cand[0] - start) if len(cand) else np.nan
