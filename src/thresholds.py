"""Threshold rules applied to the score stream of one frozen detector.

Every rule decides on point t using only the calibration scores and the stream scores
before t (past-only). None of them use labels, except `rolling_confirmed_normal`, which
is told the label of point i only once D more points have arrived (delayed labels).

Common input:
  cal - scores of clean calibration rows (seen before the stream starts)
  s   - scores of the stream, in time order
Output: boolean alarms, one per stream point.
"""
import bisect
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

BUDGET = 0.025           # alarm budget: 2.5% of normal points
Q = 1.0 - BUDGET


def budget_k(cal):
    """How many standard deviations above the mean the 97.5% point of `cal` sits.
    Used to put the SCS-style and MACS-style bounds on the same alarm budget."""
    return (np.quantile(cal, Q) - cal.mean()) / cal.std()


# ------------------------------------------------------------------ baselines
def fixed(cal, s):
    """Threshold set once on calibration scores and never changed."""
    return s > np.quantile(cal, Q)


def rolling_quantile(cal, s, window):
    """Threshold = 97.5% quantile of the last `window` scores (calibration scores fill
    the history at the start)."""
    hist = np.concatenate([cal, s])
    n, T = len(cal), len(s)
    # window for stream point t covers hist[n + t - window : n + t]
    # (when the calibration set is shorter than the window, early points use all history)
    thr = np.empty(T)
    t_full = max(0, window - n)              # first t with a full window
    for t in range(min(t_full, T)):
        thr[t] = np.quantile(hist[: n + t], Q)
    if t_full < T:
        wins = sliding_window_view(hist[n + t_full - window: n + T - 1], window)
        thr[t_full:] = np.quantile(wins, Q, axis=1)
    return s > thr


def rolling_confirmed_normal(cal, s, y, window, delay):
    """Uses delayed labels. The label of point i arrives `delay` points later. The
    threshold is the 97.5% quantile of the last `window` scores whose label has come back
    as normal (calibration rows count as normal). Recent, still-unlabeled points are not
    used at all, so a sustained anomaly cannot leak into the baseline, but a real new
    normal is only picked up `delay` points late."""
    n, T = len(cal), len(s)
    hist = np.concatenate([cal, s])
    is_normal = np.concatenate([np.ones(n, bool), y == 0])
    alarms = np.zeros(T, bool)
    # positions in hist of confirmed-normal points, in time order
    normal_pos = np.flatnonzero(is_normal)
    for t in range(T):
        known_end = max(n, n + t - delay)    # hist positions < known_end have labels
        hi = np.searchsorted(normal_pos, known_end)
        pos = normal_pos[max(0, hi - window): hi]
        alarms[t] = s[t] > np.quantile(hist[pos], Q)
    return alarms


# ------------------------------------------------------------------ SCS-style
def scs_style(cal, s, k=None, recent=100, h=0.75, pct=0.90):
    """Online, past-only version of Segmented Confidence Sequences (Li & Gautam, 2025).

    - Regimes: the score stream is split into segments online. A new segment starts when
      the mean of the last `recent` scores sits more than h segment-SDs from the current
      segment's mean. If that recent mean matches an earlier segment, that segment is
      reopened (so a regime that comes back reuses its old bounds).
    - Bound per segment: mean + k * SD of the segment's scores (upper side only, since a
      high score means anomalous).
    - Composite rule: alarm only if the score is above the segment bound AND above the
      `pct` percentile of all past scores (the paper's percentile filter).
    The paper segments with APCA or k-means over the series; this version has to find
    segments from the past only, so it uses the mean-shift test above instead.
    """
    if k is None:
        k = budget_k(cal)
    T = len(s)
    segs = [[float(cal.sum()), float((cal ** 2).sum()), len(cal)]]  # sum, sumsq, count
    cur = 0
    seg_start = 0                       # stream index where the current segment began
    sorted_past = sorted(cal.tolist())  # for the percentile filter
    alarms = np.zeros(T, bool)
    cs = np.concatenate([[0.0], np.cumsum(s)])

    def stats(g):
        S, SS, c = g
        mu = S / c
        return mu, np.sqrt(max(SS / c - mu * mu, 1e-12))

    for t in range(T):
        mu, sd = stats(segs[cur])
        filt = sorted_past[min(int(pct * len(sorted_past)), len(sorted_past) - 1)]
        alarms[t] = (s[t] > mu + k * sd) and (s[t] > filt)
        # update with the new score (unlabeled: anomalies go in too)
        g = segs[cur]; g[0] += s[t]; g[1] += s[t] ** 2; g[2] += 1
        bisect.insort(sorted_past, float(s[t]))
        # regime check on the last `recent` scores
        if t + 1 - seg_start >= recent:
            r_mean = (cs[t + 1] - cs[t + 1 - recent]) / recent
            mu, sd = stats(segs[cur])
            if abs(r_mean - mu) > h * sd:
                recent_vals = s[t + 1 - recent: t + 1]
                # move the recent points out of the old segment
                g[0] -= recent_vals.sum(); g[1] -= (recent_vals ** 2).sum(); g[2] -= recent
                match = None
                for j, gj in enumerate(segs):
                    if j == cur or gj[2] < recent:
                        continue
                    mj, sj = stats(gj)
                    if abs(r_mean - mj) <= h * sj:
                        match = j
                        break
                if match is None:
                    segs.append([float(recent_vals.sum()), float((recent_vals ** 2).sum()), recent])
                    cur = len(segs) - 1
                else:
                    gm = segs[match]
                    gm[0] += recent_vals.sum(); gm[1] += (recent_vals ** 2).sum(); gm[2] += recent
                    cur = match
                seg_start = t + 1 - recent
    return alarms


# ------------------------------------------------------------------ MACS-style
def macs_style(cal, s, k=None, windows=(50, 100, 500), pct=0.90):
    """Online, past-only version of Multi-Scale Adaptive Confidence Segments.

    - Three rolling windows (50, 100, 500 points, as in the paper) of past scores, each
      with an upper bound mean + k * SD.
    - Attention weights from the variance of the last 50 scores (scores standardized by
      the calibration SD, so the paper's 0.7 / 0.3 cut points have a fixed meaning):
      high > 0.7 -> [0.6, 0.3, 0.1], medium > 0.3 -> [0.2, 0.6, 0.2], else [0.1, 0.3, 0.6].
    - Regime change: mean of the last 500 scores vs the 500 before that; change if the
      mean moved more than 2.0 historical SDs or the SD changed by more than 1.5x.
    - Decision: normally, score > weighted bound. During a regime change, also need at
      least 2 of the 3 scale bounds violated. Then the percentile filter (score above the
      `pct` percentile of all past scores).
    """
    if k is None:
        k = budget_k(cal)
    n, T = len(cal), len(s)
    hist = np.concatenate([cal, s])
    z = (hist - cal.mean()) / cal.std()
    cs = np.concatenate([[0.0], np.cumsum(hist)])
    cs2 = np.concatenate([[0.0], np.cumsum(hist ** 2)])
    csz = np.concatenate([[0.0], np.cumsum(z)])
    csz2 = np.concatenate([[0.0], np.cumsum(z ** 2)])

    def mstd(c1, c2, lo, hi):
        m = (c1[hi] - c1[lo]) / (hi - lo)
        v = (c2[hi] - c2[lo]) / (hi - lo) - m * m
        return m, np.sqrt(np.maximum(v, 1e-12))

    sorted_past = sorted(cal.tolist())
    L = windows[-1]
    alarms = np.zeros(T, bool)
    for t in range(T):
        end = n + t                     # history index of point t; use hist[:end]
        bounds = []
        for w in windows:
            lo = max(0, end - w)
            m, sd = mstd(cs, cs2, lo, end)
            bounds.append(m + k * sd)
        bounds = np.array(bounds)
        _, zsd = mstd(csz, csz2, max(0, end - windows[0]), end)
        var = zsd ** 2
        wts = np.array([0.6, 0.3, 0.1]) if var > 0.7 else (
              np.array([0.2, 0.6, 0.2]) if var > 0.3 else np.array([0.1, 0.3, 0.6]))
        combined = float(wts @ bounds)
        regime = False
        if end - 2 * L >= 0:
            m_cur, sd_cur = mstd(cs, cs2, end - L, end)
            m_his, sd_his = mstd(cs, cs2, end - 2 * L, end - L)
            regime = (abs(m_cur - m_his) / (sd_his + 1e-8) > 2.0 or
                      abs(sd_cur - sd_his) / (sd_his + 1e-8) > 1.5)
        attn = s[t] > combined
        if regime:
            flag = attn and (np.sum(s[t] > bounds) >= 2)
        else:
            flag = attn
        filt = sorted_past[min(int(pct * len(sorted_past)), len(sorted_past) - 1)]
        alarms[t] = flag and s[t] > filt
        bisect.insort(sorted_past, float(s[t]))
    return alarms


def policies(cal, s, y):
    """All rules compared in the note, by name."""
    out = {
        "fixed": fixed(cal, s),
        "rolling-200": rolling_quantile(cal, s, 200),
        "rolling-500": rolling_quantile(cal, s, 500),
        "rolling-1000": rolling_quantile(cal, s, 1000),
        "SCS-style": scs_style(cal, s),
        "MACS-style": macs_style(cal, s),
        "SCS-style (paper k=1.8)": scs_style(cal, s, k=1.8),
        "MACS-style (paper k=1.8)": macs_style(cal, s, k=1.8),
        "confirmed-normal-500, labels 50 late": rolling_confirmed_normal(cal, s, y, 500, 50),
        "confirmed-normal-500, labels 200 late": rolling_confirmed_normal(cal, s, y, 500, 200),
    }
    return out
