"""Second follow-up: explicit false-positive confirmations, and stale feedback.

Part A - where the false-alarm rule's raises come from, and fixes. The rule of 24 November
(estimate over the 500 points before each confirmed false positive) and three alternatives:
drop confirmations older than 100, 200 or 400 points, or count only alarms and labels from
the latest flagged change on (the new default). Every delay and capacity setting of the
first follow-up, 10 seeds, plus a stream with two identical anomalies. Every raise of the
cap is logged with the age of the confirmation behind it, the part of the stream its alert
came from, and how much of the evidence came from an earlier part of the stream.

Part B - explicit confirmations only. A checked false positive is explicitly confirmed with
probability f (swept), a checked real incident is recorded as an incident with probability
g (1 or 0.5); every other ticket just closes. Rules either use explicit outcomes only (an
ambiguous closure is unknown) or read every closure without an incident record as a false
positive. Reference delay and capacity, 10 seeds.

Writes to results/:
  fix_runs.csv        Part A, one row per stream x seed x detector x rule x setting
  raise_log.csv       Part A, one row per raise of the cap
  explicit_runs.csv   Part B, one row per stream x seed x detector x rule x (f, g)
  confirm_trace_two_anomalies_seed5.csv   thresholds over time for one seed (figure 11)
With --fresh: the same at the reference setting on seeds 10 to 29, to fix_runs_fresh.csv,
raise_log_fresh.csv and explicit_runs_fresh.csv (none of these seeds was looked at before).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from streams import synthetic, T, T0
from detectors import DETECTORS
from thresholds import fixed, rolling_quantile, rolling_threshold, macs_style, Q, BUDGET
from feedback import (draw_delays, draw_outcomes, labels_seen, flag_changes,
                      every_label_returns, silence_as_normal, raise_on_false_alarms,
                      rollback_on_incidents, FACTOR, MIN_LABELS)
from metrics import false_alarm_rate, recall, time_to_recover
from run_feedback import settings, REF

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = list(range(10))
FRESH = list(range(10, 30))
STREAMS = {                      # name -> (kind, ramp length)
    "no shift": ("control", 1500),
    "sudden": ("sudden", 1500),
    "sustained anomaly": ("sustained", 1500),
    "two anomalies": ("repeat", 1500),
    "drift over 500": ("gradual", 500),
    "drift over 2000": ("gradual", 2000),
}
PARTS = {                        # stream -> (first point, name) of each part, in order
    "no shift": [(0, "normal")],
    "sudden": [(0, "before shift"), (T0, "after shift")],
    "sustained anomaly": [(0, "before anomaly"), (4000, "anomaly"), (5000, "after anomaly")],
    "two anomalies": [(0, "before anomaly"), (4000, "anomaly 1"), (5000, "between"),
                      (6000, "anomaly 2"), (7000, "after")],
    "drift over 500": [(0, "before drift"), (T0, "drift"), (T0 + 500, "after drift")],
    "drift over 2000": [(0, "before drift"), (T0, "drift"), (T0 + 2000, "after drift")],
}
FP_RULE = "raise on confirmed false alarms"
VARIANTS = {                     # Part A versions of the false-alarm rule
    FP_RULE: dict(changes=None),
    FP_RULE + ", expire after 100": dict(changes=None, max_age=100),
    FP_RULE + ", expire after 200": dict(changes=None, max_age=200),
    FP_RULE + ", expire after 400": dict(changes=None, max_age=400),
    FP_RULE + ", since last change": dict(changes="flag"),
}
FS = [0.1, 0.25, 0.5, 0.75, 1.0]  # Part B: share of checked false positives explicitly confirmed
GS = [1.0, 0.5]                   # Part B: share of checked incidents recorded as incidents


def part_of(stream, i):
    """Index and name of the part of the stream that point i belongs to."""
    k = 0
    for j, (start, _) in enumerate(PARTS[stream]):
        if i >= start:
            k = j
    return k, PARTS[stream][k][1]


def grade(kind, ramp, a, y):
    if kind == "control":
        return dict(fa=false_alarm_rate(a, y), recall=recall(a, y))
    if kind == "sudden":
        return dict(fa_pre=false_alarm_rate(a, y, slice(0, T0)),
                    fa_first500=false_alarm_rate(a, y, slice(T0, T0 + 500)),
                    recover=time_to_recover(a, y, T0),
                    fa_post=false_alarm_rate(a, y, slice(T0, T)),
                    recall_post=recall(a, y, slice(T0, T)))
    if kind == "sustained":
        return dict(fa_pre=false_alarm_rate(a, y, slice(0, 4000)),
                    missed=1 - recall(a, y, slice(4000, 5000)),
                    fa_after_block=false_alarm_rate(a, y, slice(5000, 5500)),
                    recall_after_block=recall(a, y, slice(5000, 6000)))
    if kind == "repeat":
        return dict(fa_pre=false_alarm_rate(a, y, slice(0, 4000)),
                    missed=1 - recall(a, y, slice(4000, 5000)),
                    missed2=1 - recall(a, y, slice(6000, 7000)),
                    fa_between=false_alarm_rate(a, y, slice(5000, 6000)),
                    recall_between=recall(a, y, slice(5000, 6000)))
    if kind == "gradual":
        drift = slice(T0, T0 + ramp + 500)
        return dict(fa_drift=false_alarm_rate(a, y, drift),
                    recall_drift=recall(a, y, drift),
                    fa_after=false_alarm_rate(a, y, slice(T0 + ramp + 500, T)))
    raise ValueError(kind)


def describe_raise(ev, stream, alarms, checked, y):
    """Turn one logged raise into a row: where the confirmation and its evidence came from."""
    t, i, lo = ev["t"], ev["i"], ev["lo"]
    k_i, part_i = part_of(stream, i)
    k_t, part_t = part_of(stream, t)
    s0 = max(lo, PARTS[stream][k_i][0])      # first point of the window in the alert's part
    labels, fps = ev["labels"], ev["fp_labels"]
    al = alarms[lo:i + 1]
    al_same = alarms[s0:i + 1]
    n_same = int((labels >= s0).sum())
    fp_same = int((fps >= s0).sum())
    est_same = np.nan
    if n_same >= MIN_LABELS:
        est_same = al_same.sum() * fp_same / n_same / (i + 1 - s0)
    early = slice(lo, s0)
    ch_e = checked[early][alarms[early]]
    ch_s = checked[s0:i + 1][al_same]
    normal = y[lo:i + 1] == 0
    normal_same = y[s0:i + 1] == 0
    p_lo = max(0, i - 500)                    # proposal at i = quantile of points i-500..i-1
    prop_early = (max(p_lo, min(i, PARTS[stream][k_i][0])) - p_lo) / max(1, i - p_lo)
    return dict(
        t=t, i=i, age=t - i, part_alert=part_i, part_arrival=part_t, stale=int(k_i < k_t),
        window_start=lo, window_points=i + 1 - lo,
        alarms=int(al.sum()), alarms_earlier=int(alarms[early].sum()),
        checked_share_earlier=float(ch_e.mean()) if len(ch_e) else np.nan,
        checked_share_same=float(ch_s.mean()) if len(ch_s) else np.nan,
        labels=len(labels), labels_earlier=int((labels < s0).sum()),
        fp_labels=len(fps), fp_labels_earlier=int((fps < s0).sum()),
        est=ev["est"], est_same_part=est_same,
        raise_same_part=int(n_same >= MIN_LABELS and est_same > FACTOR * BUDGET),
        true_fa_window=float(al[normal].mean()) if normal.any() else np.nan,
        true_fa_same_part=float(al_same[normal_same].mean()) if normal_same.any() else np.nan,
        proposal_window_earlier=prop_early,
        cap_before=ev["cap_before"], cap_after=ev["cap_after"])


def stream_scores(name, seed, det_name):
    kind, ramp = STREAMS[name]
    st = synthetic(kind, seed, ramp=ramp)
    det = DETECTORS[det_name](seed).fit(st["fit"])
    return kind, ramp, det.score(st["cal"]), det.score(st["X"]), st["y"]


def part_a_job(job):
    """Part A for one stream, seed and detector: every setting, every version of the rule."""
    name, seed, det_name, todo = job
    kind, ramp, cal, s, y = stream_scores(name, seed, det_name)
    last, flags = flag_changes(cal, s)
    base = dict(stream=name, seed=seed, detector=det_name)
    rows, raises, trace = [], [], None
    score_only = {"fixed": fixed(cal, s), "rolling-500": rolling_quantile(cal, s, 500),
                  "MACS-style": macs_style(cal, s)}
    for pol, a in score_only.items():
        rows.append(dict(base, policy=pol, labels="none", **grade(kind, ramp, a, y)))
    for (m, sh, c) in todo:
        d = draw_delays(seed, len(s), m, sh)
        cfg = dict(base, median=m, shape=sh, capacity=c if c else 100)
        if (m, sh, c) == tuple(REF.values()):
            others = {"every label returns": every_label_returns(cal, s, y, d, info=True),
                      "no ticket = normal": silence_as_normal(cal, s, y, d, c, wait=m, info=True),
                      "roll back on confirmed incidents": rollback_on_incidents(cal, s, y, d, c,
                                                                                info=True)}
            for pol, o in others.items():
                rows.append(dict(cfg, policy=pol, **grade(kind, ramp, o["alarms"], y)))
        outs = {}
        for pol, kw in VARIANTS.items():
            kw = dict(kw)
            if kw.get("changes") == "flag":
                kw["changes"] = last
            log = []
            o = raise_on_false_alarms(cal, s, y, d, c, info=True, log=log, **kw)
            outs[pol] = o
            n_after = 0
            for ev in log:
                r = describe_raise(ev, name, o["alarms"], o["checked"], y)
                raises.append(dict(cfg, policy=pol, **r))
                if ((name == "sustained anomaly" and ev["i"] >= 5000)
                        or (name == "two anomalies" and 5000 <= ev["i"] < 6000)):
                    n_after += 1                 # raised on a point after the (first) anomaly
            fx = float(np.quantile(cal, Q))
            cap_6000 = max([fx] + [ev["cap_after"] for ev in log if ev["t"] < 6000])
            rows.append(dict(cfg, policy=pol, n_raised=o["raised"], n_raised_after_anomaly=n_after,
                             n_checked=int(o["checked"].sum()), cap_over_fixed_at_6000=cap_6000 - fx,
                             n_flags=len(flags), **grade(kind, ramp, o["alarms"], y)))
        if (name == "two anomalies" and seed == 5 and det_name == "iforest"
                and (m, sh, c) == tuple(REF.values())):
            trace = pd.DataFrame({"t": np.arange(T), "score": s, "y": y,
                                  "fixed": np.full(T, np.quantile(cal, Q)),
                                  "rolling-500": rolling_threshold(cal, s, 500),
                                  "last_change": last,
                                  "thr_1124": outs[FP_RULE]["thr"],
                                  "thr_new": outs[FP_RULE + ", since last change"]["thr"]})
    return rows, raises, trace


def part_b_job(job):
    """Part B for one stream, seed and detector: every (f, g) at the reference setting."""
    name, seed, det_name = job
    kind, ramp, cal, s, y = stream_scores(name, seed, det_name)
    last, _ = flag_changes(cal, s)
    m, sh, c = REF["median"], REF["shape"], REF["capacity"]
    d = draw_delays(seed, len(s), m, sh)
    base = dict(stream=name, seed=seed, detector=det_name, median=m, shape=sh, capacity=c)
    rows = []
    for g in GS:
        for f in FS:
            ex = draw_outcomes(seed, y, f, g)
            seen = labels_seen(y, ex)
            runs = {
                ("no ticket = normal", "no incident record = normal"):
                    lambda: silence_as_normal(cal, s, seen, d, c, wait=m, info=True),
                (FP_RULE, "explicit only"):
                    lambda: raise_on_false_alarms(cal, s, y, d, c, explicit=ex, changes=None,
                                                  info=True),
                (FP_RULE, "closed = false positive"):
                    lambda: raise_on_false_alarms(cal, s, seen, d, c, changes=None, info=True),
                (FP_RULE + ", since last change", "explicit only"):
                    lambda: raise_on_false_alarms(cal, s, y, d, c, explicit=ex, changes=last,
                                                  info=True),
                (FP_RULE + ", since last change", "closed = false positive"):
                    lambda: raise_on_false_alarms(cal, s, seen, d, c, changes=last, info=True),
                ("roll back on confirmed incidents", "explicit only"):
                    lambda: rollback_on_incidents(cal, s, y, d, c, explicit=ex, info=True),
            }
            for (pol, reading), run in runs.items():
                o = run()
                ck = o["checked"]
                rows.append(dict(base, policy=pol, reading=reading, explicit_fp=f,
                                 explicit_inc=g, n_raised=o.get("raised", np.nan),
                                 n_checked=int(ck.sum()),
                                 n_explicit=int((ck & ex).sum()),
                                 n_explicit_fp=int((ck & ex & (y == 0)).sum()),
                                 **grade(kind, ramp, o["alarms"], y)))
    return rows


def main(fresh=False):
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    seeds = FRESH if fresh else SEEDS
    ref = [tuple(REF.values())]
    jobs = [(n, sd, "iforest", ref if fresh else settings()) for n in STREAMS for sd in seeds]
    jobs += [(n, sd, "knn", ref) for n in STREAMS for sd in seeds]
    rows, raises = [], []
    suffix = "_fresh" if fresh else ""
    with Pool(int(os.environ.get("WORKERS", "3"))) as pool:
        for k, (r, rz, tr) in enumerate(pool.imap(part_a_job, jobs)):
            rows += r
            raises += rz
            if tr is not None:
                tr.to_csv(os.path.join(RES, "confirm_trace_two_anomalies_seed5.csv"),
                          index=False, float_format="%.6g")
            if (k + 1) % 20 == 0:
                print(f"part A {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"fix_runs{suffix}.csv"), index=False,
                                  float_format="%.6g")
        pd.DataFrame(raises).to_csv(os.path.join(RES, f"raise_log{suffix}.csv"), index=False,
                                    float_format="%.6g")
        bjobs = [(n, sd, det) for det in ("iforest", "knn") for n in STREAMS for sd in seeds]
        brows = []
        for k, r in enumerate(pool.imap(part_b_job, bjobs)):
            brows += r
            if (k + 1) % 20 == 0:
                print(f"part B {k + 1}/{len(bjobs)} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(brows).to_csv(os.path.join(RES, f"explicit_runs{suffix}.csv"), index=False,
                               float_format="%.6g")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)
