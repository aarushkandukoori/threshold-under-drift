"""Variable label delays and partial feedback, on the same frozen detectors.

Runs every stream, seed and feedback setting. Writes to results/:
  feedback_runs.csv       synthetic streams, one row per stream x seed x detector x rule x setting
  feedback_gas_runs.csv   UCI gas stream at the reference setting
  feedback_trace_*.csv    threshold over time for seed 0 (Isolation Forest, reference setting)
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from streams import synthetic, gas, T, T0
from detectors import DETECTORS
from thresholds import fixed, rolling_quantile, rolling_threshold, scs_style, macs_style, Q
from feedback import feedback_policies, NAMES
from metrics import false_alarm_rate, recall, time_to_recover

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = list(range(10))
GAS_SEEDS = list(range(5))
STREAMS = {                      # name -> (kind, ramp length)
    "no shift": ("control", 1500),
    "sudden": ("sudden", 1500),
    "sustained anomaly": ("sustained", 1500),
    "drift over 500": ("gradual", 500),
    "drift over 2000": ("gradual", 2000),
}
MEDIANS = [10, 25, 50, 100, 200, 400, 800]   # median label delay, in points
SHAPES = ["steady", "heavy"]
CAPACITIES = [1, 2, 5, 10, 25, None]         # checks per 100 points; None = every alert
REF = dict(median=100, shape="heavy", capacity=5)
BLOCK = slice(4000, 5000)


def settings():
    """(median, shape, capacity) for every feedback run: the delay sweep at 5 checks per
    100 points, plus the capacity sweep at the reference delay."""
    out = [(m, sh, 5) for sh in SHAPES for m in MEDIANS]
    out += [(REF["median"], REF["shape"], c) for c in CAPACITIES if c != 5]
    return out


def score_only(cal, s):
    return {
        "fixed": fixed(cal, s),
        "rolling-200": rolling_quantile(cal, s, 200),
        "rolling-500": rolling_quantile(cal, s, 500),
        "rolling-1000": rolling_quantile(cal, s, 1000),
        "SCS-style": scs_style(cal, s),
        "MACS-style": macs_style(cal, s),
    }


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
                    missed=1 - recall(a, y, BLOCK),
                    fa_after_block=false_alarm_rate(a, y, slice(5000, 5500)),
                    recall_after_block=recall(a, y, slice(5000, 6000)))
    if kind == "gradual":
        drift = slice(T0, T0 + ramp + 500)          # the drift plus the 500 points after it
        return dict(fa_drift=false_alarm_rate(a, y, drift),
                    recall_drift=recall(a, y, drift),
                    fa_after=false_alarm_rate(a, y, slice(T0 + ramp + 500, T)))
    raise ValueError(kind)


def synthetic_job(job):
    name, seed, det_name, full = job
    kind, ramp = STREAMS[name]
    st = synthetic(kind, seed, ramp=ramp)
    det = DETECTORS[det_name](seed).fit(st["fit"])
    cal, s, y = det.score(st["cal"]), det.score(st["X"]), st["y"]
    base = dict(stream=name, seed=seed, detector=det_name)
    rows = []
    for pol, a in score_only(cal, s).items():
        rows.append(dict(base, policy=pol, labels="none", **grade(kind, ramp, a, y)))
    trace = None
    todo = settings() if full else [(REF["median"], REF["shape"], REF["capacity"])]
    for (m, sh, c) in todo:
        rules = NAMES if c == 5 else NAMES[1:]       # every-label rule does not use capacity
        out = feedback_policies(cal, s, y, seed, m, sh, c, rules=rules)
        for pol, o in out.items():
            every = pol == "every label returns"
            r = dict(base, policy=pol, labels="every point" if every else "checked alerts",
                     median=m, shape=sh, capacity=np.nan if every else (c if c else 100),
                     n_checked=np.nan if every else int(o["checked"].sum()),
                     n_raised=o.get("raised", np.nan),
                     **grade(kind, ramp, o["alarms"], y))
            rows.append(r)
        if (seed == 0 and det_name == "iforest" and (m, sh, c) == tuple(REF.values())
                and kind in ("sudden", "sustained")):
            trace = {"t": np.arange(T), "score": s, "y": y,
                     "fixed": np.full(T, np.quantile(cal, Q)),
                     "rolling-500": rolling_threshold(cal, s, 500)}
            for pol, o in out.items():
                trace[pol] = o["thr"]
    return rows, (name, trace)


def gas_job(job):
    seed, det_name = job
    st = gas(seed)
    det = DETECTORS[det_name](seed).fit(st["fit"])
    cal, s, y = det.score(st["cal"]), det.score(st["X"]), st["y"]
    pols = {k: v for k, v in score_only(cal, s).items()}
    out = feedback_policies(cal, s, y, seed, REF["median"], REF["shape"], REF["capacity"])
    pols.update({k: v["alarms"] for k, v in out.items()})
    rows = []
    for pol, a in pols.items():
        rows.append(dict(stream="gas", seed=seed, detector=det_name, policy=pol,
                         fa=false_alarm_rate(a, y), recall=recall(a, y)))
    return rows


def main():
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    jobs = [(name, seed, "iforest", True) for name in STREAMS for seed in SEEDS]
    jobs += [(name, seed, "knn", False) for name in STREAMS for seed in SEEDS]
    rows = []
    with Pool(int(os.environ.get("WORKERS", "4"))) as pool:
        for k, (r, (name, tr)) in enumerate(pool.imap(synthetic_job, jobs)):
            rows += r
            if tr is not None:
                key = "sudden" if name == "sudden" else "sustained"
                pd.DataFrame(tr).to_csv(os.path.join(RES, f"feedback_trace_{key}_seed0.csv"),
                                        index=False, float_format="%.6g")
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{len(jobs)} synthetic jobs ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RES, "feedback_runs.csv"), index=False, float_format="%.6g")
        grows = []
        for r in pool.imap(gas_job, [(seed, d) for seed in GAS_SEEDS for d in ("iforest", "knn")]):
            grows += r
    pd.DataFrame(grows).to_csv(os.path.join(RES, "feedback_gas_runs.csv"), index=False,
                               float_format="%.6g")
    print(f"feedback runs done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
