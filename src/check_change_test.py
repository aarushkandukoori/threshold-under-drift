"""Where the score-only change test fires on every synthetic stream, seed (0 to 29) and detector.

Writes results/change_flags.csv (one row per flag) and results/summary_change_flags.csv
(per stream and detector: flags per run, and how far each estimated start is from the nearest
true change point, for streams that have clean change points).
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feedback import flag_changes
from run_confirmations import STREAMS, stream_scores

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = list(range(30))
TRUE = {"no shift": [], "sudden": [2000], "sustained anomaly": [4000, 5000],
        "two anomalies": [4000, 5000, 6000, 7000], "drift over 500": None, "drift over 2000": None}


def job(args):
    name, seed, det = args
    _, _, cal, s, _ = stream_scores(name, seed, det)
    _, flags = flag_changes(cal, s)
    rows = [dict(stream=name, seed=seed, detector=det, flagged_at=f, start=st) for f, st in flags]
    return rows or [dict(stream=name, seed=seed, detector=det, flagged_at=np.nan, start=np.nan)]


def main():
    jobs = [(n, sd, det) for det in ("iforest", "knn") for n in STREAMS for sd in SEEDS]
    with Pool(int(os.environ.get("WORKERS", "3"))) as pool:
        rows = [r for rs in pool.imap(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "change_flags.csv"), index=False)
    out = []
    for (name, det), g in d.groupby(["stream", "detector"], sort=False):
        f = g.dropna(subset=["flagged_at"])
        per_run = f.groupby("seed").size().reindex(SEEDS, fill_value=0)
        r = dict(stream=name, detector=det, runs=len(SEEDS), flags_min=int(per_run.min()),
                 flags_max=int(per_run.max()), runs_without_flag=int((per_run == 0).sum()))
        cps = TRUE[name]
        if cps:
            err = [min(abs(st - c) for c in cps) for st in f.start]
            lag = [fa - min(cps, key=lambda c: abs(st - c)) for fa, st in zip(f.flagged_at, f.start)]
            r.update(runs_with_one_flag_per_change=int((per_run == len(cps)).sum()),
                     start_error_max=int(max(err)), start_error_median=float(np.median(err)),
                     flag_lag_max=int(max(lag)), flag_lag_median=float(np.median(lag)))
        out.append(r)
    pd.DataFrame(out).to_csv(os.path.join(RES, "summary_change_flags.csv"), index=False)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
