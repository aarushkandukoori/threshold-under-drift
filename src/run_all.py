"""Run every scenario, seed, detector and threshold rule. Writes CSVs to results/."""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from streams import synthetic, gas, T, T0
from detectors import DETECTORS
from thresholds import policies
from metrics import auroc, false_alarm_rate, recall, time_to_recover, trailing_fa

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = list(range(10))
GAS_SEEDS = list(range(5))
SCENARIOS = ["control", "sudden", "gradual", "recurring", "sustained"]


def synthetic_rows(kind, seed, det_name):
    st = synthetic(kind, seed)
    det = DETECTORS[det_name](seed).fit(st["fit"])
    cal, s, y = det.score(st["cal"]), det.score(st["X"]), st["y"]
    rows = []
    chg = {"sudden": T0, "gradual": 3500, "control": T0, "recurring": T0, "sustained": 4000}[kind]
    base = dict(scenario=kind, seed=seed, detector=det_name)
    # score separation first, before any threshold
    if kind == "recurring":
        on = st["m"] > 0.5
        pre, post = (~on) & (np.arange(T) < T0), on
    elif kind == "sustained":
        pre = np.arange(T) < 4000
        post = pre.copy()  # placeholder, block AUROC handled below
    else:
        pre, post = np.arange(T) < T0, np.arange(T) >= chg
    sep = dict(auroc_before=auroc(s[pre], y[pre]), auroc_after=auroc(s[post], y[post]))
    if kind == "sustained":
        blk = np.zeros(T, bool); blk[4000:5000] = True
        m = (pre & (y == 0)) | blk
        sep["auroc_after"] = auroc(s[m], y[m])  # block rows vs earlier normal rows
    pols = policies(cal, s, y)
    for name, a in pols.items():
        r = dict(base, policy=name, **sep)
        r["fa_pre"] = false_alarm_rate(a, y, slice(0, T0 if kind != "sustained" else 4000))
        r["recall_pre"] = recall(a, y, slice(0, T0 if kind != "sustained" else 4000))
        if kind == "sustained":
            r["block_recall"] = recall(a, y, slice(4000, 5000))
            r["block_recall_last_half"] = recall(a, y, slice(4500, 5000))
            r["absorbed"] = 1 - r["block_recall"]
            r["fa_after_block_500"] = false_alarm_rate(a, y, slice(5000, 5500))
            r["recall_after_block_1000"] = recall(a, y, slice(5000, 6000))
        elif kind == "recurring":
            switches = [2000, 3000, 4000, 5000, 6000, 7000]
            r["fa_first300_after_switch"] = float(np.nanmean(
                [false_alarm_rate(a, y, slice(w, w + 300)) for w in switches]))
            r["fa_post"] = false_alarm_rate(a, y, slice(T0, T))
            r["recall_post"] = recall(a, y, slice(T0, T))
        else:
            r["fa_first500"] = false_alarm_rate(a, y, slice(chg, chg + 500))
            r["fa_post"] = false_alarm_rate(a, y, slice(chg, T))
            r["recall_post"] = recall(a, y, slice(chg, T))
            r["recover"] = time_to_recover(a, y, chg)
            if kind == "gradual":
                r["fa_ramp"] = false_alarm_rate(a, y, slice(T0, 3500))
        rows.append(r)
    trace = None
    if seed == 0 and det_name == "iforest" and kind in ("sudden", "sustained", "gradual", "recurring"):
        trace = {name: trailing_fa(a, y) for name, a in pols.items()}
        trace["_t"] = np.arange(T)
    return rows, trace


def gas_rows(seed, det_name):
    st = gas(seed)
    det = DETECTORS[det_name](seed).fit(st["fit"])
    cal, s, y, b = det.score(st["cal"]), det.score(st["X"]), st["y"], st["batch"]
    rows = []
    pols = policies(cal, s, y)
    # batch 1 (before any drift): calibration normals vs batch-1 anomalies, no threshold
    s1 = det.score(st["anom1"])
    rows.append(dict(scenario="gas", seed=seed, detector=det_name, batch=1, policy="(no threshold)",
                     n=len(cal) + len(s1), n_anom=len(s1),
                     auroc=auroc(np.r_[cal, s1], np.r_[np.zeros(len(cal)), np.ones(len(s1))]),
                     fa=np.nan, recall=np.nan))
    for bb in range(2, 11):
        m = b == bb
        idx = np.flatnonzero(m)
        sl = slice(idx[0], idx[-1] + 1)
        for name, a in pols.items():
            rows.append(dict(scenario="gas", seed=seed, detector=det_name, batch=bb, policy=name,
                             n=int(m.sum()), n_anom=int(y[m].sum()),
                             auroc=auroc(s[m], y[m]),
                             fa=false_alarm_rate(a, y, sl), recall=recall(a, y, sl)))
    return rows


def main():
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    rows, traces = [], {}
    for kind in SCENARIOS:
        for seed in SEEDS:
            for det in ("iforest", "knn"):
                r, tr = synthetic_rows(kind, seed, det)
                rows += r
                if tr is not None:
                    traces[kind] = tr
        print(f"{kind} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "synthetic_runs.csv"), index=False)
    for kind, tr in traces.items():
        pd.DataFrame(tr).to_csv(os.path.join(RES, f"trace_{kind}_iforest_seed0.csv"), index=False)
    grow = []
    for seed in GAS_SEEDS:
        for det in ("iforest", "knn"):
            grow += gas_rows(seed, det)
        print(f"gas seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(grow).to_csv(os.path.join(RES, "gas_runs.csv"), index=False)
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
