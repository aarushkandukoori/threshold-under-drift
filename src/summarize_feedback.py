"""Turn results/feedback_*.csv into summary tables and figures 5-8."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import stats

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FB = ["every label returns", "no ticket = normal", "raise on confirmed false alarms",
      "roll back on confirmed incidents"]
SCORE_ONLY = ["rolling-200", "rolling-500", "rolling-1000", "SCS-style", "MACS-style"]
# Okabe-Ito colorblind-safe palette
C = {"fixed": "#000000", "rolling-500": "#7F7F7F",
     "every label returns": "#CC79A7", "no ticket = normal": "#D55E00",
     "raise on confirmed false alarms": "#0072B2", "roll back on confirmed incidents": "#009E73"}
MK = {"every label returns": "D", "no ticket = normal": "v",
      "raise on confirmed false alarms": "o", "roll back on confirmed incidents": "s"}
REF = dict(median=100, shape="heavy", capacity=5)
NEVER = 6300
# where the "10" and "800" delay labels sit next to each line in figure 5 (points)
OFF = {"every label returns": ((2, -10), (3, -10)), "no ticket = normal": ((2, 4), (5, 0)),
       "raise on confirmed false alarms": ((-3, 5), (-3, 5)),
       "roll back on confirmed incidents": ((-15, -2), (5, 0))}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return v[0], np.nan
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def pick(d, stream, policy, det="iforest", median=None, shape=None, capacity=None):
    m = (d.stream == stream) & (d.policy == policy) & (d.detector == det)
    if policy in FB:
        m &= (d["median"] == median) & (d["shape"] == shape)
        if policy == "every label returns":
            m &= d.capacity.isna()
        else:
            m &= d.capacity == capacity
    return d[m]


def row(d, policy, det="iforest", median=None, shape=None, capacity=None):
    """All metrics for one rule and one setting, mean and 95% interval over seeds."""
    g = lambda stream: pick(d, stream, policy, det, median, shape, capacity)
    su, sb, co = g("sudden"), g("sustained anomaly"), g("no shift")
    r = dict(policy=policy, detector=det, median=median, shape=shape, capacity=capacity)
    for name, col, src in [("fa_before", "fa_pre", su), ("fa_first500", "fa_first500", su),
                           ("recover", "recover", su), ("recall_after", "recall_post", su),
                           ("missed", "missed", sb), ("fa_no_shift", "fa", co),
                           ("recall_no_shift", "recall", co)]:
        r[name], r[name + "_pm"] = ci(src[col])
    r["recover_never"] = int(su.recover.isna().sum())
    r["n_seeds"] = len(su)
    for ramp in (500, 2000):
        dr = g(f"drift over {ramp}")
        r[f"fa_drift{ramp}"], r[f"fa_drift{ramp}_pm"] = ci(dr.fa_drift)
    if policy in FB and policy != "every label returns":
        r["checks_sudden"] = su.n_checked.mean()
    if policy == "raise on confirmed false alarms":
        r["raised_no_shift_max"] = co.n_raised.max()
    return r


def main():
    d = pd.read_csv(os.path.join(RES, "feedback_runs.csv"))
    gas = pd.read_csv(os.path.join(RES, "feedback_gas_runs.csv"))

    # ---------------- tables
    ref = [row(d, p) for p in ["fixed"] + SCORE_ONLY]
    ref += [row(d, p, **REF) for p in FB]
    ref = pd.DataFrame(ref)
    ref.round(4).to_csv(os.path.join(RES, "summary_feedback_ref.csv"), index=False)

    knn = [row(d, p, det="knn") for p in ["fixed"] + SCORE_ONLY]
    knn += [row(d, p, det="knn", **REF) for p in FB]
    pd.DataFrame(knn).round(4).to_csv(os.path.join(RES, "summary_feedback_knn.csv"), index=False)

    delay = []
    for sh in ("steady", "heavy"):
        for m in sorted(d["median"].dropna().unique()):
            for p in FB:
                delay.append(row(d, p, median=m, shape=sh, capacity=5))
    delay = pd.DataFrame(delay)
    delay.round(4).to_csv(os.path.join(RES, "summary_feedback_delay.csv"), index=False)

    cap = []
    for c in sorted(d.capacity.dropna().unique()):
        for p in FB[1:]:
            cap.append(row(d, p, median=REF["median"], shape=REF["shape"], capacity=c))
    cap = pd.DataFrame(cap)
    cap.round(4).to_csv(os.path.join(RES, "summary_feedback_capacity.csv"), index=False)

    gp = gas.groupby(["detector", "policy"], sort=False)[["fa", "recall"]].mean()
    gp.round(4).to_csv(os.path.join(RES, "summary_feedback_gas.csv"))

    # ---------------- figure 5: recover vs miss, feedback rules traced over the median delay
    base = ref.set_index("policy")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for k, sh in enumerate(["steady", "heavy"]):
        a = ax[k]
        r = base.loc["fixed"]
        a.scatter(NEVER, 100 * r.missed, color=C["fixed"], marker="X", s=40, zorder=4,
                  label="fixed")
        for j, p in enumerate(SCORE_ONLY):
            r = base.loc[p]
            a.scatter(r.recover, 100 * r.missed, facecolor="white", edgecolor="#7F7F7F",
                      marker="o", s=22, zorder=3, label="score-only rules" if j == 0 else None)
        for p in FB:
            q = delay[(delay.policy == p) & (delay["shape"] == sh)].sort_values("median")
            x = np.where(q.recover_never == q.n_seeds, NEVER, q.recover)
            a.plot(x, 100 * q.missed, color=C[p], marker=MK[p], ms=4, lw=1.2, label=p)
            for idx, off in zip((0, len(q) - 1), OFF[p]):
                a.annotate(f"{int(q['median'].iloc[idx])}", (x[idx], 100 * q.missed.iloc[idx]),
                           textcoords="offset points", xytext=off, fontsize=6.5, color=C[p])
        a.set_xscale("log")
        a.set_xticks([200, 500, 1000, 2000, NEVER], ["200", "500", "1000", "2000", "never"])
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a.set_xlim(150, 9000)
        a.set_ylim(50, 100)
        a.set_xlabel("Points to recover after a sudden shift")
        a.set_title(f"{'Steady' if sh == 'steady' else 'Heavy-tailed'} label delays", fontsize=9)
    ax[0].set_ylabel("Sustained anomaly missed (%)")
    ax[1].text(0.98, 0.97, "numbers = median label delay (points)", transform=ax[1].transAxes,
               ha="right", va="top", fontsize=6.5, color="#555555")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig5_feedback_recover_vs_miss.{ext}"), dpi=200)
    plt.close(fig)

    # ---------------- figure 6: false alarms during a drift vs median delay, three drift speeds
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for k, (col, title) in enumerate([("fa_first500", "Sudden shift"),
                                      ("fa_drift500", "Drift over 500 points"),
                                      ("fa_drift2000", "Drift over 2,000 points")]):
        a = ax[k]
        for p, ls_name in [("fixed", ":"), ("rolling-500", "--")]:
            a.axhline(100 * base.loc[p, col], color=C[p], ls=ls_name, lw=1,
                      label=p if k == 0 else None)
        for p in ["raise on confirmed false alarms", "every label returns"]:
            for sh, ls_name in [("steady", "-"), ("heavy", "--")]:
                q = delay[(delay.policy == p) & (delay["shape"] == sh)].sort_values("median")
                a.plot(q["median"], 100 * q[col], color=C[p], ls=ls_name, marker=MK[p], ms=3,
                       lw=1.1, label=f"{p}, {sh}" if k == 0 else None)
        a.set_xscale("log")
        a.set_xticks([10, 50, 200, 800], ["10", "50", "200", "800"])
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a.set_xlabel("Median label delay (points)")
        a.set_title(title, fontsize=9)
    ax[0].set_ylabel("False alarms during the drift\nand 500 points after (%)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.8, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig6_delay_vs_drift_speed.{ext}"), dpi=200)
    plt.close(fig)

    # ---------------- figure 7: investigation capacity
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    xs = sorted(cap.capacity.unique())
    for k, (col, ylab) in enumerate([("missed", "Sustained anomaly\nmissed (%)"),
                                     ("recover", "Points to recover\nafter a sudden shift")]):
        a = ax[k]
        scale = 100 if col == "missed" else 1
        for p in FB[1:]:
            q = cap[cap.policy == p].sort_values("capacity")
            a.plot(q.capacity, scale * q[col], color=C[p], marker=MK[p], ms=4, lw=1.2, label=p)
        for p, ls_name in [("fixed", ":"), ("rolling-500", "--")]:
            if col == "recover" and p == "fixed":
                continue                     # fixed never recovers
            a.axhline(scale * base.loc[p, col], color=C[p], ls=ls_name, lw=1, label=p)
        r = base.loc["every label returns"]
        a.axhline(scale * r[col], color=C["every label returns"], ls="-.", lw=1,
                  label="every label returns (all points)")
        a.set_xscale("log")
        a.set_xticks(xs, [("all" if x >= 100 else f"{int(x)}") for x in xs])
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a.set_xlabel("Alerts checked per 100 points (capacity)")
        a.set_ylabel(ylab)
    ax[0].set_ylim(50, 100)
    ax[0].text(1, 57, "raise on confirmed false alarms = fixed (all capacities)", fontsize=6.5,
               color=C["raise on confirmed false alarms"])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.8, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig7_checks_capacity.{ext}"), dpi=200)
    plt.close(fig)

    # ---------------- figure 8: what the thresholds do (seed 0, reference setting)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for k, (key, lo, hi, title) in enumerate([
            ("sudden", 1500, 3500, "Sudden shift of normal data at t = 2,000"),
            ("sustained", 3500, 5500, "Sustained anomaly, t = 4,000 to 4,999")]):
        tr = pd.read_csv(os.path.join(RES, f"feedback_trace_{key}_seed0.csv"))
        tr = tr[(tr.t >= lo) & (tr.t < hi)]
        a = ax[k]
        a.scatter(tr.t, tr.score, s=1.5, color="#BBBBBB", rasterized=True, label="scores")
        for p, ls_name in [("fixed", ":"), ("rolling-500", "--")]:
            a.plot(tr.t, tr[p], color=C[p], ls=ls_name, lw=1, label=p)
        for p in FB[1:]:
            a.plot(tr.t, tr[p], color=C[p], lw=1.1, label=p)
        a.set_title(title, fontsize=9)
        a.set_xlabel("Time (points)")
    ax[0].set_ylabel("Isolation Forest score\nand threshold")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.8, frameon=False, loc="lower center", ncol=3, markerscale=4)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig8_thresholds_over_time.{ext}"), dpi=200)
    plt.close(fig)
    print("feedback summaries and figures written")


if __name__ == "__main__":
    main()
