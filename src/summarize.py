"""Turn results/*.csv into summary tables and figures."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import stats

RES = os.path.join(os.path.dirname(__file__), "..", "results")
# Okabe-Ito colorblind-safe palette
C = {"fixed": "#000000", "rolling-200": "#56B4E9", "rolling-500": "#0072B2",
     "rolling-1000": "#009E73", "SCS-style": "#E69F00", "MACS-style": "#D55E00",
     "confirmed-normal-500, labels 50 late": "#CC79A7",
     "confirmed-normal-500, labels 200 late": "#999999"}
MAIN = list(C)
SHORT = {"confirmed-normal-500, labels 50 late": "confirmed-normal, labels 50 late",
         "confirmed-normal-500, labels 200 late": "confirmed-normal, labels 200 late"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h


def summary_table(d):
    rows = []
    for (det, pol), g in d.groupby(["detector", "policy"], sort=False):
        su = g[g.scenario == "sudden"]
        sb = g[g.scenario == "sustained"]
        gr = g[g.scenario == "gradual"]
        rc = g[g.scenario == "recurring"]
        co = g[g.scenario == "control"]
        r = dict(detector=det, policy=pol)
        for name, col, src in [("fa_before", "fa_pre", su), ("fa_first500", "fa_first500", su),
                               ("recover", "recover", su), ("recall_after", "recall_post", su),
                               ("absorbed", "absorbed", sb),
                               ("recall_after_block", "recall_after_block_1000", sb),
                               ("fa_ramp", "fa_ramp", gr),
                               ("fa_switch300", "fa_first300_after_switch", rc),
                               ("fa_control", "fa_post", co), ("recall_control", "recall_post", co)]:
            m, h = ci(src[col])
            r[name], r[name + "_pm"] = m, h
        r["recover_never"] = int(su["recover"].isna().sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(os.path.join(RES, "synthetic_runs.csv"))
    g = pd.read_csv(os.path.join(RES, "gas_runs.csv"))
    tab = summary_table(d)
    tab.round(4).to_csv(os.path.join(RES, "summary_synthetic.csv"), index=False)

    sep = d.groupby(["scenario", "detector"])[["auroc_before", "auroc_after"]].agg(["mean", "std"])
    sep.round(4).to_csv(os.path.join(RES, "summary_separation.csv"))
    gsep = g.groupby(["detector", "batch"]).auroc.agg(["mean", "std"]).round(4)
    gsep.to_csv(os.path.join(RES, "summary_gas_auroc.csv"))
    gp = g[g.batch >= 2].groupby(["detector", "policy", "batch"], sort=False)[["fa", "recall"]].mean()
    gp.round(4).to_csv(os.path.join(RES, "summary_gas_policies.csv"))
    # whole-stream gas numbers (batches 2-10 pooled, weighted by rows)
    gw = g[g.batch >= 2].copy()
    gw["n_norm"] = gw.n - gw.n_anom
    gw["fa_x"] = gw.fa * gw.n_norm
    gw["rec_x"] = gw.recall * gw.n_anom
    pooled = gw.groupby(["detector", "policy", "seed"], sort=False)[["fa_x", "rec_x", "n_norm", "n_anom"]].sum()
    pooled["fa"] = pooled.fa_x / pooled.n_norm
    pooled["recall"] = pooled.rec_x / pooled.n_anom
    pooled = pooled.groupby(["detector", "policy"], sort=False)[["fa", "recall"]].mean()
    pooled.round(4).to_csv(os.path.join(RES, "summary_gas_pooled.csv"))

    # ---------------- figure 1: score separation
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    scen = ["control", "sudden", "gradual", "recurring", "sustained"]
    lab = ["no shift", "sudden", "gradual", "recurring", "sustained\nanomaly"]
    for j, (det, mk) in enumerate([("iforest", "o"), ("knn", "s")]):
        b = [d[(d.scenario == s) & (d.detector == det)].auroc_before.mean() for s in scen]
        a = [d[(d.scenario == s) & (d.detector == det)].auroc_after.mean() for s in scen]
        x = np.arange(len(scen)) + (j - 0.5) * 0.25
        ax[0].scatter(x, b, marker=mk, facecolor="white", edgecolor="#0072B2", zorder=3,
                      label=f"{det}: before change")
        ax[0].scatter(x, a, marker=mk, color="#D55E00", zorder=3, label=f"{det}: after change")
        for xi, bi, ai in zip(x, b, a):
            ax[0].plot([xi, xi], [bi, ai], color="#999999", lw=0.8)
    ax[0].set_xticks(range(len(scen)), lab, fontsize=7.5)
    ax[0].set_ylabel("AUROC (normal vs anomaly)")
    ax[0].set_ylim(0.7, 1.0)
    ax[0].set_title("Synthetic streams (mean of 10 seeds)", fontsize=9)
    ax[0].legend(fontsize=6.5, frameon=False, loc="lower center", ncol=2)
    for det, mk, col in [("iforest", "o", "#0072B2"), ("knn", "s", "#D55E00")]:
        gg = gsep.loc[det]
        ax[1].errorbar(gg.index, gg["mean"], yerr=gg["std"], marker=mk, color=col, capsize=2,
                       lw=1, ms=4, label=det)
    ax[1].axhline(0.5, color="#999999", ls="--", lw=0.8)
    ax[1].text(10, 0.52, "coin flip", ha="right", fontsize=7, color="#555555")
    ax[1].set_xticks(range(1, 11))
    ax[1].set_xlabel("Batch (1 = training months)")
    ax[1].set_ylabel("AUROC")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("UCI gas sensor drift (mean $\\pm$ SD, 5 seeds)", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig1_score_separation.{ext}"), dpi=200)
    plt.close(fig)

    # ---------------- figure 2: false alarms over time (sudden + recurring, iforest seed 0)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for k, (kind, title) in enumerate([("sudden", "Sudden shift at t = 2,000"),
                                       ("recurring", "Recurring shift (on 2-3k, 4-5k, 6-7k)")]):
        tr = pd.read_csv(os.path.join(RES, f"trace_{kind}_iforest_seed0.csv"))
        for pol in ["fixed", "rolling-200", "rolling-1000", "SCS-style", "MACS-style",
                    "confirmed-normal-500, labels 50 late"]:
            ax[k].plot(tr["_t"], 100 * tr[pol], color=C[pol], lw=1.1, label=SHORT.get(pol, pol))
        ax[k].axhline(2.5, color="#555555", ls=":", lw=0.8)
        ax[k].set_title(title, fontsize=9)
        ax[k].set_xlabel("Time (points)")
        ax[k].set_yscale("symlog", linthresh=5)
        ax[k].set_yticks([0, 2.5, 5, 10, 25, 50, 100], ["0", "2.5", "5", "10", "25", "50", "100"])
    ax[0].set_ylabel("False alarms on last 200\nnormal points (%)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig2_false_alarms_over_time.{ext}"), dpi=200)
    plt.close(fig)

    # ---------------- figure 3: recover fast vs absorb a sustained anomaly
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    NEVER = 6300
    for k, det in enumerate(["iforest", "knn"]):
        t = tab[tab.detector == det]
        for pol in MAIN:
            r = t[t.policy == pol].iloc[0]
            x = NEVER if r.recover_never == 10 else r.recover
            xe = 0 if r.recover_never == 10 else r.recover_pm
            ax[k].errorbar(x, 100 * r.absorbed, xerr=xe, yerr=100 * r.absorbed_pm, fmt="o",
                           color=C[pol], ms=5, capsize=2, label=SHORT.get(pol, pol))
        ax[k].set_xscale("log")
        ax[k].set_xticks([200, 500, 1000, NEVER], ["200", "500", "1000", "never"])
        ax[k].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax[k].set_xlim(150, 9000)
        ax[k].set_xlabel("Points to recover after a sudden shift")
        ax[k].set_title(f"Frozen {'Isolation Forest' if det == 'iforest' else 'kNN'} (10 seeds)",
                        fontsize=9)
    ax[0].set_ylabel("Sustained anomaly absorbed (%)\n(share of the 1,000-point block missed)")
    ax[0].set_ylim(0, 100)
    ax[1].legend(fontsize=6.5, frameon=False, loc="center left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig3_recover_vs_absorb.{ext}"), dpi=200)
    plt.close(fig)

    # ---------------- figure 4: gas, false alarms and recall by batch (kNN)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for pol in ["fixed", "rolling-500", "SCS-style", "MACS-style", "confirmed-normal-500, labels 50 late"]:
        q = gp.xs(("knn", pol), level=("detector", "policy"))
        ax[0].plot(q.index, 100 * q.fa, marker="o", ms=3, color=C[pol], lw=1, label=SHORT.get(pol, pol))
        ax[1].plot(q.index, 100 * q.recall, marker="o", ms=3, color=C[pol], lw=1)
    ax[0].axhline(2.5, color="#555555", ls=":", lw=0.8)
    ax[0].set_ylabel("False alarms on normal rows (%)")
    ax[1].set_ylabel("Anomaly recall (%)")
    for a in ax:
        a.set_xlabel("Gas sensor batch")
        a.set_xticks(range(2, 11))
        a.set_ylim(-3, 103)
    ax[0].legend(fontsize=6.5, frameon=False, loc="center right")
    ax[0].set_title("Frozen kNN: false alarms", fontsize=9)
    ax[1].set_title("Frozen kNN: recall", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig4_gas_by_batch.{ext}"), dpi=200)
    plt.close(fig)
    print("summaries and figures written")


if __name__ == "__main__":
    main()
