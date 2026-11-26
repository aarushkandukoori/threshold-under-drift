"""Turn the second follow-up's runs into summary tables and figures 9 to 12.

Reads results/fix_runs.csv, raise_log.csv and explicit_runs.csv (and the *_fresh.csv files
from seeds 10 to 29 when they exist). Writes:
  summary_raises.csv          where the false-alarm rule's raises came from (Part A)
  summary_fix_ref.csv         every rule at the reference setting (Part A)
  summary_fix_settings.csv    the rule's versions at every delay and capacity setting
  summary_explicit.csv        explicit confirmations only, by share confirmed (Part B)
  summary_fresh.csv           the main comparisons on seeds 10 to 29, when run
  fig9 to fig12 (.pdf and .png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import stats

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FP = "raise on confirmed false alarms"
NEW = FP + ", since last change"
VARIANTS = [FP, FP + ", expire after 100", FP + ", expire after 200", FP + ", expire after 400",
            NEW]
SHORT = {FP: "24 Nov rule", FP + ", expire after 100": "expire after 100",
         FP + ", expire after 200": "expire after 200",
         FP + ", expire after 400": "expire after 400", NEW: "since last change"}
OTHERS = ["fixed", "rolling-500", "MACS-style", "every label returns", "no ticket = normal",
          "roll back on confirmed incidents"]
REF = dict(median=100, shape="heavy", capacity=5)
BAR = 3 * 0.025
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
SAVE = dict(dpi=200, metadata={"CreationDate": None})


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return v[0], np.nan
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def at(d, **kw):
    m = np.ones(len(d), bool)
    for k, v in kw.items():
        m &= (d[k] == v).to_numpy()
    return d[m]


def is_ref(d):
    """Rows at the reference setting, plus score-only rows (which have no setting)."""
    return (d["median"].isna() | ((d["median"] == REF["median"]) & (d["shape"] == REF["shape"])
                                  & (d["capacity"] == REF["capacity"])))


def metrics(d, policy, extra=None):
    """Mean and 95% interval over seeds of every metric, for one rule (rows already filtered
    to one setting and detector)."""
    g = lambda stream: d[(d.stream == stream) & (d.policy == policy)]
    su, sb, two, co = g("sudden"), g("sustained anomaly"), g("two anomalies"), g("no shift")
    r = dict(policy=policy, **(extra or {}))
    for name, col, src in [("fa_first500", "fa_first500", su), ("recover", "recover", su),
                           ("missed", "missed", sb), ("recall_after_anomaly", "recall_after_block", sb),
                           ("missed2", "missed2", two), ("recall_between", "recall_between", two),
                           ("fa_no_shift", "fa", co), ("recall_no_shift", "recall", co)]:
        r[name], r[name + "_pm"] = ci(src[col]) if col in src else (np.nan, np.nan)
    r["recover_never"] = int(su.recover.isna().sum())
    r["n_seeds"] = len(su)
    for ramp in (500, 2000):
        dr = g(f"drift over {ramp}")
        r[f"fa_drift{ramp}"], r[f"fa_drift{ramp}_pm"] = ci(dr.fa_drift)
    if "n_raised_after_anomaly" in sb and sb.n_raised_after_anomaly.notna().any():
        r["runs_raised_after_anomaly"] = int((sb.n_raised_after_anomaly > 0).sum())
        r["runs_raised_before_second"] = int((two.n_raised_after_anomaly > 0).sum())
        r["runs_raised_no_shift"] = int((co.n_raised > 0).sum())
    return r


def raise_summary(rz, fx):
    """Where the raises came from, per version of the rule and stream (Isolation Forest)."""
    out = []
    runs = fx[fx.policy.isin(VARIANTS) & (fx.detector == "iforest")]
    for (pol, stream), grp in runs.groupby(["policy", "stream"], sort=False):
        ev = rz[(rz.policy == pol) & (rz.stream == stream) & (rz.detector == "iforest")]
        r = dict(policy=pol, stream=stream, runs=len(grp), runs_with_raise=int((grp.n_raised > 0).sum()),
                 raises=len(ev))
        if stream in ("sustained anomaly", "two anomalies"):
            hi = np.inf if stream == "sustained anomaly" else 6000
            after = ev[(ev.i >= 5000) & (ev.i < hi)]
            r.update(runs_raised_after_anomaly=int((grp.n_raised_after_anomaly > 0).sum()),
                     raises_after_anomaly=len(after))
            if len(after):
                share = after.alarms_earlier / after.alarms
                r.update(
                    stale_confirmations=int(after.stale.sum()),
                    alert_after_end_min=int((after.i - 5000).min()),
                    alert_after_end_median=float((after.i - 5000).median()),
                    alert_after_end_max=int((after.i - 5000).max()),
                    age_min=int(after.age.min()), age_median=float(after.age.median()),
                    age_max=int(after.age.max()),
                    alarms_from_anomaly_share_median=float(share.median()),
                    alarms_from_anomaly_share_min=float(share.min()),
                    checked_share_anomaly_alarms_median=float(after.checked_share_earlier.median()),
                    checked_share_after_alarms_median=float(after.checked_share_same.median()),
                    labels_from_anomaly_share_median=float((after.labels_earlier / after.labels).median()),
                    fp_labels_from_anomaly=int(after.fp_labels_earlier.sum()),
                    est_median=float(after.est.median()),
                    true_fa_window_median=float(after.true_fa_window.median()),
                    raises_with_after_anomaly_evidence_only=int(after.raise_same_part.sum()),
                    est_after_anomaly_only_max=float(after.est_same_part.max()),
                    est_after_anomaly_only_missing=int(after.est_same_part.isna().sum()),
                    proposal_window_from_anomaly_median=float(after.proposal_window_earlier.median()),
                    cap_jump_median=float((after.cap_after - after.cap_before).median()))
        out.append(r)
    return pd.DataFrame(out)


def main():
    fx = pd.read_csv(os.path.join(RES, "fix_runs.csv"))
    rz = pd.read_csv(os.path.join(RES, "raise_log.csv"))
    ex = pd.read_csv(os.path.join(RES, "explicit_runs.csv"))

    # ---------------- tables
    raise_summary(rz, fx).round(4).to_csv(os.path.join(RES, "summary_raises.csv"), index=False)

    ref = []
    for det in ("iforest", "knn"):
        d = fx[(fx.detector == det) & is_ref(fx)]
        ref += [metrics(d, p, dict(detector=det)) for p in OTHERS + VARIANTS]
    ref = pd.DataFrame(ref)
    ref.round(4).to_csv(os.path.join(RES, "summary_fix_ref.csv"), index=False)

    sets = []
    d = fx[fx.detector == "iforest"]
    for (m, sh, c), grp in d[d.policy.isin(VARIANTS)].groupby(["median", "shape", "capacity"]):
        for p in VARIANTS:
            sets.append(metrics(grp, p, dict(median=m, shape=sh, capacity=c)))
    sets = pd.DataFrame(sets)
    sets.round(4).to_csv(os.path.join(RES, "summary_fix_settings.csv"), index=False)

    exs = []
    for det in ("iforest", "knn"):
        for (p, rd, f, g), grp in ex[ex.detector == det].groupby(
                ["policy", "reading", "explicit_fp", "explicit_inc"], sort=False):
            r = metrics(grp, p, dict(detector=det, reading=rd, explicit_fp=f, explicit_inc=g))
            su = grp[grp.stream == "sudden"]
            r["explicit_fp_back_sudden"] = su.n_explicit_fp.mean()
            exs.append(r)
    exs = pd.DataFrame(exs)
    exs.round(4).to_csv(os.path.join(RES, "summary_explicit.csv"), index=False)

    fresh = []
    if os.path.exists(os.path.join(RES, "fix_runs_fresh.csv")):
        ff = pd.read_csv(os.path.join(RES, "fix_runs_fresh.csv"))
        ef = pd.read_csv(os.path.join(RES, "explicit_runs_fresh.csv"))
        for det in ("iforest", "knn"):
            d = ff[(ff.detector == det) & is_ref(ff)]
            fresh += [metrics(d, p, dict(detector=det, part="A", reading="every outcome known",
                                         explicit_fp=1.0, explicit_inc=1.0))
                      for p in ["fixed"] + VARIANTS]
            for (p, rd, f, g), grp in ef[ef.detector == det].groupby(
                    ["policy", "reading", "explicit_fp", "explicit_inc"], sort=False):
                fresh.append(metrics(grp, p, dict(detector=det, part="B", reading=rd,
                                                  explicit_fp=f, explicit_inc=g)))
        pd.DataFrame(fresh).round(4).to_csv(os.path.join(RES, "summary_fresh.csv"), index=False)

    figures(fx, rz, ex, ref, exs)
    print("summaries and figures 9-12 written")


def figures(fx, rz, ex, ref, exs):
    blue, orange, green, verm, gray = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7F7F7F"

    # ---------------- figure 9: what the post-anomaly raises were made of
    after = rz[(rz.policy == FP) & (rz.stream == "sustained anomaly") & (rz.detector == "iforest")
               & (rz.i >= 5000)].copy()
    after = after.sort_values("est").reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.35]))
    a = ax[0]
    a.scatter(after.i - 5000, after.age, s=10, color=blue, alpha=0.7, edgecolor="none")
    a.set_xlabel("Alert behind the raise\n(points after the anomaly ended)")
    a.set_ylabel("Age of its confirmation\nwhen it arrived (points)")
    a.set_xlim(left=0)
    a.set_ylim(bottom=0)
    a.set_title("Every confirmation: a point after the anomaly", fontsize=8.5)
    a = ax[1]
    x = np.arange(len(after))
    same = after.est_same_part.fillna(0.0)
    a.vlines(x, same * 100, after.est * 100, color="#BBBBBB", lw=0.8)
    a.scatter(x, after.est * 100, s=9, color=blue, zorder=3,
              label="as computed (the window still holds the anomaly)")
    a.scatter(x, same * 100, s=9, facecolor="white", edgecolor=verm, zorder=3,
              label="from points after the anomaly only")
    a.axhline(BAR * 100, color="black", ls=":", lw=1, label="raise bar (3x budget)")
    a.set_xticks([])
    a.set_xlabel(f"Each raise right after the anomaly ({len(after)} raises, sorted)")
    a.set_ylabel("False-alarm estimate (%)")
    a.set_title("Its estimate: mostly the anomaly's own alarms", fontsize=8.5)
    a.legend(fontsize=6.8, frameon=False, loc="upper left")
    a.set_ylim(0, max(13.5, after.est.max() * 100 + 2))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig9_raise_evidence.{ext}"), **SAVE)
    plt.close(fig)

    # ---------------- figure 10: the fixes side by side
    d = fx[fx.detector == "iforest"]
    allset = d[d.policy.isin(VARIANTS)]
    labels = [SHORT[p] for p in VARIANTS]
    cols = [blue, gray, gray, gray, orange]
    yk = np.arange(len(VARIANTS))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    a = ax[0]
    n_runs = len(at(allset, policy=FP, stream="sustained anomaly"))
    aft = [int((at(allset, policy=p, stream="sustained anomaly").n_raised_after_anomaly > 0).sum())
           for p in VARIANTS]
    non = [int((at(allset, policy=p, stream="no shift").n_raised > 0).sum()) for p in VARIANTS]
    a.barh(yk - 0.2, aft, height=0.38, color=cols, label="right after the anomaly")
    a.barh(yk + 0.2, non, height=0.38, color="white", edgecolor=cols, hatch="////", lw=0.8,
           label="stream with no anomaly")
    for k in range(len(VARIANTS)):
        a.text(aft[k] + 2, k - 0.2, str(aft[k]), va="center", fontsize=6.8)
        a.text(non[k] + 2, k + 0.2, str(non[k]), va="center", fontsize=6.8)
    a.set_yticks(yk, labels)
    a.invert_yaxis()
    a.set_xlim(0, 115)
    a.set_xlabel(f"Runs where the cap went up\n(of {n_runs}: 19 settings x 10 seeds)")
    a.legend(fontsize=6.5, frameon=False, loc="lower right")
    r = ref[ref.detector == "iforest"].set_index("policy")
    for k, (col, xl, scale) in enumerate([("missed2", "Second anomaly missed (%)", 100),
                                          ("recover", "Points to recover\nafter a real shift", 1)]):
        a = ax[k + 1]
        for kk, p in enumerate(VARIANTS):
            v, e = r.loc[p, col] * scale, r.loc[p, col + "_pm"] * scale
            nv = int(r.loc[p, "recover_never"]) if col == "recover" else 0
            if nv:
                a.text(0.03, kk, f"never, {nv} of 10 seeds", va="center", fontsize=6.8,
                       transform=a.get_yaxis_transform())
                continue
            a.errorbar(v, kk, xerr=e, fmt="o", color=cols[kk], ms=4, lw=1, capsize=2)
        if col == "missed2":
            a.axvline(r.loc["fixed", col] * scale, color="black", ls=":", lw=1)
            a.text(r.loc["fixed", col] * scale + 0.5, len(VARIANTS) - 0.55, "fixed", fontsize=6.8)
        a.set_yticks(yk, [""] * len(yk))
        a.set_ylim(len(VARIANTS) - 0.5, -0.5)
        a.set_xlabel(xl + ",\nreference setting" if col == "missed2" else xl + ", ref. setting")
    ax[2].set_xlim(300, 620)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig10_fixes.{ext}"), **SAVE)
    plt.close(fig)

    # ---------------- figure 11: thresholds over time, two anomalies, seed 5
    tr = pd.read_csv(os.path.join(RES, "confirm_trace_two_anomalies_seed5.csv"))
    tr = tr[(tr.t >= 3500) & (tr.t < 7500)]
    fig, a = plt.subplots(figsize=(7.2, 2.5))
    for lo, hi in ((4000, 5000), (6000, 7000)):
        a.axvspan(lo, hi, color="#F2F2F2", lw=0)
    a.scatter(tr.t, tr.score, s=1.2, color="#BBBBBB", rasterized=True, label="scores")
    a.plot(tr.t, tr.fixed, color="black", ls=":", lw=1, label="fixed")
    a.plot(tr.t, tr.thr_1124, color=blue, lw=1.1, label="24 Nov rule")
    a.plot(tr.t, tr.thr_new, color=orange, lw=1.3, ls="--", label="since last change")
    flags = tr.last_change.diff().fillna(0) != 0
    for t0 in tr.t[flags]:
        a.plot([t0], [tr.score.min() - 0.01], marker="^", color=orange, ms=4, clip_on=False)
    top = tr.score.max() + 0.02
    a.text(4500, top, "anomaly 1", ha="center", va="top", fontsize=7.5)
    a.text(6500, top, "anomaly 2", ha="center", va="top", fontsize=7.5)
    a.set_xlabel("Time (points); triangles = where a change was flagged")
    a.set_ylabel("Isolation Forest score\nand threshold")
    a.set_ylim(tr.score.min() - 0.02, top + 0.005)
    h, l = a.get_legend_handles_labels()
    a.legend(h, l, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28),
             ncol=4, markerscale=4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig11_two_anomalies_seed5.{ext}"), **SAVE)
    plt.close(fig)

    # ---------------- figure 12: explicit confirmations only
    e = exs[exs.detector == "iforest"]
    base = ref[ref.detector == "iforest"].set_index("policy")
    lines = [("no ticket = normal", "no incident record = normal", 1.0, gray, "-", "no ticket = normal"),
             ("no ticket = normal", "no incident record = normal", 0.5, gray, "--", None),
             (FP, "closed = false positive", 0.5, verm, "--", "24 Nov rule, closed = false positive"),
             ("roll back on confirmed incidents", "explicit only", 1.0, green, "-",
              "roll back, explicit only"),
             ("roll back on confirmed incidents", "explicit only", 0.5, green, "--", None),
             (FP, "explicit only", 1.0, blue, "-", "24 Nov rule, explicit only"),
             (FP, "explicit only", 0.5, blue, "--", None),
             (NEW, "explicit only", 1.0, orange, "-", "since last change, explicit only"),
             (NEW, "explicit only", 0.5, orange, "--", None)]
    panels = [("recover", "Points to recover\nafter a real shift", 1),
              ("fa_first500", "False alarms, 500 points\nafter the shift (%)", 100),
              ("missed", "Sustained anomaly\nmissed (%)", 100),
              ("missed2", "Second anomaly\nmissed (%)", 100)]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.75))
    for k, (col, yl, scale) in enumerate(panels):
        a = ax[k]
        if col != "recover":
            a.axhline(base.loc["fixed", col] * scale, color="black", ls=":", lw=1,
                      label="fixed")
        for p, rd, g, c, ls_name, lab in lines:
            q = e[(e.policy == p) & (e.reading == rd) & (e.explicit_inc == g)].sort_values("explicit_fp")
            yv = np.where(q.recover_never == q.n_seeds, np.nan, q[col] * scale)
            a.plot(q.explicit_fp, yv, color=c, ls=ls_name, marker="o", ms=2.2, lw=1.0, label=lab)
        a.set_xlabel("Share of false positives\nexplicitly confirmed", fontsize=7.5)
        a.set_ylabel(yl, fontsize=7.5)
        a.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0], ["0.1", "", "0.5", "", "1"])
        a.tick_params(labelsize=7)
    ax[0].set_yscale("log")
    ax[0].set_yticks([300, 500, 1000, 2000], ["300", "500", "1000", "2000"])
    ax[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for a in ax[2:]:
        a.set_ylim(55, 100)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.6, frameon=False, loc="lower center", ncol=4)
    fig.text(0.5, 0.995, "solid - every real issue that was checked gets an incident record;  "
             "dashed - half of them just close", ha="center", va="top", fontsize=6.8, color="#555555")
    fig.tight_layout(rect=(0, 0.15, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig12_explicit_only.{ext}"), **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    main()
