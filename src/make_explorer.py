"""Build docs/index.html: a small page to explore the second follow-up's saved results.

Reads results/summary_explicit.csv, summary_raises.csv and
confirm_trace_two_anomalies_seed5.csv. Everything on the page comes from those files.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "docs", "index.html")
FP = "raise on confirmed false alarms"
NEW = FP + ", since last change"


def main():
    s = pd.read_csv(os.path.join(RES, "summary_explicit.csv"))
    s = s[s.detector == "iforest"]
    ref = pd.read_csv(os.path.join(RES, "summary_fix_ref.csv"))
    ref = ref[ref.detector == "iforest"].set_index("policy")
    rz = pd.read_csv(os.path.join(RES, "summary_raises.csv")).set_index(["policy", "stream"])
    cells = {}
    for _, r in s.iterrows():
        if r.policy not in (FP, NEW, "no ticket = normal"):
            continue
        rule = {FP: "old", NEW: "new", "no ticket = normal": "silence"}[r.policy]
        reading = {"explicit only": "unknown", "closed = false positive": "fp",
                   "no incident record = normal": "silence"}[r.reading]
        key = f"{rule}|{reading}|{r.explicit_inc:g}|{r.explicit_fp:g}"
        cells[key] = dict(missed=round(100 * r.missed), missed2=round(100 * r.missed2),
                          recover=None if r.recover_never == r.n_seeds else int(round(r.recover)),
                          fa=round(100 * r.fa_first500, 1))
    fixed = dict(missed=round(100 * ref.loc["fixed", "missed"]),
                 missed2=round(100 * ref.loc["fixed", "missed2"]),
                 fa=round(100 * ref.loc["fixed", "fa_first500"], 1))
    raises = dict(old=int(rz.loc[(FP, "sustained anomaly"), "runs_raised_after_anomaly"]),
                  new=int(rz.loc[(NEW, "sustained anomaly"), "runs_raised_after_anomaly"]),
                  none=int(rz.loc[(NEW, "no shift"), "runs_with_raise"]),
                  runs=int(rz.loc[(FP, "sustained anomaly"), "runs"]))
    tr = pd.read_csv(os.path.join(RES, "confirm_trace_two_anomalies_seed5.csv"))
    tr = tr[(tr.t >= 3500) & (tr.t < 7500)]
    trace = dict(t0=int(tr.t.iloc[0]),
                 score=[round(float(v), 3) for v in tr.score.iloc[::2]],
                 fixed=round(float(tr.fixed.iloc[0]), 4),
                 old=[round(float(v), 4) for v in tr.thr_1124],
                 new=[round(float(v), 4) for v in tr.thr_new])
    data = json.dumps(dict(cells=cells, fixed=fixed, raises=raises, trace=trace),
                      separators=(",", ":"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(PAGE.replace("__DATA__", data))
    print("wrote", OUT, f"{os.path.getsize(OUT) / 1024:.0f} KB")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explicit confirmations explorer</title>
<style>
  :root { --ink: #1f2933; --muted: #5f6b76; --line: #d9dee3; --bg: #ffffff; --panel: #f6f8fa;
          --old: #0072B2; --new: #E69F00; --bad: #D55E00; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  main { max-width: 1240px; margin: 0 auto; padding: 18px 20px 10px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13.5px; margin: 0 0 14px; }
  .grid { display: grid; grid-template-columns: 470px 1fr; gap: 18px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  h2 { font-size: 15px; margin: 0 0 10px; }
  label.ctl { display: block; font-size: 13px; color: var(--muted); margin: 10px 0 5px; }
  .slider-row { display: flex; align-items: center; gap: 12px; }
  input[type=range] { flex: 1; accent-color: var(--ink); }
  .val { font-weight: 700; font-size: 18px; min-width: 64px; text-align: right; }
  .ticks { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin: 2px 76px 0 2px; }
  .seg { display: flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fff; }
  .seg button { flex: 1; border: 0; background: transparent; padding: 7px 8px; font-size: 12.5px; cursor: pointer; color: var(--ink); }
  .seg button + button { border-left: 1px solid var(--line); }
  .seg button.on { background: var(--ink); color: #fff; }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px; }
  .card .k { font-size: 12px; color: var(--muted); }
  .card .v { font-size: 26px; font-weight: 700; margin: 2px 0; }
  .card .r { font-size: 11.5px; color: var(--muted); }
  .bar { height: 6px; background: #e8ecef; border-radius: 3px; margin-top: 5px; position: relative; }
  .bar span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--new); }
  .bar i { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--ink); }
  svg text { font-size: 11px; fill: var(--muted); }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); margin: 6px 0 0; flex-wrap: wrap; }
  .legend b { display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin-right: 5px; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }
  .stat { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .stat .v { font-size: 24px; font-weight: 700; }
  .stat .k { font-size: 12px; color: var(--muted); }
  .note { font-size: 12px; color: var(--muted); margin-top: 10px; }
  footer { color: var(--muted); font-size: 12px; margin-top: 12px; }
  footer a { color: var(--old); }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>Raising an alarm threshold only on explicit false-positive confirmations</h1>
  <p class="sub">Frozen Isolation Forest on synthetic streams, mean of 10 seeds, label delays with a median of 100 points,
  5 checks per 100 points. Every number on this page comes from the saved runs in the repo.</p>
  <div class="grid">
    <section class="panel" id="left">
      <h2>Only explicit confirmations count</h2>
      <label class="ctl" for="f">Share of checked false positives that get an explicit confirmation</label>
      <div class="slider-row"><input id="f" type="range" min="0" max="4" step="1" value="4"><span class="val" id="fval">1</span></div>
      <div class="ticks"><span>0.1</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></div>
      <label class="ctl">Real issues that get fixed</label>
      <div class="seg" id="g"><button id="g1" class="on" data-v="1">every one gets an incident record</button><button id="g05" data-v="0.5">half just close</button></div>
      <label class="ctl">A ticket that just closes is</label>
      <div class="seg" id="reading"><button id="unknown" class="on" data-v="unknown">left as unknown</button><button id="fp" data-v="fp">read as a false positive</button></div>
      <label class="ctl">Version of the rule</label>
      <div class="seg" id="rule"><button id="rule-old" data-v="old">24 November</button><button id="rule-new" class="on" data-v="new">since last change</button></div>
      <div class="cards">
        <div class="card" id="c-missed"><div class="k">Sustained anomaly missed</div><div class="v" id="missed"></div>
          <div class="bar"><span id="missed-bar"></span><i id="missed-ref"></i></div><div class="r" id="missed-r"></div></div>
        <div class="card" id="c-missed2"><div class="k">Second anomaly missed</div><div class="v" id="missed2"></div>
          <div class="bar"><span id="missed2-bar"></span><i id="missed2-ref"></i></div><div class="r" id="missed2-r"></div></div>
        <div class="card" id="c-recover"><div class="k">Points to recover after a real shift</div><div class="v" id="recover"></div><div class="r">fixed threshold never recovers</div></div>
        <div class="card" id="c-fa"><div class="k">False alarms, 500 points after the shift</div><div class="v" id="fa"></div><div class="r" id="fa-r"></div></div>
      </div>
      <div id="spacer" style="height:36px"></div>
    </section>
    <section class="panel" id="right">
      <h2>Why the 24 November rule raised its threshold right after the anomaly</h2>
      <svg id="chart" viewBox="0 0 700 300" width="100%" role="img" aria-label="Scores and thresholds over time, seed 5"></svg>
      <div class="legend"><span><b style="background:#bbb;height:6px;width:6px;border-radius:3px"></b>scores</span>
        <span><b style="border-top:2px dotted #1f2933;height:0"></b>fixed</span>
        <span><b style="background:var(--old)"></b>24 November rule</span>
        <span><b style="background:var(--new)"></b>since last change</span></div>
      <div class="seg" id="show" style="margin-top:10px"><button id="show-old" data-v="old">show 24 November rule</button><button id="show-new" data-v="new">show since last change</button><button id="show-both" class="on" data-v="both">show both</button></div>
      <div class="stats">
        <div class="stat"><div class="v" id="s-old"></div><div class="k">runs where the 24 November rule raised its cap right after the anomaly</div></div>
        <div class="stat"><div class="v" id="s-new"></div><div class="k">runs where the new version did</div></div>
        <div class="stat"><div class="v" id="s-none"></div><div class="k">runs where the new version raised it in a stream with no anomaly</div></div>
      </div>
      <p class="note">Seed 5, two identical anomalies, every outcome known. Counts are out of <span id="s-runs"></span> runs
      (19 delay and capacity settings x 10 seeds). The old rule's evidence window still held the anomaly's own alarms; the new
      version counts evidence only since the last change it detects in the scores.</p>
    </section>
  </div>
  <footer>Made by Aarush Kandukoori. Code and saved outputs at <a href="https://github.com/aarushkandukoori/threshold-under-drift">github.com/aarushkandukoori/threshold-under-drift</a></footer>
</main>
<script>
const D = __DATA__;
const F = [0.1, 0.25, 0.5, 0.75, 1];
const state = { f: 4, g: "1", reading: "unknown", rule: "new", show: "both" };
const $ = (id) => document.getElementById(id);

function seg(id, key) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $(id).querySelectorAll("button")) x.classList.toggle("on", x === b);
    state[key] = b.dataset.v; render();
  });
}
seg("g", "g"); seg("reading", "reading"); seg("rule", "rule"); seg("show", "show");
$("f").addEventListener("input", () => { state.f = +$("f").value; render(); });

function render() {
  const f = F[state.f];
  $("fval").textContent = String(f);
  const c = D.cells[`${state.rule}|${state.reading}|${state.g}|${f}`];
  const sil = D.cells[`silence|silence|${state.g}|${f}`];
  $("missed").textContent = c.missed + "%";
  $("missed2").textContent = c.missed2 + "%";
  $("recover").textContent = c.recover === null ? "never" : c.recover.toLocaleString("en-US");
  $("fa").textContent = c.fa.toFixed(1) + "%";
  $("missed-r").textContent = `fixed threshold ${D.fixed.missed}%, no ticket = normal ${sil.missed}%`;
  $("missed2-r").textContent = `fixed threshold ${D.fixed.missed2}%`;
  $("fa-r").textContent = `fixed threshold ${D.fixed.fa.toFixed(1)}%`;
  for (const k of ["missed", "missed2"]) {
    $(k + "-bar").style.width = c[k] + "%";
    $(k + "-bar").style.background = c[k] > D.fixed[k] + 5 ? "var(--bad)" : "var(--new)";
    $(k + "-ref").style.left = D.fixed[k] + "%";
  }
  draw();
}

function draw() {
  const T = D.trace, W = 700, H = 300, L = 44, R = 8, Tp = 18, B = 26;
  const n = T.old.length, t0 = T.t0;
  const lo = 0.36, hi = 0.68;
  const x = (t) => L + (t - t0) / n * (W - L - R);
  const y = (v) => Tp + (hi - Math.min(hi, Math.max(lo, v))) / (hi - lo) * (H - Tp - B);
  let s = "";
  for (const [a, b, name] of [[4000, 5000, "anomaly 1"], [6000, 7000, "anomaly 2"]]) {
    s += `<rect x="${x(a)}" y="${Tp}" width="${x(b) - x(a)}" height="${H - Tp - B}" fill="#eceff2"/>`;
    s += `<text x="${(x(a) + x(b)) / 2}" y="${Tp - 5}" text-anchor="middle">${name}</text>`;
  }
  for (let k = 0; k < T.score.length; k++)
    s += `<circle cx="${x(t0 + 2 * k).toFixed(1)}" cy="${y(T.score[k]).toFixed(1)}" r="1" fill="#bdbdbd"/>`;
  s += `<line x1="${L}" x2="${W - R}" y1="${y(T.fixed)}" y2="${y(T.fixed)}" stroke="#1f2933" stroke-dasharray="2,3"/>`;
  const path = (arr) => arr.map((v, k) => `${k ? "L" : "M"}${x(t0 + k).toFixed(1)},${y(v).toFixed(1)}`).join("");
  if (state.show !== "new") s += `<path d="${path(T.old)}" fill="none" stroke="var(--old)" stroke-width="2"/>`;
  if (state.show !== "old") s += `<path d="${path(T.new)}" fill="none" stroke="var(--new)" stroke-width="2.5" stroke-dasharray="7,4"/>`;
  for (const v of [0.4, 0.5, 0.6]) s += `<text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`;
  for (const t of [4000, 5000, 6000, 7000]) s += `<text x="${x(t)}" y="${H - 8}" text-anchor="middle">${t.toLocaleString("en-US")}</text>`;
  s += `<text x="${L}" y="${H - 8}" text-anchor="start">time</text>`;
  $("chart").innerHTML = s;
}
$("s-old").textContent = D.raises.old; $("s-new").textContent = D.raises.new;
$("s-none").textContent = D.raises.none; $("s-runs").textContent = D.raises.runs;
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
