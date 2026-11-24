"""Threshold rules that learn from delayed, partial feedback.

Feedback model
--------------
Delays. Every point gets its own label delay, drawn once per seed so that every rule
sees the same delay for the same point:
  steady        uniform between 0.5x and 1.5x the median delay
  heavy-tailed  lognormal with the same median and sigma 1.5 (about 1 label in 6 takes
                more than 4x the median, about 1 in 16 more than 10x)
A label with delay d for point i can be used from point i + d + 1 on, the same
convention as rolling_confirmed_normal in thresholds.py. Labels that would arrive after
the stream ends never arrive.

Coverage. Two cases:
  every label returns  every point gets its true label, alarm or not. This is what the
                       first version assumed.
  checked alerts only  only an alert can be checked, and only while the team has time:
                       `capacity` checks per 100 points on average (a token bucket that
                       holds at most `capacity` checks; capacity None = every alert is
                       checked). A checked alert comes back as a confirmed incident or a
                       confirmed false positive. A point that raised no alert, or an
                       alert nobody had time to check, never gets a label.

Rules (each decides on point t from the past only)
--------------------------------------------------
  every_label_returns     97.5% quantile of the last 500 points whose label came back
                          normal (the old confirmed-normal rule, with per-point delays)
  silence_as_normal       checked alerts only. A point counts as normal once `wait`
                          points pass with no incident ticket for it. This is the old
                          rule fed with real tickets, and it is wrong: no ticket does
                          not mean normal.
  raise_on_false_alarms   checked alerts only. Threshold = min(rolling-500 proposal, cap).
                          The cap starts at the fixed threshold. It only goes up when a
                          checked alert comes back as a false positive and the checked
                          alerts from the 500 points before it put the false-alarm rate
                          above 3x the budget (alarms x share of returned labels that
                          were false). It then goes up to the proposal at the time of
                          that false alarm. Silence never moves the cap.
  rollback_on_incidents   checked alerts only. Follows the rolling-500 proposal. When at
                          least 8 of the last 10 returned labels are confirmed incidents,
                          it drops back to the lowest threshold at which one of those
                          incidents fired, and stays there until fewer than 5 of the last
                          10 are incidents.
The settings (500 points and 3x budget for the first rule; last 10 labels, 8 must agree
for the second) were picked once and not tuned.
"""
import bisect
from collections import deque

import numpy as np

from thresholds import BUDGET, Q, rolling_threshold

WINDOW = 500          # points in the rolling window (proposal and confirmed-normal rules)
LAST = 10             # how many returned labels the gate looks at
AGREE = 8             # how many of them must agree to open the gate
RELEASE = 5           # rollback_on_incidents lets go when fewer than this many are incidents
RECENT = 500          # raise_on_false_alarms: points used to estimate the false-alarm rate
FACTOR = 3            # raise_on_false_alarms: raise only when that estimate is above 3x budget
MIN_LABELS = 5        # raise_on_false_alarms: returned labels needed for an estimate
SIGMA_HEAVY = 1.5     # spread of the heavy-tailed (lognormal) delays


def draw_delays(seed, T, median, shape):
    """One label delay per point, in points. The same seed gives the same underlying
    draws for every median, so a longer median stretches the same delays."""
    rng = np.random.default_rng([seed, 7])
    if shape == "steady":
        d = median * (0.5 + rng.random(T))
    elif shape == "heavy":
        d = median * np.exp(SIGMA_HEAVY * rng.standard_normal(T))
    else:
        raise ValueError(shape)
    return np.maximum(1, np.rint(d)).astype(np.int64)


class Checks:
    """Which alerts get checked and when each label comes back (checked alerts only)."""

    def __init__(self, T, delay, capacity):
        self.T = T
        self.delay = delay
        self.unlimited = capacity is None
        self.rate = 0.0 if self.unlimited else capacity / 100.0
        if self.unlimited or capacity == 0:
            self.size = 0.0                  # unlimited needs no bucket; 0 = nobody checks
        else:
            self.size = max(1.0, float(capacity))
        self.tokens = self.size
        self.arrive = [[] for _ in range(T)]   # arrive[t] = points whose label is usable from t
        self.checked = np.zeros(T, bool)

    def after_decision(self, t, alarm):
        if alarm and (self.unlimited or self.tokens >= 1.0):
            if not self.unlimited:
                self.tokens -= 1.0
            self.checked[t] = True
            a = t + int(self.delay[t]) + 1
            if a < self.T:
                self.arrive[a].append(t)
        if not self.unlimited:
            self.tokens = min(self.size, self.tokens + self.rate)


def _info(alarms, thr, checks=None):
    out = dict(alarms=alarms, thr=thr)
    if checks is not None:
        out["checked"] = checks.checked
    return out


def every_label_returns(cal, s, y, delay, window=WINDOW, info=False):
    """Full coverage: every point's true label comes back after its own delay."""
    n, T = len(cal), len(s)
    hist = np.concatenate([cal, s])
    pos = list(range(n))                     # positions in hist known to be normal, sorted
    arrive = [[] for _ in range(T)]
    for i in range(T):
        a = i + int(delay[i]) + 1
        if a < T and y[i] == 0:
            arrive[a].append(i)
    alarms = np.zeros(T, bool)
    thr = np.empty(T)
    cur = np.quantile(hist[pos[-window:]], Q)
    for t in range(T):
        if arrive[t]:
            for i in arrive[t]:
                bisect.insort(pos, n + i)
            cur = np.quantile(hist[pos[-window:]], Q)
        thr[t] = cur
        alarms[t] = s[t] > cur
    return _info(alarms, thr) if info else alarms


def silence_as_normal(cal, s, y, delay, capacity, wait, window=WINDOW, info=False):
    """Checked alerts only, but a point with no incident ticket after `wait` points is
    treated as a confirmed normal. Confirmed incidents are taken out when they arrive."""
    n, T = len(cal), len(s)
    hist = np.concatenate([cal, s])
    pos = list(range(n))
    dropped = np.zeros(T, bool)
    checks = Checks(T, delay, capacity)
    alarms = np.zeros(T, bool)
    thr = np.empty(T)
    cur = np.quantile(hist[pos[-window:]], Q)
    for t in range(T):
        changed = False
        for i in checks.arrive[t]:
            if y[i] == 1:                    # a confirmed incident: take it out
                dropped[i] = True
                j = bisect.bisect_left(pos, n + i)
                if j < len(pos) and pos[j] == n + i:
                    pos.pop(j)
                    changed = True
        k = t - wait - 1                     # point k has now gone `wait` points with no ticket
        if k >= 0 and not dropped[k]:
            pos.append(n + k)
            changed = True
        if changed:
            cur = np.quantile(hist[pos[-window:]], Q)
        thr[t] = cur
        alarms[t] = s[t] > cur
        checks.after_decision(t, alarms[t])
    return _info(alarms, thr, checks) if info else alarms


def raise_on_false_alarms(cal, s, y, delay, capacity, window=WINDOW, recent=RECENT,
                          factor=FACTOR, info=False):
    """Checked alerts only. The threshold may follow the score-only proposal down at any
    time, but it only goes up as far as confirmed false alarms allow.

    When a checked alert at point i comes back as a false positive, estimate the
    false-alarm rate over the `recent` points up to i:
        (alarms in those points) x (share of their returned labels that are false) / points
    If at least MIN_LABELS labels from those points are back and the estimate is above
    `factor` x budget, raise the cap to the proposal at point i."""
    T = len(s)
    prop = rolling_threshold(cal, s, window)
    cap = float(np.quantile(cal, Q))         # start at the fixed threshold
    checks = Checks(T, delay, capacity)
    back = np.zeros(T, np.int8)              # 1 once the label of a checked alert is back
    back_fp = np.zeros(T, np.int8)           # 1 once it is back as a false positive
    n_alarms = np.zeros(T + 1)               # n_alarms[t] = alarms before point t
    alarms = np.zeros(T, bool)
    thr = np.empty(T)
    raised = 0
    for t in range(T):
        for i in checks.arrive[t]:
            back[i] = 1
            if y[i] != 0:
                continue                     # a confirmed incident never raises the cap
            back_fp[i] = 1
            lo = max(0, i + 1 - recent)
            n_back = int(back[lo:i + 1].sum())
            if n_back < MIN_LABELS:
                continue
            share_fp = back_fp[lo:i + 1].sum() / n_back
            est = (n_alarms[i + 1] - n_alarms[lo]) * share_fp / (i + 1 - lo)
            if est > factor * BUDGET and prop[i] > cap:
                cap = float(prop[i])         # the level proposed when that false alarm fired
                raised += 1
        thr[t] = min(prop[t], cap)
        alarms[t] = s[t] > thr[t]
        n_alarms[t + 1] = n_alarms[t] + alarms[t]
        checks.after_decision(t, alarms[t])
    if info:
        out = _info(alarms, thr, checks)
        out["raised"] = raised
        return out
    return alarms


def rollback_on_incidents(cal, s, y, delay, capacity, window=WINDOW, info=False):
    """Checked alerts only. Follows the score-only proposal until confirmed incidents
    say it went too far, then drops back to where those incidents fired."""
    T = len(s)
    prop = rolling_threshold(cal, s, window)
    cap = np.inf
    checks = Checks(T, delay, capacity)
    last = deque(maxlen=LAST)                # (confirmed incident?, threshold when it fired)
    alarms = np.zeros(T, bool)
    thr = np.empty(T)
    for t in range(T):
        for i in checks.arrive[t]:
            last.append((bool(y[i] == 1), thr[i]))
            n_inc = sum(inc for inc, _ in last)
            if len(last) == LAST and n_inc >= AGREE:
                cap = min(cap, min(th for inc, th in last if inc))
            elif n_inc < RELEASE:
                cap = np.inf
        thr[t] = min(prop[t], cap)
        alarms[t] = s[t] > thr[t]
        checks.after_decision(t, alarms[t])
    return _info(alarms, thr, checks) if info else alarms


NAMES = ["every label returns", "no ticket = normal", "raise on confirmed false alarms",
         "roll back on confirmed incidents"]


def feedback_policies(cal, s, y, seed, median, shape, capacity, rules=NAMES):
    """The feedback rules for one delay setting and one capacity, by name. Each value is
    the dict returned with info=True (alarms, thr, and for checked-alerts rules the
    checked points)."""
    d = draw_delays(seed, len(s), median, shape)
    run = {
        "every label returns": lambda: every_label_returns(cal, s, y, d, info=True),
        "no ticket = normal": lambda: silence_as_normal(cal, s, y, d, capacity, wait=median,
                                                        info=True),
        "raise on confirmed false alarms": lambda: raise_on_false_alarms(cal, s, y, d, capacity,
                                                                         info=True),
        "roll back on confirmed incidents": lambda: rollback_on_incidents(cal, s, y, d, capacity,
                                                                          info=True),
    }
    return {name: run[name]() for name in rules}
