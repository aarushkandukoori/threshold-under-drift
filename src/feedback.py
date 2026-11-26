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
                       checked). A point that raised no alert, or an alert nobody had time
                       to check, never gets a label.

How a checked alert closes (second follow-up). In the first two tests every checked alert
came back as a confirmed incident or a confirmed false positive. A ticket can also just
close: a real issue that was fixed ends up closed too, so a closure alone says nothing.
draw_outcomes() decides, once per seed and point, which checked alerts close with an
explicit outcome: a false positive is explicitly confirmed with probability explicit_fp, a
real incident is recorded as an incident with probability explicit_inc. The rest close
ambiguously. Rules that take `explicit` treat an ambiguous closure as unknown and never
look at it. labels_seen() gives the other reading, where every closure without an
incident record counts as a false positive.

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
                          checked alert comes back as a false positive and the alarms and
                          returned labels behind it put the false-alarm rate above 3x the
                          budget (alarms x share of returned labels that were false). It
                          then goes up to the proposal at the time of that false alarm.
                          Silence never moves the cap.
                          By default the estimate only uses alarms and labels from the
                          latest change flagged by flag_changes() on, and drops
                          confirmations from before it. changes=None gives the version
                          of 24 November 2026, which used the 500 points before the false
                          alarm even when they straddled a change.
  rollback_on_incidents   checked alerts only. Follows the rolling-500 proposal. When at
                          least 8 of the last 10 returned labels are confirmed incidents,
                          it drops back to the lowest threshold at which one of those
                          incidents fired, and stays there until fewer than 5 of the last
                          10 are incidents.
Settings were set by hand. For raise_on_false_alarms, earlier gates and a 2x bar let the
cap creep up during normal running on trial runs (seeds 0 to 3), so it uses the 500-point
estimate with a 3x bar. Last 10 labels with 8 agreeing (rollback_on_incidents) was picked once.
The change test settings (50 recent points, 6 standard errors, at least 100 and at most
500 reference points) were picked from the score scale before any run. Two details were
changed after trial runs on seeds 0 to 3: the spread now comes from the reference scores (the
first version used the calibration scores and flagged extra changes in kNN scores after the
shift), and the estimate now starts where the change began instead of where it was flagged
(starting at the flag dropped the first checks after a shift and slowed recovery on seed 0).
Nothing else was tuned.
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
CHANGE_RECENT = 50    # flag_changes: points in the recent window
CHANGE_REF_MIN = 100  # flag_changes: reference points needed before it tests
CHANGE_REF_MAX = 500  # flag_changes: at most this many reference points
CHANGE_Z = 6.0        # flag_changes: flag when the means differ by this many standard errors


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


def draw_outcomes(seed, y, explicit_fp, explicit_inc=1.0):
    """Which tickets close with an explicit outcome, one draw per point. A false positive
    is explicitly confirmed as one with probability explicit_fp; a real incident is
    recorded as an incident with probability explicit_inc; every other ticket just closes.
    The same seed gives the same underlying draws for every probability, so a higher
    explicit_fp only adds confirmations."""
    u = np.random.default_rng([seed, 11]).random(len(y))
    return u < np.where(np.asarray(y) == 1, explicit_inc, explicit_fp)


def labels_seen(y, explicit):
    """The reading that every closed ticket without an incident record was a false
    positive: the true label where the outcome was recorded, 0 where the ticket just closed."""
    return np.where(explicit, y, 0)


def flag_changes(cal, s, recent=CHANGE_RECENT, ref_min=CHANGE_REF_MIN,
                 ref_max=CHANGE_REF_MAX, z=CHANGE_Z, edge=10):
    """A past-only test for a change in the score stream. Labels are never used.

    At point t it compares the mean of the last `recent` scores with the mean of the scores
    before them, going back at most `ref_max` points and never past the last change. It
    flags a change at t when the two means differ by more than z standard errors (from the
    spread of those reference scores). A flag comes some points after the change itself, so
    it then estimates where the change started: the split of the scores since the reference
    start that best separates the two means (at least `edge` points on each side). The next
    test starts its reference there and needs `ref_min` + `recent` points after it. `cal` is
    not used; it keeps the same call shape as the threshold rules.

    Returns (last, flags): last[t] = the estimated start of the latest change flagged at or
    before t (0 before the first flag), and a list of (flagged at, estimated start)."""
    T = len(s)
    cs = np.concatenate([[0.0], np.cumsum(s)])
    cs2 = np.concatenate([[0.0], np.cumsum(np.square(s))])
    last = np.zeros(T, np.int64)
    start = 0
    flags = []
    for t in range(T):
        r_lo = t - recent                  # recent window = s[r_lo:t], all before t
        ref_lo = max(start, r_lo - ref_max)
        n_ref = r_lo - ref_lo
        if n_ref >= ref_min:
            m_recent = (cs[t] - cs[r_lo]) / recent
            m_ref = (cs[r_lo] - cs[ref_lo]) / n_ref
            sd_ref = np.sqrt(max((cs2[r_lo] - cs2[ref_lo]) / n_ref - m_ref * m_ref, 1e-12))
            if abs(m_recent - m_ref) > z * sd_ref * np.sqrt(1.0 / recent + 1.0 / n_ref):
                k = np.arange(ref_lo + edge, t - edge + 1)
                n1, n2 = k - ref_lo, t - k
                gap = (cs[t] - cs[k]) / n2 - (cs[k] - cs[ref_lo]) / n1
                start = int(k[np.argmax(np.abs(gap) * np.sqrt(n1 * n2 / (n1 + n2)))])
                flags.append((t, start))
        last[t] = start
    return last, flags


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
    treated as a confirmed normal. Confirmed incidents are taken out when they arrive.
    (Pass labels_seen(y, explicit) as y when some tickets just close: a closure with no
    incident record is no incident ticket, so this rule counts it as normal.)"""
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
                          factor=FACTOR, explicit=None, changes="flag", max_age=None,
                          info=False, log=None):
    """Checked alerts only. The threshold may follow the score-only proposal down at any
    time, but it only goes up as far as confirmed false alarms allow.

    When a checked alert at point i comes back as a false positive, estimate the
    false-alarm rate over the points from lo to i:
        (alarms in those points) x (share of their returned labels that are false) / points
    If at least MIN_LABELS labels from those points are back and the estimate is above
    `factor` x budget, raise the cap to the proposal at point i.

    Which points count (lo):
      changes="flag" (default)  lo = the later of i + 1 - recent and the latest change that
                                flag_changes() has flagged by now. A confirmation for a point
                                before that change is dropped.
      changes=array             the same, with last[t] given (from flag_changes).
      changes=None              lo = i + 1 - recent: the version of 24 November 2026. Right
                                after a sustained anomaly those points are mostly the anomaly,
                                whose many alarms were rarely checked, so the share of false
                                positives among the few new labels got applied to the
                                anomaly's alarms and the cap went up (diagnose_raises.py).
      max_age=W                 also drop confirmations for alerts more than W points old,
                                and count only the last W points (an alternative fix that
                                was tested; off by default).
    explicit: None = every checked alert closes with its true outcome (the first two tests).
    Otherwise a boolean array from draw_outcomes(); a ticket with explicit[i] False closed
    ambiguously and is never counted, as a false positive or as an incident.
    log: a list; one dict per raise is appended (for diagnose_raises.py)."""
    T = len(s)
    prop = rolling_threshold(cal, s, window)
    if isinstance(changes, str):
        changes = flag_changes(cal, s)[0]
    cap = float(np.quantile(cal, Q))         # start at the fixed threshold
    checks = Checks(T, delay, capacity)
    back = np.zeros(T, np.int8)              # 1 once the label of a checked alert is back
    back_fp = np.zeros(T, np.int8)           # 1 once it is back as a false positive
    n_alarms = np.zeros(T + 1)               # n_alarms[t] = alarms before point t
    alarms = np.zeros(T, bool)
    thr = np.empty(T)
    raised = 0
    for t in range(T):
        start = 0                            # confirmations and alarms before this are not used
        if changes is not None:
            start = int(changes[t])
        if max_age is not None:
            start = max(start, t - max_age)
        for i in checks.arrive[t]:
            if explicit is not None and not explicit[i]:
                continue                     # closed with no explicit outcome: unknown
            back[i] = 1
            if y[i] != 0:
                continue                     # a confirmed incident never raises the cap
            back_fp[i] = 1
            if i < start:
                continue                     # from before the latest change, or too old
            lo = max(start, i + 1 - recent)
            n_back = int(back[lo:i + 1].sum())
            if n_back < MIN_LABELS:
                continue
            share_fp = back_fp[lo:i + 1].sum() / n_back
            est = (n_alarms[i + 1] - n_alarms[lo]) * share_fp / (i + 1 - lo)
            if est > factor * BUDGET and prop[i] > cap:
                if log is not None:
                    log.append(dict(t=t, i=i, lo=lo, est=float(est),
                                    n_alarms=int(n_alarms[i + 1] - n_alarms[lo]),
                                    labels=np.flatnonzero(back[lo:i + 1]) + lo,
                                    fp_labels=np.flatnonzero(back_fp[lo:i + 1]) + lo,
                                    cap_before=cap, cap_after=float(prop[i])))
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


def rollback_on_incidents(cal, s, y, delay, capacity, window=WINDOW, explicit=None,
                          info=False):
    """Checked alerts only. Follows the score-only proposal until confirmed incidents
    say it went too far, then drops back to where those incidents fired.
    explicit: as in raise_on_false_alarms; an ambiguous closure is not one of the last 10."""
    T = len(s)
    prop = rolling_threshold(cal, s, window)
    cap = np.inf
    checks = Checks(T, delay, capacity)
    last = deque(maxlen=LAST)                # (confirmed incident?, threshold when it fired)
    alarms = np.zeros(T, bool)
    thr = np.empty(T)
    for t in range(T):
        for i in checks.arrive[t]:
            if explicit is not None and not explicit[i]:
                continue
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
    """The feedback rules for one delay setting and one capacity, by name, exactly as run
    on 24 November 2026 (the false-alarm rule with changes=None). Each value is the dict
    returned with info=True (alarms, thr, and for checked-alerts rules the checked points)."""
    d = draw_delays(seed, len(s), median, shape)
    run = {
        "every label returns": lambda: every_label_returns(cal, s, y, d, info=True),
        "no ticket = normal": lambda: silence_as_normal(cal, s, y, d, capacity, wait=median,
                                                        info=True),
        "raise on confirmed false alarms": lambda: raise_on_false_alarms(cal, s, y, d, capacity,
                                                                         changes=None, info=True),
        "roll back on confirmed incidents": lambda: rollback_on_incidents(cal, s, y, d, capacity,
                                                                          info=True),
    }
    return {name: run[name]() for name in rules}
