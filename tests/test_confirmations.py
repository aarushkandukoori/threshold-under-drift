"""Tests for the second follow-up: explicit confirmations, the change test and the
since-last-change estimate."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from thresholds import BUDGET, Q, rolling_threshold
from feedback import (Checks, MIN_LABELS, draw_delays, draw_outcomes, labels_seen,
                      flag_changes, raise_on_false_alarms, rollback_on_incidents,
                      silence_as_normal)
from metrics import false_alarm_rate
from streams import synthetic
from detectors import DETECTORS

rng = np.random.default_rng(2)
CAL = rng.normal(size=1000)
FIXED = np.quantile(CAL, Q)


def stream_with_block(new_normal):
    """2,000 points, then 1,000 points shifted up by 3, then 2,000 back at the start."""
    s = np.r_[rng.normal(size=2000), rng.normal(size=1000) + 3.0, rng.normal(size=2000)]
    y = (rng.random(len(s)) < 0.02).astype(int)
    s[y == 1] += 4.0
    if not new_normal:
        y[2000:3000] = 1
    return s, y


def rule_1124(cal, s, y, delay, capacity, window=500, recent=500, factor=3):
    """Frozen copy of raise_on_false_alarms as it was on 24 November 2026."""
    T = len(s)
    prop = rolling_threshold(cal, s, window)
    cap = float(np.quantile(cal, Q))
    checks = Checks(T, delay, capacity)
    back = np.zeros(T, np.int8)
    back_fp = np.zeros(T, np.int8)
    n_alarms = np.zeros(T + 1)
    alarms = np.zeros(T, bool)
    for t in range(T):
        for i in checks.arrive[t]:
            back[i] = 1
            if y[i] != 0:
                continue
            back_fp[i] = 1
            lo = max(0, i + 1 - recent)
            n_back = int(back[lo:i + 1].sum())
            if n_back < MIN_LABELS:
                continue
            share_fp = back_fp[lo:i + 1].sum() / n_back
            est = (n_alarms[i + 1] - n_alarms[lo]) * share_fp / (i + 1 - lo)
            if est > factor * BUDGET and prop[i] > cap:
                cap = float(prop[i])
        alarms[t] = s[t] > min(prop[t], cap)
        n_alarms[t + 1] = n_alarms[t] + alarms[t]
        checks.after_decision(t, alarms[t])
    return alarms


def scores(kind, seed):
    st = synthetic(kind, seed)
    det = DETECTORS["iforest"](seed).fit(st["fit"])
    return det.score(st["cal"]), det.score(st["X"]), st["y"]


def test_changes_none_is_the_rule_of_24_november():
    for new_normal in (True, False):
        s, y = stream_with_block(new_normal)
        for shape, cap in (("heavy", 5), ("steady", 2), ("heavy", None)):
            d = draw_delays(3, len(s), 100, shape)
            assert np.array_equal(raise_on_false_alarms(CAL, s, y, d, cap, changes=None),
                                  rule_1124(CAL, s, y, d, cap))


def test_all_explicit_is_the_same_as_no_outcome_model():
    s, y = stream_with_block(new_normal=False)
    d = draw_delays(0, len(s), 100, "heavy")
    ex = np.ones(len(s), bool)
    for kw in (dict(changes=None), {}):
        assert np.array_equal(raise_on_false_alarms(CAL, s, y, d, 5, explicit=ex, **kw),
                              raise_on_false_alarms(CAL, s, y, d, 5, **kw))
    assert np.array_equal(rollback_on_incidents(CAL, s, y, d, 5, explicit=ex),
                          rollback_on_incidents(CAL, s, y, d, 5))


def test_ambiguous_closures_are_never_looked_at():
    """Flipping the true label of every ticket that closed ambiguously changes nothing."""
    s, y = stream_with_block(new_normal=False)
    d = draw_delays(0, len(s), 100, "heavy")
    ex = draw_outcomes(0, y, 0.5, 0.5)
    y2 = np.where(ex, y, 1 - y)
    for rule, kw in ((raise_on_false_alarms, dict(changes=None)), (raise_on_false_alarms, {}),
                     (rollback_on_incidents, {})):
        assert np.array_equal(rule(CAL, s, y, d, 5, explicit=ex, **kw),
                              rule(CAL, s, y2, d, 5, explicit=ex, **kw)), rule.__name__


def test_outcome_draws_have_the_right_rates_and_are_nested():
    y = (np.arange(200_000) % 10 == 0).astype(int)
    lo, hi = draw_outcomes(0, y, 0.25, 0.5), draw_outcomes(0, y, 0.75, 0.5)
    assert abs(lo[y == 0].mean() - 0.25) < 0.01 and abs(hi[y == 0].mean() - 0.75) < 0.01
    assert abs(lo[y == 1].mean() - 0.5) < 0.01
    assert np.all(hi[lo])                          # more confirmations only add, never swap
    seen = labels_seen(y, lo)
    assert np.all(seen[~lo] == 0) and np.array_equal(seen[lo], y[lo])


def test_change_test_is_past_only():
    s = np.r_[rng.normal(size=2000), rng.normal(size=3000) + 2.0]
    s2 = s.copy()
    s2[3000:] -= 4.0
    a, _ = flag_changes(CAL, s)
    b, _ = flag_changes(CAL, s2)
    assert np.array_equal(a[:3001], b[:3001])


def test_change_test_finds_a_step_and_stays_quiet_without_one():
    s = np.r_[rng.normal(size=2000), rng.normal(size=2000) + 2.0]
    last, flags = flag_changes(CAL, s)
    assert len(flags) == 1
    flagged_at, start = flags[0]
    assert 2000 < flagged_at < 2100 and abs(start - 2000) <= 10
    assert last[flagged_at - 1] == 0 and last[flagged_at] == start
    assert flag_changes(CAL, rng.normal(size=20_000))[1] == []


def test_raise_log_matches_the_count_and_changes_nothing():
    s, y = stream_with_block(new_normal=True)
    d = draw_delays(0, len(s), 50, "steady")
    log = []
    o = raise_on_false_alarms(CAL, s, y, d, 5, info=True, log=log)
    assert len(log) == o["raised"] > 0
    assert np.array_equal(o["alarms"], raise_on_false_alarms(CAL, s, y, d, 5))
    for ev in log:
        assert ev["t"] > ev["i"] >= ev["lo"] and ev["est"] > 3 * BUDGET


def test_new_estimate_does_not_raise_after_the_sustained_anomaly():
    """Seed 5 at the reference setting: the rule of 24 November raised its cap on a false
    positive right after the anomaly. Counting from the latest change, it does not."""
    cal, s, y = scores("sustained", 5)
    d = draw_delays(5, len(s), 100, "heavy")
    old, new = [], []
    a_old = raise_on_false_alarms(cal, s, y, d, 5, changes=None, log=old)
    a_new = raise_on_false_alarms(cal, s, y, d, 5, log=new)
    assert any(ev["i"] >= 5000 for ev in old)
    assert not any(ev["i"] >= 5000 for ev in new)
    fixed = s > np.quantile(cal, Q)
    assert np.array_equal(a_new[4000:5000], fixed[4000:5000])   # the anomaly stays as visible
    assert a_new[5000:6000][y[5000:6000] == 1].mean() > a_old[5000:6000][y[5000:6000] == 1].mean()


def test_new_estimate_still_recovers_from_a_real_shift():
    cal, s, y = scores("sudden", 0)
    d = draw_delays(0, len(s), 100, "heavy")
    a = raise_on_false_alarms(cal, s, y, d, 5)
    assert false_alarm_rate(a, y, slice(2600, 3000)) < 0.05
    assert false_alarm_rate(s > np.quantile(cal, Q), y, slice(2600, 3000)) > 0.3


def test_no_incident_record_counts_as_normal_for_the_silence_rule():
    """With g < 1 a fixed real issue can close with no incident record. The no-ticket rule
    then takes it in as normal, so it can only get worse than with every incident recorded."""
    s, y = stream_with_block(new_normal=False)
    d = draw_delays(0, len(s), 50, "steady")
    seen = labels_seen(y, draw_outcomes(0, y, 1.0, 0.5))
    full = silence_as_normal(CAL, s, y, d, 5, wait=50)
    half = silence_as_normal(CAL, s, seen, d, 5, wait=50)
    assert half[2000:3000].mean() <= full[2000:3000].mean()
