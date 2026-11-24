import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from thresholds import Q, rolling_confirmed_normal, rolling_threshold
from feedback import (draw_delays, every_label_returns, silence_as_normal,
                      raise_on_false_alarms, rollback_on_incidents, feedback_policies, NAMES)

rng = np.random.default_rng(1)
CAL = rng.normal(size=1000)
FIXED = np.quantile(CAL, Q)


def stream_with_block(new_normal):
    """2,000 points, then 1,000 points shifted up by 3, then 2,000 back at the start.
    new_normal=True labels the shifted points normal, False labels them an anomaly."""
    s = np.r_[rng.normal(size=2000), rng.normal(size=1000) + 3.0, rng.normal(size=2000)]
    y = (rng.random(len(s)) < 0.02).astype(int)
    s[y == 1] += 4.0
    if not new_normal:
        y[2000:3000] = 1
    return s, y


def test_every_label_returns_matches_old_rule_with_a_fixed_delay():
    s, y = stream_with_block(new_normal=False)
    d = np.full(len(s), 50)
    assert np.array_equal(every_label_returns(CAL, s, y, d),
                          rolling_confirmed_normal(CAL, s, y, 500, 50))


def test_delays_have_the_right_median():
    for shape in ("steady", "heavy"):
        d = draw_delays(0, 200_000, 100, shape)
        assert abs(np.median(d) - 100) <= 3, shape
    d = draw_delays(0, 10_000, 100, "steady")
    assert d.min() >= 50 and d.max() <= 150


def test_feedback_rules_are_past_only():
    """Changing scores or labels from t = 3000 on never changes an alarm before 3000."""
    s, y = stream_with_block(new_normal=False)
    s2, y2 = s.copy(), y.copy()
    s2[3000:] += 5.0
    y2[3000:] = 1 - y2[3000:]
    a = feedback_policies(CAL, s, y, 0, 100, "heavy", 5)
    b = feedback_policies(CAL, s2, y2, 0, 100, "heavy", 5)
    for name in NAMES:
        assert np.array_equal(a[name]["alarms"][:3000], b[name]["alarms"][:3000]), name


def test_points_nobody_checked_are_never_looked_at():
    """Flipping the true label of every point that was not checked changes nothing."""
    s, y = stream_with_block(new_normal=False)
    d = draw_delays(0, len(s), 100, "heavy")
    for rule in (silence_as_normal, raise_on_false_alarms, rollback_on_incidents):
        kw = dict(wait=100) if rule is silence_as_normal else {}
        o = rule(CAL, s, y, d, 5, info=True, **kw)
        y2 = y.copy()
        y2[~o["checked"]] = 1 - y2[~o["checked"]]
        assert np.array_equal(o["alarms"], rule(CAL, s, y2, d, 5, **kw)), rule.__name__


def test_silence_never_raises_the_threshold():
    """With nobody checking alerts, the false-alarm rule never goes above the fixed
    threshold, even when the scores move up for good."""
    s = np.r_[rng.normal(size=2000), rng.normal(size=3000) + 3.0]
    y = np.zeros(len(s), int)
    d = draw_delays(0, len(s), 50, "steady")
    o = raise_on_false_alarms(CAL, s, y, d, 0, info=True)
    assert o["checked"].sum() == 0
    assert np.all(o["thr"] <= FIXED + 1e-12)


def test_labels_are_never_used_before_they_arrive():
    s, y = stream_with_block(new_normal=True)
    d = np.full(len(s), len(s) + 10)               # no label ever comes back
    prop = rolling_threshold(CAL, s, 500)
    assert np.allclose(every_label_returns(CAL, s, y, d, info=True)["thr"],
                       np.quantile(CAL[-500:], Q))
    assert np.allclose(raise_on_false_alarms(CAL, s, y, d, None, info=True)["thr"],
                       np.minimum(prop, FIXED))
    assert np.allclose(rollback_on_incidents(CAL, s, y, d, None, info=True)["thr"], prop)


def test_no_ticket_as_normal_absorbs_a_block_the_false_alarm_rule_keeps():
    s, y = stream_with_block(new_normal=False)
    d = draw_delays(0, len(s), 50, "steady")
    naive = silence_as_normal(CAL, s, y, d, 5, wait=50)
    keep = raise_on_false_alarms(CAL, s, y, d, 5)
    assert naive[2500:3000].mean() < 0.2           # took the block in as normal
    assert keep[2500:3000].mean() > 0.7            # still alarming late in the block


def test_false_alarm_rule_follows_a_real_new_normal():
    s, y = stream_with_block(new_normal=True)
    d = draw_delays(0, len(s), 50, "steady")
    a = raise_on_false_alarms(CAL, s, y, d, 5)
    normal = y == 0
    assert a[2500:3000][normal[2500:3000]].mean() < 0.05
    assert a[3500:][normal[3500:]].mean() < 0.05   # and comes back down after it


def test_capacity_limits_checks():
    s, y = stream_with_block(new_normal=True)
    d = draw_delays(0, len(s), 50, "steady")
    for cap in (1, 2, 5):
        o = raise_on_false_alarms(CAL, s, y, d, cap, info=True)
        assert o["checked"].sum() <= cap / 100 * len(s) + cap
