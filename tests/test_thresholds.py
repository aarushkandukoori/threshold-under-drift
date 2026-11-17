import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from thresholds import (fixed, rolling_quantile, rolling_confirmed_normal, scs_style,
                        macs_style, policies)
from metrics import time_to_recover, false_alarm_rate

rng = np.random.default_rng(0)
CAL = rng.normal(size=1000)


def test_every_rule_is_past_only():
    """Changing scores after t=3000 must not change any alarm before t=3000."""
    s = rng.normal(size=5000)
    y = (rng.random(5000) < 0.02).astype(int)
    s2 = s.copy()
    s2[3000:] += 10.0
    a, b = policies(CAL, s, y), policies(CAL, s2, y)
    for name in a:
        assert np.array_equal(a[name][:3000], b[name][:3000]), name


def test_fixed_hits_budget_on_iid_scores():
    s = rng.normal(size=200_000)
    assert abs(fixed(CAL, s).mean() - 0.025) < 0.01


def test_rolling_recovers_after_a_step():
    s = np.r_[rng.normal(size=2000), rng.normal(size=3000) + 3.0]
    y = np.zeros(len(s), int)
    a = rolling_quantile(CAL, s, 200)
    assert false_alarm_rate(fixed(CAL, s), y, slice(2500, 5000)) > 0.5
    assert false_alarm_rate(a, y, slice(2500, 5000)) < 0.05


def test_scs_and_macs_recover_after_a_step():
    s = np.r_[rng.normal(size=2000), rng.normal(size=3000) + 3.0]
    y = np.zeros(len(s), int)
    for rule in (scs_style, macs_style):
        assert false_alarm_rate(rule(CAL, s), y, slice(2500, 5000)) < 0.05, rule.__name__


def test_confirmed_normal_does_not_absorb_a_labeled_block():
    s = np.r_[rng.normal(size=2000), rng.normal(size=1000) + 3.0, rng.normal(size=1000)]
    y = np.zeros(len(s), int)
    y[2000:3000] = 1
    a = rolling_confirmed_normal(CAL, s, y, 500, 50)
    assert a[2500:3000].mean() > 0.75          # still alarming late in the block (fixed: ~0.85)
    assert rolling_quantile(CAL, s, 500)[2500:3000].mean() < 0.1  # plain rolling absorbed it


def test_time_to_recover():
    alarms = np.zeros(3000, bool)
    alarms[1000:1300] = True                    # 300 points of alarms right after the change
    y = np.zeros(3000, int)
    r = time_to_recover(alarms, y, 1000)
    assert 300 <= r <= 500
    assert np.isnan(time_to_recover(np.ones(3000, bool), y, 1000))
