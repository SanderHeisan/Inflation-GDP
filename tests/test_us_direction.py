"""
Growth direction on a QoQ basis: the reversal call, its robustness to the
2020 crash/rebound pair, and the direction backtest's abstention rule.
"""
import numpy as np
import pandas as pd
import pytest

from usmodel import growth_direction as gdir
from usbacktest import direction


def _level_from_qoq(qoq_pct: pd.Series, start=15000.0) -> pd.Series:
    return pd.Series(start * np.cumprod(1 + qoq_pct.to_numpy() / 100),
                     index=qoq_pct.index)


@pytest.fixture
def ar1_level():
    """QoQ growth that is a clean AR(1): 0.6 + 0.3 * previous + small noise."""
    rng = np.random.default_rng(11)
    quarters = pd.period_range("2000Q1", "2026Q1", freq="Q")
    q = [0.6]
    for _ in range(len(quarters) - 1):
        q.append(0.6 * 0.7 + 0.3 * q[-1] + rng.normal(0, 0.15))
    return _level_from_qoq(pd.Series(q, index=quarters))


def test_ar1_fit_recovers_the_planted_persistence(ar1_level):
    a, b = gdir.fit_ar1_qoq(ar1_level, window=200)
    assert b == pytest.approx(0.3, abs=0.12)
    assert a / (1 - b) == pytest.approx(0.6, abs=0.1)     # unconditional mean


def test_ar1_fit_is_robust_to_a_crash_and_rebound(ar1_level):
    """Plant 2020Q2/Q3 (-9%, +7.5%) into the window; the winsorized fit must
    barely move, where a raw OLS fit flips the slope negative."""
    q = (ar1_level.pct_change() * 100).dropna()
    q[pd.Period("2020Q2", "Q")] = -9.0
    q[pd.Period("2020Q3", "Q")] = 7.5
    shocked = _level_from_qoq(q)
    a0, b0 = gdir.fit_ar1_qoq(ar1_level, window=200)
    a1, b1 = gdir.fit_ar1_qoq(shocked, window=200)
    raw = np.polyfit(q.to_numpy()[:-1], q.to_numpy()[1:], 1)[0]
    assert raw < 0                          # what an unguarded fit would do
    assert b1 > 0.1                         # guarded fit keeps persistence
    assert abs(b1 - 0.3) < 0.2
    assert abs(a1 / (1 - b1) - a0 / (1 - b0)) < 0.15   # AR mean barely moves


def test_reversal_call_points_toward_the_mean(ar1_level):
    """A quarter that printed far above the AR mean must be followed by a
    'decelerating' call, and one far below by 'accelerating'."""
    q = (ar1_level.pct_change() * 100).dropna()
    hot = _level_from_qoq(pd.concat([q, pd.Series(
        [2.5], index=[q.index[-1] + 1])]))
    cold = _level_from_qoq(pd.concat([q, pd.Series(
        [-1.0], index=[q.index[-1] + 1])]))
    hot_calls = gdir.qoq_direction_calls(hot)
    cold_calls = gdir.qoq_direction_calls(cold)
    assert hot_calls.iloc[0]["direction"] == "decelerating"
    assert cold_calls.iloc[0]["direction"] == "accelerating"
    assert hot_calls.iloc[0]["grade"] == "call"


def test_zigzag_lean_is_the_opposite_of_the_call(ar1_level):
    """The quarter-after lean alternates against the current-quarter call
    when the last print sits on the same side of trend as the AR mean."""
    q = (ar1_level.pct_change() * 100).dropna()
    hot = _level_from_qoq(pd.concat([q, pd.Series(
        [2.5], index=[q.index[-1] + 1])]))
    calls = gdir.qoq_direction_calls(hot, trend_qoq_pct=0.6)
    assert len(calls) == 2
    assert calls.iloc[1]["grade"] == "lean"
    assert calls.iloc[0]["direction"] != calls.iloc[1]["direction"]


def test_calls_cover_exactly_two_quarters_then_abstain(ar1_level):
    """No sequential-growth call is manufactured past q+2."""
    calls = gdir.qoq_direction_calls(ar1_level)
    assert list(calls.index) == [ar1_level.index[-1] + 1,
                                 ar1_level.index[-1] + 2]


def test_too_little_history_returns_nothing(ar1_level):
    assert gdir.fit_ar1_qoq(ar1_level.tail(10)) is None
    assert gdir.qoq_direction_calls(ar1_level.tail(10)).empty


def test_conviction_buckets_cover_the_line():
    assert direction.bucket_label(0.0) == "<0.10pp"
    assert direction.bucket_label(0.10) == "0.10-0.25pp"
    assert direction.bucket_label(0.49) == "0.25-0.50pp"
    assert direction.bucket_label(3.0) == ">0.50pp"


def test_summaries_exclude_abstentions_from_hit_rates():
    """An abstained row must count toward call share but never toward the
    hit rate."""
    df = pd.DataFrame({
        "asof": pd.Timestamp("2024-01-31"), "asof_quarter": "2024Q1",
        "horizon": [0, 1, 2, 2], "target_quarter": ["2024Q1", "2024Q2",
                                                   "2024Q3", "2024Q3"],
        "call": ["growth_qoq"] * 3 + ["growth_yoy"],
        "grade": ["call", "lean", "abstain", "call"],
        "made_call": [True, True, False, True],
        "pred_delta": [0.4, -0.2, np.nan, 0.3],
        "real_delta": [0.5, 0.1, 0.2, 0.1],
        "pred_dir": ["accelerating", "decelerating", "no call",
                     "accelerating"],
        "real_dir": ["accelerating"] * 4,
        "hit": [True, False, np.nan, True],
        "conviction_pp": [0.4, 0.2, np.nan, 0.3],
        "covid_target": [False] * 4,
    })
    by_h = direction.summarize_by_horizon(df)
    assert by_h.loc[("growth_qoq", 2), "call_share"] == 0.0
    assert by_h.loc[("growth_qoq", 2), "n_calls"] == 0
    assert np.isnan(by_h.loc[("growth_qoq", 2), "hit"])
    assert by_h.loc[("growth_qoq", 0), "hit"] == 1.0
    by_c = direction.summarize_by_conviction(df)
    assert by_c.loc[("growth_qoq", "ALL calls"), "n"] == 2
