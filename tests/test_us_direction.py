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
    assert hot_calls.iloc[0]["grade"] == "single"      # no consumer vote given


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


# ---------------------------------------------------------------------------
# The consumer's vote and the consumer state
# ---------------------------------------------------------------------------

def _pce_from_qoq(qoq_pct, start="2005-01"):
    """Monthly real PCE whose quarterly means grow by the given QoQ path."""
    months = pd.period_range(start, periods=3 * len(qoq_pct), freq="M")
    q_level = 100.0 * np.cumprod(1 + np.asarray(qoq_pct) / 100)
    return pd.Series(np.repeat(q_level, 3), index=months)


def test_pce_vote_points_toward_the_consumer_median(ar1_level):
    last = ar1_level.index[-1]
    hot = _pce_from_qoq([0.6] * 40 + [2.0])          # last quarter ran hot
    cold = _pce_from_qoq([0.6] * 40 + [-0.5])        # last quarter ran cold
    # align the synthetic PCE so its last quarter is the last GDP quarter
    for s in (hot, cold):
        s.index = pd.period_range(end=last.asfreq("M", "end"), periods=len(s),
                                  freq="M")
    assert gdir.pce_reversal_vote(hot, last) < 0
    assert gdir.pce_reversal_vote(cold, last) > 0
    assert gdir.pce_reversal_vote(None, last) is None


def test_two_votes_grade_by_agreement(ar1_level):
    q = (ar1_level.pct_change() * 100).dropna()
    hot_gdp = _level_from_qoq(pd.concat([q, pd.Series(
        [2.5], index=[q.index[-1] + 1])]))
    last = hot_gdp.index[-1]
    pce_hot = _pce_from_qoq([0.6] * 40 + [2.0])
    pce_cold = _pce_from_qoq([0.6] * 40 + [-0.5])
    for s in (pce_hot, pce_cold):
        s.index = pd.period_range(end=last.asfreq("M", "end"), periods=len(s),
                                  freq="M")
    agree = gdir.qoq_direction_calls(hot_gdp, real_pce=pce_hot)
    split = gdir.qoq_direction_calls(hot_gdp, real_pce=pce_cold)
    single = gdir.qoq_direction_calls(hot_gdp)
    assert agree.iloc[0]["grade"] == "call"
    assert agree.iloc[0]["direction"] == "decelerating"
    assert agree.iloc[0]["backtest_hit"] == gdir.HIT_AGREE
    assert split.iloc[0]["grade"] == "split"
    assert split.iloc[0]["backtest_hit"] == gdir.HIT_SPLIT
    assert single.iloc[0]["grade"] == "single"
    assert np.isnan(single.iloc[0]["pce_vote_pp"])


def test_consumer_state_flags_a_stretched_consumer(ar1_level):
    last = ar1_level.index[-1]
    calm = _pce_from_qoq([0.6] * 44 + [0.6] * 4)
    boom = _pce_from_qoq([0.6] * 44 + [1.8] * 4)     # last year far above norm
    for s in (calm, boom):
        s.index = pd.period_range(end=last.asfreq("M", "end"), periods=len(s),
                                  freq="M")
    calm_state = gdir.consumer_state({"real_pce": calm}, last)["real_pce"]
    boom_state = gdir.consumer_state({"real_pce": boom}, last)["real_pce"]
    assert not calm_state["stretched"]
    assert boom_state["stretched"]
    assert boom_state["pctl_10y"] >= gdir.STRETCH_PCTL
    assert gdir.consumer_state({}, last) == {}
