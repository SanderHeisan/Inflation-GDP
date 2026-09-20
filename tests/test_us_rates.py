"""
The rate channel: the market-implied policy path, the point-in-time
sensitivity fit, where the drag lands, and how the projection and the
vintage builder carry it.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from usmodel import gdp as gdp_mod, rates
from usmodel.data_bundle import make_demo_bundle
from usbacktest.btconfig import USVintageConfig
from usbacktest.vintage import build_vintage, published_indicators


def _level_from_qoq(qoq_pct, index, start=15000.0):
    return pd.Series(start * np.cumprod(1 + np.asarray(qoq_pct) / 100),
                     index=index)


@pytest.fixture
def policy_and_gdp():
    """A policy rate that wanders, and QoQ growth that carries a planted
    drag of -0.15pp per 1pp of hikes over the (2, 6) window."""
    rng = np.random.default_rng(3)
    quarters = pd.period_range("1958Q1", "2026Q2", freq="Q")
    ff = pd.Series(np.clip(4 + np.cumsum(rng.normal(0, 0.35, len(quarters))),
                           0.1, 15), index=quarters)
    x = (ff.shift(2) - ff.shift(6)).fillna(0.0)
    qoq = 0.7 - 0.15 * x + rng.normal(0, 0.25, len(quarters))
    months = pd.period_range("1958-01", "2026-06", freq="M")
    ff_m = pd.Series(np.repeat(ff.to_numpy(), 3), index=months)
    return ff, ff_m, _level_from_qoq(qoq, quarters)


def test_expected_path_blends_dated_steps_by_calendar_day():
    obs = pd.Series([3.63, 3.63, 3.63],
                    index=pd.period_range("2026-07", "2026-09", freq="M"))
    steps = {"2026-09-18": 3.88, "2026-10-28": 4.03}
    path = rates.expected_policy_path(obs, steps, pd.Period("2027-01", "M"))
    assert path[pd.Period("2026-08", "M")] == 3.63
    # 17 days at the observed 3.63, 13 days at 3.88
    assert path[pd.Period("2026-09", "M")] == pytest.approx(
        (17 * 3.63 + 13 * 3.88) / 30, abs=1e-9)
    assert path[pd.Period("2026-10", "M")] == pytest.approx(
        (27 * 3.88 + 4 * 4.03) / 31, abs=1e-9)
    assert path[pd.Period("2026-11", "M")] == 4.03
    assert path[pd.Period("2027-01", "M")] == 4.03          # flat after
    flat = rates.expected_policy_path(obs, None, pd.Period("2027-01", "M"))
    assert (flat == 3.63).all() and flat.index[-1] == pd.Period("2027-01", "M")


def test_sensitivity_fit_recovers_the_planted_drag(policy_and_gdp):
    ff, _, level = policy_and_gdp
    fit = rates.fit_rate_sensitivity(level, ff, override=None)
    assert fit is not None
    assert fit["beta"] == pytest.approx(-0.15, abs=0.04)
    assert fit["corr"] < -0.3
    assert fit["fit_start"] == "1960Q1"


def test_sensitivity_sign_is_imposed_and_size_estimated(policy_and_gdp):
    """A sample where hikes 'add' to growth hands back zero, not a
    stimulus, and the raw estimate is kept so the run can say so."""
    ff, _, level = policy_and_gdp
    qoq = (level.pct_change() * 100).dropna()
    x = (ff.shift(2) - ff.shift(6)).reindex(qoq.index).fillna(0.0)
    flipped = _level_from_qoq(qoq + 0.30 * x, qoq.index)
    fit = rates.fit_rate_sensitivity(flipped, ff, override=None)
    assert fit["beta"] == 0.0 and fit["beta_raw"] > 0.05
    pinned = rates.fit_rate_sensitivity(level, ff, override=-0.20)
    assert pinned["beta"] == -0.20 and pinned["overridden"]
    assert rates.fit_rate_sensitivity(level.tail(30), ff) is None


def test_drag_lands_two_to_five_quarters_after_a_hike():
    quarters = pd.period_range("2020Q1", "2025Q4", freq="Q")
    ff = pd.Series(3.0, index=quarters)
    ff[ff.index >= pd.Period("2022Q1", "Q")] = 4.0       # a 1pp step in 2022Q1
    targets = pd.period_range("2022Q1", "2023Q4", freq="Q")
    d = rates.rate_drag(ff, targets, beta=-0.10) * 100
    assert d[pd.Period("2022Q2", "Q")] == 0.0
    for q in ("2022Q3", "2022Q4", "2023Q1", "2023Q2"):
        assert d[pd.Period(q, "Q")] == pytest.approx(-0.10)
    assert d[pd.Period("2023Q3", "Q")] == 0.0
    # a path that ends early is held flat, not dropped
    late = rates.rate_drag(ff[ff.index <= "2022Q2"],
                           pd.period_range("2023Q1", "2024Q4", freq="Q"), -0.10) * 100
    assert late[pd.Period("2023Q2", "Q")] == pytest.approx(-0.10)
    assert late[pd.Period("2024Q4", "Q")] == 0.0


def test_projection_applies_the_drag_only_past_the_nowcast_quarter():
    quarters = pd.period_range("2015Q1", "2026Q2", freq="Q")
    level = _level_from_qoq([0.6] * len(quarters), quarters)
    last = quarters[-1]
    drag = pd.Series({last + 1: -0.005, last + 2: -0.002, last + 3: -0.002})
    ind = {"fitted_qoq_pct": 0.9, "fitted_trend_qoq": 0.006,
           "rate_drag_qoq": drag, "pace_mode": "trailing_median"}
    out = gdp_mod.project_gdp(level, ind, horizon_quarters=5)
    assert out.loc[last + 1, "qoq_pct"] == pytest.approx(0.9)      # nowcast untouched
    assert out.loc[last + 2, "qoq_pct"] == pytest.approx(0.4)      # 0.6 - 0.2
    assert out.loc[last + 3, "qoq_pct"] == pytest.approx(0.4)
    assert out.loc[last + 4, "qoq_pct"] == pytest.approx(0.6)      # no entry: trend
    plain = gdp_mod.project_gdp(level, {k: v for k, v in ind.items()
                                        if k != "rate_drag_qoq"}, 5)
    assert plain.loc[last + 2, "qoq_pct"] == pytest.approx(0.6)


def test_live_runs_read_the_rate_series_for_the_month_in_progress():
    months = pd.period_range("2024-01", "2026-09", freq="M")
    bundle = SimpleNamespace(indicators={
        "fed_funds": pd.Series(3.6, index=months),
        "payrolls": pd.Series(150000.0, index=months)})
    asof = pd.Timestamp("2026-09-18")
    live = published_indicators(bundle, asof, USVintageConfig(live_partial_month=True))
    bt = published_indicators(bundle, asof, USVintageConfig(live_partial_month=False))
    assert live["fed_funds"].index[-1] == pd.Period("2026-09", "M")
    assert bt["fed_funds"].index[-1] == pd.Period("2026-08", "M")
    assert live["payrolls"].index[-1] == pd.Period("2026-08", "M")   # 8-day lag: Aug is out, Sep is not


def test_vintage_carries_the_drag_and_the_switch_removes_it(policy_and_gdp):
    ff, ff_m, level = policy_and_gdp
    b = make_demo_bundle()
    b.gdp_long = level
    b.indicators["fed_funds"] = ff_m
    asof = pd.Timestamp("2025-06-30")
    on = build_vintage(b, asof, USVintageConfig(revision_mode="none",
                                                rate_channel=True))
    off = build_vintage(b, asof, USVintageConfig(revision_mode="none",
                                                 rate_channel=False))
    assert "rate_drag_qoq" in on.indicators
    assert "rate_drag_qoq" not in off.indicators and np.isnan(off.diagnostics["rate_beta"])
    assert on.diagnostics["rate_beta"] == pytest.approx(-0.15, abs=0.04)
    drag = on.indicators["rate_drag_qoq"]
    assert drag.index[0] == on.last_gdp_quarter + 1 and len(drag) == 12
    # the drag is what moves the projection past the nowcast quarter
    g_on = gdp_mod.project_gdp(on.gdp_level, on.indicators, 5)
    g_off = gdp_mod.project_gdp(off.gdp_level, off.indicators, 5)
    q2 = on.last_gdp_quarter + 2
    assert g_on.loc[q2, "qoq_pct"] - g_off.loc[q2, "qoq_pct"] == pytest.approx(
        float(drag[q2]) * 100, abs=1e-9)
    st = on.diagnostics["rate_state"]
    assert st["path_source"] == "flat at the last observation"
    assert len(st["drag"]) == 12 and st["fit"]["n"] >= 80
