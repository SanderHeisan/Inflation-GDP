"""
The point-in-time GDP nowcast: feature construction, partial quarters, and
the guard rails on the self-calibrated convergence path.
"""
import numpy as np
import pandas as pd
import pytest

from usmodel import config, gdp as gdp_mod, nowcast


@pytest.fixture
def indicators():
    months = pd.period_range("2005-01", "2026-06", freq="M")
    t = np.arange(len(months), dtype=float)
    rng = np.random.default_rng(3)
    return {
        "payrolls": pd.Series(140000 * np.cumprod(
            1 + 0.0012 + rng.normal(0, 3e-4, len(months))), index=months),
        "indpro": pd.Series(100 + 0.02 * t, index=months),
        "retail": pd.Series(500000 * np.cumprod(
            1 + 0.003 + rng.normal(0, 8e-4, len(months))), index=months),
        "claims": pd.Series(250000 + 15000 * np.sin(t / 11), index=months),
        "cfnai": pd.Series(np.sin(t / 7) * 0.3, index=months),
        "sentiment": pd.Series(85 + 8 * np.sin(t / 13), index=months),
        "yield_curve": pd.Series(1.0 + np.sin(t / 19), index=months),
        "hours": pd.Series(41 + 0.3 * np.sin(t / 15), index=months),
    }


@pytest.fixture
def gdp_level():
    quarters = pd.period_range("2005Q1", "2026Q1", freq="Q")
    rng = np.random.default_rng(5)
    return pd.Series(15000 * np.cumprod(
        1 + 0.005 + rng.normal(0, 0.003, len(quarters))), index=quarters)


def test_months_available_counts_the_binding_indicator(indicators):
    """k is set by the slowest indicator, so the design matrix is never
    ragged: drop two months from one series and k falls."""
    q = pd.Period("2026Q2", "Q")
    assert nowcast.months_available(indicators, q) == 3
    slowed = dict(indicators)
    slowed["cfnai"] = indicators["cfnai"][
        indicators["cfnai"].index < pd.Period("2026-05", "M")]
    assert nowcast.months_available(slowed, q) == 1


def test_features_for_partial_quarter_use_only_published_months(indicators):
    """With k=1, the payrolls feature must be built from month 1 of the
    target quarter -- not from the quarter's full average."""
    gdp_qoq = pd.Series(0.5, index=pd.period_range("2005Q1", "2026Q1",
                                                   freq="Q"))
    f = nowcast.quarter_features(indicators, pd.Period("2026Q1", "Q"), 1,
                                 gdp_qoq)
    pay = indicators["payrolls"]
    expected = (pay[pd.Period("2026-01", "M")]
                / pay[["2025-10", "2025-11", "2025-12"]].mean() - 1) * 100
    assert f["payrolls"] == pytest.approx(expected)


def test_k_zero_reads_one_quarter_back(indicators):
    """With nothing of the target quarter published, the features are the
    previous quarter's -- a genuine one-quarter-ahead prediction."""
    gdp_qoq = pd.Series(0.5, index=pd.period_range("2005Q1", "2026Q1",
                                                   freq="Q"))
    f0 = nowcast.quarter_features(indicators, pd.Period("2026Q2", "Q"), 0,
                                  gdp_qoq)
    f3 = nowcast.quarter_features(indicators, pd.Period("2026Q1", "Q"), 3,
                                  gdp_qoq)
    for name, _ in nowcast.FEATURE_SPEC:
        assert f0[name] == pytest.approx(f3[name])


def test_momentum_feature_is_the_previous_published_quarter(indicators,
                                                            gdp_level):
    gdp_qoq = (gdp_level.pct_change() * 100).dropna()
    target = pd.Period("2026Q2", "Q")
    f = nowcast.quarter_features(indicators, target, 1, gdp_qoq)
    assert f["momentum"] == pytest.approx(float(gdp_qoq[target - 1]))


def test_fit_returns_none_without_enough_history(indicators, gdp_level):
    assert nowcast.fit_nowcast(gdp_level.tail(10), indicators) is None
    assert nowcast.fit_nowcast(gdp_level, {}) is None


def test_fit_recovers_a_planted_relationship(indicators):
    """Build GDP so QoQ is a clean linear function of the payroll feature;
    the fit must then track it far better than the sample mean does."""
    quarters = pd.period_range("2005Q2", "2026Q1", freq="Q")
    gdp_qoq_seed = pd.Series(0.5, index=pd.period_range("2005Q1", "2026Q2",
                                                        freq="Q"))
    pay_feat = pd.Series(
        {q: nowcast.quarter_features(indicators, q, 3,
                                     gdp_qoq_seed)["payrolls"]
         for q in quarters})
    qoq = 0.2 + 1.5 * pay_feat
    level = pd.Series(15000.0, index=[pd.Period("2005Q1", "Q")])
    for q in quarters:
        level[q] = level.iloc[-1] * (1 + qoq[q] / 100)

    fit = nowcast.fit_nowcast(level, indicators, pd.Period("2026Q2", "Q"))
    assert fit is not None
    assert fit["k"] == 3
    assert fit["in_sample_mae"] < 0.5 * float(qoq.std())


def test_nowcast_qoq_prefers_the_fitted_estimate(gdp_level):
    """A supplied fitted nowcast is used directly, not re-blended with
    momentum (momentum is already one of its regressors)."""
    assert gdp_mod.nowcast_qoq(gdp_level, {"fitted_qoq_pct": 0.9}) == \
        pytest.approx(0.009)
    fallback = gdp_mod.nowcast_qoq(gdp_level, {})
    assert fallback == pytest.approx(
        float(gdp_level.pct_change().iloc[-2:].mean()))


def test_project_gdp_steps_to_trend_after_the_nowcast_quarter(gdp_level):
    """The shipped path is nowcast(q+1) then flat at trend -- deliberately a
    step, not a glide, because past the nowcast quarter QoQ is not
    forecastable and a moving path only adds noise to the known base."""
    out = gdp_mod.project_gdp(gdp_level, {"fitted_qoq_pct": 2.0,
                                          "fitted_trend_qoq": 0.001},
                              horizon_quarters=4)
    proj = out[out["projected"]]["qoq_pct"].to_numpy()
    assert proj[0] == pytest.approx(2.0, abs=1e-6)
    assert proj[1:] == pytest.approx(0.1, abs=1e-6)


def test_project_gdp_can_still_glide_when_asked(gdp_level):
    """The old geometric convergence stays reachable so --growth-variants can
    reproduce the measurement that it is worse."""
    out = gdp_mod.project_gdp(gdp_level, {"fitted_qoq_pct": 2.0,
                                          "fitted_trend_qoq": 0.001,
                                          "fitted_convergence": 0.5},
                              horizon_quarters=4)
    proj = out[out["projected"]]["qoq_pct"].to_numpy()
    assert proj[1] == pytest.approx(0.1 + (2.0 - 0.1) * 0.5, abs=1e-6)
    assert proj[2] == pytest.approx(0.1 + (2.0 - 0.1) * 0.25, abs=1e-6)


def test_trend_is_a_median_not_a_mean():
    """A crash-and-rebound pair like 2020Q2/Q3 must not move the trend. The
    mean of this series is dragged well away from its median; the estimator
    has to track the median."""
    quarters = pd.period_range("2005Q1", "2026Q1", freq="Q")
    qoq = pd.Series(0.005, index=quarters)
    qoq[pd.Period("2020Q2", "Q")] = -0.09        # crash
    qoq[pd.Period("2020Q3", "Q")] = 0.075        # rebound
    level = pd.Series(15000 * np.cumprod(1 + qoq), index=quarters)
    trend = gdp_mod.estimate_trend_qoq(level, window=len(quarters))
    assert trend == pytest.approx(0.005, abs=1e-6)
    assert abs(float(level.pct_change().dropna().mean()) - 0.005) > 1e-4


def test_trend_uses_a_trailing_window(gdp_level):
    """Only the recent window counts, so the constant tracks the data."""
    lvl = gdp_level.copy()
    short = gdp_mod.estimate_trend_qoq(lvl, window=8)
    recent = float(lvl.pct_change().dropna().tail(8).median())
    assert short == pytest.approx(min(max(recent, 0.0), 0.015))


def test_trend_is_clipped_to_a_sane_range():
    """A boom window must not extrapolate 6% quarterly growth forever."""
    quarters = pd.period_range("2010Q1", "2026Q1", freq="Q")
    level = pd.Series(15000 * np.cumprod(1 + pd.Series(0.06, index=quarters)),
                      index=quarters)
    assert gdp_mod.estimate_trend_qoq(level) == pytest.approx(0.015)


def test_estimate_convergence_clips_negative_persistence():
    """A crash-and-rebound window fits a negative AR(1), which would make
    the projection oscillate; the estimate must be clipped at zero."""
    quarters = pd.period_range("2005Q1", "2026Q1", freq="Q")
    qoq = pd.Series(0.005, index=quarters)
    qoq.iloc[::2] = -0.004                     # alternating -> rho < 0
    level = pd.Series(15000 * np.cumprod(1 + qoq), index=quarters)
    trend, rho = gdp_mod.estimate_convergence(level)
    assert rho == 0.0
    assert 0.0 <= trend <= 0.015


def test_estimate_convergence_needs_history(gdp_level):
    assert gdp_mod.estimate_trend_qoq(gdp_level.head(3)) is None


def test_convergence_falls_back_to_config(gdp_level):
    """With nothing supplied, project_gdp uses the config constants -- and
    the shipped GDP_CONVERGENCE is 0, so the path steps to trend."""
    assert config.GDP_CONVERGENCE == 0.0
    out = gdp_mod.project_gdp(gdp_level, {"fitted_qoq_pct": 1.0},
                              horizon_quarters=3)
    proj = out[out["projected"]]["qoq_pct"].to_numpy()
    assert proj[1] == pytest.approx(config.GDP_TREND_QOQ * 100, abs=1e-6)


# ---------------------------------------------------------------------------
# Direct multi-horizon fits (the machinery behind the growth-axis experiment)
# ---------------------------------------------------------------------------

def test_horizon_shifts_the_feature_window(indicators):
    """A horizon-2 row must read its features one quarter further back than a
    horizon-1 row for the same target -- that is what makes it a direct-h
    forecast rather than an iterated one."""
    gdp_qoq = pd.Series(0.5, index=pd.period_range("2005Q1", "2026Q1",
                                                   freq="Q"))
    target = pd.Period("2026Q1", "Q")
    f1 = nowcast.quarter_features(indicators, target, 3, gdp_qoq, horizon=1)
    f2 = nowcast.quarter_features(indicators, target, 3, gdp_qoq, horizon=2)
    prev = nowcast.quarter_features(indicators, target - 1, 3, gdp_qoq,
                                    horizon=1)
    for name, _ in nowcast.FEATURE_SPEC:
        assert f2[name] == pytest.approx(prev[name])
        assert f2[name] != pytest.approx(f1[name])


def test_months_available_ignores_series_the_fit_does_not_use(indicators):
    """A cached-but-unused slow series must not drag k down and cost the
    nowcast a month of data it really had."""
    q = pd.Period("2026Q2", "Q")
    assert nowcast.months_available(indicators, q) == 3
    with_slow = dict(indicators)
    with_slow["real_m2"] = indicators["cfnai"][
        indicators["cfnai"].index < pd.Period("2026-04", "M")]
    assert nowcast.months_available(with_slow, q) == 3          # not in spec
    assert nowcast.months_available(with_slow, q,
                                    nowcast.EXTENDED_SPEC) == 0


def test_extended_spec_is_reachable(indicators, gdp_level):
    """The leading panel stays wired up so the negative result stays
    reproducible, not just asserted in the README."""
    assert set(dict(nowcast.LEADING_SPEC)) <= set(
        dict(nowcast.EXTENDED_SPEC))
    assert nowcast.fit_nowcast(gdp_level, indicators,
                               spec=nowcast.EXTENDED_SPEC) is None  # no data
