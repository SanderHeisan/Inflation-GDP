"""
Point-in-time discipline for the US backtest.

These are the tests that make the accuracy numbers quotable: if any of them
fail, the backtest is reading the future and every hit rate in the README is
meaningless.
"""
import numpy as np
import pandas as pd
import pytest

from usbacktest import vintage as vt
from usbacktest.btconfig import USVintageConfig
from usbacktest.engine import model_quad_table, run_backtest
from usmodel import nowcast
from usmodel.data_bundle import USDataBundle


# ---------------------------------------------------------------------------
# A small synthetic bundle: fast, deterministic, and shaped like the real one.
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle():
    months = pd.period_range("2005-01", "2026-07", freq="M")
    quarters = pd.period_range("2005Q1", "2026Q2", freq="Q")
    rng = np.random.default_rng(7)
    t = np.arange(len(months), dtype=float)

    cpi = pd.Series(190.0 * np.cumprod(1 + 0.002 + rng.normal(0, 4e-4,
                                                              len(months))),
                    index=months)
    gdp = pd.Series(15000.0 * np.cumprod(
        1 + 0.005 + rng.normal(0, 0.004, len(quarters))), index=quarters)
    wti = pd.Series(70 + 10 * np.sin(t / 9), index=months)
    dollar = pd.Series(100 + 5 * np.sin(t / 17), index=months)
    wages = pd.Series(25 * np.cumprod(1 + 0.0028 + rng.normal(0, 3e-4,
                                                              len(months))),
                      index=months)
    rent = pd.Series(1000 * np.cumprod(1 + 0.003 + rng.normal(0, 5e-4,
                                                              len(months))),
                     index=months)
    ind = {
        "payrolls": pd.Series(140000 * np.cumprod(
            1 + 0.0012 + rng.normal(0, 4e-4, len(months))), index=months),
        "indpro": pd.Series(100 + np.cumsum(rng.normal(0, 0.2, len(months))),
                            index=months),
        "retail": pd.Series(500000 * np.cumprod(
            1 + 0.003 + rng.normal(0, 1e-3, len(months))), index=months),
        "claims": pd.Series(250000 + 20000 * np.sin(t / 11), index=months),
        "cfnai": pd.Series(rng.normal(0, 0.4, len(months)), index=months),
        "sentiment": pd.Series(85 + 10 * np.sin(t / 13), index=months),
        "yield_curve": pd.Series(1.2 + np.sin(t / 21), index=months),
        "hours": pd.Series(41 + 0.4 * np.sin(t / 15), index=months),
    }
    return USDataBundle(cpi=cpi, gdp=gdp, wti=wti, dollar=dollar,
                        market_rent=rent, wages=wages,
                        cpi_shelter=cpi * 1.1, cpi_supercore=cpi * 1.2,
                        cpi_food=cpi * 0.9, indicators=ind, source="test")


# ---------------------------------------------------------------------------
# Publication lags
# ---------------------------------------------------------------------------

def test_truncation_respects_publication_lags(bundle):
    """On 2024-02-05 the January CPI has not printed (BLS releases it around
    the 13th) and Q4 GDP has (BEA's advance estimate lands ~Jan 30)."""
    v = vt.build_vintage(bundle, "2024-02-05")
    assert v.last_cpi_month == pd.Period("2023-12", "M")
    assert v.last_gdp_quarter == pd.Period("2023Q4", "Q")


def test_cpi_appears_exactly_on_its_release_day(bundle):
    """January CPI is absent at lag-1 days and present at the lag."""
    cfg = USVintageConfig()
    release = vt.first_release_date(pd.Period("2024-01", "M"),
                                    cfg.cpi_pub_lag_days)
    before = vt.build_vintage(bundle, release - pd.Timedelta(days=1), cfg)
    after = vt.build_vintage(bundle, release, cfg)
    assert before.last_cpi_month == pd.Period("2023-12", "M")
    assert after.last_cpi_month == pd.Period("2024-01", "M")


def test_indicator_lags_differ_by_series(bundle):
    """Payrolls (employment report, ~8 days) are published for a month while
    CFNAI (~26 days) for the same month is not."""
    panel = vt.published_indicators(bundle, pd.Timestamp("2024-02-12"),
                                    USVintageConfig())
    assert pd.Period("2024-01", "M") in panel["payrolls"].index
    assert pd.Period("2024-01", "M") not in panel["cfnai"].index


# ---------------------------------------------------------------------------
# The leakage firewall
# ---------------------------------------------------------------------------

def test_mutating_the_future_cannot_change_a_prediction(bundle):
    """The decisive test. Corrupt every series *after* the as-of date; the
    vintage, the assumptions and the resulting quad table must be
    bit-identical. If this fails, the backtest is reading the future."""
    asof = pd.Timestamp("2023-06-30")
    cfg = USVintageConfig()
    before = model_quad_table(vt.build_vintage(bundle, asof, cfg), 4)

    cut_m, cut_q = pd.Period("2023-06", "M"), pd.Period("2023Q2", "Q")
    mutated = USDataBundle(
        cpi=bundle.cpi.mask(bundle.cpi.index > cut_m, 999.0),
        gdp=bundle.gdp.mask(bundle.gdp.index > cut_q, 99999.0),
        wti=bundle.wti.mask(bundle.wti.index > cut_m, 500.0),
        dollar=bundle.dollar.mask(bundle.dollar.index > cut_m, 300.0),
        market_rent=bundle.market_rent.mask(
            bundle.market_rent.index > cut_m, 9999.0),
        wages=bundle.wages.mask(bundle.wages.index > cut_m, 99.0),
        cpi_shelter=bundle.cpi_shelter.mask(
            bundle.cpi_shelter.index > cut_m, 999.0),
        cpi_supercore=bundle.cpi_supercore.mask(
            bundle.cpi_supercore.index > cut_m, 999.0),
        cpi_food=bundle.cpi_food.mask(bundle.cpi_food.index > cut_m, 999.0),
        indicators={k: s.mask(s.index > cut_m, s.iloc[0] * 5)
                    for k, s in bundle.indicators.items()},
        source="mutated")
    after = model_quad_table(vt.build_vintage(mutated, asof, cfg), 4)

    pd.testing.assert_frame_equal(before, after)


def test_forward_paths_never_contain_unobserved_prices(bundle):
    """Market paths are observed-then-carried: months already published at
    the as-of date carry their real values, every month beyond that repeats
    the last observation, and none of them is the realized future price."""
    asof = pd.Timestamp("2023-06-30")
    cfg = USVintageConfig()
    v = vt.build_vintage(bundle, asof, cfg)
    observed = vt.truncate(bundle.wti, asof, cfg.market_pub_lag_days)
    last_obs_month = observed.index[-1]
    path = {pd.Period(p, "M"): val
            for p, val in v.assumptions["wti_forward"].items()}

    # published months carry their true values
    for m, val in path.items():
        if m <= last_obs_month:
            assert val == pytest.approx(float(bundle.wti[m]))

    # unpublished months are all flat at the last observation...
    tail = {m: val for m, val in path.items() if m > last_obs_month}
    assert tail, "expected a carried tail"
    assert set(np.round(list(tail.values()), 9)) == {
        round(float(observed.iloc[-1]), 9)}
    # ...and the realized future actually moved, so a leak would have shown
    assert not np.allclose([bundle.wti[m] for m in tail],
                           float(observed.iloc[-1]))


def test_nowcast_trains_only_on_published_quarters(bundle):
    """The regression may not be fitted on a quarter BEA has not released."""
    asof = pd.Timestamp("2024-02-05")
    cfg = USVintageConfig()
    v = vt.build_vintage(bundle, asof, cfg)
    fit = nowcast.fit_nowcast(v.gdp_level, v.indicator_panel,
                              v.last_gdp_quarter + 1)
    assert fit is not None
    # training rows are QoQ observations, so one fewer than published levels
    assert fit["n_train"] <= len(v.gdp_level) - 1
    assert v.gdp_level.index[-1] == pd.Period("2023Q4", "Q")


def test_nowcast_partial_quarter_k_is_bounded(bundle):
    """k counts published months of the target quarter and must stay in
    0..2 -- by the time all three are out, BEA has released the quarter."""
    for asof in pd.date_range("2022-01-31", "2024-12-31", freq="ME"):
        v = vt.build_vintage(bundle, asof)
        assert v.diagnostics["nowcast_k"] in (-1, 0, 1, 2)


# ---------------------------------------------------------------------------
# Revision model
# ---------------------------------------------------------------------------

def test_revision_draw_is_stable_across_processes():
    """Same quarter, same seed -> same draw, so a quarter is mis-measured
    identically at every as-of date (builtin hash() would not be)."""
    a = vt.revision_draw(pd.Period("2019Q3", "Q"), 0, 0.35)
    b = vt.revision_draw(pd.Period("2019Q3", "Q"), 0, 0.35)
    assert a == b
    assert a != vt.revision_draw(pd.Period("2019Q4", "Q"), 0, 0.35)


def test_revision_noise_leaves_mature_quarters_alone(bundle):
    """Quarters older than the maturity window are treated as final."""
    cfg = USVintageConfig(revision_mode="noise")
    trunc = vt.truncate(bundle.gdp, pd.Timestamp("2024-02-05"),
                        cfg.gdp_pub_lag_days)
    noised = vt.apply_revision_noise(trunc, cfg)
    mature = trunc.index[:-cfg.revision_maturity_quarters]
    pd.testing.assert_series_equal(trunc.loc[mature], noised.loc[mature])
    assert not np.allclose(trunc.iloc[-4:], noised.iloc[-4:])


def test_revision_mode_none_returns_final_data(bundle):
    cfg = USVintageConfig(revision_mode="none")
    v = vt.build_vintage(bundle, "2024-02-05", cfg)
    expected = vt.truncate(bundle.gdp, pd.Timestamp("2024-02-05"),
                           cfg.gdp_pub_lag_days)
    pd.testing.assert_series_equal(v.gdp_level, expected)


# ---------------------------------------------------------------------------
# Backtest plumbing
# ---------------------------------------------------------------------------

def test_backtest_is_deterministic(bundle):
    cfg = USVintageConfig()
    a = run_backtest(bundle, "2022-01", "2022-12", max_horizon=2, cfg=cfg)
    b = run_backtest(bundle, "2022-01", "2022-12", max_horizon=2, cfg=cfg)
    pd.testing.assert_frame_equal(a, b)


def test_backtest_never_targets_an_unpublished_horizon(bundle):
    """Every row's target quarter must sit at the recorded horizon from the
    as-of quarter, and the vintage must be behind the as-of date."""
    preds = run_backtest(bundle, "2022-01", "2023-12", max_horizon=4)
    tq = pd.PeriodIndex(preds["target_quarter"], freq="Q")
    aq = pd.PeriodIndex(preds["asof_quarter"], freq="Q")
    assert ((tq - aq).map(lambda x: x.n) == preds["horizon"]).all()
    last_gdp = pd.PeriodIndex(preds["last_gdp_quarter"], freq="Q")
    assert (last_gdp <= aq).all()
