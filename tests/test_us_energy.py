"""
The gasoline block: retail pump prices drive it when supplied, with BLS's
seasonal added back; the WTI rule is the fallback; a live run sees the month
in progress.
"""
import numpy as np
import pandas as pd
import pytest

from usmodel import config, inflation
from usbacktest import vintage as vt
from usbacktest.btconfig import USVintageConfig


def _flat_cpi(n=40, start="2023-01"):
    return pd.Series(300.0, index=pd.period_range(start, periods=n, freq="M"))


def _gas_only_weights():
    return {**{k: 0 for k in config.CPI_WEIGHTS}, "gasoline": 34}


def test_pump_price_drives_the_gasoline_block():
    """A +5% pump month with a +1pp seasonal must show up as a +6% gasoline
    block (slope 1), i.e. 0.034 * 6 = ~0.20pp of headline."""
    cpi = _flat_cpi()
    h = pd.period_range(cpi.index[-1] + 1, periods=3, freq="M")
    assum = {"wti_recent": 80.0, "wti_forward": {},
             "gasoline_pump_recent": 4.00,
             "gasoline_pump_path": {str(h[0]): 4.20},          # +5% in month 1
             "gasoline_seasonal": {h[0].month: 1.0},
             "gasoline_pump_slope": 1.0,
             "wage_growth_pct": 0.0, "food_pipeline_yoy": 0.0,
             "market_rent_yoy_recent": 0.0, "core_goods_baseline_yoy": 0.0}
    proj = inflation.build_cpi_projection(cpi, 3, assum,
                                          weights=_gas_only_weights())
    assert proj.loc[h[0], "mom_pct"] == pytest.approx(6.0, abs=0.05)
    # carried flat afterwards: only the seasonal (none set) -> ~0
    assert proj.loc[h[1], "mom_pct"] == pytest.approx(0.0, abs=0.05)


def test_wti_rule_is_the_fallback_without_pump_data():
    cpi = _flat_cpi()
    h = pd.period_range(cpi.index[-1] + 1, periods=4, freq="M")
    strip = {str(p): 70.0 for p in h}
    strip[str(h[1])] = 80.0
    for p in h[2:]:
        strip[str(p)] = 80.0
    assum = {"wti_recent": 70.0, "wti_forward": strip,
             "wage_growth_pct": 0.0, "food_pipeline_yoy": 0.0,
             "market_rent_yoy_recent": 0.0, "core_goods_baseline_yoy": 0.0}
    proj = inflation.build_cpi_projection(cpi, 4, assum,
                                          weights=_gas_only_weights())
    total = proj.loc[h[1], "mom_pct"] + proj.loc[h[2], "mom_pct"]
    assert total == pytest.approx(0.30, abs=0.03)     # 3 bps x $10


def test_seasonal_estimate_recovers_a_planted_factor():
    """SA gasoline CPI = pump price change + a fixed December bump; the
    estimator must read that bump and a slope near 1."""
    months = pd.period_range("2008-01", "2025-12", freq="M")
    rng = np.random.default_rng(2)
    pump_mom = rng.normal(0, 3, len(months))
    pump = pd.Series(3.0 * np.cumprod(1 + pump_mom / 100), index=months)
    gas_mom = pump_mom + np.where(months.month == 12, 3.0, 0.0)
    gas = pd.Series(200.0 * np.cumprod(1 + gas_mom / 100), index=months)
    seasonal, slope = vt.estimate_gasoline_seasonal(gas, pump)
    assert seasonal[12] == pytest.approx(3.0, abs=0.4)
    assert abs(seasonal[6]) < 0.4
    assert slope == pytest.approx(1.0, abs=0.1)


def test_live_run_sees_the_month_in_progress():
    """With live_partial_month the truncation admits the current month's
    partial mean; without it the month waits until it is over."""
    s = pd.Series([1.0, 2.0, 3.0], index=pd.period_range("2026-07", periods=3,
                                                          freq="M"))
    asof = pd.Timestamp("2026-09-16")
    assert vt.truncate(s, asof, 0).index[-1] == pd.Period("2026-08", "M")
    assert vt.truncate(s, asof, -31).index[-1] == pd.Period("2026-09", "M")
    cfg_live = USVintageConfig(revision_mode="none", live_partial_month=True)
    assert cfg_live.live_partial_month
    assert not USVintageConfig().live_partial_month
