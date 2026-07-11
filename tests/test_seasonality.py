"""Calendar-seasonality overlay: profile estimation and opt-in wiring."""
import numpy as np
import pandas as pd
import pytest

from quadmap import inflation


def _seasonal_cpi(jump_month=7, jump=0.005, months=180):
    idx = pd.period_range("2010-01", periods=months, freq="M")
    mom = np.full(months, 0.002)
    mom[idx.month == jump_month] += jump
    return pd.Series(100 * np.cumprod(1 + mom), index=idx)


def test_profile_recovers_known_seasonality():
    cpi = _seasonal_cpi(jump_month=7, jump=0.005)
    prof = inflation.seasonal_mom_profile(cpi)
    assert prof[7] == pytest.approx(0.005, abs=5e-4)
    # every typical month is centred at zero: no trend smuggled in
    for m in (1, 4, 10):
        assert abs(prof[m]) < 5e-4


def test_profile_is_robust_to_oneoff_shock():
    cpi = _seasonal_cpi(jump_month=7, jump=0.005)
    # a single massive August shock must not create an August 'season'
    shocked = cpi.copy()
    shocked.loc[pd.Period("2021-08", "M"):] *= 1.05
    prof = inflation.seasonal_mom_profile(shocked)
    assert abs(prof[8]) < 1e-3


def _assumptions(last_month):
    horizon = pd.period_range(last_month + 1, periods=14, freq="M")
    return {
        "power_recent_ore_kwh": 100.0,
        "power_forward_ore_kwh": {str(p): 100.0 for p in horizon},
        "brent_recent_usd": 80.0, "brent_forward_usd": {},
        "usdnok_recent": 10.0, "usdnok_path": {},
        "i44_path": {}, "imported_goods_baseline_yoy": 1.0,
        "wage_norm_pct": 4.0, "food_pipeline_yoy": 3.0,
    }


def test_seasonality_is_opt_in_and_changes_path():
    cpi = _seasonal_cpi()
    i44 = pd.Series(100.0, index=cpi.index)
    base_assum = _assumptions(cpi.index[-1])

    off = inflation.build_cpi_projection(cpi, 12, base_assum, i44)
    off2 = inflation.build_cpi_projection(
        cpi, 12, {**base_assum, "cpi_seasonality": False}, i44)
    on = inflation.build_cpi_projection(
        cpi, 12, {**base_assum, "cpi_seasonality": True}, i44)

    # default and explicit-off are identical (backwards compatible)
    pd.testing.assert_frame_equal(off, off2)
    # opt-in adds a seasonal contribution column and moves July up
    assert "contrib_seasonal" in on.columns
    proj = on[on["contrib_seasonal"].notna()]
    july = proj[proj.index.month == 7]["mom_pct"]
    april = proj[proj.index.month == 4]["mom_pct"]
    assert july.iloc[0] > april.iloc[0] + 0.3
    assert "contrib_seasonal" not in off.columns
