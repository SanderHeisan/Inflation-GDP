"""
The pace the growth path assumes past the nowcast quarter: the four modes,
how the rate drag combines with each, and the config switch.
"""
import numpy as np
import pandas as pd
import pytest

from usmodel import config, gdp as gdp_mod
from usbacktest.btconfig import USVintageConfig


@pytest.fixture
def level():
    quarters = pd.period_range("2015Q1", "2026Q2", freq="Q")
    return pd.Series(15000 * np.cumprod([1.006] * len(quarters)), index=quarters)


def _proj(level, mode, **extra):
    ind = {"fitted_qoq_pct": 0.9, "fitted_trend_qoq": 0.0077, "pace_mode": mode}
    ind.update(extra)
    out = gdp_mod.project_gdp(level, ind, horizon_quarters=5)
    return out[out["projected"]]["qoq_pct"].to_numpy()


def test_potential_mode_steps_to_potential(level):
    p = _proj(level, "potential")
    pot = gdp_mod.potential_qoq(config.GDP_POTENTIAL_ANN_PCT) * 100
    assert p[0] == pytest.approx(0.9)
    assert p[1:] == pytest.approx(pot, abs=1e-9)
    assert pot == pytest.approx(0.4963, abs=1e-3)       # 2.0% a year


def test_nowcast_mode_carries_the_nowcast(level):
    p = _proj(level, "nowcast")
    assert p == pytest.approx(0.9, abs=1e-9)


def test_glide_mode_fades_from_the_nowcast_to_potential(level):
    p = _proj(level, "glide", glide_persistence=0.5)
    pot = gdp_mod.potential_qoq() * 100
    assert p[1] == pytest.approx(pot + (0.9 - pot) * 0.5, abs=1e-9)
    assert p[2] == pytest.approx(pot + (0.9 - pot) * 0.25, abs=1e-9)
    assert abs(p[4] - pot) < abs(p[1] - pot)


def test_trailing_median_mode_is_the_old_behaviour(level):
    p = _proj(level, "trailing_median")
    assert p[1:] == pytest.approx(0.77, abs=1e-9)


def test_rate_drag_is_relative_only_in_nowcast_mode(level):
    last = level.index[-1]
    drag = pd.Series({last + 1: -0.004, last + 2: -0.006, last + 3: -0.006})
    pot = gdp_mod.potential_qoq() * 100
    p_pot = _proj(level, "potential", rate_drag_qoq=drag)
    p_now = _proj(level, "nowcast", rate_drag_qoq=drag)
    assert p_pot[1] == pytest.approx(pot - 0.6, abs=1e-9)        # absolute drag
    assert p_now[1] == pytest.approx(0.9 - 0.2, abs=1e-9)        # change since q+1
    assert p_now[4] == pytest.approx(0.9 + 0.4, abs=1e-9)        # no entry: drag 0 vs -0.4


def test_unknown_mode_is_rejected(level):
    with pytest.raises(ValueError):
        _proj(level, "hunch")
    with pytest.raises(ValueError):
        USVintageConfig(pace_mode="hunch")
    assert USVintageConfig().pace_mode == config.GDP_PACE_MODE
