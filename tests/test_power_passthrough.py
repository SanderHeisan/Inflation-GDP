"""Self-calibrating spot->CPI power pass-through and the ex-energy
seasonal profile (the fix for the +2.4pp-vs-+0.2 overshoot)."""
import numpy as np
import pandas as pd
import pytest

from backtest import vintage as vt
from backtest.data_bundle import make_demo_bundle


def _series(months, vals):
    return pd.Series(vals, index=months, dtype=float)


def test_passthrough_recovers_known_coupling():
    """CPI built with a known effective share must yield ~that share."""
    rng = np.random.default_rng(3)
    months = pd.period_range("2014-01", "2025-12", freq="M")
    power = 60.0 + 15.0 * rng.standard_normal(len(months)).cumsum() * 0.1 \
        + 10.0 * np.sin(2 * np.pi * np.arange(len(months)) / 12)
    power = pd.Series(power, index=months).clip(20.0)
    cp = vt.consumer_power_price(power)
    true_share = 0.20
    mom = (0.002 + vt.W_ELEC * true_share * cp.pct_change().fillna(0)
           + rng.normal(0, 0.0005, len(months)))
    cpi = pd.Series(100 * np.cumprod(1 + mom), index=months)

    est = vt.estimate_power_passthrough(cpi, power)
    assert est == pytest.approx(true_share, abs=0.06)


def test_passthrough_falls_back_without_signal():
    months = pd.period_range("2014-01", "2025-12", freq="M")
    flat_power = _series(months, 80.0)                 # proxy-era: no info
    cpi = _series(months, 100.0 * 1.002 ** np.arange(len(months)))
    assert vt.estimate_power_passthrough(cpi, flat_power) == 0.35
    short = cpi.iloc[-20:]
    assert vt.estimate_power_passthrough(short, flat_power) == 0.35


def test_vintage_uses_estimated_share_and_ex_energy_profile(bundle):
    v = vt.build_vintage(bundle, "2019-08-31")
    vs = v.assumptions["power_variable_share"]
    assert 0.05 <= vs <= 0.55
    # demo CPI is coupled at 0.55 x elec weight -> estimate should be high
    assert vs > 0.3
    prof = v.assumptions["cpi_seasonal_profile"]
    assert set(prof) == set(range(1, 13))
    # the profile is ex-energy: it must differ from the raw-CPI profile
    from quadmap.inflation import seasonal_mom_profile
    raw = seasonal_mom_profile(v.cpi_index)
    assert any(abs(prof[m] - raw[m]) > 1e-5 for m in range(1, 13))


def test_extreme_power_month_no_longer_produces_wild_mom(bundle):
    """The 2025-11 failure shape: a big observed power spike in the month
    after the last print must move the projection, but by an amount scaled
    with the estimated pass-through - not the naive full variable share."""
    from quadmap import inflation
    spike = make_demo_bundle(seed=7)
    m_next = pd.Period("2019-04", freq="M")
    spike.market.loc[m_next, "power_ore_kwh"] *= 1.6   # +60% spot month

    asof = pd.Timestamp("2019-05-08")   # m0 = 2019-03, April observed
    v = vt.build_vintage(spike, asof)
    proj = inflation.build_cpi_projection(v.cpi_index, 3, v.assumptions,
                                          v.i44)
    first_mom = float(proj.loc[m_next, "mom_pct"])
    # +60% spot with damped pass-through: clearly positive, clearly bounded
    assert 0.1 < first_mom < 1.5
