"""Observed-months forward paths (the June-2026 structural fix): market
months already public at the as-of date enter the projection; live spots
extend but never overwrite them."""
import pandas as pd
import pytest

from backtest import vintage as vt
from backtest.data_bundle import make_demo_bundle
from backtest.probability import merge_assumption_overrides


def test_observed_month_after_cpi_enters_forward_path(bundle):
    # asof 2019-05-08: April CPI publishes May 10 -> m0 = 2019-03, but
    # April's market month is complete and public -> it must appear in the
    # forward path at its observed value, not the March anchor.
    asof = pd.Timestamp("2019-05-08")
    v = vt.build_vintage(bundle, asof)
    assert v.last_cpi_month == pd.Period("2019-03", freq="M")
    fwd = v.assumptions["power_forward_ore_kwh"]
    apr = bundle.market.loc[pd.Period("2019-04", "M"), "power_ore_kwh"]
    mar = bundle.market.loc[pd.Period("2019-03", "M"), "power_ore_kwh"]
    assert fwd["2019-04"] == pytest.approx(float(apr))
    assert v.assumptions["power_recent_ore_kwh"] == pytest.approx(float(mar))
    # months beyond the observed window carry the last observed value
    assert fwd["2019-07"] == pytest.approx(float(apr))
    # and nothing after the as-of date can enter (May incomplete)
    may = bundle.market.loc[pd.Period("2019-05", "M"), "power_ore_kwh"]
    if abs(may - apr) > 1e-9:
        assert fwd["2019-05"] != pytest.approx(float(may))


def test_energy_slide_into_print_month_lowers_projection():
    """A June-2026-shaped scenario: power slides hard in the month after
    the last CPI print. The projection must carry that observed drop."""
    calm = make_demo_bundle(seed=7)
    slide = make_demo_bundle(seed=7)
    m_next = pd.Period("2019-04", freq="M")
    slide.market.loc[m_next:, "power_ore_kwh"] *= 0.6   # observed collapse

    asof = pd.Timestamp("2019-05-08")   # m0 = 2019-03, April observed
    v_calm = vt.build_vintage(calm, asof)
    v_slide = vt.build_vintage(slide, asof)

    from quadmap import inflation
    p_calm = inflation.build_cpi_projection(
        v_calm.cpi_index, 4, v_calm.assumptions, v_calm.i44)
    p_slide = inflation.build_cpi_projection(
        v_slide.cpi_index, 4, v_slide.assumptions, v_slide.i44)
    first = pd.Period("2019-04", freq="M")
    assert p_slide.loc[first, "mom_pct"] < p_calm.loc[first, "mom_pct"] - 0.1


def test_override_merge_preserves_observed_months():
    assumptions = {"power_forward_ore_kwh": {"2026-06": 100.0,
                                             "2026-07": 100.0},
                   "brent_recent_usd": 80.0}
    overrides = {"power_forward_ore_kwh": {"2026-07": 150.0,
                                           "2026-08": 150.0},
                 "brent_recent_usd": 70.0}
    merged = merge_assumption_overrides(assumptions, overrides)
    fwd = merged["power_forward_ore_kwh"]
    assert fwd["2026-06"] == 100.0        # observed month untouched
    assert fwd["2026-07"] == 150.0        # override wins where it speaks
    assert fwd["2026-08"] == 150.0        # and extends the horizon
    assert merged["brent_recent_usd"] == 70.0   # scalars replace
