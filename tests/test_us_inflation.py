"""US bottom-up CPI: the oil pass-through honors the 3bps/$1 rule, and
shelter tracks market rents lagged ~12 months."""
import numpy as np
import pandas as pd
import pytest

from usmodel import config, data_bundle, inflation


def _flat_cpi(n=40, start="2023-01"):
    idx = pd.period_range(start, periods=n, freq="M")
    return pd.Series(300.0, index=idx)


def test_oil_passthrough_matches_3bps_per_dollar():
    """A +$10/bbl WTI step in one forward month must add ~3bps x $10 = 30bps
    to that month's headline CPI (split across same/next month by the
    OIL_SAME_MONTH_SHARE)."""
    cpi = _flat_cpi()
    h = pd.period_range(cpi.index[-1] + 1, periods=6, freq="M")
    step_month = str(h[2])
    strip = {str(p): 70.0 for p in h}
    strip[step_month] = 80.0                      # +$10 in month h[2]
    for p in h[3:]:
        strip[str(p)] = 80.0                      # stays elevated
    assum = {"wti_recent": 70.0, "wti_forward": strip,
             "wage_growth_pct": 0.0, "food_pipeline_yoy": 0.0,
             "market_rent_yoy_recent": 0.0, "core_goods_baseline_yoy": 0.0}
    # zero out non-energy blocks so we isolate the oil contribution
    w = {**{k: 0 for k in config.CPI_WEIGHTS}, "gasoline": 34}
    proj = inflation.build_cpi_projection(cpi, 6, assum, weights=w)

    same = config.OIL_SAME_MONTH_SHARE
    step_contrib = proj.loc[pd.Period(step_month, "M"), "mom_pct"]
    assert step_contrib == pytest.approx(0.03 * 10 * same, abs=0.02)
    # the carried remainder lands the next month
    nxt = proj.loc[pd.Period(step_month, "M") + 1, "mom_pct"]
    assert nxt == pytest.approx(0.03 * 10 * (1 - same), abs=0.02)
    # total oil effect over the two months ~ 30bps
    assert step_contrib + nxt == pytest.approx(0.30, abs=0.03)


def test_shelter_tracks_lagged_market_rents():
    """Higher market-rent momentum a year ago must lift projected shelter."""
    idx = pd.period_range("2022-01", "2025-12", freq="M")
    cpi = pd.Series(300.0 * 1.002 ** np.arange(len(idx)), index=idx)
    hot = pd.Series(100.0 * 1.10 ** (np.arange(len(idx)) / 12.0), index=idx)  # +10%/yr
    cool = pd.Series(100.0 * 1.01 ** (np.arange(len(idx)) / 12.0), index=idx) # +1%/yr

    base = dict(wti_recent=70.0, wti_forward={}, wage_growth_pct=0.0,
                food_pipeline_yoy=0.0, core_goods_baseline_yoy=0.0)
    p_hot = inflation.build_cpi_projection(cpi, 6, base,
                                           aux={"market_rent": hot})
    p_cool = inflation.build_cpi_projection(cpi, 6, base,
                                            aux={"market_rent": cool})
    hot_sh = p_hot["contrib_shelter"].dropna().iloc[-6:].mean()
    cool_sh = p_cool["contrib_shelter"].dropna().iloc[-6:].mean()
    assert hot_sh > cool_sh + 0.05          # clearly higher shelter contribution


def test_shelter_is_the_largest_block(bundle_us):
    """US structural fact: shelter is the dominant CPI contributor."""
    assum = data_bundle.assumptions_from_bundle(bundle_us)
    aux = {"market_rent": bundle_us.market_rent, "dollar": bundle_us.dollar}
    proj = inflation.build_cpi_projection(bundle_us.cpi, 6, assum, aux)
    contribs = proj.filter(like="contrib_").dropna().iloc[0].abs()
    assert contribs.idxmax() == "contrib_shelter"


def test_projection_is_deterministic_and_shaped(bundle_us):
    assum = data_bundle.assumptions_from_bundle(bundle_us)
    aux = {"market_rent": bundle_us.market_rent, "dollar": bundle_us.dollar}
    a = inflation.build_cpi_projection(bundle_us.cpi, 12, assum, aux)
    b = inflation.build_cpi_projection(bundle_us.cpi, 12, assum, aux)
    pd.testing.assert_frame_equal(a, b)
    assert a["yoy_pct"].dropna().between(-5, 15).all()
