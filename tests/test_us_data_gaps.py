"""
The October 2025 hole. BLS never published that month's CPI, FRED carries
it as missing, and a positional 12-row shift then reads every 12-month rate
spanning the hole as a 13-month one. These tests pin the two defenses: the
loader fills an isolated month and records it, and every YoY is computed by
calendar alignment so a hole can only ever yield NaN, never a wrong number.
"""
import numpy as np
import pandas as pd
import pytest

from quadmap import quads
from usmodel import data_bundle, inflation


@pytest.fixture
def holed():
    idx = pd.period_range("2023-01", "2026-07", freq="M")
    s = pd.Series(100.0 * (1.003 ** np.arange(len(idx))), index=idx)
    return s.drop(pd.Period("2025-10", "M"))


def test_positional_shift_is_wrong_across_the_hole(holed):
    """Documents the failure mode: pct_change(12) on the holed series gives a
    13-month change for July 2026."""
    p = pd.Period("2026-07", "M")
    positional = float(holed.pct_change(12)[p]) * 100
    true = (float(holed[p] / holed[p - 12]) - 1) * 100
    assert positional > true + 0.2          # ~13 months of 0.3%/m vs 12


def test_calendar_pct_change_is_right_across_the_hole(holed):
    p = pd.Period("2026-07", "M")
    got = quads.calendar_pct_change(holed, 12)
    assert got[p] == pytest.approx((holed[p] / holed[p - 12] - 1) * 100)
    # and it is NaN exactly where the base is missing, never a wrong number:
    # extend the series to October 2026, whose base is the hole
    ext = pd.concat([holed, pd.Series(
        [float(holed.iloc[-1]) * 1.003 ** i for i in range(1, 4)],
        index=pd.period_range("2026-08", periods=3, freq="M"))])
    got_ext = quads.calendar_pct_change(ext, 12)
    assert np.isnan(got_ext[pd.Period("2026-10", "M")])
    assert not np.isnan(got_ext[pd.Period("2026-09", "M")])


def test_fill_single_month_gap_is_geometric_and_recorded(holed):
    filled, gaps = data_bundle.fill_single_month_gaps(holed)
    assert gaps == ["2025-10"]
    sep, nov = holed[pd.Period("2025-09", "M")], holed[pd.Period("2025-11", "M")]
    assert filled[pd.Period("2025-10", "M")] == pytest.approx(np.sqrt(sep * nov))
    assert len(filled) == len(holed) + 1


def test_longer_gaps_are_not_invented(holed):
    two = holed.drop(pd.Period("2025-11", "M"))
    filled, gaps = data_bundle.fill_single_month_gaps(two)
    assert gaps == []
    assert pd.Period("2025-10", "M") not in filled.index


def test_projection_yoy_is_calendar_aligned_across_a_hole(holed):
    """The CPI projection's YoY must use the true year-ago month even when
    the history it compounds from has a hole."""
    assum = {"wti_recent": 70.0, "wti_forward": {}, "wage_growth_pct": 3.0,
             "market_rent_yoy_recent": 3.0, "food_pipeline_yoy": 2.0}
    full = inflation.build_cpi_projection(holed, 6, assum)
    p = pd.Period("2026-09", "M")
    expected = (full.loc[p, "cpi_index"] / holed[p - 12] - 1) * 100
    assert full.loc[p, "yoy_pct"] == pytest.approx(expected)
    assert np.isnan(full.loc[pd.Period("2026-10", "M"), "yoy_pct"])
