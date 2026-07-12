"""Food / imported-goods sub-index momentum entering the assumptions:
point-in-time discipline, fallback behavior, and directional response."""
import numpy as np
import pandas as pd
import pytest

from backtest import vintage as vt
from backtest.data_bundle import make_demo_bundle
from backtest.fetch_data import _division_col


def test_subindex_trends_enter_assumptions(bundle):
    v = vt.build_vintage(bundle, "2019-08-31")
    food = vt.truncate(bundle.cpi_food, pd.Timestamp("2019-08-31"), 10)
    expected = (food.iloc[-1] / food.iloc[-13] - 1) * 100.0
    assert v.assumptions["food_pipeline_yoy"] == pytest.approx(
        float(np.clip(expected, -2.0, 8.0)))
    assert v.assumptions["imported_goods_baseline_yoy"] != 1.0


def test_subindices_respect_publication_lag():
    a = make_demo_bundle(seed=7)
    b = make_demo_bundle(seed=7)
    cutoff = pd.Period("2018-07", freq="M")
    b.cpi_food.loc[b.cpi_food.index >= cutoff] *= 0.5      # future crash
    b.cpi_imported.loc[b.cpi_imported.index >= cutoff] *= 0.5
    va = vt.build_vintage(a, "2018-06-30")
    vb = vt.build_vintage(b, "2018-06-30")
    assert va.assumptions["food_pipeline_yoy"] \
        == vb.assumptions["food_pipeline_yoy"]
    assert va.assumptions["imported_goods_baseline_yoy"] \
        == vb.assumptions["imported_goods_baseline_yoy"]


def test_falling_food_prices_lower_the_projection():
    """The June-2026 failure mode: a grocery price war visible in the food
    sub-index must pull the projected CPI path down."""
    calm = make_demo_bundle(seed=7)
    war = make_demo_bundle(seed=7)
    start = pd.Period("2019-01", freq="M")
    idx = war.cpi_food.index >= start
    decay = 0.99 ** np.arange(idx.sum())
    war.cpi_food.loc[idx] = war.cpi_food.loc[idx] * decay

    asof = "2019-12-20"
    v_calm = vt.build_vintage(calm, asof)
    v_war = vt.build_vintage(war, asof)
    assert v_war.assumptions["food_pipeline_yoy"] \
        < v_calm.assumptions["food_pipeline_yoy"] - 2.0

    from quadmap import inflation
    p_calm = inflation.build_cpi_projection(
        v_calm.cpi_index, 6, v_calm.assumptions, v_calm.i44)
    p_war = inflation.build_cpi_projection(
        v_war.cpi_index, 6, v_war.assumptions, v_war.i44)
    proj = p_calm.index[p_calm["mom_pct"].notna()]
    assert (p_war.loc[proj, "mom_pct"]
            < p_calm.loc[proj, "mom_pct"]).all()


def test_fallback_when_subindices_absent():
    b = make_demo_bundle(seed=7)
    b.cpi_food = None
    b.cpi_imported = None
    v = vt.build_vintage(b, "2019-08-31")
    cpi_yoy = (v.cpi_index.iloc[-1] / v.cpi_index.iloc[-13] - 1) * 100.0
    assert v.assumptions["food_pipeline_yoy"] == pytest.approx(
        float(np.clip(cpi_yoy, 0.0, 6.0)))
    assert v.assumptions["imported_goods_baseline_yoy"] == 1.0


def test_division_col_matches_table_specific_codes():
    df = pd.DataFrame({"TOTAL": [1.0], "JA01": [2.0], "01.1": [3.0],
                       "JA03": [4.0], "05": [5.0]})
    assert _division_col(df, "01").iloc[0] == 2.0     # JA01 -> 01
    assert _division_col(df, "03").iloc[0] == 4.0
    assert _division_col(df, "05").iloc[0] == 5.0
    assert _division_col(df, "09") is None