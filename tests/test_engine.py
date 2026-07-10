"""Walk-forward engine: prediction bookkeeping and end-to-end no-leakage."""
import pandas as pd
import pytest

from backtest import engine
from backtest.btconfig import VintageConfig
from backtest.data_bundle import make_demo_bundle


CFG = VintageConfig(revision_mode="none")   # deterministic for these tests


def test_predictions_cover_requested_horizons(bundle):
    preds = engine.run_backtest(bundle, "2018-01-01", "2018-12-31",
                                freq="Q", max_horizon=4, cfg=CFG)
    assert set(preds["horizon"]) == {0, 1, 2, 3, 4}
    one = preds[(preds["asof_quarter"] == "2018Q2")
                & (preds["horizon"] == 3)]
    assert (one["target_quarter"] == "2019Q1").all()
    for col in ("pred_quad", "pred_d_growth", "pred_d_inflation",
                "low_conviction", "bench_persistence_quad",
                "bench_base_quad", "revision_mode"):
        assert col in preds.columns
    assert preds["pred_quad"].isin([1, 2, 3, 4]).all()


def test_engine_output_is_point_in_time(bundle):
    """Changing the future changes nothing: rerun the walk-forward with all
    data after the last as-of date shocked, predictions must be identical."""
    a = make_demo_bundle(seed=7)
    b = make_demo_bundle(seed=7)
    cutoff_m, cutoff_q = pd.Period("2019-01", "M"), pd.Period("2019Q1", "Q")
    b.gdp_final.loc[b.gdp_final.index >= cutoff_q] *= 5.0
    b.cpi.loc[b.cpi.index >= cutoff_m] *= 2.0
    b.market.loc[b.market.index >= cutoff_m] *= 4.0
    b.i44.loc[b.i44.index >= cutoff_m] += 300.0

    kw = dict(start="2018-01-01", end="2018-11-30", freq="M",
              max_horizon=4, cfg=CFG)
    pd.testing.assert_frame_equal(engine.run_backtest(a, **kw),
                                  engine.run_backtest(b, **kw))


def test_last_gdp_quarter_never_at_or_after_asof(bundle):
    preds = engine.run_backtest(bundle, "2017-01-01", "2019-12-31",
                                freq="M", max_horizon=2, cfg=CFG)
    for _, row in preds.iterrows():
        last_pub = pd.Period(row["last_gdp_quarter"], freq="Q")
        assert last_pub.end_time + pd.Timedelta(days=40) <= row["asof"]


def test_revision_mode_recorded(bundle):
    preds = engine.run_backtest(bundle, "2018-01-01", "2018-06-30",
                                freq="Q", max_horizon=1,
                                cfg=VintageConfig(revision_mode="noise"))
    assert (preds["revision_mode"] == "noise").all()
