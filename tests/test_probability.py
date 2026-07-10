"""Calibrated quad probabilities: calibration math and the forward view."""
import pandas as pd
import pytest

from backtest import engine, probability, scoring
from backtest.btconfig import VintageConfig


def test_horizon_calibration_math():
    # 4 predictions at h=1: called quad 2 three times (realized 2,2,3) and
    # quad 4 once (realized 4).
    preds = pd.DataFrame([
        {"target_quarter": t, "horizon": 1, "pred_quad": p,
         "pred_d_growth": 1.0, "pred_d_inflation": 1.0,
         "low_conviction": False}
        for t, p in [("2020Q1", 2), ("2020Q2", 2), ("2020Q3", 2),
                     ("2020Q4", 4)]])
    realized = pd.DataFrame(
        {"quad": [2, 2, 3, 4], "d_growth": 1.0, "d_inflation": 1.0},
        index=pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q3", "2020Q4"],
                             freq="Q"))
    cal = probability.horizon_calibration(preds, realized, horizons=[1],
                                          alpha=1.0)[1]
    # called 2: counts (0,2,1,0) + alpha -> (1,3,2,1)/7
    assert cal.loc[2].tolist() == pytest.approx([1/7, 3/7, 2/7, 1/7])
    # called 4: counts (0,0,0,1) + alpha -> (1,1,1,2)/5
    assert cal.loc[4].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.4])
    # never-called quads fall back to the flat prior
    assert cal.loc[1].tolist() == pytest.approx([0.25] * 4)
    assert cal.sum(axis=1).tolist() == pytest.approx([1.0] * 4)


def test_current_probabilities_end_to_end(bundle):
    cfg = VintageConfig(revision_mode="noise")
    preds = engine.run_backtest(bundle, "2015-01-01", "2020-12-31",
                                freq="Q", max_horizon=4, cfg=cfg)
    realized = scoring.realized_quads_final(bundle)
    prob, calls, asof = probability.current_quad_probabilities(
        bundle, preds, realized, cfg)

    assert list(prob.index) == [1, 2, 3, 4]
    assert len(prob.columns) == 4
    assert prob.sum(axis=0).tolist() == pytest.approx([1.0] * 4)
    # target quarters are the four after the as-of quarter
    asof_q = pd.Period(asof, freq="Q")
    assert list(prob.columns) == [asof_q + h for h in range(1, 5)]
    for tq, call in calls.items():
        assert call in (1, 2, 3, 4)
        # the point call must be the (weak) modal outcome of its own column
        # only when calibration says the model tends to be right; no strict
        # assertion - but its probability must never be zero
        assert prob.loc[call, tq] > 0

    monthly = probability.monthly_probabilities(prob)
    assert len(monthly.columns) == 12
    assert monthly.sum(axis=0).tolist() == pytest.approx([1.0] * 12)
    # months of one quarter share that quarter's distribution
    first_q = prob.columns[0]
    m = first_q.asfreq("M", "start")
    pd.testing.assert_series_equal(monthly[m], prob[first_q],
                                   check_names=False)


def test_forecast_asof_sees_all_published_data(bundle):
    asof = probability.latest_full_asof(bundle)
    from backtest.vintage import build_vintage
    v = build_vintage(bundle, asof,
                      VintageConfig(revision_mode="none"))
    assert v.last_gdp_quarter == bundle.gdp_final.index[-1]
    assert v.last_cpi_month == bundle.cpi.index[-1]
