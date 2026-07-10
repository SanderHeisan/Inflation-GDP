"""Residual probability method and the live-override path (no network)."""
import pandas as pd
import pytest

from backtest import engine, probability, scoring
from backtest.btconfig import VintageConfig


def test_residual_probability_quadrant_shares():
    resid = pd.DataFrame({"eg": [0.5, 0.5, -0.5, -0.5],
                          "ei": [0.5, -0.5, 0.5, -0.5]})
    # dg=0, di=0: one error pair per quadrant
    p = probability.residual_quad_probability(0.0, 0.0, resid)
    assert p.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    # strong positive growth delta: growth is up in every draw
    p = probability.residual_quad_probability(2.0, 0.0, resid)
    assert p[1] + p[2] == pytest.approx(1.0)
    assert p.sum() == pytest.approx(1.0)


def test_residual_probability_moves_continuously():
    """The reactivity requirement: a higher predicted inflation delta must
    raise P(quad 2 or 3) monotonically - no jump-only behavior."""
    rng_resid = pd.DataFrame({"eg": [0.3, -0.2, 0.1, -0.4, 0.25, -0.05],
                              "ei": [0.2, 0.3, -0.15, -0.3, 0.05, -0.25]})
    shares = [probability.residual_quad_probability(0.1, di, rng_resid)
              .loc[[2, 3]].sum()
              for di in (-0.4, -0.1, 0.0, 0.1, 0.4)]
    assert all(b >= a for a, b in zip(shares, shares[1:]))
    assert shares[0] < shares[-1]


def test_live_overrides_change_the_forecast(bundle):
    cfg = VintageConfig(revision_mode="noise")
    preds = engine.run_backtest(bundle, "2016-01-01", "2019-12-31",
                                freq="Q", max_horizon=4, cfg=cfg)
    realized = scoring.realized_quads_final(bundle)

    base_prob, base_calls, asof = probability.current_quad_probabilities(
        bundle, preds, realized, cfg, method="residual")

    # simulate an energy shock arriving through the live feeds today:
    # forward paths jump to the live spot, anchors stay at the level
    # embedded in the last CPI print (as gather_live_spots does)
    horizon = pd.period_range(pd.Period(asof, freq="M"), periods=18,
                              freq="M")
    shock = {
        "power_forward_ore_kwh": {str(p): 400.0 for p in horizon},
        "brent_forward_usd": {str(p): 180.0 for p in horizon},
    }
    hot_prob, _, _ = probability.current_quad_probabilities(
        bundle, preds, realized, cfg, method="residual",
        assumption_overrides=shock, asof=asof)

    # the shock must raise the inflation-accelerating share (quads 2+3)
    # at least at the front of the curve
    front = base_prob.columns[0]
    assert hot_prob.loc[[2, 3], front].sum() \
        > base_prob.loc[[2, 3], front].sum()
    # and probabilities stay valid
    assert hot_prob.sum(axis=0).tolist() == pytest.approx([1.0] * 4)


def test_methods_agree_on_structure(bundle):
    cfg = VintageConfig(revision_mode="noise")
    preds = engine.run_backtest(bundle, "2016-01-01", "2018-12-31",
                                freq="Q", max_horizon=4, cfg=cfg)
    realized = scoring.realized_quads_final(bundle)
    for method in ("residual", "calibration"):
        prob, calls, _ = probability.current_quad_probabilities(
            bundle, preds, realized, cfg, method=method)
        assert list(prob.index) == [1, 2, 3, 4]
        assert prob.sum(axis=0).tolist() == pytest.approx([1.0]
                                                          * len(prob.columns))
