"""Quad scoring, confusion matrix, and flip precision/recall on hand-built
fixtures with known answers."""
import numpy as np
import pandas as pd
import pytest

from backtest import scoring
from quadmap import quads


def _realized(quads_by_quarter: dict) -> pd.DataFrame:
    idx = pd.PeriodIndex(list(quads_by_quarter), freq="Q")
    df = pd.DataFrame({"quad": list(quads_by_quarter.values())}, index=idx)
    df["d_growth"] = [1.0 if q in (1, 2) else -1.0 for q in df["quad"]]
    df["d_inflation"] = [1.0 if q in (2, 3) else -1.0 for q in df["quad"]]
    return df


def _pred_row(target, pred, horizon=1, low_conviction=False):
    return {"asof": pd.Timestamp("2020-01-31"), "asof_quarter": "2019Q4",
            "target_quarter": target, "horizon": horizon, "pred_quad": pred,
            "pred_d_growth": 1.0 if pred in (1, 2) else -1.0,
            "pred_d_inflation": 1.0 if pred in (2, 3) else -1.0,
            "low_conviction": low_conviction,
            "bench_persistence_quad": 2, "bench_base_quad": 3}


def test_hit_rate_and_direction():
    realized = _realized({"2020Q1": 2, "2020Q2": 3, "2020Q3": 4, "2020Q4": 2})
    preds = pd.DataFrame([
        _pred_row("2020Q1", 2),          # hit
        _pred_row("2020Q2", 3),          # hit
        _pred_row("2020Q3", 1),          # miss: growth wrong, inflation right
        _pred_row("2020Q4", 4),          # miss: both directions wrong
    ])
    s = scoring.score(preds, realized, "final", [1]).loc["final"]
    m = s.loc[("model", 1)]
    assert m["hit_rate"] == pytest.approx(0.5)
    assert m["growth_dir_hit"] == pytest.approx(0.5)
    assert m["inflation_dir_hit"] == pytest.approx(0.75)
    # persistence predicted quad 2 everywhere -> hits 2020Q1 and Q4
    assert s.loc[("persistence", 1)]["hit_rate"] == pytest.approx(0.5)
    assert s.loc[("random", 1)]["hit_rate"] == pytest.approx(0.25)
    assert m["edge_vs_random"] == pytest.approx(0.25)
    assert m["edge_vs_persistence"] == pytest.approx(0.0)


def test_low_conviction_exclusion():
    realized = _realized({"2020Q1": 1, "2020Q2": 1})
    preds = pd.DataFrame([
        _pred_row("2020Q1", 1, low_conviction=False),
        _pred_row("2020Q2", 3, low_conviction=True),   # miss but excluded
    ])
    m = scoring.score(preds, realized, "final", [1]).loc[
        ("final", "model", 1)]
    assert m["hit_rate"] == pytest.approx(0.5)
    assert m["n_high_conviction"] == 1
    assert m["hit_rate_high_conviction"] == pytest.approx(1.0)


def test_flip_precision_recall():
    # realized: 1 -> 1 -> 3 -> 3 -> 2 : flips at Q3 (1->3) and 2021Q1 (3->2)
    realized = _realized({"2020Q1": 1, "2020Q2": 1, "2020Q3": 3,
                          "2020Q4": 3, "2021Q1": 2})
    preds = pd.DataFrame([
        _pred_row("2020Q2", 4),   # predicted flip (prev=1) - did not happen
        _pred_row("2020Q3", 4),   # predicted flip (prev=1) - flip happened,
                                  #   but to 3 not 4: counts for precision,
                                  #   not precision_exact
        _pred_row("2020Q4", 3),   # no flip predicted (prev=3), none happened
        _pred_row("2021Q1", 2),   # predicted flip (prev=3) - exact hit
    ])
    joined = scoring._attach_realized(preds, realized)
    f = scoring.flip_metrics(joined)
    assert f["n_pred_flips"] == 3
    assert f["n_real_flips"] == 2
    assert f["flip_precision"] == pytest.approx(2 / 3)
    assert f["flip_precision_exact"] == pytest.approx(1 / 3)
    assert f["flip_recall"] == pytest.approx(1.0)


def test_confusion_matrix_shape_and_counts():
    realized = _realized({"2020Q1": 1, "2020Q2": 2, "2020Q3": 2})
    preds = pd.DataFrame([_pred_row("2020Q1", 1), _pred_row("2020Q2", 4),
                          _pred_row("2020Q3", 2)])
    m = scoring.confusion_by_horizon(preds, realized, [1])[1]
    assert m.shape == (4, 4)
    assert m.loc[1, 1] == 1 and m.loc[2, 4] == 1 and m.loc[2, 2] == 1
    assert m.to_numpy().sum() == 3


def test_realized_quads_match_manual_classification(bundle):
    realized = scoring.realized_quads_final(bundle)
    g_yoy = (bundle.gdp_final / bundle.gdp_final.shift(4) - 1) * 100
    cpi_yoy_q = quads.monthly_to_quarterly_yoy(
        (bundle.cpi.pct_change(12) * 100).dropna())
    q = pd.Period("2020Q2", freq="Q")
    expected_g_up = (g_yoy[q] - g_yoy[q - 1]) > 0
    row = realized.loc[q]
    assert (row["quad"] in (1, 2)) == expected_g_up
    assert realized["quad"].isin([1, 2, 3, 4]).all()


def test_first_release_quads_differ_under_noise(bundle):
    from backtest.btconfig import VintageConfig
    final = scoring.realized_quads_final(bundle)
    first = scoring.realized_quads_first_release(
        bundle, VintageConfig(revision_mode="noise", revision_sigma_pp=0.5))
    common = final.index.intersection(first.index)
    assert len(common) > 30
    # revisions must flip at least some quads at the margin...
    diff = (final.loc[common, "quad"] != first.loc[common, "quad"]).mean()
    assert diff > 0.0
    # ...but a 0.5pp revision error must not scramble the map entirely
    assert diff < 0.6
