"""
The monthly growth measure: the index from its components, the nowcast of
a month with some components in, the near-term calls, and monthly quads.
"""
import numpy as np
import pandas as pd
import pytest

from usmodel import monthly_growth as mg


def _series(start, n, mom, seed=0, level=100.0):
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, periods=n, freq="M")
    return pd.Series(level * np.cumprod(1 + mom / 100 + rng.normal(0, 0.0005, n)), index=months)


@pytest.fixture
def panel():
    n = 60
    return {"real_pce": _series("2021-01", n, 0.25, 1), "indpro": _series("2021-01", n, 0.10, 2),
            "real_income_ex_transfers": _series("2021-01", n, 0.20, 3),
            "real_retail": _series("2021-01", n, 0.15, 4), "payrolls": _series("2021-01", n, 0.12, 5),
            "hours_all": pd.Series(34.5, index=pd.period_range("2021-01", periods=n, freq="M"))}


def test_index_weights_and_status(panel):
    idx = mg.activity_index(panel, horizon_months=2)
    assert idx.attrs["weights"]["real_pce"] == pytest.approx(0.55)
    assert idx.attrs["last_complete"] == pd.Period("2025-12", "M")
    assert list(idx["status"].tail(2)) == ["forecast", "forecast"]
    assert (idx["status"].loc[:"2025-12"] == "actual").all()
    # the index MoM is the weighted mean of component MoMs
    mom = sum(w * (np.log(panel[k] if k != "labor_input" else panel["payrolls"] * panel["hours_all"]).diff() * 100)
              for k, w in idx.attrs["weights"].items())
    assert idx["mom_pct"].loc["2025-12"] == pytest.approx(float(mom.loc["2025-12"]), abs=1e-9)


def test_nowcast_uses_the_components_that_are_in(panel):
    # PCE and income stop a month early: the last month is a nowcast with 3 of 5 in
    p = dict(panel)
    p["real_pce"] = panel["real_pce"].iloc[:-1]; p["real_income_ex_transfers"] = panel["real_income_ex_transfers"].iloc[:-1]
    idx = mg.activity_index(p, horizon_months=2)
    assert idx.attrs["last_complete"] == pd.Period("2025-11", "M")
    assert idx.loc["2025-12", "status"] == "nowcast" and idx.loc["2025-12", "n_actual"] == 3
    assert idx.loc["2026-01", "status"] == "forecast" and idx.loc["2026-01", "n_actual"] == 0
    # the missing components take their trailing-6-month mean MoM
    fill = float((np.log(p["real_pce"]).diff() * 100).tail(6).mean())
    assert abs(fill - 0.25) < 0.05


def test_calls_are_base_effect_arithmetic(panel):
    idx = mg.activity_index(panel, horizon_months=2)
    calls = mg.growth_calls(idx, 2)
    assert list(calls["horizon"]) == [1, 2]
    t = calls.index[0]
    yoy = idx["yoy_pct"]
    assert calls.loc[t, "d_yoy_pp"] == pytest.approx(float(yoy[t] - yoy[t - 1]), abs=1e-9)
    assert calls.loc[t, "base_mom_pct"] == pytest.approx(float(idx.loc[t - 12, "mom_pct"]), abs=1e-9)
    assert calls.loc[t, "bucket"] in [b[2] for b in mg.BUCKETS]
    assert mg.growth_calls(None, 2).empty


def test_monthly_quads_follow_the_two_directions():
    months = pd.period_range("2025-01", periods=6, freq="M")
    g = pd.Series([2.0, 2.2, 2.1, 2.3, 2.3, 2.0], index=months)
    c = pd.Series([3.0, 3.1, 3.0, 2.9, 3.2, 3.1], index=months)
    q = mg.monthly_quads(g, c, deadband_pp=0.05)
    assert list(q["quad"]) == [2, 4, 1, 3, 4]        # (up,up) (down,down) (up,down) (flat->down? no: +0.0 growth, +0.3 infl)
    assert bool(q.loc["2025-05", "close"]) is True   # growth change 0.0 inside the deadband


def test_too_few_components_returns_none():
    assert mg.activity_index({"real_pce": _series("2021-01", 40, 0.2)}) is None
    assert mg.bucket_label(0.5) == "strong (>0.30pp)" and mg.bucket_label(0.0) == "toss-up (<0.05pp)"
