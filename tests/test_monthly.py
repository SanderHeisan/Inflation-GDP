"""One-print-ahead inflation direction backtest."""
import pandas as pd
import pytest

from backtest import monthly
from backtest.btconfig import VintageConfig
from backtest.data_bundle import make_demo_bundle


@pytest.fixture(scope="module")
def mdir(bundle):
    return monthly.monthly_direction_backtest(
        bundle, "2017-01-01", "2019-12-31", VintageConfig())


def test_one_month_ahead_and_point_in_time(mdir):
    for _, r in mdir.iterrows():
        m0 = pd.Period(r["last_print"], freq="M")
        m1 = pd.Period(r["target_print"], freq="M")
        assert m1 == m0 + 1
        # the target print must NOT have been published at the as-of date:
        # CPI for month m1 appears ~10 days after m1 ends
        assert r["asof"] < m1.end_time + pd.Timedelta(days=10)


def test_realized_delta_matches_series(bundle, mdir):
    ryoy = (bundle.cpi.pct_change(12) * 100).dropna()
    r = mdir.iloc[5]
    m0 = pd.Period(r["last_print"], freq="M")
    m1 = pd.Period(r["target_print"], freq="M")
    assert r["real_delta"] == pytest.approx(ryoy[m1] - ryoy[m0])
    assert r["hit"] == ((r["pred_delta"] > 0) == (r["real_delta"] > 0))


def test_prediction_ignores_future_cpi(bundle):
    """Shock the CPI after the as-of window: predicted deltas must not move
    (realized ones of course do)."""
    a = make_demo_bundle(seed=7)
    b = make_demo_bundle(seed=7)
    b.cpi.loc[b.cpi.index >= pd.Period("2019-01", "M")] *= 1.5
    kw = dict(start="2018-01-01", end="2018-10-31", cfg=VintageConfig())
    da = monthly.monthly_direction_backtest(a, **kw)
    db = monthly.monthly_direction_backtest(b, **kw)
    pd.testing.assert_series_equal(da["pred_delta"], db["pred_delta"])


def test_summary_buckets(mdir):
    s = monthly.summarize_monthly_direction(mdir)
    assert "ALL months" in s.index
    all_row = s.loc["ALL months"]
    assert all_row["n"] == len(mdir)
    assert 0.0 <= all_row["hit_rate"] <= 1.0
    # bucket ns partition the total
    bucket_rows = s[s.index.str.contains("pp")]
    bucket_rows = bucket_rows[~bucket_rows.index.str.startswith("callable")]
    assert bucket_rows["n"].sum() == len(mdir)


def test_direction_hit_rate_lookup(mdir):
    acc, n = monthly.direction_hit_rate_for(0.5, mdir)
    assert 0.0 <= acc <= 1.0 and n > 0
    assert monthly.bucket_label(0.02).startswith("coin-flip")
    assert monthly.bucket_label(0.5).startswith("high conviction")
