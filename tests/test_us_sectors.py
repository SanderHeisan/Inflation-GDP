"""Sector returns by regime (us_sectors.py): the statistics on synthetic
inputs, and the shape of the block the feed carries.

The regime series themselves come from the data bundle and the vintage
builder, which the other test files cover; here the question is whether a
known set of returns and regimes comes out as the right averages, counts,
excess returns and rankings, and whether the compact block a website reads
is the shape it expects. The integration run over the real data is skipped
when the cached prices are not on disk (data/ is not committed).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import us_sectors as us
from usbacktest.btconfig import RESULTS_DIR


def _frame():
    """36 months, four funds. SPY +1% in R1/R2 months and -1% in R3/R4; XLK
    beats SPY by 0.5 everywhere; XLU is flat; TLT mirrors SPY."""
    idx = pd.period_range("2020-01", periods=36, freq="M")
    regime = pd.Series([1, 2, 3, 4] * 9, index=idx)
    spy = np.where(regime.isin([1, 2]), 1.0, -1.0)
    rets = pd.DataFrame({"SPY": spy, "XLK": spy + 0.5, "XLU": np.zeros(36), "TLT": -spy}, index=idx)
    close = pd.Series([False] * 36, index=idx)
    close.iloc[::3] = True                     # every third month is too close to call
    return rets, regime, close


def test_cell_stats_are_plain_averages_with_counts_and_excess():
    rets, regime, close = _frame()
    st = us.cell_stats(rets, regime, close, basis="t")
    r1_spy = st[(st.regime == 1) & (st.fund == "SPY")].iloc[0]
    assert r1_spy.n == 9 and r1_spy["mean"] == 1.0 and r1_spy.hit == 1.0 and pd.isna(r1_spy.excess)
    r3_xlk = st[(st.regime == 3) & (st.fund == "XLK")].iloc[0]
    assert r3_xlk["mean"] == -0.5 and r3_xlk.excess == 0.5 and r3_xlk.hit == 0.0
    r4_tlt = st[(st.regime == 4) & (st.fund == "TLT")].iloc[0]
    assert r4_tlt["mean"] == 1.0 and r4_tlt.excess == 2.0
    allrow = st[(st.regime == 0) & (st.fund == "SPY")].iloc[0]
    assert allrow.n == 36 and abs(allrow["mean"]) < 1e-12
    # "clear only" drops the too-close months
    clear = us.cell_stats(rets, regime, close, basis="c", clear_only=True)
    assert clear[(clear.regime == 0) & (clear.fund == "SPY")].iloc[0].n == 24


def test_the_block_ranks_funds_by_excess_over_spy_per_regime():
    rets, regime, close = _frame()
    st = us.cell_stats(rets, regime, close, basis="t")
    blk = us._block(st, regime, ["XLK", "XLU", "TLT"], top=2)
    r4 = blk["regimes"]["4"]
    assert r4["n"] == 9 and r4["spy_mean"] == -1.0 and r4["spy_hit"] == 0.0
    assert [b["fund"] for b in r4["best"]] == ["TLT", "XLU"] and [b["fund"] for b in r4["worst"]] == ["XLK", "XLU"]
    r1 = blk["regimes"]["1"]
    assert [b["fund"] for b in r1["best"]] == ["XLK", "XLU"] and r1["best"][0]["excess"] == 0.5
    table = {row["fund"]: row for row in blk["table"]}
    assert table["SPY"]["r1"] == 1.0 and table["SPY"]["r3"] == -1.0 and table["XLK"]["all"] == 0.5
    assert table["TLT"]["r4_n"] == 9 and table["TLT"]["all_n"] == 36
    assert json.dumps(blk)                       # serialisable as is


def test_returns_come_from_month_end_closes():
    idx = pd.period_range("2024-01", periods=4, freq="M")
    closes = pd.DataFrame({"SPY": [100.0, 110.0, 99.0, 99.0]}, index=idx)
    m = us.monthly_returns(closes)
    assert m["SPY"].round(6).tolist()[1:] == [10.0, -10.0, 0.0]
    q = us.quarterly_returns(closes)
    assert list(q.index.astype(str)) == ["2024Q1", "2024Q2"] and round(float(q["SPY"].iloc[-1]), 6) == 0.0


@pytest.mark.skipif(not us._prices_path().exists() or not (RESULTS_DIR / "us_sectors_feed.json").exists(),
                    reason="needs the cached fund prices and a prior run")
def test_the_committed_block_has_every_regime_and_names_no_other_vendor():
    blk = json.loads((RESULTS_DIR / "us_sectors_feed.json").read_text())
    for basis in ("realized_monthly", "realized_quarterly"):
        assert set(blk[basis]["regimes"]) == {"1", "2", "3", "4"} and blk[basis]["n"] > 50
        assert all(len(blk[basis]["regimes"][r]["best"]) == 3 for r in "1234")
    assert "hedgeye" not in json.dumps(blk).lower() and "quad" not in json.dumps(blk).lower()
