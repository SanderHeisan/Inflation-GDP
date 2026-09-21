"""Which wage series drives the supercore block, and the plumbing that lets
the question be asked at all.

Core services ex shelter is the one CPI block the model builds from wages:
`supercore_yoy = a + b * wage_yoy`, with a and b refitted on each vintage's
own published history. The series behind `wage_yoy` was average hourly
earnings, a MEAN across whoever is on payrolls, so it moves when the
composition of employment changes rather than when anyone is paid more --
April 2020 printed +8.1% while the Atlanta Fed's matched-person tracker
printed +3.7%, because low-wage workers had been laid off.

These tests pin the switch that lets either series drive the block, the fact
that the tracker is already a RATE (differencing it again would be silently
wrong), the publication lag, and the invariant that matters most: the default
is unchanged, so nothing published moves unless the switch is flipped.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from usbacktest import btconfig, vintage as vt
from usmodel.data_bundle import USDataBundle, load_bundle
from usmodel.fetch_data_us import FRED_SERIES, INDICATOR_PUB_LAG_DAYS


def _bundle():
    try:
        return load_bundle(start="1995-01")
    except FileNotFoundError:
        pytest.skip("needs the cached US data (data/ is not committed)")


def test_the_default_is_still_average_hourly_earnings():
    """The published model must not move because a research switch exists."""
    cfg = btconfig.USVintageConfig()
    assert cfg.wage_source == "ahe"
    assert "ahe" in btconfig.WAGE_SOURCES and "tracker" in btconfig.WAGE_SOURCES


def test_the_tracker_is_a_rate_and_is_never_differenced_again():
    b = _bundle()
    if b.wage_tracker is None:
        pytest.skip("wage_tracker.csv not cached")
    tr = b.wage_tracker
    # a growth rate in percent, not an index: no value near 100, and the
    # whole history sits in a plausible wage-growth band
    assert tr.between(0.0, 12.0).all(), tr.describe()
    growth, src = vt.wage_growth_vintage(
        b, pd.Timestamp("2026-09-20"),
        btconfig.USVintageConfig(wage_source="tracker"))
    assert src == "tracker"
    # the series is passed through, not turned into a 12-month change of itself
    common = growth.index.intersection(tr.index)
    assert len(common) > 100
    assert np.allclose(growth.loc[common].to_numpy(float),
                       tr.loc[common].to_numpy(float))


def test_ahe_is_differenced_and_the_two_sources_disagree():
    b = _bundle()
    asof = pd.Timestamp("2026-09-20")
    ahe, src_a = vt.wage_growth_vintage(b, asof,
                                        btconfig.USVintageConfig())
    assert src_a == "ahe"
    # a 12-month change of an index, so far from the index's own level
    assert ahe.between(-5.0, 12.0).all()
    if b.wage_tracker is None:
        pytest.skip("wage_tracker.csv not cached")
    trk, _ = vt.wage_growth_vintage(
        b, asof, btconfig.USVintageConfig(wage_source="tracker"))
    # they are different objects: a mean over everyone vs a median of matched
    # individuals. If these ever coincide, the switch is not doing anything.
    assert abs(float(ahe.iloc[-1]) - float(trk.iloc[-1])) > 0.2


def test_a_missing_tracker_falls_back_to_the_published_basis():
    """A vintage before the tracker's history, or a checkout without the
    file, stays on average hourly earnings instead of failing."""
    b = _bundle()
    stripped = replace(b, wage_tracker=None)
    growth, src = vt.wage_growth_vintage(
        stripped, pd.Timestamp("2026-09-20"),
        btconfig.USVintageConfig(wage_source="tracker"))
    assert src == "ahe" and len(growth) > 12


def test_the_tracker_respects_its_publication_lag():
    b = _bundle()
    if b.wage_tracker is None:
        pytest.skip("wage_tracker.csv not cached")
    cfg = btconfig.USVintageConfig(wage_source="tracker")
    assert cfg.wage_tracker_pub_lag_days >= btconfig.WAGE_PUB_LAG_DAYS, \
        "the tracker is published after the report it is built from"
    # the day after a month ends, that month cannot be in the information set
    early, _ = vt.wage_growth_vintage(b, pd.Timestamp("2026-09-02"), cfg)
    later, _ = vt.wage_growth_vintage(b, pd.Timestamp("2026-09-25"), cfg)
    assert early.index[-1] < pd.Period("2026-08", "M")
    assert later.index[-1] >= early.index[-1]


def test_the_passthrough_is_fitted_on_a_growth_series():
    """supercore_yoy = a + b * wage_yoy, recovered from a clean line, with
    the wage side handed in as a RATE rather than an index."""
    idx = pd.period_range("2010-01", periods=180, freq="M")
    wage = pd.Series(np.linspace(2.0, 6.0, len(idx)), index=idx)
    # build a supercore INDEX whose 12-month change is 1.0 + 0.5 * wage
    target = 1.0 + 0.5 * wage
    lvl = [100.0]
    for i in range(1, len(idx)):
        lvl.append(lvl[-1] * (1 + target.iloc[i] / 100.0 / 12.0))
    sc = pd.Series(lvl, index=idx)
    b, a = vt.estimate_supercore_passthrough(sc, wage)
    assert 0.35 < b < 0.65, b
    assert -0.5 < a < 2.5, a
    # no series at all -> the config default, never a crash
    assert vt.estimate_supercore_passthrough(None, wage)[0] == \
        pytest.approx(btconfig.uconfig.SUPERCORE_WAGE_PASSTHROUGH)


def test_the_series_and_its_lag_are_declared_together():
    for key in ("wage_tracker", "wage_switcher", "wage_stayer"):
        assert key in FRED_SERIES, f"{key} has no FRED id"
        assert key in INDICATOR_PUB_LAG_DAYS, f"{key} has no publication lag"
    assert FRED_SERIES["wage_tracker"][0] == "FRBATLWGTUMHWGO"


def test_switching_the_source_changes_the_supercore_call():
    """The end-to-end point: the same as-of date, two wage series, two
    different supercore paths -- with the coefficient refitted each time
    rather than a constant tuned to the other series."""
    b = _bundle()
    if b.wage_tracker is None:
        pytest.skip("wage_tracker.csv not cached")
    seen = {}
    for src in ("ahe", "tracker"):
        v = vt.build_vintage(b, "2026-09-20",
                             btconfig.USVintageConfig(wage_source=src))
        d = v.diagnostics
        assert d["wage_source"] == src
        seen[src] = (d["wage_yoy"], d["supercore_b"], d["supercore_a"])
    assert seen["ahe"] != seen["tracker"]
    # the fit moves with the series: a coefficient carried over unchanged
    # would be the bug this round exists to avoid
    assert seen["ahe"][1] != seen["tracker"][1]


def test_the_feed_shows_the_tracker_but_the_model_still_runs_on_ahe():
    """The round's outcome as a contract: a reader gets the matched-person
    number, with average hourly earnings beside it and the switcher/stayer
    split, while `wage_source` on the sheet still says the model ran on
    average hourly earnings."""
    import us_feed
    sheet = {
        "meta": {"wage_yoy": 3.09, "wage_source": "ahe", "wage_tracker_yoy": 4.1,
                 "wage_switcher_yoy": 4.4, "wage_stayer_yoy": 3.6,
                 "wti_now": 97.0, "wti_last_cpi": 83.0, "cpi_through": "Aug 2026",
                 "pump_now": 4.24, "pump_mom": 1.2, "rent_yoy": 3.4,
                 "pace_ann": 2.0, "k": 2},
        "quarters": [{"realized": False, "label": "Q3 2026", "qoq_ann": 1.6,
                      "bar_ann": 1.4, "dg": 0.2, "g_yoy": 1.6, "q": "2026Q3"}],
        "months": [{"realized": False, "label": "Sep 2026",
                    "contrib": {"gasoline": 0.02}}],
        "rate": {}, "consumer": {},
    }
    row = next(d for d in us_feed.drivers(sheet) if d["key"] == "wages")
    assert "+4.1%" in row["value"] and "same people" in row["label"].lower()
    assert "average hourly earnings +3.1%" in row["change"]
    assert "switchers +4.4%" in row["level"] and "stayers +3.6%" in row["level"]
    assert "layoffs cannot flatter it" in row["note"]
    # no vendor names reach the feed, as everywhere else
    blob = " ".join(row.values()).lower()
    assert "hedgeye" not in blob and "quad" not in blob

    # a sheet without the tracker cached falls back to the published basis
    bare = {**sheet, "meta": {**sheet["meta"], "wage_tracker_yoy": None}}
    row2 = next(d for d in us_feed.drivers(bare) if d["key"] == "wages")
    assert row2["value"] == "+3.1% year over year"
    assert "Average hourly earnings" in row2["note"]
