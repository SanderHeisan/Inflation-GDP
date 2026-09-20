"""The compact feed (us_feed.py): the contract a consumer relies on, built
from the committed sheet in results_us/, no model run needed.

The two blocks added on 2026-09-20 -- `hit_rates` (the backtest's scorecard
in one place) and `notes` (the story behind the numbers, computed from the
sheet) -- are for a website that shows the regimes to the public. So the
notes are plain text with no markup, and neither block may carry another
vendor's product names: the site says "regime" and the four names, nothing
else. The per-month field that records that vendor's published path stays
in `months` for the model's own scoring; the consumer strips it.
"""
from __future__ import annotations

import json
import math
import re

import pytest

import us_feed
from usbacktest.btconfig import RESULTS_DIR

SHEET = RESULTS_DIR / "us_sheet.json"
pytestmark = pytest.mark.skipif(not SHEET.exists(), reason="no committed sheet")

MONTH_KEYS = {"month", "label", "regime", "regime_name", "too_close_to_call", "status",
              "growth_measure_yoy", "growth_dir", "growth_conviction", "growth_hit",
              "cpi_yoy", "cpi_mom", "cpi_dir", "cpi_conviction", "cpi_hit", "hedgeye_regime"}
QUARTER_KEYS = {"quarter", "label", "regime", "regime_name", "too_close_to_call", "realized",
                "growth_yoy", "growth_dir", "inflation_yoy", "inflation_dir", "backtest_hit"}
VENDOR = re.compile(r"hedgeye|\bquads?\b|\bgip\b", re.I)


@pytest.fixture(scope="module")
def feed():
    return us_feed.build_feed(json.load(open(SHEET)))


def _rates(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _rates(v)
        elif isinstance(v, float):
            yield v


def test_the_rows_keep_their_contract(feed):
    assert feed["regimes"] == {1: "Sweet spot", 2: "Heating", 3: "Squeeze", 4: "Cooling"}
    assert feed["quarters"] and all(set(q) == QUARTER_KEYS for q in feed["quarters"])
    assert feed["months"] and all(set(m) == MONTH_KEYS for m in feed["months"])
    assert all(q["regime"] in (1, 2, 3, 4) for q in feed["quarters"])
    assert all(m["status"] in ("actual", "nowcast", "forecast") for m in feed["months"])


def test_hit_rates_are_the_backtests_scorecard(feed):
    hr = feed["hit_rates"]
    assert hr["n_asof"] > 100 and hr["span"].startswith("January 2017 to ")
    assert set(hr["regime_by_horizon"]) >= {"0", "1", "2", "3"}
    assert set(hr["cpi_month"]) == {"all", "strong", "good", "lean", "toss-up"}
    assert {"h1", "h2", "h3", "nowcast", "strong", "good", "lean", "toss-up", "vs_quarter_gdp", "n"} <= set(hr["growth_month"])
    rates = [r for r in _rates(hr) if r is not None]
    assert rates and all(0.0 <= r <= 1.0 and math.isfinite(r) for r in rates)
    # the block says what the sheet's stats say, not a rounding of its own
    st = json.load(open(SHEET))["stats"]
    assert hr["regime_by_horizon"]["0"] == round(st["quad_hit"]["0"], 3)
    assert hr["cpi_month"]["strong"] == round(st["cpi"]["high conviction (>0.30pp)"], 3)
    assert hr["growth_month"]["h1"] == round(st["mg"]["h1"], 3)


def test_notes_are_plain_sentences_computed_from_the_sheet(feed):
    notes = feed["notes"]
    keys = [n["key"] for n in notes]
    for k in ("inflation_next", "inflation_path", "energy", "rents_wages", "growth_now", "growth_after", "rates", "consumer"):
        assert k in keys, k
    for n in notes:
        assert set(n) == {"key", "title", "text"}
        assert n["title"] and n["text"].endswith(".")
        assert "<" not in n["text"] and ">" not in n["text"], "markup in a note"
        assert "—" not in n["text"] and "—" not in n["title"]
    sheet = json.load(open(SHEET))
    nxt = next(m for m in sheet["months"] if not m["realized"])
    assert nxt["label"] in next(n for n in notes if n["key"] == "inflation_next")["title"]
    assert f"{nxt['yoy']:.2f}%" in next(n for n in notes if n["key"] == "inflation_next")["text"]
    assert sheet["meta"]["peak"]["label"] in next(n for n in notes if n["key"] == "inflation_path")["text"]


def test_drivers_are_rows_of_figures(feed):
    rows = feed["drivers"]
    keys = [d["key"] for d in rows]
    for k in ("oil", "pump", "rents", "wages", "growth_pace", "pace_after", "fed_funds", "spending"):
        assert k in keys, k
    for d in rows:
        assert set(d) == {"key", "label", "value", "change", "level", "note"}
        assert d["label"] and d["value"] and all(isinstance(v, str) for v in d.values())
        assert "<" not in json.dumps(d) and "\u2014" not in json.dumps(d)
    sheet = json.load(open(SHEET))
    oil = next(d for d in rows if d["key"] == "oil")
    assert f"${sheet['meta']['wti_now']:.0f} a barrel" == oil["value"]


def test_no_other_vendors_names_in_the_public_blocks(feed):
    blob = json.dumps(feed["hit_rates"]) + json.dumps(feed["notes"]) + json.dumps(feed["drivers"]) + json.dumps(feed["regimes"])
    assert not VENDOR.search(blob), VENDOR.findall(blob)
    for row in feed["quarters"] + feed["months"]:
        for k, v in row.items():
            if k != "hedgeye_regime" and isinstance(v, str):
                assert not VENDOR.search(v), (k, v)


def test_a_block_that_cannot_be_built_does_not_stop_the_feed():
    sheet = json.load(open(SHEET))
    del sheet["stats"]
    sheet["rate"] = {}
    feed = us_feed.build_feed(sheet)
    assert "hit_rates" not in feed and feed["quarters"] and feed["months"]
    assert [n["key"] for n in feed["notes"]].count("rates") == 0 and len(feed["notes"]) >= 6
