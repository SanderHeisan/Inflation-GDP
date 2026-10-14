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
# Present only on a month that has BEGUN: there is no call to record for a
# month still ahead. Named rather than waved through, so a typo or a stray
# field still fails the contract.
MONTH_CALL_KEYS = {"called_regime", "called_asof", "called_too_close"}
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
    assert feed["months"]
    for m in feed["months"]:
        extra = set(m) - MONTH_KEYS
        assert not extra - MONTH_CALL_KEYS, f"{m['month']}: unexpected keys {sorted(extra - MONTH_CALL_KEYS)}"
        assert MONTH_KEYS <= set(m), f"{m['month']}: missing {sorted(MONTH_KEYS - set(m))}"
        # the call keys arrive together or not at all
        assert extra in (set(), MONTH_CALL_KEYS), f"{m['month']}: half a call {sorted(extra)}"
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


# ── the call that was made before the month began ───────────────────────────

def test_each_month_that_has_begun_carries_the_call_made_before_it(feed, tmp_path, monkeypatch):
    """Owner, 2026-10-14, of a month that had already turned out: *"So when did
    I call this? ... it is hard to see when I called it."*

    The live feed is force-published on every run and keeps no history, so it
    cannot answer that by remembering. `us_sectors.py` RECONSTRUCTS it instead
    - one vintage per month-end, each seeing only what had been published by
    then - and `attach_calls` stamps it onto the months.
    """
    import csv as _csv
    months = [m["month"] for m in feed["months"]]
    assert len(months) >= 3

    rows = [{"month": months[0], "asof": "2026-05-31", "quad": 4, "close": "True", "horizon": 2, "status": "forecast"},
            {"month": months[1], "asof": "2026-06-30", "quad": 1, "close": "False", "horizon": 2, "status": "forecast"},
            # a month the record has no call for, and one that is not in the feed
            {"month": "1999-01", "asof": "1998-12-31", "quad": 2, "close": "False", "horizon": 2, "status": "forecast"}]
    out = tmp_path / "us_regime_monthly_called.csv"
    with out.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["month", "asof", "quad", "close", "horizon", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    monkeypatch.setattr(us_feed, "RESULTS_DIR", tmp_path)

    f = json.loads(json.dumps(feed))
    for m in f["months"]:                       # start from a feed with no calls on it
        for k in ("called_regime", "called_asof", "called_too_close"):
            m.pop(k, None)
    assert us_feed.attach_calls(f) == 2

    by = {m["month"]: m for m in f["months"]}
    assert by[months[0]]["called_regime"] == 4 and by[months[0]]["called_asof"] == "2026-05-31"
    assert by[months[0]]["called_too_close"] is True
    assert by[months[1]]["called_regime"] == 1 and by[months[1]]["called_too_close"] is False
    # a month the record says nothing about is left alone rather than guessed at
    assert "called_regime" not in by[months[2]]
    # the call is the call, never the outcome: nothing here grades it
    assert not any(k.startswith("called_right") or k == "called_hit" for k in by[months[0]])


def test_a_missing_or_broken_call_record_does_not_stop_the_feed(feed, tmp_path, monkeypatch):
    """`us_sectors.py` is continue-on-error in the workflow, so the record can
    simply be absent. A feed that refused to publish over it would take the
    regimes down with it."""
    monkeypatch.setattr(us_feed, "RESULTS_DIR", tmp_path)
    f = json.loads(json.dumps(feed))
    for m in f["months"]:                       # the fixture arrives already stamped
        for k in ("called_regime", "called_asof", "called_too_close"):
            m.pop(k, None)
    assert us_feed.attach_calls(f) == 0                      # no file at all

    import csv as _csv
    out = tmp_path / "us_regime_monthly_called.csv"
    with out.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["month", "asof", "quad", "close", "horizon", "status"])
        w.writeheader()
        w.writerow({"month": f["months"][0]["month"], "asof": "", "quad": "", "close": "", "horizon": "", "status": ""})
        w.writerow({"month": f["months"][0]["month"], "asof": "2026-05-31", "quad": "7", "close": "", "horizon": "", "status": ""})
    assert us_feed.attach_calls(f) == 0                      # blank and out-of-range rows are dropped
    assert "called_regime" not in f["months"][0]
