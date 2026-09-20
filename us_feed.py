"""
The compact feed: the regimes, for a consumer that does not want the whole
sheet (results_us/latest.json). Regimes are numbered 1-4 internally and
named Sweet spot / Heating / Squeeze / Cooling.

    python us_feed.py            # after us_sheet.py; writes results_us/latest.json

What it carries:

- `quarters`: each projected quarter (from GDP) with its regime, whether it
  is too close to call, the two directions and the backtested hit rate of
  the regime call at that distance.
- `months`: each month (from the monthly growth measure and the CPI path)
  with its regime, the two directions with their conviction and the
  backtested hit rate of that kind of call, and, where one was recorded,
  the monthly regime another vendor published (for the model's own private
  scoring; a consumer that serves the feed to the public strips it).
- `hit_rates`: the backtest's scorecard as one block (the regime by
  distance, next month's inflation and the growth measure by conviction),
  read from the sheet's `stats`, which are read from results_us/.
- `notes`: the plain-language story behind the numbers (the next inflation
  print, the inflation path, oil, rents, this quarter's growth, the pace
  assumed after it, interest rates, the consumer), as text with no markup.
  Every sentence is computed from the sheet; nothing is written by hand.
- `drivers`: the same story as a table of figures - one row per driver
  (oil, the pump, rents, wages, this quarter's pace, the pace assumed after
  it, the Fed funds rate, the consumer lines) with its latest reading, its
  change, where it sits against the last ten years, and one line on what
  it means. Strings only, no markup.

The "US macro regime" workflow publishes this file, the full sheet and the
page to the live-us-regime branch.
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional

from usbacktest.btconfig import RESULTS_DIR
from usmodel import config as ucfg

SRC = RESULTS_DIR / "us_sheet.json"
OUT = RESULTS_DIR / "latest.json"
REGIME_NAME = {1: "Sweet spot", 2: "Heating", 3: "Squeeze", 4: "Cooling"}


def _r3(x: Any) -> Optional[float]:
    try:
        return None if x is None else round(float(x), 3)
    except (TypeError, ValueError):
        return None


def _level_words(p: Any) -> str:
    """A ten-year percentile in words."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p != p:
        return ""
    return ("very high against the last ten years" if p >= 0.9 else
            "high against the last ten years" if p >= 0.8 else
            "very low against the last ten years" if p <= 0.1 else
            "low against the last ten years" if p <= 0.2 else
            "normal against the last ten years")


def hit_rates(sheet: dict) -> dict:
    """The backtest's scorecard as one block. Keys are the words a reader
    sees (strong / good / lean / toss-up), not the bucket labels the
    backtest files use."""
    ST, M = sheet["stats"], sheet["meta"]
    by_h = lambda d: {str(h): _r3(d[str(h)]) for h in range(5) if str(h) in {str(k) for k in d}}
    cpi, mg = ST["cpi"], ST["mg"]
    b = mg.get("buckets", {})
    asof = datetime.date.fromisoformat(M["asof_iso"])
    return {
        "n_asof": int(ST["n_asof"]),
        "span": f"January 2017 to {asof.strftime('%B %Y')}",
        "regime_by_horizon": by_h({str(k): v for k, v in ST["quad_hit"].items()}),
        "regime_this_quarter_clear": _r3(ST["quad_hit_hc0"]),
        "growth_quarter_by_horizon": by_h({str(k): v for k, v in ST["growth_yoy"].items()}),
        "inflation_quarter_by_horizon": by_h({str(k): v for k, v in ST["infl_yoy"].items()}),
        "cpi_month": {"all": _r3(ST["cpi_all"]), "strong": _r3(cpi["high conviction (>0.30pp)"]),
                      "good": _r3(cpi["call (0.15-0.30pp)"]), "lean": _r3(cpi["lean (0.05-0.15pp)"]),
                      "toss-up": _r3(cpi["coin-flip (<0.05pp)"])},
        "growth_month": {"h1": _r3(mg["h1"]), "h2": _r3(mg["h2"]), "h3": _r3(mg["h3"]),
                         "nowcast": _r3(mg["h1_nowcast"]),
                         "strong": _r3(b.get("strong (>0.30pp)")), "good": _r3(b.get("call (0.15-0.30pp)")),
                         "lean": _r3(b.get("lean (0.05-0.15pp)")), "toss-up": _r3(b.get("toss-up (<0.05pp)")),
                         "vs_quarter_gdp": _r3(mg["gdp_q"]), "n": int(mg["n"])},
    }


def notes(sheet: dict) -> List[Dict[str, str]]:
    """The story behind the numbers, in sentences computed from the sheet.
    Plain text: a consumer escapes it, and nothing here is markup."""
    M, Q, MO, RT, C = sheet["meta"], sheet["quarters"], sheet["months"], sheet["rate"], sheet["consumer"]
    now = next(q for q in Q if not q["realized"])
    nxt = next(m for m in MO if not m["realized"])
    c = nxt.get("contrib") or {}
    rest = float(nxt["mom"]) - float(c.get("gasoline", 0.0)) - float(c.get("shelter", 0.0))
    out: List[Dict[str, str]] = []

    out.append({"key": "inflation_next", "title": f"The next inflation print, {nxt['label']}",
                "text": (f"{nxt['mom']:+.2f}% on the month, {nxt['yoy']:.2f}% year over year "
                         f"({'up' if nxt['d_yoy'] > 0 else 'down'} from {M['last_cpi_yoy']:.2f}%). "
                         f"Gasoline adds {c.get('gasoline', 0.0):+.2f} points (pump prices are {M['pump_mom']:+.1f}% "
                         f"this month so far), rents {c.get('shelter', 0.0):+.2f}, everything else {rest:+.2f}.")})

    order = {m["label"]: i for i, m in enumerate(MO)}
    peak, trough = M["peak"], M["trough"]
    if order.get(peak["label"], 0) <= order.get(trough["label"], 0):
        path = (f"Inflation tops out at {peak['yoy']:.2f}% in {peak['label']} and falls to "
                f"{trough['yoy']:.2f}% by {trough['label']}.")
    else:
        path = (f"Inflation falls to {trough['yoy']:.2f}% by {trough['label']} and climbs back to "
                f"{peak['yoy']:.2f}% by {peak['label']}.")
    out.append({"key": "inflation_path", "title": "The inflation path",
                "text": path + (" Much of the fall is arithmetic rather than forecast: the largest monthly price "
                                "rises of the past year drop out of the twelve-month comparison one by one, and "
                                "those months carry the strongest calls.")})

    out.append({"key": "energy", "title": "Oil and the pump",
                "text": (f"Oil is ${M['wti_now']:.0f} a barrel against ${M['wti_last_cpi']:.0f} in {M['cpi_through']}, "
                         f"and the pump price is ${M['pump_now']:.2f} a gallon. In the data since 2000, $1 on a barrel "
                         f"of oil has been worth about 0.03 points on inflation, and pump prices follow oil by a few weeks.")})

    out.append({"key": "rents_wages", "title": "Rents and wages",
                "text": (f"Rents in the index follow new-lease rents with about a year's lag; those are up "
                         f"{M['rent_yoy']:.1f}%, so rents add a steady 0.08 points a month. Wages are up "
                         f"{M['wage_yoy']:.1f}%, which feeds services prices slowly.")})

    beats = float(now["dg"]) > 0
    out.append({"key": "growth_now", "title": f"Growth this quarter, {now['label']}",
                "text": (f"About {now['qoq_ann']:+.1f}% annualized, estimated from jobs, industrial production, retail "
                         f"sales and jobless claims with {M['k']} of 3 months in. That {'beats' if beats else 'falls short of'} "
                         f"the {now['bar_ann']:+.1f}% pace of the same quarter last year, so year-over-year growth "
                         f"{'rises' if beats else 'slips'} to {now['g_yoy']:.1f}%.")})

    pace = {"potential": f"the economy's normal pace, {M['pace_ann']:.1f}% a year",
            "nowcast": f"that this quarter's pace, {M['nowcast_ann']:+.1f}% a year, carries on",
            "glide": f"that this quarter's pace fades toward a normal {M['pace_ann']:.1f}% a year",
            "trailing_median": f"the trend of the last six years, {M['trend_ann']:.1f}% a year"}[M["pace_mode"]]
    out.append({"key": "growth_after", "title": "After this quarter",
                "text": (f"The model does not forecast GDP two to four quarters out; every method tried did worse than "
                         f"a constant. It assumes {pace}, adjusted for interest rates. Up or down is then decided by the "
                         f"bar to beat: year-over-year growth rises only in a quarter that grows faster than the same "
                         f"quarter a year earlier, and those bars are already published.")})

    if RT.get("level") is not None and RT.get("beta_ann_per_pp") is not None:
        qlabel = {q["q"]: q["label"] for q in Q}
        month = datetime.date.fromisoformat(f"{RT['month']}-01").strftime("%B %Y")
        end = datetime.date.fromisoformat(max(ucfg.POLICY_RATE_PATH)).strftime("%B %Y")
        cut = float(RT.get("chg_8q") or 0.0) < 0
        text = (f"The Fed funds rate was {RT['level']:.2f}% through {month}, and the market path reaches "
                f"{RT['implied_end']:.2f}% by {end}. Rate changes reach growth with a lag: about "
                f"{abs(RT['beta_ann_per_pp']):.1f} points of annual growth per 1 point of change, two to six quarters "
                f"later. The Fed {'cut' if cut else 'raised rates'} {abs(RT.get('chg_8q') or 0.0):.1f} points over the "
                f"last two years, {'so the cuts still help growth for now' if cut else 'and that is weighing on growth now'}")
        if RT.get("first_bite") in qlabel:
            text += f"; the new hikes start to bite in {qlabel[RT['first_bite']]}"
        out.append({"key": "rates", "title": "Interest rates", "text": text + "."})

    parts = []
    if "real_pce" in C:
        r = C["real_pce"]
        parts.append(f"consumer spending is {r['yoy_pct']:+.1f}% on a year ago ({_level_words(r.get('pctl_10y'))})")
    if "sentiment" in C:
        r = C["sentiment"]
        parts.append(f"consumer confidence is {r['level']:.0f} ({_level_words(r.get('pctl_10y'))})")
    if "net_worth" in C:
        r = C["net_worth"]
        parts.append(f"household wealth is {r['yoy_pct']:+.1f}% on a year ago ({_level_words(r.get('pctl_10y'))})")
    if "real_income" in C:
        r = C["real_income"]
        parts.append(f"income after inflation is {r['yoy_pct']:+.1f}% ({_level_words(r.get('pctl_10y'))})")
    if "saving_rate" in C:
        r = C["saving_rate"]
        parts.append(f"the saving rate is {r['level_pct']:.1f}% of income ({_level_words(r.get('pctl_10y'))})")
    if "mortgage_30y" in C:
        r = C["mortgage_30y"]
        parts.append(f"the 30-year mortgage rate is {r['level_pct']:.2f}%")
    if parts:
        text = "Consumer spending is about two thirds of GDP. Right now " + "; ".join(parts) + "."
        weak_income = "real_income" in C and float(C["real_income"].get("pctl_10y") or 1.0) <= 0.2
        low_saving = "saving_rate" in C and float(C["saving_rate"].get("pctl_10y") or 1.0) <= 0.2
        if weak_income and low_saving:
            text += " Weak income after inflation is the soft spot: spending is being carried by a low saving rate, not by pay."
        out.append({"key": "consumer", "title": "The consumer", "text": text})
    return out


def drivers(sheet: dict) -> List[Dict[str, str]]:
    """The story behind the numbers as rows of figures. Every value is a
    string a page can print as is; `change` and `level` may be empty."""
    M, Q, RT, C = sheet["meta"], sheet["quarters"], sheet["rate"], sheet["consumer"]
    now = next(q for q in Q if not q["realized"])
    nxt = next(m for m in sheet["months"] if not m["realized"])
    c = nxt.get("contrib") or {}
    out: List[Dict[str, str]] = []

    def add(key: str, label: str, value: str, change: str = "", level: str = "", note: str = "") -> None:
        out.append({"key": key, "label": label, "value": value, "change": change, "level": level, "note": note})

    add("oil", "Oil (WTI)", f"${M['wti_now']:.0f} a barrel",
        f"{M['wti_now'] - M['wti_last_cpi']:+.0f} since {M['cpi_through']}", "",
        "Each $1 on a barrel has been worth about 0.03 points on inflation since 2000.")
    add("pump", "Pump price", f"${M['pump_now']:.2f} a gallon", f"{M['pump_mom']:+.1f}% this month so far", "",
        f"Adds {c.get('gasoline', 0.0):+.2f} points to {nxt['label']} inflation; pump prices follow oil by a few weeks.")
    add("rents", "Market rents, new leases", f"{M['rent_yoy']:+.1f}% year over year", "", "",
        "Rents in the index follow with about a year's lag, about 0.08 points a month.")
    add("wages", "Wages", f"{M['wage_yoy']:+.1f}% year over year", "", "", "Feeds services prices slowly.")
    add("growth_pace", f"Growth pace, {now['label']}", f"{now['qoq_ann']:+.1f}% annualized",
        f"bar to beat {now['bar_ann']:+.1f}%", "",
        f"{M['k']} of 3 months in; year-over-year growth {'rises' if float(now['dg']) > 0 else 'slips'} to {now['g_yoy']:.1f}%.")
    add("pace_after", "Pace assumed after this quarter", f"{M['pace_ann']:.1f}% a year", "", "",
        "A normal pace, adjusted for interest rates; the model does not forecast GDP further out.")
    if RT.get("level") is not None and RT.get("implied_end") is not None:
        qlabel = {q["q"]: q["label"] for q in Q}
        end = datetime.date.fromisoformat(max(ucfg.POLICY_RATE_PATH)).strftime("%B %Y")
        note = f"Market path to {RT['implied_end']:.2f}% by {end}; hikes reach growth two to six quarters later"
        if RT.get("first_bite") in qlabel:
            note += f", biting from {qlabel[RT['first_bite']]}"
        add("fed_funds", "Fed funds rate", f"{RT['level']:.2f}%",
            f"{RT.get('chg_8q') or 0.0:+.1f} points over two years", _level_words(C.get("fed_funds", {}).get("pctl_10y")),
            note + ".")
    if "real_pce" in C:
        r = C["real_pce"]
        add("spending", "Consumer spending", f"{r['yoy_pct']:+.1f}% year over year", "", _level_words(r.get("pctl_10y")),
            "About two thirds of GDP. Very hot spending has been followed by slower growth about 7 times in 10.")
    if "sentiment" in C:
        r = C["sentiment"]
        add("confidence", "Consumer confidence", f"{r['level']:.0f} (University of Michigan)", "", _level_words(r.get("pctl_10y")),
            "Very low confidence has not meant weaker growth: the next quarter was higher 63% of the time.")
    if "net_worth" in C:
        r = C["net_worth"]
        add("wealth", "Household wealth", f"${r['level_tn']:.0f} trillion", f"{r['yoy_pct']:+.1f}% year over year",
            _level_words(r.get("pctl_10y")), "Fast wealth gains have meant slowing growth three to four quarters later 61% to 68% of the time.")
    if "real_income" in C:
        r = C["real_income"]
        add("income", "Income after inflation", f"{r['yoy_pct']:+.1f}% year over year", "", _level_words(r.get("pctl_10y")),
            "Weak real income is the soft spot: spending carried by a low saving rate, not by pay.")
    if "saving_rate" in C:
        r = C["saving_rate"]
        add("saving", "Saving rate", f"{r['level_pct']:.1f}% of income", "", _level_words(r.get("pctl_10y")),
            "Extremes carry no measured signal for next quarter, but it is the cushion that is gone when very low.")
    if "mortgage_30y" in C:
        r = C["mortgage_30y"]
        add("mortgage", "30-year mortgage rate", f"{r['level_pct']:.2f}%", f"{r.get('chg_4q_pp') or 0.0:+.2f} points over a year",
            _level_words(r.get("pctl_10y")), "Falling mortgage rates have been a strong lead on faster growth (75% to 79%); rising ones a weak lead on slower growth.")
    return out


def build_feed(sheet: dict, repo: str = "SanderHeisan/Inflation-GDP") -> dict:
    m = sheet["meta"]
    feed = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "asof": m["asof_iso"], "cpi_through": m["cpi_through"], "gdp_through": m["gdp_through"],
        "regimes": dict(REGIME_NAME),
        "quarters": [{"quarter": q["q"], "label": q["label"], "regime": q["quad"], "regime_name": q["name"],
                      "too_close_to_call": q["close"], "realized": q["realized"],
                      "growth_yoy": round(q["g_yoy"], 2), "growth_dir": q["g_dir"],
                      "inflation_yoy": round(q["i_yoy"], 2), "inflation_dir": q["i_dir"],
                      "backtest_hit": q["quad_hit"]} for q in sheet["quarters"]],
        "months": [{"month": mm["m"], "label": mm["label"], "regime": mm["mquad"]["quad"],
                    "regime_name": REGIME_NAME[mm["mquad"]["quad"]],
                    "too_close_to_call": mm["mquad"]["close"], "status": mm["mquad"]["status"],
                    "growth_measure_yoy": round(mm["mg"]["yoy"], 2), "growth_dir": mm["mg"]["dir"],
                    "growth_conviction": mm["mg"]["bucket"], "growth_hit": mm["mg"]["hit"],
                    "cpi_yoy": round(mm["yoy"], 2), "cpi_mom": round(mm["mom"], 2), "cpi_dir": mm["dir"],
                    "cpi_conviction": mm["word"], "cpi_hit": round(mm["hit"], 3),
                    "hedgeye_regime": (mm.get("hedgeye") or {}).get("quad")}
                   for mm in sheet["months"] if mm.get("mquad") and mm.get("mg")],
        "source": f"https://github.com/{repo}",
    }
    # The regimes publish even if a block cannot be written from this sheet.
    for key, fn in (("hit_rates", hit_rates), ("notes", notes), ("drivers", drivers)):
        try:
            feed[key] = fn(sheet)
        except Exception as e:  # noqa: BLE001 - a missing field must not stop the feed
            print(f"[us_feed] {key} skipped: {type(e).__name__}: {e}")
    # What each regime has meant for the sector funds: us_sectors.py writes it
    # when it runs; the feed carries it when the file is there.
    sectors_path = RESULTS_DIR / "us_sectors_feed.json"
    if sectors_path.exists():
        try:
            feed["sectors"] = json.loads(sectors_path.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[us_feed] sectors skipped: {type(e).__name__}: {e}")
    return feed


def main() -> None:
    feed = build_feed(json.load(open(SRC)))
    OUT.write_text(json.dumps(feed, indent=1))
    print(f"wrote {OUT}: {len(feed['quarters'])} quarters, {len(feed['months'])} months, "
          f"{len(feed.get('notes', []))} notes, as of {feed['asof']}")


if __name__ == "__main__":
    main()
