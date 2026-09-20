"""
The compact feed: the quads only, for a consumer that does not want the
whole sheet (results_us/latest.json).

    python us_feed.py            # after us_sheet.py; writes results_us/latest.json

Each projected quarter (from GDP) and each month (from the monthly growth
measure) with its quad, whether it is too close to call, the two directions
with their conviction and the backtested hit rate of that kind of call, and
Hedgeye's monthly quad where one was recorded. The "US quad sheet" workflow
publishes this file, the full sheet and the page to the live-us-quad
branch; until then it is readable from the branch it was built on.
"""
from __future__ import annotations

import datetime
import json

from usbacktest.btconfig import RESULTS_DIR

SRC = RESULTS_DIR / "us_sheet.json"
OUT = RESULTS_DIR / "latest.json"


def build_feed(sheet: dict, repo: str = "SanderHeisan/Inflation-GDP") -> dict:
    m = sheet["meta"]
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "asof": m["asof_iso"], "cpi_through": m["cpi_through"], "gdp_through": m["gdp_through"],
        "quarters": [{"quarter": q["q"], "label": q["label"], "quad": q["quad"], "name": q["name"],
                      "too_close_to_call": q["close"], "realized": q["realized"],
                      "growth_yoy": round(q["g_yoy"], 2), "growth_dir": q["g_dir"],
                      "inflation_yoy": round(q["i_yoy"], 2), "inflation_dir": q["i_dir"],
                      "backtest_hit": q["quad_hit"]} for q in sheet["quarters"]],
        "months": [{"month": mm["m"], "label": mm["label"], "quad": mm["mquad"]["quad"],
                    "too_close_to_call": mm["mquad"]["close"], "status": mm["mquad"]["status"],
                    "growth_measure_yoy": round(mm["mg"]["yoy"], 2), "growth_dir": mm["mg"]["dir"],
                    "growth_conviction": mm["mg"]["bucket"], "growth_hit": mm["mg"]["hit"],
                    "cpi_yoy": round(mm["yoy"], 2), "cpi_mom": round(mm["mom"], 2), "cpi_dir": mm["dir"],
                    "cpi_conviction": mm["word"], "cpi_hit": round(mm["hit"], 3),
                    "hedgeye_quad": (mm.get("hedgeye") or {}).get("quad")}
                   for mm in sheet["months"] if mm.get("mquad") and mm.get("mg")],
        "source": f"https://github.com/{repo}",
    }


def main() -> None:
    feed = build_feed(json.load(open(SRC)))
    OUT.write_text(json.dumps(feed, indent=1))
    print(f"wrote {OUT}: {len(feed['quarters'])} quarters, {len(feed['months'])} months, as of {feed['asof']}")


if __name__ == "__main__":
    main()
