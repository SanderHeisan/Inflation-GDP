"""
Daily point-in-time indicator archive.

    python -m backtest.snapshots --out results/indicator_snapshots.csv

The growth-side upgrade (Regional Network, jobs, industrial production,
retail in the GDP nowcast) is blocked on one thing: honest historical data.
Backtesting against today's *revised* series is cheating; what's needed is
the value as it stood on each date. Public archives of that barely exist -
so we build our own: every daily run appends the CURRENT latest value of
each indicator, stamped with today's date. In a year this file is a true
real-time archive, and the nowcast weights can be calibrated on it without
leakage.

Indicators are resolved by StatBank search (table ids rot - see the CPI
rebase) and cached; every fetch is best-effort so one broken indicator
never blocks the others.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

# search: StatBank query to (re)locate the table; must: words the table
# title must contain; contents_must: words to pick the ContentsCode by.
INDICATORS = [
    {"name": "retail_volume_index", "table": "07129",
     "search": "retail sales volume index",
     "must": ("retail",), "contents_must": ("volume", "index")},
    {"name": "industrial_production_index", "table": None,
     "search": "index of industrial production",
     "must": ("industrial", "production"), "contents_must": ("index",)},
    {"name": "registered_unemployed", "table": None,
     "search": "registered unemployed nav",
     "must": ("unemploy",), "contents_must": ()},
]


def _resolve_table(ds, spec: dict, cache: dict) -> str | None:
    if cache.get(spec["name"]):
        return cache[spec["name"]]
    if spec.get("table"):
        cache[spec["name"]] = spec["table"]
        return spec["table"]
    from .fetch_data import _search_ssb_tables
    for tid, title in _search_ssb_tables(spec["search"]):
        t = title.lower()
        if all(w in t for w in spec["must"]) \
                and not any(x in t for x in ("closed", "discontinued")):
            cache[spec["name"]] = tid
            return tid
    return None


def _latest_value(ds, table_id: str, spec: dict) -> tuple[str, float]:
    """Newest published (period, value) for the indicator: every non-time
    dimension collapses to its first category (conventionally the total),
    the ContentsCode is matched by label, and Tid uses the 'top' filter."""
    meta = ds.get_table_metadata(table_id)
    query = []
    for v in meta.get("variables", []):
        code = v.get("code", "")
        if code.lower() in ("tid", "time"):
            query.append({"code": code,
                          "selection": {"filter": "top", "values": ["1"]}})
        elif code == "ContentsCode":
            ccode = ds._match_value(v, must=spec["contents_must"]) \
                or v["values"][0]
            query.append({"code": code,
                          "selection": {"filter": "item", "values": [ccode]}})
        else:
            query.append({"code": code,
                          "selection": {"filter": "item",
                                        "values": [v["values"][0]]}})
    js = ds._post_query(table_id, query)
    df = ds._jsonstat_to_frame(js)
    tcol = next(c for c in df.columns if str(c).lower() in ("tid", "time"))
    row = df.dropna(subset=["value"]).iloc[-1]
    return str(row[tcol]), float(row["value"])


def snapshot_indicators(out_file: str | Path,
                        cache_file: str | Path | None = None,
                        snap_date: str | None = None) -> pd.DataFrame:
    from quadmap import data_sources as ds

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_file) if cache_file \
        else out_file.with_suffix(".tables.json")
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    snap_date = snap_date or dt.date.today().isoformat()

    rows = []
    for spec in INDICATORS:
        try:
            tid = _resolve_table(ds, spec, cache)
            if tid is None:
                print(f"  {spec['name']}: no table found")
                continue
            period, value = _latest_value(ds, tid, spec)
            rows.append({"snap_date": snap_date, "indicator": spec["name"],
                         "table_id": tid, "period": period, "value": value})
            print(f"  {spec['name']}: {period} = {value} (table {tid})")
        except Exception as e:
            print(f"  {spec['name']}: snapshot failed ({e})")
    cache_file.write_text(json.dumps(cache, indent=1))

    new = pd.DataFrame(rows)
    if out_file.exists() and len(new):
        old = pd.read_csv(out_file)
        new = (pd.concat([old, new], ignore_index=True)
               .drop_duplicates(subset=["snap_date", "indicator"],
                                keep="last"))
    if len(new):
        new.to_csv(out_file, index=False)
        print(f"indicator archive: {len(new)} rows -> {out_file}")
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/indicator_snapshots.csv")
    args = ap.parse_args(argv)
    snapshot_indicators(args.out)


if __name__ == "__main__":
    main()
