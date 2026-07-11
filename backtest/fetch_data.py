"""
Populate data/ with the cached series the backtest needs. Run this on a
machine with network access - e.g. Google Colab or your laptop (the hosted
Claude sandbox blocks data.ssb.no and norges-bank.no):

    python -m backtest.fetch_data [--data-dir data]

Fetched automatically:
    gdp_mainland_q.csv   SSB 09190 mainland GDP, constant prices, SA (current vintage)
    cpi_monthly.csv      SSB 03013 total CPI index (unrevised by construction)
    i44_monthly.csv      Norges Bank I-44 krone index
    market_monthly.csv   BEST-EFFORT PROXY, built unless the file already exists:
                           usdnok  - Norges Bank EXR API (real data)
                           brent   - FRED DCOILBRENTEU, monthly mean (real data)
                           power   - SSB CPI electricity sub-index rescaled to
                                     ore/kWh (PROXY: post-2021 it is measured
                                     after stromstotte, so the subsidy formula
                                     partly double-counts; replace with true
                                     Nord Pool history for production use)
    wage_norms.csv       approximate frontfag settlement norms, written only
                         if absent - verify against TBU reports before
                         trusting the rent/services components

Cannot be fetched for you:

  gdp_vintages.csv - TRUE GDP VINTAGES (Norges Bank real-time database).
    Norges Bank maintains real-time vintages of the Norwegian QNA (the
    dataset behind their nowcasting/forecast-evaluation research; see
    norges-bank.no -> Statistics, or contact their statistics unit).
    Convert whatever export you obtain to a CSV shaped as:
        first column: quarter ('2012Q1'), one row per quarter
        remaining columns: vintage/publication dates ('2012-05-15'),
        values: mainland GDP level as published in that vintage (blank where
        the quarter did not exist yet).
    If the file is present the backtest switches to revision_mode='realtime'
    automatically; otherwise it runs the flagged revision-noise model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import btconfig

NORGES_BANK_EXR = ("https://data.norges-bank.no/api/data/EXR/M.{base}.NOK.SP"
                   "?format=sdmx-json&startPeriod={start}")
FRED_BRENT = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

# Approximate frontfag settlement norms (annual wage growth, %). These are
# from memory of TBU/NHO-LO reporting, NOT verified figures - good enough to
# anchor the rent/services blocks in a backtest, but check the TBU reports
# before quoting results that hinge on them.
WAGE_NORMS_APPROX = {
    2005: 3.3, 2006: 4.1, 2007: 5.4, 2008: 5.6, 2009: 4.2, 2010: 3.7,
    2011: 3.9, 2012: 4.1, 2013: 3.4, 2014: 3.3, 2015: 2.7, 2016: 2.4,
    2017: 2.4, 2018: 2.8, 2019: 3.2, 2020: 1.7, 2021: 2.7, 2022: 3.7,
    2023: 5.2, 2024: 5.2, 2025: 4.4,
}


def fetch_norges_bank_monthly(base: str, start: str = "2000-01") -> pd.Series:
    """Monthly spot series from the Norges Bank EXR API (e.g. base='USD')."""
    import requests
    url = NORGES_BANK_EXR.format(base=base, start=start)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    series = js["data"]["dataSets"][0]["series"]
    obs = next(iter(series.values()))["observations"]
    periods = js["data"]["structure"]["dimensions"]["observation"][0]["values"]
    idx = pd.PeriodIndex([p["id"] for p in periods], freq="M")
    vals = [obs.get(str(i), [None])[0] for i in range(len(idx))]
    return pd.Series(pd.to_numeric(vals, errors="coerce"), index=idx).dropna()


def fetch_brent_monthly() -> pd.Series:
    """Brent front price from FRED (no API key), averaged to monthly."""
    df = pd.read_csv(FRED_BRENT)
    dates = pd.to_datetime(df.iloc[:, 0])
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=pd.PeriodIndex(dates, freq="M"))
    return s.groupby(level=0).mean().dropna()


def power_proxy_from_cpi(cpi_groups: pd.DataFrame,
                         anchor_2015_ore: float = 30.0) -> pd.Series | None:
    """Rescale the CPI electricity sub-index to an ore/kWh-shaped series
    (2015 average anchored). A proxy, clearly - see module docstring."""
    for col in cpi_groups.columns:
        code = str(col).replace(".", "").replace("_", "")
        if "0451" in code or "elek" in str(col).lower():
            idx = cpi_groups[col].dropna()
            base = idx[idx.index.year == 2015].mean()
            if base and base == base:
                return idx / base * anchor_2015_ore
    return None


def build_market_proxy(data_dir: Path,
                       cpi_groups: pd.DataFrame | None = None) -> None:
    usdnok = fetch_norges_bank_monthly("USD")
    brent = fetch_brent_monthly()

    power = None
    if cpi_groups is not None:
        power = power_proxy_from_cpi(cpi_groups)
    if power is None:
        print("WARNING: could not locate the CPI electricity sub-index; "
              "using a flat 80 ore/kWh placeholder. The electricity block "
              "will carry no signal until you supply real power history.")
        power = pd.Series(80.0, index=usdnok.index)
    else:
        print("NOTE: power_ore_kwh is a PROXY rescaled from the CPI "
              "electricity sub-index; replace with Nord Pool history for "
              "production use.")

    market = pd.DataFrame({"power_ore_kwh": power, "brent_usd": brent,
                           "usdnok": usdnok}).dropna()
    market.rename_axis("period").to_csv(
        data_dir / btconfig.DATA_FILES["market"])
    print(f"wrote {btconfig.DATA_FILES['market']} "
          f"({market.index[0]}..{market.index[-1]})")


def _total_col(groups: pd.DataFrame) -> pd.Series:
    cols = [c for c in groups.columns
            if str(c).upper() in ("TOTAL", "TOTALT", "ALLGRUPPER")]
    return (groups[cols[0]] if cols else groups.iloc[:, 0]).dropna()


SSB_SEARCH = "https://data.ssb.no/api/v0/en/table/?query={q}"


def _splice_series(old: pd.Series, new: pd.Series) -> pd.Series:
    """Continue `old` with `new` (a rebased/successor series): rescale the
    old history onto the new base at the overlap so MoM/YoY arithmetic
    stays continuous across the junction."""
    if new.index[0] <= old.index[0]:
        return new.dropna()
    overlap = old.index.intersection(new.index)
    if not len(overlap):
        print("  WARNING: no overlap between old and new CPI series - "
              "keeping the longer one; YoY near the junction unreliable")
        return (new if len(new) >= 15 * 12 else old).dropna()
    ratio = float((new.loc[overlap] / old.loc[overlap]).mean())
    junction = new.index[0]
    spliced = pd.concat([old.loc[old.index < junction] * ratio, new])
    print(f"  spliced pre-{junction} history onto the new base "
          f"(ratio {ratio:.4f})")
    return spliced.dropna()


def _search_ssb_tables(query: str) -> list[tuple[str, str]]:
    import requests
    from urllib.parse import quote
    r = requests.get(SSB_SEARCH.format(q=quote(query)), timeout=30)
    r.raise_for_status()
    js = r.json()
    hits = js if isinstance(js, list) else js.get("tables", [])
    out = []
    for h in hits:
        tid = str(h.get("id", "")).strip()
        title = str(h.get("title") or h.get("text") or h.get("label") or "")
        if tid:
            out.append((tid, title))
    return out


def _try_cpi_from_table(ds, table_id: str) -> pd.Series | None:
    """Best-effort: pull a monthly total-CPI index series out of an
    arbitrary CPI-shaped StatBank table."""
    meta = ds.get_table_metadata(table_id)
    variables = meta.get("variables", [])
    tvar = next((v for v in variables
                 if v.get("code", "").lower() in ("tid", "time")), None)
    if not tvar or "M" not in str(tvar.get("values", ["?"])[-1]).upper():
        return None   # not a monthly table

    query = []
    gvar = next((v for v in variables
                 if "konsum" in v.get("code", "").lower()
                 or "coicop" in v.get("code", "").lower()), None)
    if gvar:
        total = next((val for val, t in zip(gvar["values"],
                                            gvar["valueTexts"])
                      if str(val).upper() in ("TOTAL", "TOTALT")
                      or "all-item" in str(t).lower()
                      or str(t).lower() in ("total", "cpi total")),
                     gvar["values"][0])
        query.append({"code": gvar["code"],
                      "selection": {"filter": "item", "values": [total]}})
    contents = ds._get_variable(meta, "ContentsCode")
    if contents:
        ccode = ds._match_value(contents, must=("index",),
                                exclude=("change", "rate"))
        if ccode:
            query.append({"code": "ContentsCode",
                          "selection": {"filter": "item", "values": [ccode]}})

    js = ds._post_query(table_id, query)
    df = ds._jsonstat_to_frame(js)
    tcol = next((c for c in df.columns
                 if str(c).lower() in ("tid", "time")), None)
    if tcol is None:
        return None
    df["period"] = pd.PeriodIndex(df[tcol].str.replace("M", "-"), freq="M")
    s = df.groupby("period")["value"].first().sort_index().dropna()
    return s if len(s) > 24 else None


def _discover_replacement_table(ds, old_cpi: pd.Series) -> pd.Series | None:
    """The frozen-table case (e.g. the 2026 base-year rebase retired the
    old CPI table): search StatBank for successor tables, probe the most
    plausible ones, adopt the freshest, splice histories."""
    print("  searching StatBank for a successor CPI table...")
    try:
        hits = _search_ssb_tables("consumer price index")
    except Exception as e:
        print(f"  table search failed: {e}")
        return None
    best, best_id, best_title = None, None, None
    checked = 0
    for tid, title in hits:
        t = title.lower()
        if tid == ds.config.SSB_TABLES["cpi"] or "consumer price" not in t:
            continue
        if any(x in t for x in ("harmonis", "delivery", "seasonal",
                                "closed", "discontinued")):
            continue
        if checked >= 8:
            break
        checked += 1
        try:
            s = _try_cpi_from_table(ds, tid)
        except Exception:
            continue
        if s is not None and (best is None or s.index[-1] > best.index[-1]):
            best, best_id, best_title = s, tid, title
    if best is None or best.index[-1] <= old_cpi.index[-1]:
        print("  no fresher CPI table found automatically. Manual escape "
              "hatch: export the current CPI index to data/cpi_monthly.csv "
              "- a hand-supplied file that is fresher than the fetch is "
              "kept, never overwritten.")
        return None
    print(f"  ADOPTED table {best_id} ('{best_title[:70]}') - runs to "
          f"{best.index[-1]}")
    print(f"  -> make it permanent: set SSB_TABLES['cpi'] = {best_id!r} "
          "in quadmap/config.py")
    return _splice_series(old_cpi, best)


def _fetch_cpi_freshest(ds) -> tuple[pd.Series, pd.DataFrame]:
    """Fetch the CPI index; if the default content series has gone stale
    (typically a base-year rebase), first probe the same table's other
    index series, then search StatBank for the successor table. Whatever
    wins gets the old history spliced on so YoY arithmetic is continuous."""
    groups = ds.fetch_cpi_by_group()
    cpi = _total_col(groups)
    age = (pd.Timestamp.today() - cpi.index[-1].end_time).days
    if age <= 75:
        return cpi, groups

    print(f"  default CPI series ends {cpi.index[-1]} ({age} days old) - "
          "probing alternative index series (base-year rebase?)")
    try:
        meta = ds.get_table_metadata(ds.config.SSB_TABLES["cpi"])
        contents = ds._get_variable(meta, "ContentsCode") or {}
        cands = [v for v, t in zip(contents.get("values", []),
                                   contents.get("valueTexts", []))
                 if "index" in str(t).lower()
                 and not any(x in str(t).lower() for x in ("change", "rate"))]
        best, best_groups = cpi, groups
        for code in cands[:6]:
            try:
                g = ds.fetch_cpi_by_group(content_code=code)
                s = _total_col(g)
            except Exception:
                continue
            if s.index[-1] > best.index[-1]:
                best, best_groups = s, g
                print(f"  ContentsCode {code!r} runs to {s.index[-1]} - "
                      "using it")
        if best is not cpi:
            return _splice_series(cpi, best), best_groups
    except Exception as e:
        print(f"  alternative-series probe failed: {e}")

    try:   # same table exhausted -> the table itself is frozen
        replaced = _discover_replacement_table(ds, cpi)
        if replaced is not None:
            return replaced, groups
    except Exception as e:
        print(f"  successor-table discovery failed: {e}")
    return cpi, groups


def _market_stale(market_file: Path, cpi: pd.Series) -> bool:
    try:
        m = pd.read_csv(market_file)
        last = pd.Period(m["period"].iloc[-1], freq="M")
        return last < cpi.index[-1] - 1
    except Exception:
        return True


def _staleness_check(name: str, last_period, max_age_days: int,
                     ds, table_id: str) -> None:
    """A series ending long before today means either the table was frozen
    (SSB restructure) or our chosen variable code stopped being populated.
    Compare against the table's own latest time value to tell which."""
    age = (pd.Timestamp.today() - last_period.end_time).days
    if age <= max_age_days:
        return
    print(f"\nWARNING: {name} ends at {last_period} - {age} days old. ")
    try:
        meta = ds.get_table_metadata(table_id)
        tid = next((v for v in meta.get("variables", [])
                    if v.get("code", "").lower() in ("tid", "time")), None)
        latest_in_table = tid["values"][-1] if tid else "?"
        print(f"  Table {table_id} itself runs to: {latest_in_table}")
        if tid and str(last_period.year) not in str(latest_in_table):
            print("  -> the TABLE has newer data than our series: the "
                  "selected variable code stopped being populated; inspect "
                  f"get_table_metadata('{table_id}') and update the query.")
        else:
            print("  -> the table appears frozen/discontinued: SSB likely "
                  "replaced it. Search ssb.no/en/statbank for the successor "
                  f"table and update SSB_TABLES in quadmap/config.py.")
    except Exception as e:
        print(f"  (could not fetch table metadata for diagnosis: {e})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(btconfig.DATA_DIR))
    ap.add_argument("--skip-market-proxy", action="store_true",
                    help="do not build the best-effort market_monthly.csv")
    args = ap.parse_args(argv)
    d = Path(args.data_dir)
    d.mkdir(parents=True, exist_ok=True)

    from quadmap import data_sources as ds

    print("fetching SSB mainland GDP (09190)...")
    gdp = ds.fetch_mainland_gdp()
    gdp.rename("value").rename_axis("period").to_csv(
        d / btconfig.DATA_FILES["gdp"])

    print("fetching SSB CPI (03013)...")
    cpi, cpi_groups = _fetch_cpi_freshest(ds)
    cpi_path = d / btconfig.DATA_FILES["cpi"]
    if cpi_path.exists():   # manual escape hatch: never clobber fresher data
        try:
            from .data_bundle import _read_period_series
            existing = _read_period_series(cpi_path, "M")
            if existing.index[-1] > cpi.index[-1]:
                print(f"keeping existing {cpi_path.name} (ends "
                      f"{existing.index[-1]}, fresher than the fetch)")
                cpi = existing
        except Exception:
            pass
    cpi.rename("value").rename_axis("period").to_csv(cpi_path)

    print("fetching Norges Bank I-44...")
    i44 = ds.fetch_i44()
    i44.rename("value").rename_axis("period").to_csv(
        d / btconfig.DATA_FILES["i44"])
    print(f"wrote GDP/CPI/I-44 to {d.resolve()}")
    print(f"series end:  GDP {gdp.index[-1]} | CPI {cpi.index[-1]} | "
          f"I-44 {i44.index[-1]}")
    try:   # diagnostics must never kill the fetch
        _staleness_check("CPI", cpi.index[-1], 75,
                         ds, ds.config.SSB_TABLES["cpi"])
        _staleness_check("GDP", gdp.index[-1], 160,
                         ds, ds.config.SSB_TABLES["gdp_qna"])
    except Exception as e:
        print(f"(staleness diagnostics failed: {e})")

    market_file = d / btconfig.DATA_FILES["market"]
    marker = d / ".market_is_proxy"   # only files WE built get rebuilt
    rebuild = (not market_file.exists()
               or (marker.exists() and _market_stale(market_file, cpi)))
    if market_file.exists() and not marker.exists():
        print(f"{market_file.name} is user-supplied - not overwriting")
    elif args.skip_market_proxy:
        print(f"MANUAL: {market_file.name} skipped - see module docstring")
    elif rebuild:
        try:
            build_market_proxy(d, cpi_groups)
            marker.touch()
        except Exception as e:   # non-fatal: backtest needs it, but the user
            print(f"market proxy build failed ({e}) - see module docstring "
                  "for the manual format")
    else:
        print(f"{market_file.name} proxy is current - keeping it")

    norms_file = d / btconfig.DATA_FILES["wage_norms"]
    if not norms_file.exists():
        pd.Series(WAGE_NORMS_APPROX, name="norm_pct").rename_axis(
            "year").to_csv(norms_file)
        print(f"wrote {norms_file.name} (APPROXIMATE frontfag norms - "
              "verify against TBU reports)")

    vintages = d / btconfig.DATA_FILES["gdp_vintages"]
    if not vintages.exists():
        print(f"MANUAL: {vintages.name} not present - backtest will run in "
              "revision-noise mode. See module docstring for how to wire in "
              "the Norges Bank real-time database.")


if __name__ == "__main__":
    main()
