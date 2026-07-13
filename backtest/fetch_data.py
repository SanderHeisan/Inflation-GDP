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
POWER_API = ("https://www.hvakosterstrommen.no/api/v1/prices/"
             "{y}/{m:02d}-{d:02d}_{area}.json")
POWER_AREAS = ("NO1", "NO2", "NO5")   # southern areas, most CPI-relevant
REAL_POWER_START = "2021-12"          # hvakosterstrommen history begins

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
    labels = cpi_groups.attrs.get("labels", {}) \
        if hasattr(cpi_groups, "attrs") else {}
    for col in cpi_groups.columns:
        code = str(col).replace(".", "").replace("_", "")
        text = str(labels.get(col, col)).lower()
        if "0451" in code or "elek" in text or "electricity" in text \
                or "elek" in str(col).lower():
            idx = cpi_groups[col].dropna()
            base = idx[idx.index.year == 2015].mean()
            if base and base == base:
                return idx / base * anchor_2015_ore
    return None


def area_day_mean_ore(d) -> float | None:
    """Mean hourly spot across the southern price areas for one day, in
    ore/kWh incl VAT, or None if not available."""
    import requests
    prices = []
    for area in POWER_AREAS:
        url = POWER_API.format(y=d.year, m=d.month, d=d.day, area=area)
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        prices.extend(h["NOK_per_kWh"] for h in r.json())
    return (sum(prices) / len(prices)) * 100.0 * 1.25 if prices else None


def real_power_monthly(start: str = REAL_POWER_START, end=None,
                       sample_days=(4, 11, 18, 25),
                       day_mean_fn=None) -> pd.Series:
    """Real Nord Pool area prices as monthly means (sampled days), in
    ore/kWh incl VAT. This is what replaces the CPI-subindex power proxy
    for 2021+ - the June-2026 miss was driven by an energy move the stale
    proxy could not see."""
    import datetime as dt
    day_mean_fn = day_mean_fn or area_day_mean_ore
    end = pd.Period(end, freq="M") if end \
        else pd.Period(dt.date.today(), freq="M") - 1
    vals = {}
    for m in pd.period_range(pd.Period(start, freq="M"), end, freq="M"):
        samples = [v for dom in sample_days
                   if (v := day_mean_fn(dt.date(m.year, m.month, dom)))
                   is not None]
        if samples:
            vals[m] = sum(samples) / len(samples)
    return pd.Series(vals, dtype=float)


def build_market_proxy(data_dir: Path,
                       cpi_groups: pd.DataFrame | None = None,
                       day_mean_fn=None) -> str:
    usdnok = fetch_norges_bank_monthly("USD")
    brent = fetch_brent_monthly()

    power = None
    if cpi_groups is not None:
        power = power_proxy_from_cpi(cpi_groups)
    if power is None:
        print("WARNING: could not locate the CPI electricity sub-index; "
              "using a flat 80 ore/kWh placeholder for the pre-2021 years.")
        power = pd.Series(80.0, index=usdnok.index)

    mode = "proxy"
    try:
        print("sampling real Nord Pool area prices (2021+, a few days per "
              "month - takes a minute)...")
        real = real_power_monthly(day_mean_fn=day_mean_fn)
        if len(real) >= 12:
            overlap = power.index.intersection(real.index)
            if len(overlap):   # scale the pre-real proxy into real units
                ratio = float((real.loc[overlap] / power.loc[overlap]).mean())
                pre = power.loc[power.index < real.index[0]] * ratio
                power = pd.concat([pre, real])
            else:
                power = real
            mode = "real-2021"
            print(f"power column: REAL ore/kWh from {real.index[0]} "
                  f"(proxy rescaled before that)")
    except Exception as e:
        print(f"real power sampling failed ({e}) - keeping the CPI proxy; "
              "the electricity block stays blurry")
    if mode == "proxy":
        print("NOTE: power_ore_kwh is a PROXY rescaled from the CPI "
              "electricity sub-index; replace with Nord Pool history for "
              "production use.")

    market = pd.DataFrame({"power_ore_kwh": power, "brent_usd": brent,
                           "usdnok": usdnok}).dropna()
    market.rename_axis("period").to_csv(
        data_dir / btconfig.DATA_FILES["market"])
    print(f"wrote {btconfig.DATA_FILES['market']} "
          f"({market.index[0]}..{market.index[-1]})")
    return mode


def _total_col(groups: pd.DataFrame) -> pd.Series:
    cols = [c for c in groups.columns
            if str(c).upper() in ("TOTAL", "TOTALT", "ALLGRUPPER")]
    return (groups[cols[0]] if cols else groups.iloc[:, 0]).dropna()


def _division_col(groups: pd.DataFrame, division: str,
                  keywords: tuple = ()) -> pd.Series | None:
    """COICOP division sub-index from a group frame: match by embedded
    numeric code ('01', 'JA01', '01.1'), else by label keywords (the
    rebased tables carry codes with no recognizable numbers - matching the
    Norwegian/English labels is what survives)."""
    for c in groups.columns:
        code = "".join(ch for ch in str(c) if ch.isdigit() or ch == ".")
        if code == division:
            return groups[c].dropna()
    labels = groups.attrs.get("labels", {}) if hasattr(groups, "attrs") \
        else {}
    for c in groups.columns:
        text = str(labels.get(c, c)).lower()
        if keywords and any(k in text for k in keywords):
            return groups[c].dropna()
    return None


def _write_subindices(d: Path, groups: pd.DataFrame, ds=None) -> None:
    """Food and imported-goods momentum series: the observable inputs for
    the two CPI blocks whose static assumptions caused the June-2026 level
    miss. Best-effort - absent columns just keep the old constants. If the
    active table exposes no division breakdown (Kpi10 returns TOTAL only),
    fall back to the legacy table's divisions - stale at the rebase
    boundary, but trailing momentum beats a constant."""
    if len(groups.columns) < 3 and ds is not None:
        legacy = ds.config.SSB_TABLES.get("cpi_legacy", "03013")
        try:
            legacy_groups = ds.fetch_cpi_by_group(table_id=legacy)
            if len(legacy_groups.columns) >= 3:
                print(f"  division breakdown unavailable in the active "
                      f"table - using legacy {legacy} (ends "
                      f"{legacy_groups.index[-1]}) for sub-indices")
                groups = legacy_groups
        except Exception as e:
            print(f"  legacy sub-index fetch failed: {e}")
    food = _division_col(groups, "01", ("matvarer", "food"))
    if food is not None and len(food) > 24:
        food.rename("value").rename_axis("period").to_csv(
            d / btconfig.DATA_FILES["cpi_food"])
        print(f"wrote {btconfig.DATA_FILES['cpi_food']} "
              f"(food sub-index, ends {food.index[-1]})")
    clothing = _division_col(groups, "03", ("klær", "clothing"))
    furnishings = _division_col(groups, "05", ("møbler", "furnish"))
    # Information & communication: where electronics deflation lives (ICT
    # equipment printed -8.5 MoM in June 2026, one of the two components
    # behind that month's miss).
    ict = _division_col(groups, "08", ("informasjon", "communication"))
    parts = [s for s in (clothing, furnishings, ict) if s is not None]
    if parts:
        imported = pd.concat(parts, axis=1).mean(axis=1).dropna()
        if len(imported) > 24:
            imported.rename("value").rename_axis("period").to_csv(
                d / btconfig.DATA_FILES["cpi_imported"])
            print(f"wrote {btconfig.DATA_FILES['cpi_imported']} "
                  f"(imported-goods proxy, ends {imported.index[-1]})")
    if food is None and not parts:
        labels = groups.attrs.get("labels", {})
        shown = [f"{c!r}={labels.get(c, '?')!r}"
                 for c in list(groups.columns)[:10]]
        print("sub-index columns not recognized - available columns: "
              + "; ".join(shown))


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


def _discover_replacement_table(
        ds, old_cpi: pd.Series) -> tuple[pd.Series, pd.DataFrame | None] | None:
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
          f"{best.index[-1]}; remembered for future runs")
    _discover_replacement_table.last_adopted = best_id
    groups = None
    try:   # group frame from the adopted table keeps the power proxy fresh
        groups = ds.fetch_cpi_by_group(table_id=best_id)
    except Exception as e:
        print(f"  (could not fetch group breakdown from {best_id}: {e})")
    return _splice_series(old_cpi, best), groups


def _fetch_cpi_freshest(ds,
                        data_dir: Path | None = None
                        ) -> tuple[pd.Series, pd.DataFrame]:
    """Fetch the CPI index; if the default table errors or has gone stale
    (base-year rebases both freeze old tables and break ids), fall back to
    the legacy table for history, probe alternative content series, then
    search StatBank for the successor table. Whatever wins gets the old
    history spliced on so YoY arithmetic is continuous. A successful
    discovery persists its table id so later runs skip the detour."""
    override_file = (Path(data_dir) / ".cpi_table_override") if data_dir \
        else None
    if override_file is not None and override_file.exists():
        adopted = override_file.read_text().strip()
        try:
            groups = ds.fetch_cpi_by_group(table_id=adopted)
            cpi = _total_col(groups)
            age = (pd.Timestamp.today() - cpi.index[-1].end_time).days
            if age <= 75:
                print(f"  using previously adopted table {adopted!r} "
                      f"(ends {cpi.index[-1]})")
                return cpi, groups
        except Exception as e:
            print(f"  previously adopted table {adopted!r} failed ({e})")

    default_failed = None
    try:
        groups = ds.fetch_cpi_by_group()
        cpi = _total_col(groups)
    except Exception as e:
        default_failed = e
        print(f"  default CPI table {ds.config.SSB_TABLES['cpi']!r} "
              f"failed ({e}) - falling back to legacy table")
        legacy = ds.config.SSB_TABLES.get("cpi_legacy", "03013")
        groups = ds.fetch_cpi_by_group(table_id=legacy)
        cpi = _total_col(groups)

    age = (pd.Timestamp.today() - cpi.index[-1].end_time).days
    if age <= 75:
        return cpi, groups

    active_table = ds.config.SSB_TABLES.get("cpi_legacy", "03013") \
        if default_failed else ds.config.SSB_TABLES["cpi"]
    print(f"  CPI series from table {active_table!r} ends {cpi.index[-1]} "
          f"({age} days old) - probing alternative index series "
          "(base-year rebase?)")
    try:
        meta = ds.get_table_metadata(active_table)
        contents = ds._get_variable(meta, "ContentsCode") or {}
        cands = [v for v, t in zip(contents.get("values", []),
                                   contents.get("valueTexts", []))
                 if "index" in str(t).lower()
                 and not any(x in str(t).lower() for x in ("change", "rate"))]
        best, best_groups = cpi, groups
        for code in cands[:6]:
            try:
                g = ds.fetch_cpi_by_group(content_code=code,
                                          table_id=active_table)
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
        result = _discover_replacement_table(ds, cpi)
        if result is not None:
            replaced, new_groups = result
            if override_file is not None and getattr(
                    _discover_replacement_table, "last_adopted", None):
                override_file.write_text(
                    _discover_replacement_table.last_adopted)
            return replaced, (new_groups if new_groups is not None
                              else groups)
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

    print(f"fetching SSB CPI ({ds.config.SSB_TABLES['cpi']})...")
    cpi, cpi_groups = _fetch_cpi_freshest(ds, data_dir=d)
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
    try:
        _write_subindices(d, cpi_groups, ds=ds)
    except Exception as e:
        print(f"(sub-index extraction failed: {e} - block assumptions "
              "fall back to constants)")

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
            mode = build_market_proxy(d, cpi_groups)
            marker.write_text(mode)
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
