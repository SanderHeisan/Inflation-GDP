"""
Data acquisition layer.

Two live sources:
  * SSB StatBank PxWebApi (POST json queries, json-stat2 responses)
  * Norges Bank open data API (SDMX-JSON) for the I-44 krone index

Everything returns tidy pandas objects indexed by pandas.Period
(monthly 'M' or quarterly 'Q') so the model layer never touches raw API
payloads.
"""
from __future__ import annotations

import requests
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# SSB PxWeb helpers
# ---------------------------------------------------------------------------

def get_table_metadata(table_id: str) -> dict:
    """GET the table metadata: lists variable codes and valid values.

    Run this once per table when setting up -- it is how you confirm the
    exact 'Konsumgrp' / 'ContentsCode' values used in the queries below.
    """
    r = requests.get(config.SSB_BASE + table_id, timeout=30)
    r.raise_for_status()
    return r.json()


def _get_variable(meta: dict, code: str) -> dict | None:
    """Find one variable block (code + values + valueTexts) in PxWeb table
    metadata, case-insensitively."""
    for v in meta.get("variables", []):
        if v.get("code", "").lower() == code.lower():
            return v
    return None


def _match_value(var: dict, must: tuple = (), exclude: tuple = ()) -> str | None:
    """First value whose English label contains every `must` keyword and no
    `exclude` keyword. SSB renames ContentsCodes when the base year rolls,
    but the labels stay descriptive - so resolving by label survives table
    restructurings that break hardcoded codes."""
    for val, text in zip(var.get("values", []), var.get("valueTexts", [])):
        t = str(text).lower()
        if all(k in t for k in must) and not any(k in t for k in exclude):
            return val
    return None


def _post_query(table_id: str, query: list[dict]) -> dict:
    payload = {"query": query, "response": {"format": "json-stat2"}}
    r = requests.post(config.SSB_BASE + table_id, json=payload, timeout=60)
    if not r.ok:
        # SSB puts the reason (e.g. which variable code is invalid) in the
        # body; surface it instead of a bare status code.
        raise requests.HTTPError(
            f"SSB query for table {table_id} failed with "
            f"{r.status_code}: {r.text[:300]}", response=r)
    return r.json()


def _jsonstat_to_frame(js: dict) -> pd.DataFrame:
    """Minimal json-stat2 -> long DataFrame converter (no pyjstat needed)."""
    dims = js["id"]
    sizes = js["size"]
    cats = []
    for d in dims:
        cat = js["dimension"][d]["category"]
        order = sorted(cat["index"], key=cat["index"].get)
        cats.append(order)
    idx = pd.MultiIndex.from_product(cats, names=dims)
    df = pd.DataFrame({"value": js["value"]}, index=idx).reset_index()
    return df


def fetch_cpi_by_group(start_year: int = 2000,
                       content_code: str | None = None,
                       table_id: str | None = None) -> pd.DataFrame:
    """Monthly CPI index level for total + COICOP divisions.

    Returns DataFrame: index = Period[M], columns = consumption groups,
    values = index levels. The consumption-group variable name and the
    ContentsCode are resolved from the table metadata (tables get replaced
    on base-year rebases and rename both); content_code / table_id override
    the automatic choices so callers can probe alternatives.
    """
    table = table_id or config.SSB_TABLES["cpi"]
    group_code = "Konsumgrp"
    ccode = content_code or "KpiIndMnd"
    gvar = None
    try:
        meta = get_table_metadata(table)
        gvar = next((v for v in meta.get("variables", [])
                     if "konsum" in v.get("code", "").lower()
                     or "coicop" in v.get("code", "").lower()), None)
        if gvar:
            group_code = gvar["code"]
        if content_code is None:
            contents = _get_variable(meta, "ContentsCode")
            if contents and ccode not in contents.get("values", []):
                ccode = _match_value(contents, must=("index",),
                                     exclude=("change", "rate")) or ccode
    except Exception:
        pass   # metadata lookup is best-effort; the query may still work
    query = [
        # Empty values + 'all' filter = every consumption group in the table
        {"code": group_code,
         "selection": {"filter": "all", "values": ["*"]}},
        {"code": "ContentsCode",
         "selection": {"filter": "item", "values": [ccode]}},
    ]
    try:
        js = _post_query(table, query)
    except requests.HTTPError:
        # Some rebased/dataset-style tables reject the 'all' wildcard with
        # 'Parameter error'. Retry with the explicit value list from the
        # metadata; as a last resort drop the group dimension entirely
        # (total-only frame - callers degrade gracefully).
        if gvar is not None and gvar.get("values"):
            query[0] = {"code": group_code,
                        "selection": {"filter": "item",
                                      "values": list(gvar["values"])}}
            js = _post_query(table, query)
        else:
            js = _post_query(table, query[1:])
    df = _jsonstat_to_frame(js)
    tcol = next(c for c in df.columns if str(c).lower() in ("tid", "time"))
    df["period"] = pd.PeriodIndex(df[tcol].str.replace("M", "-"), freq="M")
    if group_code in df.columns:
        wide = df.pivot_table(index="period", columns=group_code,
                              values="value")
    else:   # total-only fallback: no group dimension in the response
        wide = df.groupby("period")[["value"]].first()
        wide.columns = ["TOTAL"]
    wide = wide[wide.index.year >= start_year]
    # code -> human label map, so downstream sub-index matching survives
    # tables whose codes carry no recognizable COICOP numbers
    if gvar is not None:
        wide.attrs["labels"] = dict(zip(gvar.get("values", []),
                                        gvar.get("valueTexts", [])))
    return wide


def fetch_cpi_delivery_sector(start_year: int = 2000) -> pd.DataFrame:
    """CPI by delivery sector -- gives the 'imported consumer goods' series
    that the FX pass-through regression is estimated on."""
    query = [
        {"code": "Leveringssektor",
         "selection": {"filter": "all", "values": ["*"]}},
        {"code": "ContentsCode",
         "selection": {"filter": "item", "values": ["KpiIndeks"]}},
    ]
    js = _post_query(config.SSB_TABLES["cpi_delivery_sector"], query)
    df = _jsonstat_to_frame(js)
    df["period"] = pd.PeriodIndex(df["Tid"].str.replace("M", "-"), freq="M")
    wide = df.pivot_table(index="period", columns="Leveringssektor",
                          values="value")
    return wide[wide.index.year >= start_year]


def resolve_mainland_gdp_codes(table_id: str | None = None) -> tuple[str, str]:
    """Resolve the (Makrost, ContentsCode) pair for mainland GDP, constant
    prices, seasonally adjusted, from the table's live metadata. SSB renames
    ContentsCodes when the national-accounts base year rolls, so hardcoded
    codes 400 sooner or later; the labels stay descriptive."""
    meta = get_table_metadata(table_id or config.SSB_TABLES["gdp_qna"])

    makro_code = "bnpb.nrfast"
    makro = _get_variable(meta, "Makrost")
    if makro and makro_code not in makro.get("values", []):
        makro_code = (_match_value(makro, must=("mainland", "market values"))
                      or _match_value(makro, must=("mainland",),
                                      exclude=("excluding",))
                      or makro_code)

    ccode = None
    contents = _get_variable(meta, "ContentsCode")
    if contents:
        for must in (("seasonally adjusted", "constant"),
                     ("seasonally adjusted", "volume"),
                     ("seasonally adjusted",)):
            ccode = _match_value(contents, must=must,
                                 exclude=("trend", "current", "price ind"))
            if ccode:
                break
    return makro_code, ccode or "Sesongjustert"


def fetch_mainland_gdp(start_year: int = 2000) -> pd.Series:
    """Quarterly mainland-Norway GDP, constant prices, seasonally adjusted.

    Variable codes are resolved from the table metadata at call time (with
    the historical codes as fallback) - see resolve_mainland_gdp_codes.
    """
    try:
        makro_code, ccode = resolve_mainland_gdp_codes()
    except Exception:
        makro_code, ccode = "bnpb.nrfast", "Sesongjustert"
    query = [
        {"code": "Makrost",
         "selection": {"filter": "item", "values": [makro_code]}},
        {"code": "ContentsCode",
         "selection": {"filter": "item", "values": [ccode]}},
    ]
    js = _post_query(config.SSB_TABLES["gdp_qna"], query)
    df = _jsonstat_to_frame(js)
    df["period"] = pd.PeriodIndex(df["Tid"].str.replace("K", "Q"), freq="Q")
    s = df.set_index("period")["value"].sort_index()
    return s[s.index.year >= start_year]


# ---------------------------------------------------------------------------
# Norges Bank: I-44 import-weighted krone index
# ---------------------------------------------------------------------------

def fetch_i44(start: str = "2000-01") -> pd.Series:
    """Monthly I-44. Higher = weaker NOK (index of import-weighted FX)."""
    url = config.NORGES_BANK_I44.format(start=start)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    series = js["data"]["dataSets"][0]["series"]
    obs = next(iter(series.values()))["observations"]
    periods = js["data"]["structure"]["dimensions"]["observation"][0]["values"]
    idx = pd.PeriodIndex([p["id"] for p in periods], freq="M")
    vals = [obs[str(i)][0] for i in range(len(idx))]
    return pd.Series(vals, index=idx, name="I44", dtype=float)
