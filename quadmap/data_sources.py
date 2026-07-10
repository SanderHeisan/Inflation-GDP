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


def _post_query(table_id: str, query: list[dict]) -> dict:
    payload = {"query": query, "response": {"format": "json-stat2"}}
    r = requests.post(config.SSB_BASE + table_id, json=payload, timeout=60)
    r.raise_for_status()
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


def fetch_cpi_by_group(start_year: int = 2000) -> pd.DataFrame:
    """Monthly CPI index level for total + COICOP divisions.

    Returns DataFrame: index = Period[M], columns = consumption groups,
    values = index levels (2015=100).
    """
    query = [
        # Empty values + 'all' filter = every consumption group in the table
        {"code": "Konsumgrp", "selection": {"filter": "all", "values": ["*"]}},
        {"code": "ContentsCode",
         "selection": {"filter": "item", "values": ["KpiIndMnd"]}},
    ]
    js = _post_query(config.SSB_TABLES["cpi"], query)
    df = _jsonstat_to_frame(js)
    df["period"] = pd.PeriodIndex(df["Tid"].str.replace("M", "-"), freq="M")
    wide = df.pivot_table(index="period", columns="Konsumgrp", values="value")
    return wide[wide.index.year >= start_year]


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


def fetch_mainland_gdp(start_year: int = 2000) -> pd.Series:
    """Quarterly mainland-Norway GDP, constant prices, seasonally adjusted.

    NOTE: run get_table_metadata('09190') once to confirm the exact
    Makrost code for 'bnpb.nrfast' (GDP Mainland Norway, market values)
    and the ContentsCode for seasonally adjusted constant prices.
    """
    query = [
        {"code": "Makrost",
         "selection": {"filter": "item", "values": ["bnpb.nrfast"]}},
        {"code": "ContentsCode",
         "selection": {"filter": "item", "values": ["Sesongjustert"]}},
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
