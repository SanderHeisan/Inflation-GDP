"""Frozen-table recovery: successor-table discovery, splicing, and the
never-crash guarantee of the staleness diagnostics (all offline, mocked)."""
import types

import numpy as np
import pandas as pd
import pytest

from backtest import fetch_data
from quadmap import data_sources as real_ds


def _index_series(start, end, base=100.0):
    idx = pd.period_range(start, end, freq="M")
    return pd.Series(base * np.cumprod([1.002] * len(idx)), index=idx)


def _fake_ds(old_end="2025-12", new_start="2020-01", new_end=None,
             new_base=50.0, search_hits=None):
    """A stand-in for quadmap.data_sources with one frozen CPI table and
    (optionally) a fresh successor table discoverable via search."""
    old = _index_series("2000-01", old_end)
    new = _index_series(new_start, new_end, base=new_base) \
        if new_end else None

    ds = types.SimpleNamespace()
    ds.config = types.SimpleNamespace(SSB_TABLES={"cpi": "03013",
                                                  "gdp_qna": "09190"})
    ds._get_variable = real_ds._get_variable
    ds._match_value = real_ds._match_value

    def fetch_cpi_by_group(start_year=2000, content_code=None):
        return old.to_frame("TOTAL")

    def get_table_metadata(table_id):
        if table_id == "03013":
            return {"variables": [
                {"code": "ContentsCode", "values": ["KpiIndMnd"],
                 "valueTexts": ["Consumer Price Index (2015=100)"]},
                {"code": "Tid", "values": [str(old.index[-1])],
                 "valueTexts": [str(old.index[-1])]}]}
        if table_id == "99999" and new is not None:
            return {"variables": [
                {"code": "Konsumgrp", "values": ["TOTAL", "01"],
                 "valueTexts": ["Total", "Food"]},
                {"code": "ContentsCode", "values": ["KpiNy"],
                 "valueTexts": ["Consumer Price Index (2025=100)"]},
                {"code": "Tid", "values": ["2026M06"],
                 "valueTexts": ["2026M06"]}]}
        raise RuntimeError(f"unknown table {table_id}")

    def _post_query(table_id, query):
        assert table_id == "99999" and new is not None
        return {"__series__": new}

    def _jsonstat_to_frame(js):
        s = js["__series__"]
        return pd.DataFrame({"Tid": [str(p).replace("-", "M")
                                     for p in s.index],
                             "value": s.to_numpy()})

    ds.fetch_cpi_by_group = fetch_cpi_by_group
    ds.get_table_metadata = get_table_metadata
    ds._post_query = _post_query
    ds._jsonstat_to_frame = _jsonstat_to_frame
    ds.__search_hits__ = search_hits if search_hits is not None else [
        ("03013", "Consumer Price Index (old)"),
        ("11111", "Harmonised consumer price index"),   # filtered out
        ("99999", "Consumer Price Index, by consumption group (new base)"),
    ]
    return ds


def test_discovery_adopts_and_splices_successor(monkeypatch):
    ds = _fake_ds(new_end="2026-06", new_base=50.0)
    monkeypatch.setattr(fetch_data, "_search_ssb_tables",
                        lambda q: ds.__search_hits__)
    cpi, _ = fetch_data._fetch_cpi_freshest(ds)
    assert cpi.index[-1] == pd.Period("2026-06", freq="M")
    assert cpi.index[0] == pd.Period("2000-01", freq="M")   # history kept
    # splice leaves no jump: MoM at the junction stays ~0.2%
    mom = cpi.pct_change().dropna()
    assert abs(mom.loc[pd.Period("2020-01", "M")] - 0.002) < 5e-4


def test_discovery_skips_harmonised_and_reports_none(monkeypatch):
    ds = _fake_ds(new_end=None,   # successor exists in search but is broken
                  search_hits=[("11111", "Harmonised consumer price index")])
    monkeypatch.setattr(fetch_data, "_search_ssb_tables",
                        lambda q: ds.__search_hits__)
    cpi, _ = fetch_data._fetch_cpi_freshest(ds)
    assert cpi.index[-1] == pd.Period("2025-12", freq="M")   # graceful keep


def test_staleness_check_never_raises():
    broken = types.SimpleNamespace(
        get_table_metadata=lambda t: (_ for _ in ()).throw(RuntimeError()),)
    fetch_data._staleness_check("CPI", pd.Period("2020-01", freq="M"), 75,
                                broken, "03013")


def test_splice_no_overlap_keeps_longer():
    old = _index_series("2000-01", "2025-12")
    new = _index_series("2026-01", "2026-06", base=210.0)
    out = fetch_data._splice_series(old, new)
    assert out.index[-1] == pd.Period("2025-12", freq="M")   # old is longer
