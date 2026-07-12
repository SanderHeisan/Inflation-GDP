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


def test_power_proxy_scale_bridges_units():
    from forecast import power_proxy_scale
    # proxy says 60 'ore' for the anchor month; real sampled days say ~120
    scale = power_proxy_scale(pd.Period("2026-05", freq="M"), 60.0,
                              day_mean_fn=lambda d: 120.0)
    assert scale == pytest.approx(0.5)
    # a 132.61 real live spot then enters the projection as ~66 proxy-ore:
    # the same *relative* level, no fabricated doubling
    assert 132.61 * scale == pytest.approx(66.3, abs=0.1)
    with pytest.raises(RuntimeError):
        power_proxy_scale(pd.Period("2026-05", freq="M"), 60.0,
                          day_mean_fn=lambda d: None)


def test_discovery_returns_groups_for_power_proxy(monkeypatch):
    ds = _fake_ds(new_end="2026-06", new_base=50.0)
    calls = {}

    def fetch_groups(start_year=2000, content_code=None, table_id=None):
        calls["table_id"] = table_id
        if table_id == "99999":
            return _index_series("2020-01", "2026-06",
                                 base=50.0).to_frame("TOTAL")
        return _index_series("2000-01", "2025-12").to_frame("TOTAL")

    ds.fetch_cpi_by_group = fetch_groups
    monkeypatch.setattr(fetch_data, "_search_ssb_tables",
                        lambda q: ds.__search_hits__)
    cpi, groups = fetch_data._fetch_cpi_freshest(ds)
    assert calls["table_id"] == "99999"          # groups came from successor
    assert groups.index[-1] == pd.Period("2026-06", freq="M")


def test_broken_default_table_falls_back_to_legacy_then_discovers(monkeypatch):
    """The 'Kpi10 vs 14710' failure mode: the configured table id 400s
    outright. The fetch must recover via the legacy table's history and
    then adopt the successor through discovery - never die."""
    ds = _fake_ds(new_end="2026-06", new_base=50.0)
    ds.config.SSB_TABLES = {"cpi": "14710", "cpi_legacy": "03013",
                            "gdp_qna": "09190"}
    old = _index_series("2000-01", "2025-12")

    def fetch_cpi_by_group(start_year=2000, content_code=None,
                           table_id=None):
        if table_id is None:      # the broken configured default
            raise RuntimeError("400 Parameter error")
        if table_id == "03013":
            return old.to_frame("TOTAL")
        if table_id == "99999":
            return _index_series("2020-01", "2026-06",
                                 base=50.0).to_frame("TOTAL")
        raise RuntimeError(f"unknown table {table_id}")

    ds.fetch_cpi_by_group = fetch_cpi_by_group
    monkeypatch.setattr(fetch_data, "_search_ssb_tables",
                        lambda q: ds.__search_hits__)
    cpi, groups = fetch_data._fetch_cpi_freshest(ds)
    assert cpi.index[-1] == pd.Period("2026-06", freq="M")
    assert cpi.index[0] == pd.Period("2000-01", freq="M")


def test_splice_no_overlap_keeps_longer():
    old = _index_series("2000-01", "2025-12")
    new = _index_series("2026-01", "2026-06", base=210.0)
    out = fetch_data._splice_series(old, new)
    assert out.index[-1] == pd.Period("2025-12", freq="M")   # old is longer
