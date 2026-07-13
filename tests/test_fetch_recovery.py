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


def test_adopted_table_is_persisted_and_reused(tmp_path, monkeypatch):
    ds = _fake_ds(new_end="2026-06", new_base=50.0)
    fetched_tables = []
    orig = ds.fetch_cpi_by_group

    def tracking_fetch(start_year=2000, content_code=None, table_id=None):
        fetched_tables.append(table_id)
        if table_id == "99999":
            return _index_series("2000-01", "2026-06",
                                 base=50.0).to_frame("TOTAL")
        return orig(start_year, content_code)

    ds.fetch_cpi_by_group = tracking_fetch
    monkeypatch.setattr(fetch_data, "_search_ssb_tables",
                        lambda q: ds.__search_hits__)

    cpi1, _ = fetch_data._fetch_cpi_freshest(ds, data_dir=tmp_path)
    assert (tmp_path / ".cpi_table_override").read_text() == "99999"

    fetched_tables.clear()
    cpi2, _ = fetch_data._fetch_cpi_freshest(ds, data_dir=tmp_path)
    # second run goes straight to the remembered table - no discovery
    assert fetched_tables == ["99999"]
    assert cpi2.index[-1] == pd.Period("2026-06", freq="M")


def test_wildcard_rejection_retries_with_explicit_values(monkeypatch):
    """The 'Parameter error' on filter-all: fetch_cpi_by_group must retry
    with the explicit group values, then total-only."""
    import requests
    from quadmap import data_sources as real_ds

    calls = []

    def fake_post(table, query):
        calls.append([q["selection"]["filter"] for q in query])
        if any(q["selection"]["filter"] == "all" for q in query):
            raise requests.HTTPError("400 Parameter error")
        return {"ok": True}

    def fake_meta(table):
        return {"variables": [
            {"code": "KonsumgrpNy", "values": ["TOTAL", "01"],
             "valueTexts": ["Total", "Food"]},
            {"code": "ContentsCode", "values": ["Ind"],
             "valueTexts": ["Consumer price index (2025=100)"]},
            {"code": "Tid", "values": ["2026M05"], "valueTexts": ["2026M05"]},
        ]}

    def fake_frame(js):
        return pd.DataFrame({"Tid": ["2026M04", "2026M05"],
                             "KonsumgrpNy": ["TOTAL", "TOTAL"],
                             "value": [120.0, 121.0]})

    monkeypatch.setattr(real_ds, "_post_query", fake_post)
    monkeypatch.setattr(real_ds, "get_table_metadata", fake_meta)
    monkeypatch.setattr(real_ds, "_jsonstat_to_frame", fake_frame)

    wide = real_ds.fetch_cpi_by_group(table_id="Kpi00")
    assert list(wide.columns) == ["TOTAL"]
    assert wide.index[-1] == pd.Period("2026-05", freq="M")
    # first attempt used 'all', the retry used explicit items
    assert calls[0][0] == "all" and calls[1][0] == "item"


def test_real_power_overlay_produces_real_units(tmp_path, monkeypatch):
    """build_market_proxy with a working day-sampler must emit real
    ore/kWh from 2021 on, rescale the proxy years onto real units, and
    report mode 'real-2021' (which disables the live-feed bridge)."""
    months = pd.period_range("2018-01", "2026-06", freq="M")
    groups = pd.DataFrame(
        {"TOTAL": np.linspace(100, 130, len(months)),
         "JA045.1_elektrisitet": np.full(len(months), 100.0)}, index=months)

    monkeypatch.setattr(fetch_data, "fetch_norges_bank_monthly",
                        lambda base: pd.Series(10.0, index=months))
    monkeypatch.setattr(fetch_data, "fetch_brent_monthly",
                        lambda: pd.Series(75.0, index=months))
    mode = fetch_data.build_market_proxy(
        tmp_path, groups, day_mean_fn=lambda d: 120.0)
    assert mode == "real-2021"
    m = pd.read_csv(tmp_path / "market_monthly.csv")
    m.index = pd.PeriodIndex(m.pop("period"), freq="M")
    # real months carry the sampled real level...
    assert m.loc[pd.Period("2024-06", "M"),
                 "power_ore_kwh"] == pytest.approx(120.0)
    # ...and pre-2021 proxy months are rescaled into the same units
    assert m.loc[pd.Period("2019-06", "M"),
                 "power_ore_kwh"] == pytest.approx(120.0, rel=0.05)


def test_real_power_failure_keeps_proxy_mode(tmp_path, monkeypatch):
    months = pd.period_range("2018-01", "2026-06", freq="M")
    groups = pd.DataFrame({"TOTAL": np.full(len(months), 100.0)},
                          index=months)
    monkeypatch.setattr(fetch_data, "fetch_norges_bank_monthly",
                        lambda base: pd.Series(10.0, index=months))
    monkeypatch.setattr(fetch_data, "fetch_brent_monthly",
                        lambda: pd.Series(75.0, index=months))
    mode = fetch_data.build_market_proxy(
        tmp_path, groups, day_mean_fn=lambda d: None)   # sampler always dry
    assert mode == "proxy"


def test_splice_no_overlap_keeps_longer():
    old = _index_series("2000-01", "2025-12")
    new = _index_series("2026-01", "2026-06", base=210.0)
    out = fetch_data._splice_series(old, new)
    assert out.index[-1] == pd.Period("2025-12", freq="M")   # old is longer
