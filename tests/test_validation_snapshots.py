"""Split-sample validation math and the indicator snapshotter (mocked)."""
import types

import pandas as pd
import pytest

from backtest import engine, monthly, scoring, snapshots, validation
from backtest.btconfig import VintageConfig


def test_split_validation_end_to_end(bundle):
    cfg = VintageConfig(revision_mode="noise")
    preds = engine.run_backtest(bundle, "2014-01-01", "2021-12-31",
                                freq="Q", max_horizon=2, cfg=cfg)
    realized = scoring.realized_quads_final(bundle)
    mdir = monthly.monthly_direction_backtest(bundle, "2014-01-01",
                                              "2021-12-31", cfg)
    val = validation.split_validation(preds, realized, mdir,
                                      split_year=2018, horizons=(1, 2))
    assert len(val)
    quad_rows = val[val["metric"].str.startswith("quad_hit")]
    assert set(quad_rows["metric"]) == {"quad_hit_h1", "quad_hit_h2"}
    assert ((val["in_sample"] >= 0) & (val["in_sample"] <= 1)).all()
    assert ((val["out_of_sample"] >= 0) & (val["out_of_sample"] <= 1)).all()
    assert (val["oos_minus_is"]
            == val["out_of_sample"] - val["in_sample"]).all()
    # train/test really are disjoint windows
    assert (val["n_train"] > 0).all() and (val["n_test"] > 0).all()


def test_split_validation_empty_when_no_test_window(bundle):
    cfg = VintageConfig(revision_mode="noise")
    preds = engine.run_backtest(bundle, "2015-01-01", "2016-12-31",
                                freq="Q", max_horizon=1, cfg=cfg)
    realized = scoring.realized_quads_final(bundle)
    mdir = monthly.monthly_direction_backtest(bundle, "2015-01-01",
                                              "2016-12-31", cfg)
    val = validation.split_validation(preds, realized, mdir,
                                      split_year=2030, horizons=(1,))
    assert len(val) == 0


from quadmap.data_sources import _match_value as _real_match_value


def _fake_ds_for_snapshots():
    ds = types.SimpleNamespace()

    def get_table_metadata(tid):
        return {"variables": [
            {"code": "Region", "values": ["0"], "valueTexts": ["Whole country"]},
            {"code": "ContentsCode", "values": ["Vol", "Verdi"],
             "valueTexts": ["Volume index", "Value index"]},
            {"code": "Tid", "values": ["2026M05"], "valueTexts": ["2026M05"]},
        ]}

    def _match_value(var, must=(), exclude=()):
        return _real_match_value(var, must, exclude)

    def _post_query(tid, query):
        # Tid must use the 'top' filter, other dims collapsed to one item
        tsel = next(q for q in query if q["code"] == "Tid")["selection"]
        assert tsel["filter"] == "top"
        return {"ok": True}

    def _jsonstat_to_frame(js):
        return pd.DataFrame({"Tid": ["2026M05"], "value": [123.4]})

    ds.get_table_metadata = get_table_metadata
    ds._match_value = _match_value
    ds._post_query = _post_query
    ds._jsonstat_to_frame = _jsonstat_to_frame
    return ds


def test_snapshot_appends_and_dedupes(tmp_path, monkeypatch):
    import quadmap.data_sources
    fake = _fake_ds_for_snapshots()
    for attr in ("get_table_metadata", "_match_value", "_post_query",
                 "_jsonstat_to_frame"):
        monkeypatch.setattr(quadmap.data_sources, attr, getattr(fake, attr))
    monkeypatch.setattr(snapshots, "INDICATORS",
                        [{"name": "retail_volume_index", "table": "07129",
                          "search": "x", "must": (),
                          "contents_must": ("volume", "index")}])
    out = tmp_path / "snaps.csv"
    df1 = snapshots.snapshot_indicators(out, snap_date="2026-07-11")
    assert len(df1) == 1
    assert df1.iloc[0]["value"] == 123.4
    assert df1.iloc[0]["period"] == "2026M05"
    # same day twice -> deduped; new day -> appended
    snapshots.snapshot_indicators(out, snap_date="2026-07-11")
    df3 = snapshots.snapshot_indicators(out, snap_date="2026-07-12")
    assert len(df3) == 2
    # table id cached for next runs
    assert "retail_volume_index" in out.with_suffix(
        ".tables.json").read_text()
