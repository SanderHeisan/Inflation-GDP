"""SSB variable-code resolution from table metadata (no network needed)."""
from quadmap import data_sources as ds


FAKE_META_09190 = {
    "title": "09190: Final expenditure and gross domestic product",
    "variables": [
        {"code": "Makrost",
         "values": ["bnpb.nr23_9", "bnpb.nr23_9fn", "makrok.konsum"],
         "valueTexts": [
             "Gross domestic product, market values",
             "Gross domestic product Mainland Norway, market values",
             "Final consumption expenditure of households"]},
        {"code": "ContentsCode",
         "values": ["Priser", "Faste", "SesongFaste", "SesongTrend"],
         "valueTexts": [
             "Current prices (NOK million)",
             "Constant 2023 prices (NOK million)",
             "Seasonally adjusted, constant 2023 prices (NOK million)",
             "Trend, seasonally adjusted, constant 2023 prices"]},
        {"code": "Tid", "values": ["2023K1"], "valueTexts": ["2023K1"]},
    ],
}


def test_resolver_finds_mainland_sa_constant(monkeypatch):
    monkeypatch.setattr(ds, "get_table_metadata",
                        lambda table_id: FAKE_META_09190)
    makro, ccode = ds.resolve_mainland_gdp_codes("09190")
    assert makro == "bnpb.nr23_9fn"       # mainland, market values
    assert ccode == "SesongFaste"          # SA constant prices, not trend


def test_resolver_keeps_known_codes_when_still_valid(monkeypatch):
    meta = {"variables": [
        {"code": "Makrost", "values": ["bnpb.nrfast"],
         "valueTexts": ["Gross domestic product Mainland Norway"]},
        {"code": "ContentsCode", "values": ["Sesongjustert"],
         "valueTexts": ["Seasonally adjusted, constant prices"]},
    ]}
    monkeypatch.setattr(ds, "get_table_metadata", lambda table_id: meta)
    makro, ccode = ds.resolve_mainland_gdp_codes("09190")
    assert makro == "bnpb.nrfast"
    assert ccode == "Sesongjustert"


def test_resolver_falls_back_to_historical_codes(monkeypatch):
    monkeypatch.setattr(ds, "get_table_metadata",
                        lambda table_id: {"variables": []})
    makro, ccode = ds.resolve_mainland_gdp_codes("09190")
    assert makro == "bnpb.nrfast"
    assert ccode == "Sesongjustert"


def test_match_value_excludes():
    var = {"values": ["a", "b"],
           "valueTexts": ["Seasonally adjusted, trend",
                          "Seasonally adjusted, constant prices"]}
    assert ds._match_value(var, must=("seasonally adjusted",),
                           exclude=("trend",)) == "b"
    assert ds._match_value(var, must=("nope",)) is None
