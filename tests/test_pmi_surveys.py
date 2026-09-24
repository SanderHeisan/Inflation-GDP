"""The regional Fed manufacturing surveys, and the dead weight they replaced.

`GDP_NOWCAST_WEIGHTS` carried an `ism` slot at 0.30 from the day the model was
written and it never fired once: ISM restricted redistribution, FRED dropped
the NAPM series, and nothing ever supplied one. The blend renormalises over
the slots it actually has, so the number was never wrong -- but anyone reading
the config was told a third of the growth fallback rested on an input that did
not exist.

The free substitutes are the regional Fed surveys, the same family of
diffusion index and the series forecasters use to nowcast the ISM itself. They
are wired in as a research panel and were scored on 2026-09-24: they cut the
nowcast's RMSE and made the growth DIRECTION call worse, which is the half the
regime depends on. So they stay off by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from usmodel import config as uconfig, gdp as gdp_mod, nowcast
from usmodel.data_bundle import INDICATOR_NAMES, load_bundle
from usmodel.fetch_data_us import FRED_SERIES, INDICATOR_PUB_LAG_DAYS
from usbacktest import btconfig, vintage as vt

SURVEYS = ("philly_fed", "philly_orders", "empire_fed", "dallas_fed")


def _bundle():
    try:
        return load_bundle(start="1995-01")
    except FileNotFoundError:
        pytest.skip("needs the cached US data (data/ is not committed)")


def test_the_dead_ism_weight_is_gone():
    assert "ism" not in uconfig.GDP_NOWCAST_WEIGHTS
    src = (__import__("pathlib").Path(gdp_mod.__file__)).read_text()
    assert '"ism" in indicators' not in src, "the branch outlived the weight"


def test_removing_it_changed_no_number():
    """The blend renormalises over the signals it HAS, so a slot with no
    series can never enter the sum. Pinned so a future reader does not have
    to take that on trust."""
    idx = pd.period_range("2016Q1", periods=40, freq="Q")
    lvl = pd.Series(np.cumprod(1 + np.full(40, 0.005)), index=idx)
    w_now = dict(uconfig.GDP_NOWCAST_WEIGHTS)
    for ind in ({}, {"payrolls": 150.0}, {"payrolls": 150.0, "retail": 0.004}):
        now = gdp_mod.nowcast_qoq(lvl, dict(ind))
        try:
            uconfig.GDP_NOWCAST_WEIGHTS = {**w_now, "ism": 0.30}
            then = gdp_mod.nowcast_qoq(lvl, dict(ind))
        finally:
            uconfig.GDP_NOWCAST_WEIGHTS = w_now
        assert now == then, f"the ism slot moved the blend for {sorted(ind)}"


def test_every_survey_is_declared_end_to_end():
    for k in SURVEYS:
        assert k in FRED_SERIES, f"{k} has no FRED id"
        assert k in INDICATOR_PUB_LAG_DAYS, f"{k} has no publication lag"
        assert k in INDICATOR_NAMES, f"{k} never reaches the bundle"
        # published DURING the month they describe, so a month-end run has
        # them; zero is the conservative reading of that, never more
        assert INDICATOR_PUB_LAG_DAYS[k] == 0, k
    assert FRED_SERIES["philly_fed"][0] == "GACDFSA066MSFRBPHI"


def test_a_diffusion_index_enters_as_a_level():
    """It is already a rate of change in disguise (the share of firms
    reporting improvement), so differencing it asks for the change in the
    change. Every survey slot must be 'lvl'."""
    spec = dict(nowcast.SURVEY_SPEC)
    assert set(spec) == set(SURVEYS)
    assert set(spec.values()) == {"lvl"}
    assert dict(nowcast.BASE_PLUS_SURVEYS) == {**dict(nowcast.FEATURE_SPEC), **spec}


def test_the_production_panel_is_still_the_base_one():
    cfg = btconfig.USVintageConfig()
    assert cfg.nowcast_spec == "base"
    assert nowcast.NOWCAST_SPECS["base"] is nowcast.FEATURE_SPEC
    assert set(nowcast.NOWCAST_SPECS) == {"base", "surveys", "philly", "extended"}


def test_the_panel_switch_actually_moves_the_nowcast():
    b = _bundle()
    if not all(k in b.indicators for k in SURVEYS):
        pytest.skip("survey CSVs not cached")
    seen = {}
    for spec in ("base", "philly", "surveys"):
        v = vt.build_vintage(b, "2026-09-20",
                             btconfig.USVintageConfig(nowcast_spec=spec))
        seen[spec] = v.diagnostics["nowcast_qoq_pct"]
    assert seen["base"] != seen["philly"] != seen["surveys"]
    # an unknown name falls back to the production panel rather than crashing
    v = vt.build_vintage(b, "2026-09-20",
                         btconfig.USVintageConfig(nowcast_spec="nonsense"))
    assert v.diagnostics["nowcast_qoq_pct"] == pytest.approx(seen["base"])
