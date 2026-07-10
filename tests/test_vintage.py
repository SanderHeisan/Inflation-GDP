"""Point-in-time discipline: publication-lag truncation, no-future-leakage,
revision model behavior, and spot-carry assumptions."""
import numpy as np
import pandas as pd
import pytest

from backtest import vintage as vt
from backtest.btconfig import VintageConfig
from backtest.data_bundle import make_demo_bundle, synthesize_vintage_panel


def test_gdp_truncation_respects_publication_lag(bundle):
    # 2020Q1 GDP publishes ~May 10 2020; a mid-March as-of must not see it.
    v = vt.build_vintage(bundle, "2020-03-15")
    assert v.last_gdp_quarter == pd.Period("2019Q4", freq="Q")
    assert pd.Period("2020Q1", freq="Q") not in v.gdp_level.index


def test_gdp_not_available_inside_lag_window(bundle):
    # 2019Q4 publishes ~Feb 9 2020 (Dec 31 + 40 days): invisible on Feb 5,
    # visible on Feb 10.
    early = vt.build_vintage(bundle, "2020-02-05")
    assert early.last_gdp_quarter == pd.Period("2019Q3", freq="Q")
    late = vt.build_vintage(bundle, "2020-02-10")
    assert late.last_gdp_quarter == pd.Period("2019Q4", freq="Q")


def test_cpi_truncation_respects_publication_lag(bundle):
    # Feb 2020 CPI publishes ~Mar 10 (Feb 29 + 10 days).
    v = vt.build_vintage(bundle, "2020-03-15")
    assert v.last_cpi_month == pd.Period("2020-02", freq="M")
    assert vt.build_vintage(bundle, "2020-03-09").last_cpi_month \
        == pd.Period("2020-01", freq="M")


def test_no_future_leakage_anywhere():
    """The strongest guarantee: mutate everything after the as-of date and
    the vintage (and hence any prediction built from it) must not change."""
    asof = pd.Timestamp("2018-06-30")
    a = make_demo_bundle(seed=7)
    b = make_demo_bundle(seed=7)

    cutoff_m = pd.Period("2018-07", freq="M")
    cutoff_q = pd.Period("2018Q2", freq="Q")   # unpublished at 2018-06-30
    b.gdp_final.loc[b.gdp_final.index >= cutoff_q] *= 7.0
    b.cpi.loc[b.cpi.index >= cutoff_m] *= 3.0
    b.i44.loc[b.i44.index >= cutoff_m] += 500.0
    b.market.loc[b.market.index >= cutoff_m] *= 10.0

    va, vb = (vt.build_vintage(x, asof) for x in (a, b))
    pd.testing.assert_series_equal(va.gdp_level, vb.gdp_level)
    pd.testing.assert_series_equal(va.cpi_index, vb.cpi_index)
    pd.testing.assert_series_equal(va.i44, vb.i44)
    assert va.assumptions == vb.assumptions


def test_spot_carry_forwards_are_flat_at_last_spot(bundle):
    asof = pd.Timestamp("2019-05-20")
    v = vt.build_vintage(bundle, asof)
    last_m = pd.Period("2019-04", freq="M")   # last complete month
    spot = bundle.market.loc[last_m, "power_ore_kwh"]
    fwd = v.assumptions["power_forward_ore_kwh"]
    assert v.assumptions["power_recent_ore_kwh"] == pytest.approx(spot)
    assert all(val == pytest.approx(spot) for val in fwd.values())
    # and the realized future spot (2022 spike) never appears
    future_max = bundle.market.loc[bundle.market.index > last_m,
                                   "power_ore_kwh"].max()
    assert max(fwd.values()) < future_max


def test_wage_norm_point_in_time(bundle):
    # Before the spring settlement, year Y's norm must not be visible.
    pre = vt.build_vintage(bundle, "2021-03-01")
    post = vt.build_vintage(bundle, "2021-05-01")
    assert pre.assumptions["wage_norm_pct"] \
        == pytest.approx(bundle.wage_norms[2020])
    assert post.assumptions["wage_norm_pct"] \
        == pytest.approx(bundle.wage_norms[2021])


def test_noise_mode_is_deterministic_and_flagged(bundle):
    cfg = VintageConfig(revision_mode="noise", seed=3)
    v1 = vt.build_vintage(bundle, "2019-08-31", cfg)
    v2 = vt.build_vintage(bundle, "2019-08-31", cfg)
    assert v1.revision_mode == "noise"
    pd.testing.assert_series_equal(v1.gdp_level, v2.gdp_level)
    # different seed -> different simulated first releases
    v3 = vt.build_vintage(bundle, "2019-08-31", VintageConfig(
        revision_mode="noise", seed=4))
    assert not np.allclose(v1.gdp_level.tail(4), v3.gdp_level.tail(4))


def test_noise_only_touches_immature_quarters(bundle):
    cfg = VintageConfig(revision_mode="noise")
    clean = vt.build_vintage(bundle, "2019-08-31",
                             VintageConfig(revision_mode="none"))
    noisy = vt.build_vintage(bundle, "2019-08-31", cfg)
    mature = clean.gdp_level.index[:-cfg.revision_maturity_quarters]
    pd.testing.assert_series_equal(clean.gdp_level.loc[mature],
                                   noisy.gdp_level.loc[mature])
    assert not np.allclose(clean.gdp_level.tail(3), noisy.gdp_level.tail(3))


def test_noise_converges_to_final_as_quarter_matures(bundle):
    """The same quarter's estimate must drift toward final across as-of
    dates (revisions settle), matching the persistent-error design."""
    cfg = VintageConfig(revision_mode="noise", seed=1)
    q = pd.Period("2018Q1", freq="Q")
    final_qoq = bundle.gdp_final.pct_change().loc[q]
    errs = []
    for asof in ("2018-06-01", "2019-06-01", "2021-06-01"):
        v = vt.build_vintage(bundle, asof, cfg)
        errs.append(abs(v.gdp_level.pct_change().loc[q] - final_qoq))
    assert errs[0] > errs[1] > errs[2] or errs[2] == 0.0
    assert errs[2] == pytest.approx(0.0, abs=1e-12)


def test_realtime_mode_uses_vintage_panel(bundle):
    panel = synthesize_vintage_panel(bundle.gdp_final, seed=11)
    bundle2 = make_demo_bundle(seed=7)
    bundle2.gdp_vintages = panel
    cfg = VintageConfig(revision_mode="auto")
    v = vt.build_vintage(bundle2, "2019-08-31", cfg)
    assert v.revision_mode == "realtime"
    # first releases differ from final for young quarters...
    assert not np.allclose(v.gdp_level.tail(3),
                           bundle.gdp_final.loc[v.gdp_level.index].tail(3))
    # ...and the panel must not reveal unpublished quarters either
    assert v.last_gdp_quarter == pd.Period("2019Q2", freq="Q")


def test_realtime_mode_requires_panel(bundle):
    with pytest.raises(ValueError, match="realtime"):
        vt.build_vintage(bundle, "2019-08-31",
                         VintageConfig(revision_mode="realtime"))
