"""
US GIP Quad Map -- pipeline runner (mirrors run.py for Norway).

    python run_us.py            # real cached FRED + Zillow data
    python run_us.py --demo     # synthetic data, validates the pipeline offline

Real mode needs the cache: `python -m usmodel.fetch_data_us` first. The same
point-in-time machinery the backtest uses builds today's information set, so
the live call is produced exactly the way every backtested call was -- no
separate live code path to drift out of sync.

The quad engine (quadmap.quads) and the base-effect logic are shared with the
Norwegian model; only the inflation component model (usmodel.inflation), the
GDP projection (usmodel.gdp / usmodel.nowcast) and the data are US-specific.
"""
from __future__ import annotations

import argparse

import pandas as pd

from quadmap import quads
from usmodel import config, data_bundle, gdp as gdp_mod, inflation


def _demo_inputs(horizon_months: int):
    bundle = data_bundle.make_demo_bundle()
    assumptions = data_bundle.assumptions_from_bundle(bundle)
    aux = {"market_rent": bundle.market_rent, "dollar": bundle.dollar}
    return (bundle.cpi, bundle.gdp, assumptions, aux, {},
            bundle.gdp.index[-1], "synthetic demo")


def _live_inputs(horizon_months: int):
    """Today's vintage, built by the backtest's own vintage machinery."""
    from usbacktest.btconfig import USVintageConfig
    from usbacktest.vintage import build_vintage

    bundle = data_bundle.load_bundle()
    cfg = USVintageConfig(revision_mode="none")   # current vintage is the live one
    v = build_vintage(bundle, pd.Timestamp.today().normalize(), cfg,
                      horizon_months=horizon_months + 2)
    d = v.diagnostics
    note = (f"real data | CPI through {v.last_cpi_month}, GDP through "
            f"{v.last_gdp_quarter} | shelter b={d['shelter_b']:.2f} "
            f"a={d['shelter_a']:.2f}, supercore b={d['supercore_b']:.2f}, "
            f"nowcast {d['nowcast_qoq_pct']:.2f}% QoQ (k={d['nowcast_k']})")
    return (v.cpi_index, v.gdp_level, v.assumptions, v.aux, v.indicators,
            v.last_gdp_quarter, note)


def run(demo: bool = False, horizon_months: int = 15):
    cpi_hist, gdp_hist, assumptions, aux, indicators, last_real, note = (
        _demo_inputs(horizon_months) if demo else _live_inputs(horizon_months))
    print(f"\nUS GIP Quad Map | {note}")

    # --- Inflation: bottom-up component projection -> YoY path
    cpi_full = inflation.build_cpi_projection(cpi_hist, horizon_months,
                                              assumptions, aux)
    cpi_yoy_q = quads.monthly_to_quarterly_yoy(cpi_full["yoy_pct"].dropna())

    # --- GDP: nowcast + convergence -> YoY path
    gdp_full = gdp_mod.project_gdp(gdp_hist, indicators, horizon_quarters=6)

    # --- Quads (shared engine)
    table = quads.classify(gdp_full["yoy_pct"], cpi_yoy_q)
    table["projected"] = table.index > last_real

    pd.set_option("display.width", 150)
    print("\n=== US quad table (last realized + projected) ===")
    print(table.tail(9)[["growth_yoy", "inflation_yoy", "d_growth",
                         "d_inflation", "quad", "label", "low_conviction",
                         "projected"]].round(2).to_string())

    flips = quads.quad_flips(table[table["projected"]
                                   | (table.index >= last_real - 3)])
    if len(flips):
        print("\n=== Quad flips (the tradeable events) ===")
        print(flips[["quad", "label", "projected"]].to_string())

    print("\n=== Next 6 CPI prints (bottom-up decomposition, MoM %) ===")
    cols = ["mom_pct", "contrib_gasoline", "contrib_shelter",
            "contrib_core_goods", "contrib_supercore", "yoy_pct"]
    print(cpi_full[cols].tail(horizon_months).head(6).round(3).to_string())

    print(f"\nDeadband for low_conviction: +/-{config.QUAD_DEADBAND_PP}pp. "
          f"Measured accuracy of these calls is in the README.")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="synthetic data instead of the real cache")
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()
    run(demo=args.demo, horizon_months=args.horizon)
