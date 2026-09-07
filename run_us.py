"""
US GIP Quad Map -- pipeline runner (mirrors run.py for Norway).

    python run_us.py --demo    # synthetic US data, validates the pipeline
    python run_us.py           # live: FRED/BLS (needs a fetch, see fetch_data_us.py)

The quad engine (quadmap.quads) and the base-effect logic are shared with the
Norwegian model; only the inflation component model (usmodel.inflation), the
GDP constants (usmodel.gdp) and the data are US-specific.
"""
from __future__ import annotations

import argparse

import pandas as pd

from quadmap import quads
from usmodel import config, data_bundle, gdp as gdp_mod, inflation


def run(demo: bool = True, horizon_months: int = 15):
    if not demo:
        raise SystemExit("live mode needs fetch_data_us.py (FRED/BLS) - run "
                         "with --demo here, or on Colab with data fetched.")
    bundle = data_bundle.make_demo_bundle()
    assumptions = data_bundle.assumptions_from_bundle(bundle)
    aux = {"market_rent": bundle.market_rent, "dollar": bundle.dollar}

    # --- Inflation: bottom-up component projection -> YoY path
    cpi_full = inflation.build_cpi_projection(bundle.cpi, horizon_months,
                                              assumptions, aux)
    cpi_yoy_q = quads.monthly_to_quarterly_yoy(cpi_full["yoy_pct"].dropna())

    # --- GDP: nowcast + convergence -> YoY path
    gdp_full = gdp_mod.project_gdp(bundle.gdp, indicators={},
                                   horizon_quarters=5)

    # --- Quads (shared engine)
    table = quads.classify(gdp_full["yoy_pct"], cpi_yoy_q)
    table["projected"] = table.index > bundle.gdp.index[-1]

    pd.set_option("display.width", 150)
    print("\n=== US quad table (last realized + projected) ===")
    print(table.tail(9)[["growth_yoy", "inflation_yoy", "d_growth",
                         "d_inflation", "quad", "label", "low_conviction",
                         "projected"]].round(2).to_string())

    flips = quads.quad_flips(table[table["projected"] |
                                   (table.index >= bundle.gdp.index[-1] - 3)])
    if len(flips):
        print("\n=== Quad flips (the tradeable events) ===")
        print(flips[["quad", "label", "projected"]].to_string())

    print("\n=== Next 6 CPI prints (bottom-up decomposition, MoM %) ===")
    cols = ["mom_pct", "contrib_gasoline", "contrib_shelter",
            "contrib_core_goods", "contrib_supercore", "yoy_pct"]
    print(cpi_full[cols].tail(horizon_months).head(6).round(3).to_string())
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", default=True)
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()
    run(demo=args.demo, horizon_months=args.horizon)
