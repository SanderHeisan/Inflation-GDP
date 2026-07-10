"""
Norway GIP Quad Map -- pipeline runner.

  python run.py --demo    # synthetic data, validates the full pipeline offline
  python run.py           # live mode: pulls SSB + Norges Bank (needs internet)

Live mode requires assumptions.yaml to be updated with your current market
inputs (power forwards, Brent, NOK path, wage norm). Those are the levers
you touch every month; everything else is mechanical.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import yaml

from quadmap import config, gdp as gdp_mod, inflation, quads, plotting


def load_assumptions(path: str = "assumptions.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Demo data: synthetic but economically plausible series so the whole
# pipeline (projection -> base effects -> quad classification -> chart)
# can be validated without network access.
# ---------------------------------------------------------------------------

def make_demo_data(seed: int = 7):
    rng = np.random.default_rng(seed)

    months = pd.period_range("2015-01", "2026-05", freq="M")
    # CPI: 2% trend + an inflation wave peaking mid-sample + noise
    t = np.arange(len(months))
    mom = 0.02 / 12 + 0.025 / 12 * np.exp(-((t - 90) / 20.0) ** 2) \
        + rng.normal(0, 0.0012, len(months))
    cpi = pd.Series(100 * np.cumprod(1 + mom), index=months, name="cpi")

    quarters = pd.period_range("2015Q1", "2026Q1", freq="Q")
    tq = np.arange(len(quarters))
    qoq = 0.004 + 0.004 * np.sin(tq / 5.0) + rng.normal(0, 0.002, len(quarters))
    gdp = pd.Series(1000 * np.cumprod(1 + qoq), index=quarters, name="gdp")

    i44 = pd.Series(100 + np.cumsum(rng.normal(0.05, 0.6, len(months))),
                    index=months, name="I44")
    return cpi, gdp, i44


def demo_assumptions(last_month: pd.Period) -> dict:
    horizon = pd.period_range(last_month + 1, periods=15, freq="M")
    return {
        "power_recent_ore_kwh": 110.0,
        "power_forward_ore_kwh": {str(p): 110.0 - i * 1.5
                                  for i, p in enumerate(horizon)},
        "power_variable_share": 0.55,
        "brent_recent_usd": 78.0,
        "brent_forward_usd": {str(p): 78.0 + i * 0.4
                              for i, p in enumerate(horizon)},
        "usdnok_recent": 10.6,
        "usdnok_path": {str(p): 10.6 for p in horizon},
        "i44_path": {},
        "imported_goods_baseline_yoy": 1.0,
        "wage_norm_pct": 4.5,
        "food_pipeline_yoy": 3.0,
        "fuel_passthrough": 0.40,
        "services_wage_haircut": 0.75,
    }


def run(demo: bool, horizon_months: int = 15):
    if demo:
        cpi_hist, gdp_hist, i44_hist = make_demo_data()
        assumptions = demo_assumptions(cpi_hist.index[-1])
        indicators = {"regional_network_index": 0.8, "pmi": 51.5,
                      "retail_qoq": 0.002}
    else:
        from quadmap import data_sources as ds
        cpi_groups = ds.fetch_cpi_by_group()
        cpi_hist = cpi_groups.iloc[:, 0].dropna()   # total CPI column
        gdp_hist = ds.fetch_mainland_gdp()
        i44_hist = ds.fetch_i44()
        assumptions = load_assumptions()
        indicators = assumptions.get("gdp_indicators", {})

    # --- Inflation: bottom-up component projection -> YoY path
    cpi_full = inflation.build_cpi_projection(
        cpi_hist, horizon_months, assumptions, i44_hist)
    cpi_yoy_q = quads.monthly_to_quarterly_yoy(cpi_full["yoy_pct"].dropna())

    # --- GDP: nowcast + convergence -> YoY path
    gdp_full = gdp_mod.project_gdp(gdp_hist, indicators, horizon_quarters=5)

    # --- Quads
    table = quads.classify(gdp_full["yoy_pct"], cpi_yoy_q)
    table["projected"] = table.index > gdp_hist.index[-1]

    flips = quads.quad_flips(table[table["projected"] |
                                   (table.index >= gdp_hist.index[-1] - 3)])

    # --- Output
    pd.set_option("display.width", 140)
    recent = table.tail(9)
    print("\n=== Norway quad table (last realized + projected) ===")
    print(recent[["growth_yoy", "inflation_yoy", "d_growth", "d_inflation",
                  "quad", "label", "low_conviction", "projected"]]
          .round(2).to_string())
    if len(flips):
        print("\n=== Quad flips (the tradeable events) ===")
        print(flips[["quad", "label", "projected"]].to_string())

    table.to_csv("quad_projections.csv")
    cpi_full.tail(horizon_months + 3).round(3).to_csv("cpi_decomposition.csv")
    plotting.plot_quad_map(table, "quad_map.png")
    print("\nWrote quad_projections.csv, cpi_decomposition.csv, quad_map.png")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data (no network needed)")
    ap.add_argument("--horizon", type=int, default=15,
                    help="CPI projection horizon in months")
    args = ap.parse_args()
    run(demo=args.demo, horizon_months=args.horizon)
