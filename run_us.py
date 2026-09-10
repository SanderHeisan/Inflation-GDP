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
from usmodel import (config, data_bundle, gdp as gdp_mod,
                     growth_direction, inflation)


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
    trend_ann = (1 + d["trend_qoq_pct"] / 100) ** 4 * 100 - 100
    note = (f"real data | CPI through {v.last_cpi_month}, GDP through "
            f"{v.last_gdp_quarter}\n"
            f"  CPI blocks : shelter b={d['shelter_b']:.2f} a={d['shelter_a']:.2f}"
            f"   supercore b={d['supercore_b']:.2f}"
            f"   market rents {d['market_rent_yoy']:.1f}% YoY\n"
            f"  growth     : nowcast {d['nowcast_qoq_pct']:.2f}% QoQ "
            f"(k={d['nowcast_k']} months in), then flat at trend "
            f"{d['trend_qoq_pct']:.2f}% QoQ = {trend_ann:.1f}% annualized")
    return (v.cpi_index, v.gdp_level, v.assumptions, v.aux, v.indicators,
            v.last_gdp_quarter, note)


def _direction_block(table, gdp_hist, gdp_full, cpi_full, last_real,
                     trend_qoq_pct=None) -> str:
    """The four direction calls for the current quarter and the next, each
    with its conviction and the backtested hit rate for that kind of call
    (README, 'Direction calls'). Growth on a QoQ basis abstains past q+2."""
    cpi_q = cpi_full["cpi_index"].groupby(cpi_full.index.asfreq("Q")).mean()
    infl_qoq_d = (cpi_q.pct_change() * 100).diff()
    gd = growth_direction.qoq_direction_calls(gdp_hist, trend_qoq_pct)
    lines = [f"{'quarter':8s} {'growth YoY':>22s} {'growth QoQ':>22s} "
             f"{'inflation YoY':>22s} {'inflation QoQ':>22s}"]

    def fmt(delta, grade="call"):
        if delta is None or delta != delta:
            return f"{'no call':>22s}"
        arrow = "up" if delta > 0 else "down"
        return f"{arrow:>4s} {abs(delta):5.2f}pp {grade:>9s}"

    for tq in [last_real + 1, last_real + 2, last_real + 3]:
        if tq not in table.index:
            continue
        gy = table.loc[tq, "d_growth"]
        gy_grade = ("strong" if abs(gy)
                    >= growth_direction.YOY_HIGH_CONVICTION_PP else "weak")
        gq, gq_grade = ((gd.loc[tq, "delta_pp"], gd.loc[tq, "grade"])
                        if not gd.empty and tq in gd.index else (None, ""))
        iy = table.loc[tq, "d_inflation"]
        iy_grade = "strong" if abs(iy) >= 0.30 else "weak"
        iq = infl_qoq_d.get(tq, None)
        iq_grade = "strong" if iq is not None and abs(iq) >= 0.25 else "weak"
        lines.append(f"{str(tq):8s} {fmt(gy, gy_grade)} {fmt(gq, gq_grade)} "
                     f"{fmt(iy, iy_grade)} {fmt(iq, iq_grade)}")
    lines.append("  backtested hit rates by call and conviction: "
                 "results_us/us_direction_conviction.csv")
    return "\n".join(lines)


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

    print("\n=== Direction calls (accelerating / decelerating) ===")
    trend = indicators.get("fitted_trend_qoq")
    print(_direction_block(table, gdp_hist, gdp_full, cpi_full, last_real,
                           trend * 100 if trend is not None else None))

    print("\n=== Next 6 CPI prints (bottom-up decomposition, MoM %) ===")
    cols = ["mom_pct", "contrib_gasoline", "contrib_shelter",
            "contrib_core_goods", "contrib_supercore", "yoy_pct"]
    print(cpi_full[cols].tail(horizon_months).head(6).round(3).to_string())

    print(f"\nDeadband for low_conviction: +/-{config.QUAD_DEADBAND_PP}pp. "
          f"Measured accuracy of these calls is in the README.")
    if not demo:
        print("Note: the growth path past the nowcast quarter is FLAT at the "
              "trend above.\n  It is a trailing median, so it reflects the "
              "last six years rather than any\n  estimate of potential - "
              "read the projected growth LEVEL with that in mind.\n  The "
              "quad only uses the direction, which is far less sensitive to "
              "it.")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="synthetic data instead of the real cache")
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()
    run(demo=args.demo, horizon_months=args.horizon)
