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
            bundle.gdp.index[-1], "synthetic demo", {}, None, None)


def _live_inputs(horizon_months: int):
    """Today's vintage, built by the backtest's own vintage machinery."""
    from usbacktest.btconfig import USVintageConfig
    from usbacktest.vintage import build_vintage

    bundle = data_bundle.load_bundle()
    # current vintage is the live one; and a live run may use the month in
    # progress for oil, the dollar and the pump price (see btconfig)
    # ... and the market-implied policy path extends the rate channel
    # forward (config.POLICY_RATE_PATH, dated; the backtest never sees it)
    cfg = USVintageConfig(
        revision_mode="none", live_partial_month=True,
        policy_rate_path=(config.POLICY_RATE_PATH
                          if config.RATE_CHANNEL_ENABLED else None))
    v = build_vintage(bundle, pd.Timestamp.today().normalize(), cfg,
                      horizon_months=horizon_months + 2)
    d = v.diagnostics
    trend_ann = (1 + d["trend_qoq_pct"] / 100) ** 4 * 100 - 100
    rs = d.get("rate_state")
    rate_line = ""
    if rs:
        rs["pace_qoq_pct"] = float(d["pace_qoq_pct"])
        f = rs["fit"]
        rate_line = (f"\n  rates      : fed funds {rs['level_pct']:.2f}% "
                     f"({rs['latest_month']}), {rs['chg_4q_pp']:+.2f}pp over 4q, "
                     f"{rs['chg_8q_pp']:+.2f}pp over 8q; drag sensitivity "
                     f"{f['beta']:.3f}pp/q per 1pp (raw fit {f['beta_raw']:+.3f}, "
                     f"n={f['n']}, {f['fit_start']}..{f['fit_end']}); path "
                     f"{rs['path_source']}")
    note = (f"real data | CPI through {v.last_cpi_month}, GDP through "
            f"{v.last_gdp_quarter}\n"
            f"  CPI blocks : shelter b={d['shelter_b']:.2f} a={d['shelter_a']:.2f}"
            f"   supercore b={d['supercore_b']:.2f}"
            f"   market rents {d['market_rent_yoy']:.1f}% YoY\n"
            f"  growth     : nowcast {d['nowcast_qoq_pct']:.2f}% QoQ "
            f"(k={d['nowcast_k']} months in), then pace mode "
            f"'{d['pace_mode']}' = {d['pace_qoq_pct']:.2f}% QoQ "
            f"({(1 + d['pace_qoq_pct'] / 100) ** 4 * 100 - 100:.1f}% annualized"
            f"{', persistence ' + format(d['pace_persistence'], '.2f') if d['pace_persistence'] else ''})"
            f"; trailing median was {d['trend_qoq_pct']:.2f}% = {trend_ann:.1f}%\n"
            f"  energy     : WTI ${v.assumptions['wti_recent']:.0f} at the last CPI "
            f"month, {d['gasoline_pump_months_observed']} later month(s) of pump "
            f"prices observed (incl. the month in progress); "
            f"pump slope {d['gasoline_pump_slope']:.2f}{rate_line}")
    consumer = growth_direction.consumer_state(v.indicator_panel,
                                               v.last_gdp_quarter)
    real_pce = v.indicator_panel.get("real_pce")
    return (v.cpi_index, v.gdp_level, v.assumptions, v.aux, v.indicators,
            v.last_gdp_quarter, note, consumer, real_pce, rs)


def _rates_block(rs: dict, trend_qoq_pct: float) -> str:
    """The rate channel quarter by quarter: the policy path the projection
    assumes and where each number comes from, the drag it puts on QoQ
    growth past the nowcast quarter, and the cumulative level effect.
    The channel is on because it was asked for; its measured cost to the
    growth call is in the README and results_us/us_rate_channel.csv."""
    if not rs:
        return "  (rate channel off, or no policy-rate data)"
    lines = [f"  {'quarter':8s} {'policy rate':>12s} {'source':>20s} "
             f"{'drag pp/q':>10s} {'ann.':>7s} {'path QoQ':>9s} {'cum level':>10s}"
             f"   (path = pace {trend_qoq_pct:.2f}% + drag)"]
    drag = {r["quarter"]: r for r in rs.get("drag", [])}
    for r in rs["path"]:
        q = r["quarter"]
        if q not in drag:
            continue
        d = drag[q]
        lines.append(f"  {q:8s} {r['rate_pct']:11.2f}% {r['source']:>20s} "
                     f"{d['drag_qoq_pp']:+10.2f} {d['drag_ann_pp']:+6.1f}% "
                     f"{trend_qoq_pct + d['drag_qoq_pp']:8.2f}% "
                     f"{d['cum_level_pct']:+9.2f}%")
    f = rs["fit"]
    lines.append(f"  Window: the change in the policy rate over the year ending "
                 f"{rs['lags'][0]} quarters before the target ({rs['lags'][0]}-"
                 f"{rs['lags'][1]}q lag); sensitivity {f['beta']:.3f}pp of "
                 f"quarterly growth per 1pp, fitted point-in-time on "
                 f"{f['fit_start']}..{f['fit_end']} (n={f['n']}, corr "
                 f"{f['corr']:+.2f}).")
    lines.append("  Backtested 2017-2026 the drag LOWERS growth-direction "
                 "accuracy (README, 'The rate channel'); set "
                 "config.RATE_CHANNEL_ENABLED = False for the flat path.")
    return "\n".join(lines)


def _consumer_block(c: dict) -> str:
    """Where the consumer stands against its own trailing decade, with the
    realized base rates behind each flag (growth_direction.consumer_state)."""
    if not c:
        return "  (no consumer data in this bundle)"
    lines = []
    if "real_pce" in c:
        p = c["real_pce"]
        flag = (" STRETCHED (top decile: next-quarter GDP QoQ lower 70%, "
                "YoY decelerating 70-75% historically)" if p["stretched"]
                else " depressed (bottom decile)" if p["depressed"] else "")
        lines.append(f"  real consumer spending : {p['yoy_pct']:+.2f}% YoY "
                     f"({p['latest_month']}); last quarter {p['last_quarter_saar_pct']:+.1f}% "
                     f"ann. vs trailing median "
                     f"{((1 + p['trailing_median_qoq_pct'] / 100) ** 4 - 1) * 100:+.1f}%; "
                     f"10y percentile {p['pctl_10y']:.0%}{flag}")
    for key, r in c.get("detail", {}).items():
        lines.append(f"    {r['label']:24s}: {r['yoy_pct']:+.1f}% YoY, "
                     f"{r['m3_ann_pct']:+.1f}% 3m ann. ({r['latest_month']}); "
                     f"10y percentile {r['pctl_10y']:.0%}")
    if "sentiment" in c:
        sn = c["sentiment"]
        flag = (" elevated (top decile: no historical signal)" if sn["elevated"]
                else " depressed (bottom decile: next-quarter GDP QoQ HIGHER 63% historically)"
                if sn["depressed"] else "")
        lines.append(f"  consumer sentiment     : {sn['level']:.1f} ({sn['latest_month']}); "
                     f"10y percentile {sn['pctl_10y']:.0%}{flag}")
    if "net_worth" in c:
        n = c["net_worth"]
        flag = (" ELEVATED (top quintile: growth YoY decelerating 3-4q later "
                "61-68% historically)" if n["elevated"] else
                " depressed (bottom quintile: decelerating only 32-36%)"
                if n["depressed"] else "")
        lines.append(f"  household net worth    : ${n['level_tn']:.1f}T "
                     f"({n['latest_quarter']}), {n['qoq_change_tn']:+.1f}T on the "
                     f"quarter, {n['yoy_pct']:+.1f}% YoY; 10y percentile "
                     f"{n['pctl_10y']:.0%}{flag}")
    if "mortgage_30y" in c:
        m = c["mortgage_30y"]
        flag = (" falling (bottom quintile: growth accelerating 2-3q later 75-79%)"
                if m["falling"] else " rising (top quintile: weak signal, 57%)"
                if m["rising"] else "")
        lines.append(f"  30y mortgage rate      : {m['level_pct']:.2f}% "
                     f"({m['latest_month']}), {m['chg_4q_pp']:+.2f}pp over 4q; "
                     f"10y percentile of the change {m['pctl_10y']:.0%}{flag}")
    if "fed_funds" in c:
        f = c["fed_funds"]
        lines.append(f"  fed funds              : {f['level_pct']:.2f}% "
                     f"({f['latest_month']}), {f['chg_4q_pp']:+.2f}pp over 4q")
    if "real_income" in c:
        i = c["real_income"]
        lines.append(f"  real disposable income : {i['yoy_pct']:+.2f}% YoY "
                     f"({i['latest_month']}); 10y percentile {i['pctl_10y']:.0%}")
    if "saving_rate" in c:
        sv = c["saving_rate"]
        lines.append(f"  saving rate            : {sv['level_pct']:.1f}% "
                     f"({sv['latest_month']}); 10y percentile {sv['pctl_10y']:.0%}")
    return "\n".join(lines)


def _direction_block(table, gdp_hist, gdp_full, cpi_full, last_real,
                     trend_qoq_pct=None, real_pce=None) -> str:
    """The four direction calls for the current quarter and the next, each
    with its conviction and the backtested hit rate for that kind of call
    (README, 'Direction calls'). Growth on a QoQ basis abstains past q+2."""
    cpi_q = cpi_full["cpi_index"].groupby(cpi_full.index.asfreq("Q")).mean()
    infl_qoq_d = (cpi_q.pct_change() * 100).diff()
    gd = growth_direction.qoq_direction_calls(gdp_hist, trend_qoq_pct,
                                              real_pce)
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
    if not gd.empty:
        r = gd.iloc[0]
        votes = (f"  growth QoQ votes for {gd.index[0]}: GDP reversal "
                 f"{r['gdp_vote_pp']:+.2f}pp")
        if r["pce_vote_pp"] == r["pce_vote_pp"]:
            votes += (f", real-PCE reversal {r['pce_vote_pp']:+.2f}pp -> "
                      f"{'AGREE: call' if r['grade'] == 'call' else 'SPLIT: coin flip'}"
                      f" ({r['backtest_hit']:.0%} backtested)")
        lines.append(votes)
    lines.append("  backtested hit rates by call and conviction: "
                 "results_us/us_direction_conviction.csv")
    return "\n".join(lines)


def run(demo: bool = False, horizon_months: int = 15):
    (cpi_hist, gdp_hist, assumptions, aux, indicators, last_real, note,
     consumer, real_pce, rate_state) = (_demo_inputs(horizon_months) if demo
                                        else _live_inputs(horizon_months))
    print(f"\nUS GIP Quad Map | {note}")
    if not demo:
        print("\n=== The consumer (published data, vs its own trailing decade) ===")
        print(_consumer_block(consumer))
        print("\n=== The rate channel (policy rate -> growth, 2-6 quarters later) ===")
        print(_rates_block(rate_state, rate_state.get("pace_qoq_pct",
                                                      config.GDP_TREND_QOQ * 100)))

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
                           trend * 100 if trend is not None else None,
                           real_pce))

    print("\n=== Next 6 CPI prints (bottom-up decomposition, MoM %) ===")
    cols = ["mom_pct", "contrib_gasoline", "contrib_shelter",
            "contrib_core_goods", "contrib_supercore", "yoy_pct"]
    print(cpi_full[cols].tail(horizon_months).head(6).round(3).to_string())

    print(f"\nDeadband for low_conviction: +/-{config.QUAD_DEADBAND_PP}pp. "
          f"Measured accuracy of these calls is in the README.")
    if not demo:
        print("Note: the growth path past the nowcast quarter is the pace "
              "above plus the rate\n  channel's drag (see the rates block; "
              "zero when the channel is off). It is an\n  assumption, not "
              "a forecast - read the projected growth LEVEL with that in "
              "mind.\n  The quad only uses the direction against a known "
              "year-ago base.")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="synthetic data instead of the real cache")
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()
    run(demo=args.demo, horizon_months=args.horizon)
