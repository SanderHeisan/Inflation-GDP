"""
Walk-forward US backtest loop.

At each as-of date: reconstruct the vintage information set, run the
*existing* usmodel projection code on it (no forked model logic), and record
the predicted quad and (d_growth, d_inflation) for the current quarter and
horizons t+1..t+H, where t is the calendar quarter containing the as-of
date. Benchmark predictions are computed from the same vintage, so every
strategy sees an identical information set.

Benchmarks:
  persistence   - "next quarter looks like the last one you can see"
  base_effects  - freeze all sequential rates at point-in-time trend and let
                  the known year-ago levels drive the YoY path. Anything it
                  scores is pure base-effect arithmetic, not projection
                  skill, so the model's edge over it is the honest measure
                  of what the component model adds.
  random        - 25% by construction (uniform over four quads)
"""
from __future__ import annotations

import pandas as pd

from quadmap import quads
from usmodel import gdp as gdp_mod, inflation
from usmodel.data_bundle import USDataBundle

from .btconfig import USVintageConfig
from .vintage import USVintage, build_vintage


def model_quad_table(vintage: USVintage, max_horizon: int) -> pd.DataFrame:
    """Run the usmodel pipeline on a vintage and classify. Projection
    lengths are computed so the table reaches asof_quarter + max_horizon
    even when publication lags leave the model a quarter or two behind the
    calendar."""
    asof_q = pd.Period(vintage.asof, freq="Q")
    target_q = asof_q + max_horizon

    gdp_quarters = (target_q - vintage.last_gdp_quarter).n + 1
    cpi_months = (pd.Period(target_q.end_time, freq="M")
                  - vintage.last_cpi_month).n + 1

    cpi_full = inflation.build_cpi_projection(
        vintage.cpi_index, cpi_months, vintage.assumptions, vintage.aux)
    cpi_yoy_q = quads.monthly_to_quarterly_yoy(cpi_full["yoy_pct"].dropna())

    gdp_full = gdp_mod.project_gdp(vintage.gdp_level, vintage.indicators,
                                   horizon_quarters=gdp_quarters)
    return quads.classify(gdp_full["yoy_pct"], cpi_yoy_q)


def base_effects_quad_table(vintage: USVintage,
                            max_horizon: int) -> pd.DataFrame:
    """Benchmark (b): sequential rates frozen at point-in-time trend, known
    base periods do the rest."""
    asof_q = pd.Period(vintage.asof, freq="Q")
    target_q = asof_q + max_horizon

    gdp = vintage.gdp_level
    trend_qoq = gdp.pct_change().tail(20).mean()
    gq = max((target_q - gdp.index[-1]).n + 1, 1)
    glevels = list(gdp)
    for _ in range(gq):
        glevels.append(glevels[-1] * (1.0 + trend_qoq))
    gidx = gdp.index.append(pd.period_range(gdp.index[-1] + 1, periods=gq,
                                            freq="Q"))
    g_yoy = (pd.Series(glevels, index=gidx).pct_change(4)) * 100

    cpi = vintage.cpi_index
    trend_mom = cpi.pct_change().tail(60).mean()
    cm = max((pd.Period(target_q.end_time, freq="M") - cpi.index[-1]).n + 1, 1)
    clevels = list(cpi)
    for _ in range(cm):
        clevels.append(clevels[-1] * (1.0 + trend_mom))
    cidx = cpi.index.append(pd.period_range(cpi.index[-1] + 1, periods=cm,
                                            freq="M"))
    c_yoy_q = quads.monthly_to_quarterly_yoy(
        (pd.Series(clevels, index=cidx).pct_change(12) * 100).dropna())

    return quads.classify(g_yoy.dropna(), c_yoy_q)


def asof_dates(start: str, end: str, freq: str = "M") -> pd.DatetimeIndex:
    """Month-end or quarter-end as-of dates. start/end accept a year
    ('2017'), a month ('2017-01') or any date-ish string."""
    def _ts(v, end_of: bool):
        v = str(v)
        if len(v) == 4:
            return pd.Timestamp(f"{v}-12-31" if end_of else f"{v}-01-01")
        if len(v) == 7:
            p = pd.Period(v, freq="M")
            return p.end_time.normalize() if end_of else p.start_time
        return pd.Timestamp(v)
    pandas_freq = {"M": "ME", "Q": "QE"}[freq.upper().rstrip("E")]
    return pd.date_range(_ts(start, False), _ts(end, True), freq=pandas_freq)


def run_backtest(bundle: USDataBundle, start: str, end: str,
                 freq: str = "M", max_horizon: int = 4,
                 cfg: USVintageConfig | None = None) -> pd.DataFrame:
    """One tidy row per (as-of date, target quarter), recording the model's
    call, its deltas, conviction, the benchmark calls, and the vintage
    bookkeeping needed to audit point-in-time discipline."""
    cfg = cfg or USVintageConfig()
    rows = []
    for asof in asof_dates(start, end, freq):
        try:
            vintage = build_vintage(bundle, asof, cfg)
        except ValueError:
            continue
        table = model_quad_table(vintage, max_horizon)
        bench_base = base_effects_quad_table(vintage, max_horizon)

        # Persistence: the quad of the last quarter fully observed at the
        # time, read off the same vintage the model starts from.
        last_q = vintage.last_gdp_quarter
        persistence_quad = (int(table.loc[last_q, "quad"])
                            if last_q in table.index else None)

        asof_q = pd.Period(asof, freq="Q")
        for h in range(0, max_horizon + 1):
            tq = asof_q + h
            if tq not in table.index:
                continue
            r = table.loc[tq]
            rows.append({
                "asof": asof,
                "asof_quarter": str(asof_q),
                "target_quarter": str(tq),
                "horizon": h,
                "pred_quad": int(r["quad"]),
                "pred_label": r["label"],
                "pred_d_growth": float(r["d_growth"]),
                "pred_d_inflation": float(r["d_inflation"]),
                "pred_growth_yoy": float(r["growth_yoy"]),
                "pred_inflation_yoy": float(r["inflation_yoy"]),
                "low_conviction": bool(r["low_conviction"]),
                "bench_persistence_quad": persistence_quad,
                "bench_base_quad": (int(bench_base.loc[tq, "quad"])
                                    if tq in bench_base.index else None),
                "last_gdp_quarter": str(vintage.last_gdp_quarter),
                "last_cpi_month": str(vintage.last_cpi_month),
                "revision_mode": vintage.revision_mode,
                "shelter_b": vintage.diagnostics["shelter_b"],
                "supercore_b": vintage.diagnostics["supercore_b"],
                "nowcast_qoq_pct": vintage.diagnostics["nowcast_qoq_pct"],
                "nowcast_k": vintage.diagnostics["nowcast_k"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("backtest produced no predictions - check the "
                         "start/end range against the available data")
    return df
