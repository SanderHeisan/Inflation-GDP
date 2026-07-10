"""
Walk-forward backtest loop.

At each as-of date: reconstruct the vintage information set, run the
*existing* quadmap projection code on it (no forked model logic), and record
the predicted quad and (d_growth, d_inflation) for the current quarter and
horizons t+1..t+H, where t is the calendar quarter containing the as-of
date. Benchmark predictions (persistence, base-effects-only) are computed
from the same vintage so every strategy sees identical information.
"""
from __future__ import annotations

import pandas as pd

from quadmap import gdp as gdp_mod, inflation, quads

from .btconfig import VintageConfig
from .data_bundle import RawDataBundle
from .vintage import VintageDataset, build_vintage


def model_quad_table(vintage: VintageDataset,
                     max_horizon: int) -> pd.DataFrame:
    """Run the quadmap pipeline on a vintage dataset and classify.

    Projection lengths are computed so the table covers asof_quarter +
    max_horizon even when GDP publication lags leave the model 1-2 quarters
    behind the calendar.
    """
    asof_q = pd.Period(vintage.asof, freq="Q")
    target_q = asof_q + max_horizon

    gdp_quarters = (target_q - vintage.last_gdp_quarter).n + 1
    cpi_months = (pd.Period(target_q.end_time, freq="M")
                  - vintage.last_cpi_month).n + 1

    cpi_full = inflation.build_cpi_projection(
        vintage.cpi_index, cpi_months, vintage.assumptions, vintage.i44)
    cpi_yoy_q = quads.monthly_to_quarterly_yoy(cpi_full["yoy_pct"].dropna())

    gdp_full = gdp_mod.project_gdp(vintage.gdp_level, vintage.indicators,
                                   horizon_quarters=gdp_quarters)
    return quads.classify(gdp_full["yoy_pct"], cpi_yoy_q)


def base_effects_quad_table(vintage: VintageDataset,
                            max_horizon: int) -> pd.DataFrame:
    """Benchmark (b): freeze all sequential rates at point-in-time trend and
    let the known base periods drive the YoY path. Any accuracy here is pure
    base-effect arithmetic, not projection skill."""
    asof_q = pd.Period(vintage.asof, freq="Q")
    target_q = asof_q + max_horizon

    gdp = vintage.gdp_level
    trend_qoq = gdp.pct_change().tail(20).mean()
    gq = (target_q - gdp.index[-1]).n + 1
    glevels = list(gdp)
    for _ in range(gq):
        glevels.append(glevels[-1] * (1.0 + trend_qoq))
    gidx = gdp.index.append(pd.period_range(gdp.index[-1] + 1, periods=gq,
                                            freq="Q"))
    gfull = pd.Series(glevels, index=gidx)
    g_yoy = (gfull / gfull.shift(4) - 1) * 100

    cpi = vintage.cpi_index
    trend_mom = cpi.pct_change().tail(60).mean()
    cm = (pd.Period(target_q.end_time, freq="M") - cpi.index[-1]).n + 1
    clevels = list(cpi)
    for _ in range(cm):
        clevels.append(clevels[-1] * (1.0 + trend_mom))
    cidx = cpi.index.append(pd.period_range(cpi.index[-1] + 1, periods=cm,
                                            freq="M"))
    cfull = pd.Series(clevels, index=cidx)
    c_yoy_q = quads.monthly_to_quarterly_yoy(
        ((cfull / cfull.shift(12) - 1) * 100).dropna())

    return quads.classify(g_yoy, c_yoy_q)


def asof_dates(start: str, end: str, freq: str = "M") -> pd.DatetimeIndex:
    """Month-end or quarter-end as-of dates. --start/--end accept a year
    ('2012') or any date-ish string."""
    start_ts = pd.Timestamp(f"{start}-01-01" if len(str(start)) == 4 else start)
    end_ts = pd.Timestamp(f"{end}-12-31" if len(str(end)) == 4 else end)
    pandas_freq = {"M": "ME", "Q": "QE"}[freq.upper().rstrip("E")]
    return pd.date_range(start_ts, end_ts, freq=pandas_freq)


def run_backtest(bundle: RawDataBundle, start: str, end: str,
                 freq: str = "M", max_horizon: int = 4,
                 cfg: VintageConfig | None = None) -> pd.DataFrame:
    """One tidy row per (as-of date, target quarter). Columns record the
    model's call, its deltas, conviction, and the benchmark calls, plus the
    vintage bookkeeping needed to audit point-in-time discipline."""
    cfg = cfg or VintageConfig()
    rows = []
    for asof in asof_dates(start, end, freq):
        vintage = build_vintage(bundle, asof, cfg)
        table = model_quad_table(vintage, max_horizon)
        bench_base = base_effects_quad_table(vintage, max_horizon)

        # Persistence benchmark: the quad of the last quarter fully observed
        # at the time (same vintage the model starts from).
        last_q = vintage.last_gdp_quarter
        persistence_quad = int(table.loc[last_q, "quad"]) \
            if last_q in table.index else None

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
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("backtest produced no predictions - check the "
                         "start/end range against the available data")
    return df
