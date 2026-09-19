"""
A MONTHLY growth measure, and the near-term growth call it makes possible.

Why. The quad's growth axis is quarterly real GDP, which prints once a
quarter, a month late, and is then revised for years. A subscriber who
wants a growth call one or two months ahead -- the same shape as the CPI
call, and what a monthly quad map (Hedgeye's is monthly) is built on --
needs a monthly measure of real activity whose next month can be called
the way next month's CPI is: next month's YoY change = next month's MoM
minus the MoM that drops out of the 12-month window, and only the first
term is a forecast.

The measure. A geometric index of the NBER-style coincident set, weighted
toward the consumer because that is what real GDP is made of:

    real PCE 0.55, industrial production 0.15, real personal income ex
    transfers 0.10, real retail sales 0.10, labor input (payrolls x hours) 0.10

Measured against real GDP (2008-2026, final data): correlation of the
quarterly-average YoY with GDP YoY 0.97; the direction of the change
agrees with the quad axis two months in three -- the third month is
inventories, imports and government, which no monthly measure carries.
So the measure is the consumer-and-production trend, not the GDP print;
the quarterly quad still comes from GDP, and this is the month-by-month
read underneath it.

What can be called. On its own YoY the next month's direction is right
~78% of the time on the base-effect rule with a trailing-mean MoM, ~90% in
the more convinced half, and two months out ~75% / ~87% (1995-2026
ex-COVID, study in the README). Point-in-time the components arrive at
different lags (payrolls and hours ~8 days, IP and retail ~17, PCE and
income ~30), so the first unpublished month is a NOWCAST with most of its
components in hand, and the months after it are forecasts. The walk-forward
numbers, with those lags, are in results_us/us_growth_monthly.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = (("real_pce", 0.55), ("indpro", 0.15),
              ("real_income_ex_transfers", 0.10), ("real_retail", 0.10),
              ("labor_input", 0.10))
MOM_WINDOW = 6            # trailing months the MoM forecast averages over
BUCKETS = [(0.00, 0.05, "toss-up (<0.05pp)"), (0.05, 0.15, "lean (0.05-0.15pp)"),
           (0.15, 0.30, "call (0.15-0.30pp)"), (0.30, np.inf, "strong (>0.30pp)")]


def bucket_label(conviction_pp: float) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= conviction_pp < hi:
            return label
    return BUCKETS[-1][2]


def labor_input(panel: dict) -> pd.Series | None:
    """Payrolls x average weekly hours (all private employees); payrolls
    alone before hours exist."""
    pay = panel.get("payrolls")
    if pay is None or pay.empty:
        return None
    hrs = panel.get("hours_all")
    if hrs is None or hrs.empty:
        return pay
    idx = pay.index.intersection(hrs.index)
    if len(idx) < 24:
        return pay
    return (pay.reindex(idx) * hrs.reindex(idx)).dropna()


def component_series(panel: dict) -> dict[str, pd.Series]:
    out = {}
    for name, _ in COMPONENTS:
        s = labor_input(panel) if name == "labor_input" else panel.get(name)
        if s is not None and len(s) > 24:
            out[name] = s.dropna()
    return out


def activity_index(panel: dict, horizon_months: int = 2,
                   mom_window: int = MOM_WINDOW) -> pd.DataFrame | None:
    """The index, month by month: published months, then `horizon_months`
    beyond the last month every component has published.

    Columns: level, mom_pct, yoy_pct, status ('actual' where every
    component is published, 'nowcast' where some are, 'forecast' where
    none are), n_actual (components published that month). A component's
    missing month takes its own trailing-`mom_window` mean MoM, so the
    nowcast month leans on the components that are in."""
    comps = component_series(panel)
    if len(comps) < 3 or "real_pce" not in comps:
        return None
    weights = {n: w for n, w in COMPONENTS if n in comps}
    wsum = sum(weights.values())
    weights = {n: w / wsum for n, w in weights.items()}
    start = max(s.index[0] for s in comps.values())
    last_complete = min(s.index[-1] for s in comps.values())
    last_any = max(s.index[-1] for s in comps.values())
    end = max(last_any, last_complete + horizon_months)
    months = pd.period_range(start, end, freq="M")
    moms = {}
    for n, s in comps.items():
        m = (np.log(s) - np.log(s.shift(1))).reindex(months)
        hist = m.dropna()
        fill = float(hist.tail(mom_window).mean()) if len(hist) else 0.0
        moms[n] = m.fillna(fill)
    frame = pd.DataFrame(moms)
    idx_mom = sum(frame[n] * w for n, w in weights.items())
    idx_mom.iloc[0] = 0.0
    level = 100.0 * np.exp(idx_mom.cumsum())
    n_actual = pd.Series({m: sum(int(m <= s.index[-1]) for s in comps.values())
                          for m in months})
    status = pd.Series(np.where(months <= last_complete, "actual",
                                np.where(n_actual > 0, "nowcast", "forecast")),
                       index=months)
    out = pd.DataFrame({"level": level, "mom_pct": idx_mom * 100,
                        "yoy_pct": (level / level.shift(12) - 1) * 100,
                        "status": status, "n_actual": n_actual})
    out.attrs["last_complete"] = last_complete
    out.attrs["weights"] = weights
    return out


def growth_calls(index: pd.DataFrame, horizon_months: int = 2) -> pd.DataFrame:
    """The near-term calls on the measure's YoY: for each month past the
    last complete one, the predicted YoY change, its direction and
    conviction, and what drops out of the 12-month window (the base).
    Indexed by target month; empty if the index is too short."""
    if index is None or len(index) < 15:
        return pd.DataFrame()
    last = index.attrs["last_complete"]
    yoy, mom = index["yoy_pct"], index["mom_pct"]
    rows = []
    for h in range(1, horizon_months + 1):
        t = last + h
        if t not in yoy.index or (t - 12) not in mom.index or np.isnan(yoy.get(last, np.nan)):
            continue
        d = float(yoy[t] - yoy[t - 1])
        rows.append({"target": t, "horizon": h, "status": index.loc[t, "status"],
                     "n_actual": int(index.loc[t, "n_actual"]),
                     "yoy_pct": float(yoy[t]), "pred_mom_pct": float(mom[t]),
                     "base_mom_pct": float(mom[t - 12]), "d_yoy_pp": d,
                     "direction": "up" if d > 0 else "down",
                     "conviction_pp": abs(d), "bucket": bucket_label(abs(d))})
    return pd.DataFrame(rows).set_index("target") if rows else pd.DataFrame()


def monthly_quads(growth_yoy: pd.Series, cpi_yoy: pd.Series,
                  deadband_pp: float = 0.0) -> pd.DataFrame:
    """A quad for every month both series cover, from the month-on-month
    change in each YoY rate (the monthly analogue of the quarterly quad).
    `deadband_pp` flags months where either change is smaller than it."""
    idx = growth_yoy.dropna().index.intersection(cpi_yoy.dropna().index)
    g = growth_yoy.reindex(idx); c = cpi_yoy.reindex(idx)
    dg, dc = g.diff(), c.diff()
    quad = np.where(dg > 0, np.where(dc > 0, 2, 1), np.where(dc > 0, 3, 4))
    out = pd.DataFrame({"growth_yoy": g, "inflation_yoy": c, "d_growth": dg,
                        "d_inflation": dc, "quad": quad}, index=idx)
    out["close"] = (dg.abs() < deadband_pp) | (dc.abs() < deadband_pp)
    return out.iloc[1:]
