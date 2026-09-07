"""
US real-GDP projection: nowcast + geometric convergence to trend, then
compound onto the last level so the YoY path falls out against known
year-ago levels. Same mechanics as quadmap/gdp.py, with US constants and
US nowcast indicators (ISM, payrolls, retail).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def nowcast_qoq(gdp_level: pd.Series, indicators: dict) -> float:
    """One-quarter-ahead real GDP QoQ (decimal).

    Preferred path: `indicators['fitted_qoq_pct']`, the point-in-time ridge
    nowcast from usmodel.nowcast, which already carries trailing momentum as
    one of its regressors -- so it is used directly rather than blended
    again. Backtested on 2017-2026 it roughly halves the QoQ error of the
    momentum fallback (see README).

    Fallback: the trailing-momentum blend, used when there is not enough
    published history to fit (early vintages) or no indicators at all. The
    scalar slots below are the original rough multipliers; they are only
    reached in the fallback, and the blend renormalizes over whatever is
    supplied so it degrades gracefully to momentum-only."""
    if "fitted_qoq_pct" in indicators:
        return float(indicators["fitted_qoq_pct"]) / 100.0

    w = config.GDP_NOWCAST_WEIGHTS
    signals = {"momentum": gdp_level.pct_change().iloc[-2:].mean()}
    if "ism" in indicators:            # 50 = neutral; calibrate the slope
        signals["ism"] = (indicators["ism"] - 50.0) * 0.0006
    if "payrolls" in indicators:       # monthly payroll gain (k) -> QoQ proxy
        signals["payrolls"] = (indicators["payrolls"] - 100.0) * 0.00002
    if "retail" in indicators:
        signals["retail"] = indicators["retail"] * 0.5
    used = {k: v for k, v in signals.items() if k in w}
    total_w = sum(w[k] for k in used)
    return sum(w[k] * v for k, v in used.items()) / total_w


def estimate_convergence(gdp_level: pd.Series,
                         min_quarters: int = 40) -> tuple[float, float] | None:
    """AR(1) of published QoQ growth: (long-run mean, persistence). Lets the
    convergence path be read off the vintage's own history instead of the
    config constants. Returns None when history is too short.

    The estimate is deliberately clipped: a window containing the 2020 crash
    and rebound fits a NEGATIVE persistence, which would have the projection
    oscillate rather than converge."""
    qoq = gdp_level.pct_change().dropna()
    if len(qoq) < min_quarters:
        return None
    x, y = qoq.shift(1).dropna(), qoq.iloc[1:]
    idx = x.index.intersection(y.index)
    if len(idx) < min_quarters or float(x[idx].var()) < 1e-12:
        return None
    rho = float(np.polyfit(x[idx].to_numpy(float), y[idx].to_numpy(float),
                           1)[0])
    return float(np.clip(qoq.mean(), 0.0, 0.015)), float(np.clip(rho, 0.0, 0.9))


def project_gdp(gdp_level: pd.Series, indicators: dict,
                horizon_quarters: int = 5) -> pd.DataFrame:
    """Extend the real-GDP level and compute the YoY path. Returns a frame
    indexed by Period[Q] with level, qoq_pct, yoy_pct, projected.

    The path is nowcast(q+1) then geometric convergence toward trend. Pass
    indicators['fitted_trend_qoq'] / ['fitted_convergence'] to use a
    point-in-time AR(1) estimate instead of the config constants."""
    q1 = nowcast_qoq(gdp_level, indicators)
    trend = float(indicators.get("fitted_trend_qoq", config.GDP_TREND_QOQ))
    conv = float(indicators.get("fitted_convergence", config.GDP_CONVERGENCE))

    path, qoq = [], q1
    for _ in range(horizon_quarters):
        path.append(qoq)
        qoq = trend + (qoq - trend) * conv

    last = gdp_level.index[-1]
    horizon = pd.period_range(last + 1, periods=horizon_quarters, freq="Q")
    level, proj = gdp_level.iloc[-1], []
    for g in path:
        level = level * (1.0 + g)
        proj.append(level)

    full = pd.concat([gdp_level, pd.Series(proj, index=horizon)])
    out = full.to_frame("level")
    out["qoq_pct"] = out["level"].pct_change() * 100
    out["yoy_pct"] = (out["level"] / out["level"].shift(4) - 1) * 100
    out["projected"] = out.index > last
    return out
