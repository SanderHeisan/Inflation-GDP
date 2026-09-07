"""
US real-GDP projection: nowcast + geometric convergence to trend, then
compound onto the last level so the YoY path falls out against known
year-ago levels. Same mechanics as quadmap/gdp.py, with US constants and
US nowcast indicators (ISM, payrolls, retail).
"""
from __future__ import annotations

import pandas as pd

from . import config


def nowcast_qoq(gdp_level: pd.Series, indicators: dict) -> float:
    """One-quarter-ahead real GDP QoQ (decimal). Blends trailing momentum
    with US activity indicators; renormalizes over whatever is supplied, so
    it degrades gracefully to momentum-only."""
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


def project_gdp(gdp_level: pd.Series, indicators: dict,
                horizon_quarters: int = 5) -> pd.DataFrame:
    """Extend the real-GDP level and compute the YoY path. Returns a frame
    indexed by Period[Q] with level, qoq_pct, yoy_pct, projected."""
    q1 = nowcast_qoq(gdp_level, indicators)
    trend, conv = config.GDP_TREND_QOQ, config.GDP_CONVERGENCE

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
