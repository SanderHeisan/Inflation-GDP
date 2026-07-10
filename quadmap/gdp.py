"""
Mainland-Norway GDP projection.

Two-stage structure:

  1. NOWCAST (next 1-2 quarters): blend of
       - trailing QoQ momentum (the 'base case is continuation' prior)
       - Norges Bank Regional Network output index (best single real-time
         growth indicator for Norway; published ~5x/year, scale roughly
         maps: index of 1.0 ~ 0.25% QoQ over the next quarter... calibrate!)
       - PMI (NIMA/DNB): (PMI - 50) * beta as a QoQ proxy
       - retail volume momentum
     Weights in config.GDP_NOWCAST_WEIGHTS.

  2. CONVERGENCE (quarters 3+): geometric convergence from the nowcast to
     trend growth (config.GDP_TREND_QOQ). You can override the far quarters
     with Norges Bank's MPR forecast path if you prefer an external anchor.

The projected QoQ path is compounded onto the last observed *level*, and the
YoY series falls out mechanically against known year-ago levels -- this is
where the base-effect logic lives.
"""
from __future__ import annotations

import pandas as pd

from . import config


def nowcast_qoq(gdp_level: pd.Series, indicators: dict) -> float:
    """One-quarter-ahead QoQ growth (decimal, e.g. 0.003 = 0.3%)."""
    w = config.GDP_NOWCAST_WEIGHTS
    momentum = gdp_level.pct_change().iloc[-2:].mean()

    signals = {"momentum": momentum}
    # Regional Network: survey index (~ -2..+2) -> QoQ. Calibrate the 0.0025
    # multiplier by regressing history of the RN index on realized QoQ.
    if "regional_network_index" in indicators:
        signals["regional_network"] = indicators["regional_network_index"] * 0.0025
    if "pmi" in indicators:
        signals["pmi"] = (indicators["pmi"] - 50.0) * 0.0004
    if "retail_qoq" in indicators:
        signals["retail"] = indicators["retail_qoq"] * 0.5

    used = {k: v for k, v in signals.items() if k in w}
    total_w = sum(w[k] for k in used)
    return sum(w[k] * v for k, v in used.items()) / total_w


def project_gdp(gdp_level: pd.Series, indicators: dict,
                horizon_quarters: int = 5) -> pd.DataFrame:
    """Extend the GDP level series and compute the YoY path.

    Returns DataFrame indexed by Period[Q] with columns:
    level, qoq_pct, yoy_pct, projected (bool).
    """
    q1 = nowcast_qoq(gdp_level, indicators)
    trend = config.GDP_TREND_QOQ
    conv = config.GDP_CONVERGENCE

    path = []
    qoq = q1
    for h in range(horizon_quarters):
        path.append(qoq)
        qoq = trend + (qoq - trend) * conv   # geometric convergence

    last = gdp_level.index[-1]
    horizon = pd.period_range(last + 1, periods=horizon_quarters, freq="Q")
    level = gdp_level.iloc[-1]
    proj_levels = []
    for g in path:
        level = level * (1.0 + g)
        proj_levels.append(level)

    full = pd.concat([gdp_level, pd.Series(proj_levels, index=horizon)])
    out = full.to_frame("level")
    out["qoq_pct"] = out["level"].pct_change() * 100
    out["yoy_pct"] = (out["level"] / out["level"].shift(4) - 1) * 100
    out["projected"] = out.index > last
    return out
