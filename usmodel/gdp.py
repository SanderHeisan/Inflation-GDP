"""
US real-GDP projection: nowcast the first unpublished quarter, hold the rest
flat at trend, and compound onto the last level so the YoY path falls out
against known year-ago levels.

Why flat rather than a converging path. The quad needs the sign of

    d_growth(q) = yoy(q) - yoy(q-1) ~= qoq(q) - qoq(q-4)

and qoq(q-4) is already PUBLISHED for most of the horizons the quad table
covers. So the growth call is "will next year's QoQ come in above or below a
number we already know". Backtesting says the first half of that is the only
part with any signal: past the nowcast quarter, every projected-QoQ variant
tried here -- geometric convergence, a point-in-time AR(1), and direct ridge
regressions at each horizon on a coincident panel, a leading panel (financial
conditions, credit spreads, permits, capex orders, equities, real M2, housing
starts) and both together -- had essentially zero correlation with realized
QoQ and did worse than a constant. A path that moves without carrying
information just adds noise to the known base effect, so the projection
stops moving once the nowcast quarter is past.

That leaves a hard ceiling, and it is worth stating: with zero QoQ skill and
a perfectly calibrated constant, the growth-direction call is right
E[Phi(|Z|)] = 75% of the time. Measured on the 2017-2026 sample the ceiling
is 73.7%. Chasing growth accuracy above that requires forecasting quarterly
GDP two to four quarters out, which nothing in this repo (or the literature)
can do.
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


def estimate_trend_qoq(gdp_level: pd.Series,
                       window: int = config.GDP_TREND_WINDOW_Q) -> float | None:
    """Trailing MEDIAN of published QoQ growth (decimal) -- the constant the
    projection holds past the nowcast quarter, and therefore the number every
    multi-quarter growth call is measured against.

    Median rather than mean because 2020Q2/2020Q3 are a crash-and-rebound
    pair that drags a mean for years; the median ignores them. Returns None
    when there is not enough published history."""
    qoq = gdp_level.pct_change().dropna()
    if len(qoq) < 8:
        return None
    return float(np.clip(qoq.tail(window).median(), 0.0, 0.015))


def estimate_convergence(gdp_level: pd.Series,
                         min_quarters: int = 40) -> tuple[float, float] | None:
    """(trend, persistence) for the projected path.

    Persistence is measured as an AR(1) on published QoQ and clipped to
    [0, 0.9], but the shipped default ignores it (config.GDP_CONVERGENCE = 0)
    because the backtest scores it as harmful -- see the module docstring.
    The estimator is kept so `--growth-variants` can reproduce that result.
    """
    trend = estimate_trend_qoq(gdp_level)
    if trend is None:
        return None
    qoq = gdp_level.pct_change().dropna()
    if len(qoq) < min_quarters:
        return trend, config.GDP_CONVERGENCE
    x, y = qoq.shift(1).dropna(), qoq.iloc[1:]
    idx = x.index.intersection(y.index)
    if len(idx) < min_quarters or float(x[idx].var()) < 1e-12:
        return trend, config.GDP_CONVERGENCE
    rho = float(np.polyfit(x[idx].to_numpy(float), y[idx].to_numpy(float),
                           1)[0])
    return trend, float(np.clip(rho, 0.0, 0.9))


def project_gdp(gdp_level: pd.Series, indicators: dict,
                horizon_quarters: int = 5) -> pd.DataFrame:
    """Extend the real-GDP level and compute the YoY path. Returns a frame
    indexed by Period[Q] with level, qoq_pct, yoy_pct, projected.

    The path is nowcast(q+1), then trend from q+2 on. With the shipped
    config.GDP_CONVERGENCE = 0 that is a step, not a glide -- deliberately,
    see the module docstring. Pass indicators['fitted_trend_qoq'] /
    ['fitted_convergence'] to override either with a point-in-time
    estimate."""
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
