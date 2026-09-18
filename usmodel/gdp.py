"""
US real-GDP projection: nowcast the first unpublished quarter, then a
constant pace (config.GDP_PACE_MODE: potential, the nowcast carried, a
glide between them, or the trailing median), plus the rate channel's drag,
compounded onto the last level so the YoY path falls out against known
year-ago levels.

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

Which constant, then. The trailing 24-quarter median scored best on
2017-2026 -- 3.1% annualized today, because the window is the post-2020
boom -- and it scored best precisely because that sample's past kept
repeating. It is backward-looking by construction, and the subscriber's
brief is to be forward-looking. So the pace past the nowcast quarter is a
config choice (config.GDP_PACE_MODE), shipped as a potential pace of 2.0%
a year, with the nowcast carried forward and a glide between the two as
alternatives; `python us_backtest.py` measures all four on identical
vintages (us_pace_modes.csv) and the README states what the switch costs on
the sample it can be measured on.
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


def potential_qoq(ann_pct: float = config.GDP_POTENTIAL_ANN_PCT) -> float:
    """A long-run annualized pace as a quarterly decimal."""
    return (1.0 + ann_pct / 100.0) ** 0.25 - 1.0


def pace_past_nowcast(q1: float, indicators: dict,
                      mode: str | None = None) -> tuple[float, float, str]:
    """(anchor, persistence, mode) for the path from q+2 on, per
    config.GDP_PACE_MODE or indicators['pace_mode']:
      trailing_median  anchor = the fitted trailing median, persistence 0
      potential        anchor = potential, persistence 0
      nowcast          anchor = the nowcast itself (carried forward)
      glide            anchor = potential, persistence GDP_GLIDE_PERSISTENCE
    """
    mode = mode or indicators.get("pace_mode", config.GDP_PACE_MODE)
    if mode not in config.GDP_PACE_MODES:
        raise ValueError(f"pace_mode must be one of {config.GDP_PACE_MODES}, got {mode!r}")
    if mode == "trailing_median":
        return (float(indicators.get("fitted_trend_qoq", config.GDP_TREND_QOQ)),
                float(indicators.get("fitted_convergence", config.GDP_CONVERGENCE)), mode)
    pot = potential_qoq(float(indicators.get("potential_ann_pct",
                                             config.GDP_POTENTIAL_ANN_PCT)))
    if mode == "potential":
        return pot, 0.0, mode
    if mode == "nowcast":
        return float(q1), 0.0, mode
    return pot, float(indicators.get("glide_persistence",
                                     config.GDP_GLIDE_PERSISTENCE)), mode


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

    The path is nowcast(q+1), then the pace config.GDP_PACE_MODE selects
    (indicators['pace_mode'] overrides): potential, the nowcast carried, a
    glide from the nowcast to potential, or the trailing median. Pass
    indicators['fitted_trend_qoq'] / ['fitted_convergence'] for the
    trailing-median mode's point-in-time estimates, and
    indicators['rate_drag_qoq'] ({quarter: decimal}) to add the rate
    channel's drag (usmodel.rates) to every quarter past the nowcast
    quarter -- the nowcast itself already sees the rate environment through
    its indicators, so the drag is not applied there."""
    q1 = nowcast_qoq(gdp_level, indicators)
    trend, conv, mode = pace_past_nowcast(q1, indicators)
    drag = indicators.get("rate_drag_qoq")      # {Period[Q]: decimal}, optional

    last = gdp_level.index[-1]
    horizon = pd.period_range(last + 1, periods=horizon_quarters, freq="Q")
    # In "nowcast" mode the carried pace already reflects the rate
    # environment of the nowcast quarter, so only the CHANGE in drag from
    # that quarter is added; the other anchors are rate-neutral.
    drag0 = float(drag.get(last + 1, 0.0)) if (drag is not None and mode == "nowcast") else 0.0
    path, base = [q1], q1
    for tq in horizon[1:]:
        base = trend + (base - trend) * conv
        extra = (float(drag.get(tq, 0.0)) - drag0) if drag is not None else 0.0
        path.append(base + extra)
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
