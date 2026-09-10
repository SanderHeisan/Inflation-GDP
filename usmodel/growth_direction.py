"""
Growth DIRECTION calls on a QoQ basis -- the sequential question ("will this
quarter's GDP growth print above or below last quarter's?"), as opposed to
the quad's YoY rate-of-change question.

Why this is a separate model from the level path. The backtest measured
that the activity-indicator nowcast, which cuts the QoQ *level* error
substantially, is a coin flip (50%) on the SIGN of this quarter's deviation
from trend. It knows roughly where the quarter lands, not which side of
trend it lands on. Yet the sequential direction is one of the more
predictable things about US GDP: quarterly growth mean-reverts hard
(autocorrelation of dQoQ ~ -0.4), so a quarter that printed above trend is
followed by a lower print ~72% of the time, from published data alone. The
nowcast adds nothing to that call and, blended in, dilutes it.

So the QoQ direction call is a point-in-time AR(1) on published QoQ growth:

    E[qoq(q)] = a + b * qoq(q-1),   b < 1

and the call for the current (nowcast) quarter is sign(E[qoq(q)] - qoq(q-1))
= -sign(qoq(q-1) - a/(1-b)): reversal toward the AR mean. The fit is
winsorized at 3 MADs because 2020Q2/Q3 would otherwise own it.

Backtested 2017-2026 for the current quarter: 74% direction hit (81% on
ex-COVID targets), against 68% for the nowcast-based call. Uniform across
conviction buckets -- this call does not need grading.

One quarter further out the same structure gives a weak "zigzag" lean
(dQoQ alternates in sign, so the call for q+1 is the OPPOSITE of the call
for q): 57% measured, which is a lean, not a call, and is labelled so. Past
that, nothing tested beats a coin flip and the model abstains rather than
manufacture a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AR_WINDOW_Q = 80          # quarters of published history the AR(1) is fitted on
AR_WINSOR_MAD = 3.0       # clip training sample at this many MADs from the median
MIN_AR_QUARTERS = 24

# Conviction the growth *YoY* call needs before it is worth acting on. The
# backtest found it two-regime rather than smoothly graded: above this
# threshold ~80% right, below it ~55-60% regardless of size. CPI's direction
# call grades smoothly and uses 0.30pp; growth is a coarser instrument.
YOY_HIGH_CONVICTION_PP = 0.50


def fit_ar1_qoq(gdp_level: pd.Series, window: int = AR_WINDOW_Q,
                winsor_mad: float = AR_WINSOR_MAD) -> tuple[float, float] | None:
    """(intercept, slope) of a robust AR(1) on published QoQ growth (%).
    Point-in-time: pass a vintage-truncated level series. Returns None when
    there is not enough history."""
    qoq = (gdp_level.pct_change() * 100).dropna().tail(window)
    if len(qoq) < MIN_AR_QUARTERS:
        return None
    med = float(qoq.median())
    mad = float((qoq - med).abs().median()) * 1.4826
    if mad > 0:
        qoq = qoq.clip(med - winsor_mad * mad, med + winsor_mad * mad)
    x, y = qoq.to_numpy(float)[:-1], qoq.to_numpy(float)[1:]
    if float(x.var()) < 1e-12:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(np.clip(slope, -0.5, 0.95))


def qoq_direction_calls(gdp_level: pd.Series,
                        trend_qoq_pct: float | None = None) -> pd.DataFrame:
    """Sequential-growth direction calls for the first two unpublished
    quarters, from published data only.

    Returns a frame indexed by target quarter with columns
      pred_qoq_pct, last_qoq_pct, delta_pp, direction, conviction_pp, grade
    where grade is 'call' (the reversal call for the current quarter) or
    'lean' (the zigzag for the quarter after). Empty if history is too short.
    """
    fit = fit_ar1_qoq(gdp_level)
    if fit is None:
        return pd.DataFrame()
    a, b = fit
    last_q = gdp_level.index[-1]
    qoq = (gdp_level.pct_change() * 100).dropna()
    q_last = float(qoq.iloc[-1])
    ar_mean = a / (1 - b) if abs(1 - b) > 1e-9 else q_last
    trend = float(trend_qoq_pct) if trend_qoq_pct is not None else ar_mean

    rows = []
    # h=0 / current quarter: reversal toward the AR mean.
    pred1 = a + b * q_last
    d1 = pred1 - q_last
    rows.append({"target": last_q + 1, "pred_qoq_pct": pred1,
                 "last_qoq_pct": q_last, "delta_pp": d1,
                 "direction": "accelerating" if d1 > 0 else "decelerating",
                 "conviction_pp": abs(d1), "grade": "call",
                 "backtest_hit": 0.74})
    # h=1 / next quarter: the zigzag -- dQoQ alternates, so the lean for
    # q+1 is the OPPOSITE of the call for q, i.e. +sign(q_last - trend).
    d2 = q_last - trend
    rows.append({"target": last_q + 2, "pred_qoq_pct": np.nan,
                 "last_qoq_pct": pred1, "delta_pp": d2,
                 "direction": "accelerating" if d2 > 0 else "decelerating",
                 "conviction_pp": abs(d2), "grade": "lean",
                 "backtest_hit": 0.57})
    return pd.DataFrame(rows).set_index("target")
