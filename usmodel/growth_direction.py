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

A second vote comes from the consumer. Real personal consumption is ~68%
of GDP and its quarterly growth mean-reverts the same way, and it is
published monthly -- so the last full quarter of real PCE growth, against
its own trailing median, is an independent reversal signal that is known
before GDP is. Backtested 2017-2026 on final data the PCE vote alone is
right 68% of the time (75% ex-COVID) against 63% (66%) for the GDP vote.
The two AGREE in roughly two months out of three, and then the call is
right 73% of the time on final GDP and 79% under simulated revisions (81%
and 91% ex-COVID) -- robust to the revision assumption, which is what makes
it a call. When they SPLIT the blended direction scored 67% on final data
and 30% under revisions on a dozen distinct quarters: unreliable either
way, so it is graded a coin flip. That agreement grade is the whole value
of the second vote: the single-vote call has no useful conviction scale,
the two-vote one does.

Two honesty notes. First, the single GDP-vote hit rate depends on the
revision assumption: 63% on final data, 63-76% across simulated-revision
seeds (an earlier README quoted the top of that range). The PCE vote does
not move with the seed. Second, the "stretched consumer" idea -- real PCE
growth in the top decile of its trailing decade -- does NOT improve the
sequential call as an override; it belongs on the YoY axis, where a
stretched consumer plus a model call of "decelerating" has been right
79-84% of the time against 64-66% otherwise (43-51 rows, ~15 distinct
quarters), and `consumer_state` reports it as a supporting flag rather
than a call.

One quarter further out the same structure gives a "zigzag" lean (dQoQ
alternates in sign, so the lean for q+1 is the OPPOSITE of the call for
q): 51% on final data, 57% under simulated revisions -- close to nothing,
labelled weak. Past that, nothing tested beats a coin flip and the model
abstains rather than manufacture a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AR_WINDOW_Q = 80          # quarters of published history the AR(1) is fitted on
AR_WINSOR_MAD = 3.0       # clip training sample at this many MADs from the median
MIN_AR_QUARTERS = 24

PCE_MEDIAN_WINDOW_Q = 24   # trailing window for the consumer's own trend
STRETCH_WINDOW_Q = 40      # trailing decade for the percentile rank
STRETCH_PCTL = 0.90        # top decile = "stretched", bottom decile = "depressed"

# Backtested hit rates (2017-2026, final-vintage GDP) for the live block.
HIT_AGREE, HIT_SPLIT, HIT_LEAN, HIT_SINGLE = 0.73, 0.50, 0.54, 0.63

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


def pce_reversal_vote(real_pce: pd.Series | None, last_gdp_quarter: pd.Period,
                      window: int = PCE_MEDIAN_WINDOW_Q) -> float | None:
    """The consumer's vote on next quarter's sequential growth: minus the
    gap between the last FULL quarter of real PCE growth (quarterly mean of
    the monthly level) and its trailing median. Positive = PCE ran below
    its norm, so growth is more likely to pick up. None if unavailable."""
    if real_pce is None or real_pce.empty:
        return None
    q = real_pce.groupby(real_pce.index.asfreq("Q")).mean()
    growth = (q.pct_change() * 100).dropna()
    growth = growth[growth.index <= last_gdp_quarter]
    if len(growth) < 12:
        return None
    return float(growth.tail(window).median() - growth.iloc[-1])


def qoq_direction_calls(gdp_level: pd.Series,
                        trend_qoq_pct: float | None = None,
                        real_pce: pd.Series | None = None) -> pd.DataFrame:
    """Sequential-growth direction calls for the first two unpublished
    quarters, from published data only.

    Two votes for the current quarter: the GDP reversal and, when real PCE
    is supplied, the consumer reversal. Returns a frame indexed by target
    quarter with columns pred_qoq_pct, last_qoq_pct, delta_pp, direction,
    conviction_pp, grade, gdp_vote_pp, pce_vote_pp, backtest_hit, where
    grade is 'call' (votes agree), 'split' (votes disagree: a coin flip,
    graded as one), 'single' (no PCE available) or 'lean' (the zigzag for
    the quarter after). Empty if history is too short.
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
    # h=0 / current quarter: reversal toward the AR mean, plus the consumer.
    pred1 = a + b * q_last
    gdp_vote = pred1 - q_last
    pce_vote = pce_reversal_vote(real_pce, last_q)
    if pce_vote is None:
        d1, grade, hit = gdp_vote, "single", HIT_SINGLE
    elif np.sign(pce_vote) == np.sign(gdp_vote):
        d1, grade, hit = (gdp_vote + pce_vote) / 2, "call", HIT_AGREE
    else:
        d1, grade, hit = (gdp_vote + pce_vote) / 2, "split", HIT_SPLIT
    rows.append({"target": last_q + 1, "pred_qoq_pct": pred1,
                 "last_qoq_pct": q_last, "delta_pp": d1,
                 "direction": "accelerating" if d1 > 0 else "decelerating",
                 "conviction_pp": abs(d1), "grade": grade,
                 "gdp_vote_pp": gdp_vote,
                 "pce_vote_pp": pce_vote if pce_vote is not None else np.nan,
                 "backtest_hit": hit})
    # h=1 / next quarter: the zigzag -- dQoQ alternates, so the lean for
    # q+1 is the OPPOSITE of the call for q, i.e. +sign(q_last - trend).
    d2 = q_last - trend
    rows.append({"target": last_q + 2, "pred_qoq_pct": np.nan,
                 "last_qoq_pct": pred1, "delta_pp": d2,
                 "direction": "accelerating" if d2 > 0 else "decelerating",
                 "conviction_pp": abs(d2), "grade": "lean",
                 "gdp_vote_pp": d2, "pce_vote_pp": np.nan,
                 "backtest_hit": HIT_LEAN})
    return pd.DataFrame(rows).set_index("target")


def _trailing_pctl(series: pd.Series, window: int = STRETCH_WINDOW_Q) -> float:
    """Where the latest value sits in its own trailing window, 0..1."""
    s = series.dropna()
    if len(s) < 20:
        return float("nan")
    w = s.tail(window)
    return float((w.iloc[:-1] < w.iloc[-1]).mean())


def consumer_state(indicators: dict[str, pd.Series],
                   last_gdp_quarter: pd.Period) -> dict:
    """The consumer block as a reader wants to see it: where spending,
    confidence, income and saving stand against their own trailing decade,
    and whether spending is stretched. Everything from published data.

    What the realized record (1995-2026, ex-COVID) says about extremes, so
    the flags carry their base rates rather than a story:
      real PCE growth in its top decile  -> next-quarter GDP QoQ lower 70%,
                                            YoY growth decelerating 70-75%
      sentiment in its top decile        -> no signal (47-60%)
      sentiment in its bottom decile     -> next-quarter GDP QoQ HIGHER 63%
      real income growth in top decile   -> next-quarter GDP QoQ HIGHER 67%
    """
    out: dict = {}
    pce = indicators.get("real_pce")
    if pce is not None and len(pce) > 16:
        q = pce.groupby(pce.index.asfreq("Q")).mean()
        yoy = (q.pct_change(4) * 100).dropna()
        yoy = yoy[yoy.index <= last_gdp_quarter]
        qoq = (q.pct_change() * 100).dropna()
        qoq = qoq[qoq.index <= last_gdp_quarter]
        mpce_yoy = (pce / pce.shift(12) - 1) * 100
        pctl = _trailing_pctl(yoy)
        out["real_pce"] = {
            "latest_month": str(pce.index[-1]),
            "yoy_pct": float(mpce_yoy.dropna().iloc[-1]),
            "last_quarter": str(qoq.index[-1]),
            "last_quarter_qoq_pct": float(qoq.iloc[-1]),
            "last_quarter_saar_pct": float(((1 + qoq.iloc[-1] / 100) ** 4 - 1) * 100),
            "trailing_median_qoq_pct": float(qoq.tail(PCE_MEDIAN_WINDOW_Q).median()),
            "pctl_10y": pctl,
            "stretched": bool(pctl >= STRETCH_PCTL),
            "depressed": bool(pctl <= 1 - STRETCH_PCTL),
        }
    sent = indicators.get("sentiment")
    if sent is not None and len(sent) > 12:
        pctl = _trailing_pctl(sent.groupby(sent.index.asfreq("Q")).mean())
        out["sentiment"] = {"latest_month": str(sent.index[-1]),
                            "level": float(sent.iloc[-1]),
                            "pctl_10y": pctl,
                            "elevated": bool(pctl >= STRETCH_PCTL),
                            "depressed": bool(pctl <= 1 - STRETCH_PCTL)}
    inc = indicators.get("real_income")
    if inc is not None and len(inc) > 16:
        yoy = ((inc / inc.shift(12) - 1) * 100).dropna()
        out["real_income"] = {"latest_month": str(inc.index[-1]),
                              "yoy_pct": float(yoy.iloc[-1]),
                              "pctl_10y": _trailing_pctl(
                                  yoy.groupby(yoy.index.asfreq("Q")).mean())}
    sav = indicators.get("saving_rate")
    if sav is not None and len(sav) > 12:
        out["saving_rate"] = {"latest_month": str(sav.index[-1]),
                              "level_pct": float(sav.iloc[-1]),
                              "pctl_10y": _trailing_pctl(
                                  sav.groupby(sav.index.asfreq("Q")).mean())}
    return out
