"""
Calibrated quad probabilities for the year ahead.

The projection model is deterministic - one quad per quarter. Its own
walk-forward track record supplies the uncertainty: for each horizon h,
P(realized quad = q | model called p at horizon h) is estimated from the
backtest's confusion matrix (Laplace-smoothed). Today's point calls are then
read through that calibration, giving a probability for each quad in each
of the next four quarters that means something concrete: "when this model
made this call at this distance, here is what actually happened."
"""
from __future__ import annotations

import pandas as pd

from quadmap.quads import QUAD_LABELS

from . import btconfig
from .btconfig import VintageConfig
from .data_bundle import RawDataBundle
from .engine import model_quad_table
from .scoring import _attach_realized
from .vintage import build_vintage, first_release_date

QUADS = [1, 2, 3, 4]


def horizon_calibration(preds: pd.DataFrame, realized: pd.DataFrame,
                        horizons=(1, 2, 3, 4),
                        alpha: float = 1.0) -> dict[int, pd.DataFrame]:
    """Per-horizon calibration matrices: rows = predicted quad, columns =
    realized quad, values = P(realized | predicted, horizon). alpha is the
    Laplace prior that keeps rare calls away from hard 0%/100%."""
    joined = _attach_realized(preds, realized)
    out = {}
    for h in horizons:
        sub = joined[joined["horizon"] == h]
        m = (pd.crosstab(sub["pred_quad"].astype(int),
                         sub["real_quad"].astype(int))
             .reindex(index=QUADS, columns=QUADS, fill_value=0))
        sm = m + alpha
        out[h] = sm.div(sm.sum(axis=1), axis=0)
    return out


def delta_residuals(preds: pd.DataFrame, realized: pd.DataFrame,
                    horizons=(1, 2, 3, 4)) -> dict[int, pd.DataFrame]:
    """Per-horizon forecast errors on the two axes that define the quad:
    (realized - predicted) for d_growth and d_inflation, one row per
    backtest prediction. This is the model's empirical error cloud."""
    joined = _attach_realized(preds, realized)
    out = {}
    for h in horizons:
        sub = joined[joined["horizon"] == h]
        out[h] = pd.DataFrame({
            "eg": (sub["real_d_growth"] - sub["pred_d_growth"]).astype(float),
            "ei": (sub["real_d_inflation"]
                   - sub["pred_d_inflation"]).astype(float),
        }).dropna()
    return out


def residual_quad_probability(dg: float, di: float,
                              residuals: pd.DataFrame) -> pd.Series:
    """P(quad) given predicted deltas (dg, di): place the model's historical
    error cloud around today's point forecast and count the share landing in
    each quadrant. Joint resampling preserves the growth/inflation error
    correlation, and the result moves *continuously* with the inputs - a
    higher oil price that nudges d_inflation up nudges P(quad 2/3) up the
    same day, no regime flip needed."""
    g_up = (dg + residuals["eg"]) > 0
    i_up = (di + residuals["ei"]) > 0
    n = len(residuals)
    return pd.Series({
        1: float((g_up & ~i_up).sum()) / n,
        2: float((g_up & i_up).sum()) / n,
        3: float((~g_up & i_up).sum()) / n,
        4: float((~g_up & ~i_up).sum()) / n,
    })


def latest_full_asof(bundle: RawDataBundle,
                     cfg: VintageConfig | None = None) -> pd.Timestamp:
    """First date on which every observation in the bundle is published -
    the natural 'today' for a forecast built from this dataset."""
    cfg = cfg or VintageConfig()
    return max(
        first_release_date(bundle.gdp_final.index[-1], cfg.gdp_pub_lag_days),
        first_release_date(bundle.cpi.index[-1], cfg.cpi_pub_lag_days),
        first_release_date(bundle.market.index[-1], cfg.market_pub_lag_days),
    ) + pd.Timedelta(days=1)


def current_quad_probabilities(bundle: RawDataBundle, preds: pd.DataFrame,
                               realized: pd.DataFrame,
                               cfg: VintageConfig | None = None,
                               asof: pd.Timestamp | str | None = None,
                               horizons=(1, 2, 3, 4), alpha: float = 1.0,
                               method: str = "residual",
                               assumption_overrides: dict | None = None,
                               indicator_overrides: dict | None = None):
    """Probability of each quad in each of the next `horizons` quarters.

    Returns (prob, calls, asof): prob is a DataFrame indexed by quad 1..4
    with one column per target quarter (columns sum to 1), calls maps each
    target quarter to the model's point call.

    method='residual' (default): the model's per-horizon backtest error
    cloud on (d_growth, d_inflation) is placed around today's predicted
    deltas and P(quad) is the share of that cloud in each quadrant. This
    responds continuously to the inputs: fresher oil/NOK/power spots shift
    the deltas and the probabilities move the same day.
    method='calibration': the per-horizon confusion row for today's point
    call - P(realized | called, horizon) - which only moves when the call
    itself flips.

    assumption_overrides lets a live runner replace spot-carry market
    anchors (power/brent/NOK spots, forward dicts) with today's prices;
    indicator_overrides does the same for the GDP nowcast indicators.

    The forecast vintage runs with revision_mode='none': unlike the
    backtest's reconstruction of the past, today's data already *is* the
    first release, so no revision simulation belongs on top of it.
    """
    if method not in ("residual", "calibration"):
        raise ValueError(f"unknown method {method!r}")
    cfg = cfg or VintageConfig()
    fcast_cfg = VintageConfig(
        gdp_pub_lag_days=cfg.gdp_pub_lag_days,
        cpi_pub_lag_days=cfg.cpi_pub_lag_days,
        market_pub_lag_days=cfg.market_pub_lag_days,
        revision_mode="none", seed=cfg.seed)

    asof = pd.Timestamp(asof) if asof is not None else \
        latest_full_asof(bundle, cfg)
    vintage = build_vintage(bundle, asof, fcast_cfg)
    if assumption_overrides:
        vintage.assumptions.update(assumption_overrides)
    if indicator_overrides:
        vintage.indicators.update(indicator_overrides)
    table = model_quad_table(vintage, max(horizons))

    if method == "residual":
        resid = delta_residuals(preds, realized, horizons)
    else:
        cal = horizon_calibration(preds, realized, horizons, alpha=alpha)

    asof_q = pd.Period(asof, freq="Q")
    cols, calls = {}, {}
    for h in horizons:
        tq = asof_q + h
        if tq not in table.index:
            continue
        row = table.loc[tq]
        calls[tq] = int(row["quad"])
        if method == "residual" and len(resid[h]):
            cols[tq] = residual_quad_probability(
                float(row["d_growth"]), float(row["d_inflation"]), resid[h])
        else:
            cols[tq] = cal[h].loc[calls[tq]] if method == "calibration" \
                else pd.Series(0.25, index=QUADS)
    prob = pd.DataFrame(cols)
    prob.index.name = "quad"
    return prob, calls, asof


def monthly_probabilities(prob: pd.DataFrame) -> pd.DataFrame:
    """Expand the quarterly distributions to the months of each target
    quarter. The quad is a quarterly object, so the three months of a
    quarter share its distribution - this is a display convenience, not
    extra resolution."""
    cols = {}
    for q in prob.columns:
        for m in pd.period_range(q.asfreq("M", "start"),
                                 q.asfreq("M", "end"), freq="M"):
            cols[m] = prob[q]
    out = pd.DataFrame(cols)
    out.index.name = "quad"
    return out


def format_probability_table(prob: pd.DataFrame,
                             calls: dict[pd.Period, int]) -> pd.DataFrame:
    """Human-readable version: labelled rows, % cells, point call marked."""
    show = pd.DataFrame(index=[f"Q{q} {QUAD_LABELS[q]}" for q in QUADS])
    for tq in prob.columns:
        vals = []
        for q in QUADS:
            mark = " <- call" if calls.get(tq) == q else ""
            vals.append(f"{prob.loc[q, tq] * 100:.0f}%{mark}")
        show[str(tq)] = vals
    return show
