"""
Point-in-time GDP nowcast from real-time activity indicators.

Why this exists: the first cut of the US model nowcast blended trailing GDP
momentum with indicator *slots* that had no data behind them, so in practice
it was momentum-only. The backtest said plainly what that costs -- with the
inflation axis calling direction ~70-80% of the time, the growth axis was
near a coin flip, and the quad is the AND of the two. Growth was the binding
constraint, so it is the thing worth fixing.

The fix is a small ridge regression from published monthly activity data to
the current quarter's real GDP QoQ, refitted at every as-of date on only the
history published at that date. Nothing is calibrated on the full sample:
the coefficients a live run would use are the coefficients the backtest uses.

Partial quarters are handled explicitly. `k` is the number of months of the
target quarter that are already published (0, 1 or 2 in practice, because
BEA's advance estimate lands ~28 days after quarter end):

    k >= 1   features compare the first k months of the target quarter with
             all three months of the quarter before it
    k == 0   nothing of the target quarter is out yet, so the same features
             are computed one quarter back and the regression becomes a
             genuine one-quarter-ahead model, carried by the leading series
             (claims, sentiment, the yield curve)

A separate fit is run for each k, so training features and prediction
features always have the same construction and the same scale.

KNOWN LIMITATION: training features use current-vintage indicator values.
Payrolls and industrial production are revised, so the fit sees slightly
cleaner history than a forecaster had. The *prediction* features and the
regression target are strictly point-in-time, and `backtest/snapshots.py`
is accumulating a real indicator archive to close this gap; until it spans
enough quarters, treat the growth numbers as a modest upper bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# indicator -> how it enters the design matrix
#   pct     percent change of the period mean vs the prior quarter
#   logpct  same in logs (for claims, which move multiplicatively)
#   diff    difference of the period mean vs the prior quarter
#   lvl     the period mean itself (already a rate or a diffusion-style index)
FEATURE_SPEC = (
    ("payrolls",    "pct"),
    ("indpro",      "pct"),
    ("retail",      "pct"),
    ("claims",      "logpct"),
    ("cfnai",       "lvl"),
    ("sentiment",   "diff"),
    ("yield_curve", "lvl"),
    ("hours",       "diff"),
)

# Genuinely LEADING series, kept as an OPTIONAL extended panel. Adding them
# was the obvious first attempt at forecasting growth two to four quarters
# out. It does not work: at horizons past the nowcast quarter these carry no
# more signal than the coincident block, which is to say none (README). They
# stay here so `us_backtest.py --growth-variants` can reproduce that result
# rather than leaving it as a claim.
LEADING_SPEC = (
    ("nfci",           "lvl"),
    ("credit_spread",  "lvl"),
    ("permits",        "pct"),
    ("capex_orders",   "pct"),
    ("stocks",         "pct"),
    ("real_m2",        "pct"),
    ("housing_starts", "pct"),
)
EXTENDED_SPEC = FEATURE_SPEC + LEADING_SPEC

MIN_TRAIN_QUARTERS = 40      # ~10 years before the regression is trusted
RIDGE_LAMBDA = 1.0           # on standardized features


def months_available(indicators: dict[str, pd.Series], quarter: pd.Period,
                     spec: tuple = FEATURE_SPEC) -> int:
    """How many months of `quarter` are published across every indicator the
    fit actually uses (the binding one sets k, so the design matrix is never
    ragged).

    Restricted to `spec` on purpose: a slow series that is cached but unused
    -- real M2 lands ~30 days after month end -- would otherwise drag k down
    and cost the nowcast a month of data it really had.
    """
    used = [indicators[n] for n, _ in spec if n in indicators]
    if not used:
        return 0
    months = pd.period_range(quarter.start_time, periods=3, freq="M")
    return min(sum(1 for m in months if m in s.index) for s in used)


def _quarter_means(series: pd.Series, k: int) -> tuple[pd.Series, pd.Series]:
    """(mean of the first k months of each quarter, mean of all three).
    Vectorized on purpose: the walk-forward refits this at every as-of date,
    and a per-quarter reindex loop dominated the backtest's runtime."""
    idx = series.index
    full = series.groupby(idx.asfreq("Q")).mean()
    if k >= 1:
        sel = (((idx.month - 1) % 3) + 1) <= k
        partial = series[sel].groupby(idx[sel].asfreq("Q")).mean()
    else:
        partial = full
    return partial, full


def feature_frame(indicators: dict[str, pd.Series], k: int,
                  gdp_qoq: pd.Series, spec: tuple = FEATURE_SPEC,
                  horizon: int = 1) -> pd.DataFrame:
    """Design matrix for every quarter at once, indexed by target quarter.

    `horizon` is how many quarters past the last published one the target
    sits (1 = the nowcast quarter). Features always come from the most recent
    data available at forecast time, which is `horizon - 1` quarters before
    the target when k >= 1, and one further back when k == 0. Training rows
    use the same construction, so a horizon-h fit is a genuine direct-h
    forecast rather than an iterated one.
    """
    shift = (horizon - 1) if k >= 1 else horizon
    # One common quarterly index, extended past the last published quarter so
    # the target quarter always has a row even when k == 0 (nothing of it is
    # published, and its features are read one quarter back).
    last_q = max(s.index[-1].asfreq("Q") for s in indicators.values()
                 if s is not None and not s.empty)
    first_q = min(s.index[0].asfreq("Q") for s in indicators.values()
                  if s is not None and not s.empty)
    qidx = pd.period_range(first_q, last_q + horizon + 2, freq="Q")

    cols: dict[str, pd.Series] = {}
    for name, kind in spec:
        s = indicators.get(name)
        if s is None or s.empty:
            return pd.DataFrame()
        partial, full = _quarter_means(s, k)
        partial, full = partial.reindex(qidx), full.reindex(qidx)
        cur = partial.shift(shift)
        prev = full.shift(1 + shift)
        if kind == "pct":
            cols[name] = (cur / prev - 1) * 100
        elif kind == "logpct":
            safe = (cur > 0) & (prev > 0)
            cols[name] = (np.log(cur.where(safe))
                          - np.log(prev.where(safe))) * 100
        elif kind == "diff":
            cols[name] = cur - prev
        else:
            cols[name] = cur
    # Previous quarter's published GDP QoQ, so the regression subsumes the
    # momentum benchmark instead of being blended with it after the fact.
    # Re-labelling the index (rather than shifting values) keeps the row for
    # the target quarter, whose predecessor IS published.
    momentum = gdp_qoq.copy()
    momentum.index = momentum.index + shift + 1
    cols["momentum"] = momentum
    return pd.DataFrame(cols).dropna()


def quarter_features(indicators: dict[str, pd.Series], quarter: pd.Period,
                     k: int, gdp_qoq: pd.Series, spec: tuple = FEATURE_SPEC,
                     horizon: int = 1) -> dict[str, float] | None:
    """One feature row, or None when any input is missing."""
    frame = feature_frame(indicators, k, gdp_qoq, spec, horizon)
    if frame.empty or quarter not in frame.index:
        return None
    return frame.loc[quarter].to_dict()


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA):
    """Standardize, ridge-solve, return a predictor for one raw feature row.
    Standardization keeps a single lambda meaningful across features whose
    units range from index points to percentage changes; the intercept is
    left unpenalized."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    P = np.eye(A.shape[1]) * lam
    P[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + P, A.T @ y)

    def predict(row: np.ndarray) -> float:
        return float(coef[0] + ((row - mu) / sd) @ coef[1:])
    return predict, coef


def fit_nowcast(gdp_level: pd.Series, indicators: dict[str, pd.Series],
                target_quarter: pd.Period | None = None,
                min_train: int = MIN_TRAIN_QUARTERS,
                lam: float = RIDGE_LAMBDA,
                spec: tuple = FEATURE_SPEC) -> dict | None:
    """Nowcast `target_quarter`'s real GDP QoQ from published data alone.

    gdp_level  : published real GDP levels (already vintage-truncated)
    indicators : published monthly activity series (already truncated, each
                 with its own publication lag)
    Returns {'qoq_pct', 'k', 'n_train', 'in_sample_mae', 'coef'} or None when
    there is not enough published history to fit.
    """
    if not indicators or len(gdp_level) < min_train + 2:
        return None
    target_quarter = target_quarter or (gdp_level.index[-1] + 1)
    gdp_qoq = (gdp_level.pct_change() * 100).dropna()
    horizon = (target_quarter - gdp_level.index[-1]).n
    if horizon < 1:
        return None
    k = months_available(indicators, gdp_level.index[-1] + 1, spec)

    frame = feature_frame(indicators, k, gdp_qoq, spec, horizon)
    if frame.empty or target_quarter not in frame.index:
        return None
    train = frame.loc[frame.index.intersection(gdp_qoq.index)]
    if len(train) < min_train:
        return None
    rows = train.to_numpy(float)
    targets = gdp_qoq.reindex(train.index).to_numpy(float)
    x = frame.loc[target_quarter]

    X, y = np.asarray(rows, float), np.asarray(targets, float)
    predict, coef = _fit_ridge(X, y, lam)
    fitted = np.array([predict(r) for r in X])
    return {
        "qoq_pct": predict(x.to_numpy(float)),
        "k": k,
        "n_train": len(train),
        "in_sample_mae": float(np.abs(fitted - y).mean()),
        "coef": dict(zip(["intercept"] + list(frame.columns), coef.tolist())),
    }
