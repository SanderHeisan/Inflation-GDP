"""
One-print-ahead inflation direction backtest.

The sharpest question a Nordic GIP service must answer every month: will
the next CPI print show YoY inflation ACCELERATING or DECELERATING?

The mechanics favor the forecaster: with the CPI index through month m0
published, next month's YoY change is

    yoy(m1) - yoy(m0)  ~  mom(m1) - mom(m1 - 12)

and mom(m1 - 12) - the hurdle - is already known. Only one monthly change
needs forecasting, and when the known hurdle is extreme (an energy-spike
month dropping out of the YoY window) the call is nearly certain before
any modelling happens. Conviction is therefore |predicted delta|: the
further the forecast sits from the hurdle, the harder it is to be wrong.

This module walks monthly as-of dates, makes that call using only the
vintage information set, scores it against the (unrevised) CPI series, and
summarizes hit rates by conviction bucket - the numbers behind "we call
the direction of every print, and on flagged high-conviction months we are
almost never wrong."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quadmap import inflation

from .btconfig import VintageConfig
from .data_bundle import RawDataBundle
from .engine import asof_dates
from .vintage import build_vintage

CONVICTION_BUCKETS = [(0.00, 0.05, "coin-flip (<0.05pp)"),
                      (0.05, 0.15, "lean (0.05-0.15pp)"),
                      (0.15, 0.30, "call (0.15-0.30pp)"),
                      (0.30, np.inf, "high conviction (>0.30pp)")]


def monthly_direction_backtest(bundle: RawDataBundle, start: str, end: str,
                               cfg: VintageConfig | None = None
                               ) -> pd.DataFrame:
    """One row per as-of month: the model's next-print YoY direction call
    and what actually printed. Only vintage data enters the prediction;
    scoring uses the final series (Norwegian CPI is unrevised, so first
    release and final coincide)."""
    cfg = cfg or VintageConfig()
    ryoy = (bundle.cpi.pct_change(12) * 100).dropna()
    rmom = (bundle.cpi.pct_change() * 100).dropna()

    rows = []
    for asof in asof_dates(start, end, "M"):
        try:
            v = build_vintage(bundle, asof, cfg)
        except ValueError:
            continue
        m0, m1 = v.last_cpi_month, v.last_cpi_month + 1
        proj = inflation.build_cpi_projection(
            v.cpi_index, 3, v.assumptions, v.i44)
        yoy = proj["yoy_pct"]
        if m1 not in yoy.index or m1 not in ryoy.index:
            continue
        pred_delta = float(yoy[m1] - yoy[m0])
        real_delta = float(ryoy[m1] - ryoy[m0])
        pred_mom = float(proj["mom_pct"].get(m1, np.nan))
        real_mom = float(rmom.get(m1, np.nan))
        rows.append({
            "asof": asof,
            "last_print": str(m0),
            "target_print": str(m1),
            "pred_yoy": float(yoy[m1]),
            "real_yoy": float(ryoy[m1]),
            "yoy_error": float(yoy[m1] - ryoy[m1]),
            "pred_delta": pred_delta,
            "real_delta": real_delta,
            "pred_mom": pred_mom,
            "real_mom": real_mom,
            "mom_error": pred_mom - real_mom,
            "hurdle_mom": float(rmom.get(m1 - 12, np.nan)),
            "pred_dir": "accelerating" if pred_delta > 0 else "decelerating",
            "real_dir": "accelerating" if real_delta > 0 else "decelerating",
            "hit": (pred_delta > 0) == (real_delta > 0),
            "conviction_pp": abs(pred_delta),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no monthly direction calls in the window")
    return df


def bucket_label(conviction_pp: float) -> str:
    for lo, hi, label in CONVICTION_BUCKETS:
        if lo <= conviction_pp < hi:
            return label
    return CONVICTION_BUCKETS[-1][2]


def summarize_monthly_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Hit rates overall and per conviction bucket. The product question:
    how often is the direction right when the model actually leans?"""
    rows = [{"bucket": "ALL months", "n": len(df),
             "share_of_months": 1.0,
             "hit_rate": float(df["hit"].mean()),
             "avg_conviction_pp": float(df["conviction_pp"].mean())}]
    for lo, hi, label in CONVICTION_BUCKETS:
        sub = df[(df["conviction_pp"] >= lo) & (df["conviction_pp"] < hi)]
        if not len(sub):
            continue
        rows.append({"bucket": label, "n": len(sub),
                     "share_of_months": len(sub) / len(df),
                     "hit_rate": float(sub["hit"].mean()),
                     "avg_conviction_pp": float(sub["conviction_pp"].mean())})
    callable_ = df[df["conviction_pp"] >= 0.05]
    if len(callable_):
        rows.append({"bucket": "callable (>=0.05pp)", "n": len(callable_),
                     "share_of_months": len(callable_) / len(df),
                     "hit_rate": float(callable_["hit"].mean()),
                     "avg_conviction_pp":
                         float(callable_["conviction_pp"].mean())})
    return pd.DataFrame(rows).set_index("bucket")


def prediction_error_stats(df: pd.DataFrame) -> pd.Series:
    """How close the level forecasts get, one print ahead. MAE in
    percentage points; bias positive = model runs hot."""
    return pd.Series({
        "n_months": len(df),
        "yoy_mae_pp": float(df["yoy_error"].abs().mean()),
        "yoy_bias_pp": float(df["yoy_error"].mean()),
        "yoy_within_0.1pp": float((df["yoy_error"].abs() <= 0.10).mean()),
        "yoy_within_0.2pp": float((df["yoy_error"].abs() <= 0.20).mean()),
        "mom_mae_pp": float(df["mom_error"].abs().mean()),
        "mom_bias_pp": float(df["mom_error"].mean()),
        "direction_hit_rate": float(df["hit"].mean()),
        "direction_hit_ex_coinflip": float(
            df.loc[df["conviction_pp"] >= 0.05, "hit"].mean())
        if (df["conviction_pp"] >= 0.05).any() else float("nan"),
    })


def direction_hit_rate_for(conviction_pp: float,
                           history: pd.DataFrame) -> tuple[float, int]:
    """Empirical accuracy of past calls in the same conviction bucket -
    the number to quote next to a live directional call."""
    label = bucket_label(conviction_pp)
    for lo, hi, lbl in CONVICTION_BUCKETS:
        if lbl == label:
            sub = history[(history["conviction_pp"] >= lo)
                          & (history["conviction_pp"] < hi)]
            if len(sub):
                return float(sub["hit"].mean()), len(sub)
    return float(history["hit"].mean()), len(history)
