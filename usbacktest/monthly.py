"""
Monthly CPI backtest: the print-level numbers.

Two questions, scored separately:

1. **How far off is the CPI forecast?** For every as-of month, project the
   CPI index forward and record the YoY and MoM error at 1, 3, 6 and 12
   months ahead. Reported as mean absolute error in percentage points
   against two naive benchmarks that see the same vintage:
     random_walk  - YoY stays where it last printed
     seasonal_naive - each future MoM equals the trailing 12-month mean MoM
   Anything the model does that these two do not is the component model's
   contribution.

2. **Which way does the next print move?** The sharpest recurring call: is
   next month's YoY rate accelerating or decelerating? The arithmetic
   favors the forecaster, because

       yoy(m1) - yoy(m0)  ~  mom(m1) - mom(m1 - 12)

   and the hurdle mom(m1-12) is already published. Only one monthly change
   needs forecasting, so months with an extreme known hurdle are near-certain
   before any modelling happens. Conviction is |predicted delta|; hit rates
   are bucketed by it, which is what tells a reader how much to trust any
   individual live call.

Scoring note: the model projects the seasonally adjusted index (CPIAUCSL),
whose seasonal factors BLS revises each February. The final SA series is
therefore not exactly what printed. Headline YoY is almost unaffected by
that, but to keep the claim honest every YoY statistic is also computed
against CPIAUCNS, which is never revised (`basis='nsa'`).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quadmap import quads
from usmodel import inflation
from usmodel.data_bundle import USDataBundle

from .btconfig import USVintageConfig
from .engine import asof_dates
from .vintage import build_vintage

CONVICTION_BUCKETS = [(0.00, 0.05, "coin-flip (<0.05pp)"),
                      (0.05, 0.15, "lean (0.05-0.15pp)"),
                      (0.15, 0.30, "call (0.15-0.30pp)"),
                      (0.30, np.inf, "high conviction (>0.30pp)")]

ERROR_HORIZONS = (1, 3, 6, 12)


def _yoy(series: pd.Series) -> pd.Series:
    return quads.calendar_pct_change(series, 12).dropna()


def monthly_backtest(bundle: USDataBundle, start: str, end: str,
                     cfg: USVintageConfig | None = None) -> pd.DataFrame:
    """One row per (as-of month, forecast horizon in months). Only vintage
    data enters the prediction."""
    # Only the CPI projection is scored here, so skip fitting the GDP
    # nowcast at every as-of date.
    cfg = replace(cfg or USVintageConfig(), use_indicator_nowcast=False)
    real_yoy = _yoy(bundle.cpi)
    real_mom = quads.calendar_pct_change(bundle.cpi, 1).dropna()
    real_yoy_nsa = _yoy(bundle.cpi_nsa) if bundle.cpi_nsa is not None else None

    rows = []
    for asof in asof_dates(start, end, "M"):
        try:
            v = build_vintage(bundle, asof, cfg)
        except ValueError:
            continue
        m0 = v.last_cpi_month
        proj = inflation.build_cpi_projection(
            v.cpi_index, max(ERROR_HORIZONS), v.assumptions, v.aux)
        pred_yoy, pred_mom = proj["yoy_pct"], proj["mom_pct"]

        # Benchmarks from the same vintage.
        rw_yoy = float(_yoy(v.cpi_index).iloc[-1])
        trail_mom = float(quads.calendar_pct_change(v.cpi_index, 1)
                          .tail(12).mean())

        for h in ERROR_HORIZONS:
            m = m0 + h
            if m not in pred_yoy.index or m not in real_yoy.index:
                continue
            # seasonal-naive YoY at horizon h: compound trailing mean MoM
            # forward off the last observed level, against the known base.
            lvl = float(v.cpi_index.iloc[-1]) * (1 + trail_mom / 100) ** h
            base = bundle.cpi.get(m - 12, np.nan)
            sn_yoy = (lvl / float(base) - 1) * 100 if base == base else np.nan

            rows.append({
                "asof": asof,
                "last_print": str(m0),
                "target_print": str(m),
                "h_months": h,
                "pred_yoy": float(pred_yoy[m]),
                "real_yoy": float(real_yoy[m]),
                "err_yoy": float(pred_yoy[m] - real_yoy[m]),
                "err_yoy_rw": rw_yoy - float(real_yoy[m]),
                "err_yoy_seasonal_naive": sn_yoy - float(real_yoy[m]),
                "pred_mom": float(pred_mom.get(m, np.nan)),
                "real_mom": float(real_mom.get(m, np.nan)),
                "err_mom": float(pred_mom.get(m, np.nan)
                                 - real_mom.get(m, np.nan)),
                "err_yoy_nsa": (float(pred_yoy[m] - real_yoy_nsa[m])
                                if real_yoy_nsa is not None
                                and m in real_yoy_nsa.index else np.nan),
                "revision_mode": v.revision_mode,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no monthly CPI forecasts in the window")
    return df


def direction_backtest(bundle: USDataBundle, start: str, end: str,
                       cfg: USVintageConfig | None = None) -> pd.DataFrame:
    """The one-print-ahead YoY direction call, with its known hurdle."""
    cfg = replace(cfg or USVintageConfig(), use_indicator_nowcast=False)
    real_yoy = _yoy(bundle.cpi)
    real_mom = quads.calendar_pct_change(bundle.cpi, 1).dropna()

    rows = []
    for asof in asof_dates(start, end, "M"):
        try:
            v = build_vintage(bundle, asof, cfg)
        except ValueError:
            continue
        m0, m1 = v.last_cpi_month, v.last_cpi_month + 1
        proj = inflation.build_cpi_projection(v.cpi_index, 3, v.assumptions,
                                              v.aux)
        yoy = proj["yoy_pct"]
        if m1 not in yoy.index or m1 not in real_yoy.index:
            continue
        pred_delta = float(yoy[m1] - yoy[m0])
        real_delta = float(real_yoy[m1] - real_yoy[m0])
        rows.append({
            "asof": asof,
            "last_print": str(m0),
            "target_print": str(m1),
            "pred_yoy": float(yoy[m1]),
            "real_yoy": float(real_yoy[m1]),
            "pred_delta": pred_delta,
            "real_delta": real_delta,
            "hurdle_mom": float(real_mom.get(m1 - 12, np.nan)),
            "pred_dir": "accelerating" if pred_delta > 0 else "decelerating",
            "real_dir": "accelerating" if real_delta > 0 else "decelerating",
            "hit": (pred_delta > 0) == (real_delta > 0),
            "conviction_pp": abs(pred_delta),
            "revision_mode": v.revision_mode,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no monthly direction calls in the window")
    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon MAE/RMSE/bias for the model and both naive benchmarks."""
    rows = []
    for h, sub in df.groupby("h_months"):
        row = {
            "h_months": int(h), "n": len(sub),
            "mae_yoy": float(sub["err_yoy"].abs().mean()),
            "rmse_yoy": float(np.sqrt((sub["err_yoy"] ** 2).mean())),
            "bias_yoy": float(sub["err_yoy"].mean()),
            "mae_yoy_nsa": float(sub["err_yoy_nsa"].abs().mean()),
            "mae_mom": float(sub["err_mom"].abs().mean()),
            "mae_yoy_random_walk": float(sub["err_yoy_rw"].abs().mean()),
            "mae_yoy_seasonal_naive":
                float(sub["err_yoy_seasonal_naive"].abs().mean()),
        }
        row["skill_vs_random_walk"] = (
            1 - row["mae_yoy"] / row["mae_yoy_random_walk"]
            if row["mae_yoy_random_walk"] else np.nan)
        row["skill_vs_seasonal_naive"] = (
            1 - row["mae_yoy"] / row["mae_yoy_seasonal_naive"]
            if row["mae_yoy_seasonal_naive"] else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("h_months").sort_index()


def bucket_label(conviction_pp: float) -> str:
    for lo, hi, label in CONVICTION_BUCKETS:
        if lo <= conviction_pp < hi:
            return label
    return CONVICTION_BUCKETS[-1][2]


def summarize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Hit rates overall and per conviction bucket: how often the direction
    is right when the model actually leans."""
    rows = [{"bucket": "ALL months", "n": len(df), "share_of_months": 1.0,
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
