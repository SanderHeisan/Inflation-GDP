"""
The direction backtest: four calls, scored the way a subscriber uses them.

For every as-of date and every horizon the quad table covers, the model
makes a direction call -- accelerating or decelerating -- on

    growth,    YoY basis   sign of d(yoy GDP growth)        the quad's growth axis
    growth,    QoQ basis   sign of d(sequential QoQ growth)  "will this print beat last quarter's?"
    inflation, YoY basis   sign of d(quarterly-avg CPI YoY)  the quad's inflation axis
    inflation, QoQ basis   sign of d(quarterly CPI rate)     "is the sequential pace picking up?"

and each call carries a conviction (|predicted change|, pp). Hit rates are
reported by horizon and by conviction bucket, plus the share of the time the
model actually makes a call: on the QoQ growth basis it abstains past one
quarter out, because nothing tested there beats a coin flip and a number
without skill behind it is worse than no number.

The YoY calls come straight off the same projection the quad uses. The
growth-QoQ call comes from usmodel.growth_direction (a published-data
reversal model that beats the nowcast-based call 74% to 68%). The inflation
calls come off the bottom-up CPI path. Truth is the final vintage; the
growth calls therefore carry the simulated-revision caveat in the README.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quadmap import quads
from usmodel import gdp as gdp_mod, growth_direction, inflation
from usmodel.data_bundle import USDataBundle

from .btconfig import USVintageConfig
from .engine import asof_dates
from .vintage import build_vintage

CALLS = ("growth_yoy", "growth_qoq", "inflation_yoy", "inflation_qoq")
CALL_LABELS = {"growth_yoy": "growth, YoY basis",
               "growth_qoq": "growth, QoQ basis",
               "inflation_yoy": "inflation, YoY basis",
               "inflation_qoq": "inflation, QoQ basis"}
BUCKETS = [(0.00, 0.10, "<0.10pp"), (0.10, 0.25, "0.10-0.25pp"),
           (0.25, 0.50, "0.25-0.50pp"), (0.50, np.inf, ">0.50pp")]
COVID_TARGETS = pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q3", "2020Q4",
                                "2021Q1", "2021Q2"], freq="Q")


def realized_deltas(bundle: USDataBundle) -> dict[str, pd.Series]:
    """Realized change series for each call, final vintage, Period[Q]."""
    g_qoq = (bundle.gdp.pct_change() * 100).dropna()
    g_yoy = (bundle.gdp.pct_change(4) * 100).dropna()
    cpi_q = bundle.cpi.groupby(bundle.cpi.index.asfreq("Q")).mean()
    i_qoq = (cpi_q.pct_change() * 100).dropna()
    i_yoy = quads.monthly_to_quarterly_yoy(
        (bundle.cpi.pct_change(12) * 100).dropna())
    return {"growth_yoy": g_yoy.diff().dropna(),
            "growth_qoq": g_qoq.diff().dropna(),
            "inflation_yoy": i_yoy.diff().dropna(),
            "inflation_qoq": i_qoq.diff().dropna()}


def direction_backtest(bundle: USDataBundle, start: str, end: str,
                       max_horizon: int = 4,
                       cfg: USVintageConfig | None = None) -> pd.DataFrame:
    """One row per (as-of date, horizon, call). `made_call` is False where
    the model abstains; those rows carry no prediction and are excluded from
    hit rates but counted in the call-share statistics."""
    cfg = cfg or USVintageConfig()
    real = realized_deltas(bundle)
    rows = []
    for asof in asof_dates(start, end, "M"):
        try:
            v = build_vintage(bundle, asof, cfg)
        except ValueError:
            continue
        aq = pd.Period(asof, freq="Q")
        target_q = aq + max_horizon

        # the quad's own projection: growth YoY and inflation YoY / QoQ
        gq = (target_q - v.last_gdp_quarter).n + 1
        gf = gdp_mod.project_gdp(v.gdp_level, v.indicators, horizon_quarters=gq)
        cm = (pd.Period(target_q.end_time, freq="M") - v.last_cpi_month).n + 1
        cf = inflation.build_cpi_projection(v.cpi_index, cm, v.assumptions,
                                            v.aux)
        cpi_q = cf["cpi_index"].groupby(cf.index.asfreq("Q")).mean()
        pred = {
            "growth_yoy": gf["yoy_pct"].diff(),
            "inflation_yoy": quads.monthly_to_quarterly_yoy(
                cf["yoy_pct"].dropna()).diff(),
            "inflation_qoq": (cpi_q.pct_change() * 100).diff(),
        }
        # the sequential growth call: its own model, published data only
        gd = growth_direction.qoq_direction_calls(
            v.gdp_level, v.diagnostics.get("trend_qoq_pct"))

        for h in range(max_horizon + 1):
            tq = aq + h
            for call in CALLS:
                if call == "growth_qoq":
                    if gd.empty or tq not in gd.index:
                        p, grade = np.nan, "abstain"
                    else:
                        p, grade = float(gd.loc[tq, "delta_pp"]), \
                            str(gd.loc[tq, "grade"])
                else:
                    p = float(pred[call].get(tq, np.nan))
                    grade = "call"
                r = float(real[call].get(tq, np.nan))
                if np.isnan(r):
                    continue
                made = not np.isnan(p) and abs(p) > 1e-12
                rows.append({
                    "asof": asof, "asof_quarter": str(aq), "horizon": h,
                    "target_quarter": str(tq), "call": call, "grade": grade,
                    "made_call": made,
                    "pred_delta": p if made else np.nan,
                    "real_delta": r,
                    "pred_dir": ("accelerating" if p > 0 else "decelerating")
                    if made else "no call",
                    "real_dir": "accelerating" if r > 0 else "decelerating",
                    "hit": float(np.sign(p) == np.sign(r)) if made else np.nan,
                    "conviction_pp": abs(p) if made else np.nan,
                    "covid_target": tq in COVID_TARGETS,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no direction calls in the window")
    return df


def bucket_label(conviction_pp: float) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= conviction_pp < hi:
            return label
    return BUCKETS[-1][2]


def summarize_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Hit rate per call x horizon (calls only), share of rows with a call,
    and the same hit rate on ex-COVID targets."""
    made = df[df["made_call"]].copy()
    made["hit"] = made["hit"].astype(float)
    hit = made.pivot_table(index="call", columns="horizon", values="hit",
                           aggfunc="mean")
    hit_ex = made[~made["covid_target"]].pivot_table(
        index="call", columns="horizon", values="hit", aggfunc="mean")
    share = df.pivot_table(index="call", columns="horizon",
                           values="made_call", aggfunc="mean")
    n = made.pivot_table(index="call", columns="horizon", values="hit",
                         aggfunc="size")
    tidy = []
    for call in CALLS:
        if call not in share.index:
            continue
        for h in share.columns:
            tidy.append({"call": call, "horizon": int(h),
                         "n_calls": int(n.loc[call, h]) if call in n.index
                         and not pd.isna(n.loc[call, h]) else 0,
                         "call_share": float(share.loc[call, h]),
                         "hit": float(hit.loc[call, h]) if call in hit.index
                         and h in hit.columns
                         and not pd.isna(hit.loc[call, h]) else np.nan,
                         "hit_ex_covid": float(hit_ex.loc[call, h])
                         if call in hit_ex.index and h in hit_ex.columns
                         else np.nan})
    return pd.DataFrame(tidy).set_index(["call", "horizon"])


def summarize_by_conviction(df: pd.DataFrame) -> pd.DataFrame:
    """Hit rate per call x conviction bucket, all horizons pooled."""
    made = df[df["made_call"]].copy()
    made["hit"] = made["hit"].astype(float)
    made["bucket"] = made["conviction_pp"].map(bucket_label)
    order = [b[2] for b in BUCKETS]
    rows = []
    for call in CALLS:
        sub = made[made["call"] == call]
        if sub.empty:
            continue
        rows.append({"call": call, "bucket": "ALL calls", "n": len(sub),
                     "share_of_calls": 1.0, "hit": float(sub["hit"].mean()),
                     "hit_ex_covid": float(
                         sub[~sub["covid_target"]]["hit"].mean())})
        for bkt in order:
            s = sub[sub["bucket"] == bkt]
            if s.empty:
                continue
            rows.append({"call": call, "bucket": bkt, "n": len(s),
                         "share_of_calls": len(s) / len(sub),
                         "hit": float(s["hit"].mean()),
                         "hit_ex_covid": float(
                             s[~s["covid_target"]]["hit"].mean())
                         if (~s["covid_target"]).any() else np.nan})
    return pd.DataFrame(rows).set_index(["call", "bucket"])
