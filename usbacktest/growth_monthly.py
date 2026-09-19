"""
Walk-forward backtest of the monthly growth call (usmodel.monthly_growth).

At each month-end as-of date the component panel is truncated with each
series' own publication lag, the activity index is built from what was
published, and its calls for the first months past the last complete one
are scored against three truths:

  own      the direction of the final-data index's YoY change that month
           (what a monthly quad map needs)
  bbk      the direction of the Chicago Fed's monthly real GDP (BBK) YoY
           change that month, final vintage -- an independent monthly GDP
  gdp_q    the direction of the quarterly real-GDP YoY change for the
           quarter containing the target month (the quad axis itself)

Rows carry the horizon (months past the last complete month), the status
of the target month (nowcast: some components in; forecast: none) and the
conviction, so hit rates can be read by horizon and by bucket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from usmodel import monthly_growth as mg
from usmodel.data_bundle import USDataBundle

from .btconfig import USVintageConfig
from .engine import asof_dates
from .vintage import published_indicators

COVID_MONTHS = pd.period_range("2020-02", "2021-06", freq="M")


def truths(bundle: USDataBundle) -> dict[str, pd.Series]:
    """Final-data direction series (signed YoY changes) for the three truths."""
    final = mg.activity_index(bundle.indicators, horizon_months=0)
    own = final["yoy_pct"].diff() if final is not None else pd.Series(dtype=float)
    own = own[final["status"] == "actual"] if final is not None else own
    bbk = bundle.indicators.get("bbk_gdp")
    if bbk is not None and len(bbk):
        lvl = pd.Series(np.cumprod((1 + bbk / 100) ** (1 / 12)), index=bbk.index)
        bbk_d = ((lvl / lvl.shift(12) - 1) * 100).diff()
    else:
        bbk_d = pd.Series(dtype=float)
    g = bundle.gdp
    gdp_d = ((g / g.shift(4) - 1) * 100).diff()
    return {"own": own.dropna(), "bbk": bbk_d.dropna(), "gdp_q": gdp_d.dropna()}


def monthly_asof_dates(start: str, end: str, days: tuple[int, ...] = (20, 31)) -> pd.DatetimeIndex:
    """As-of dates on the given days of each month (31 = month end). The
    20th is the point where payrolls, hours, IP and retail sales for the
    previous month are in and PCE is not -- the information set a live
    reader has for most of the month."""
    ends = asof_dates(start, end, "M")
    out = []
    for e in ends:
        for d in days:
            out.append(e if d >= 28 else e.replace(day=d))
    return pd.DatetimeIndex(sorted(set(out)))


def monthly_growth_backtest(bundle: USDataBundle, start: str, end: str,
                            horizon_months: int = 3,
                            cfg: USVintageConfig | None = None,
                            asof_days: tuple[int, ...] = (20, 31)) -> pd.DataFrame:
    cfg = cfg or USVintageConfig()
    tr = truths(bundle)
    rows = []
    for asof in monthly_asof_dates(start, end, asof_days):
        panel = published_indicators(bundle, asof, cfg)
        idx = mg.activity_index(panel, horizon_months=horizon_months)
        calls = mg.growth_calls(idx, horizon_months)
        if calls.empty:
            continue
        for t, r in calls.iterrows():
            own = tr["own"].get(t, np.nan)
            bbk = tr["bbk"].get(t, np.nan)
            gq = tr["gdp_q"].get(t.asfreq("Q"), np.nan)
            rows.append({
                "asof": asof, "asof_day": int(asof.day),
                "last_complete": str(idx.attrs["last_complete"]),
                "target": str(t), "horizon": int(r["horizon"]),
                "status": r["status"], "n_actual": int(r["n_actual"]),
                "pred_d_yoy": float(r["d_yoy_pp"]), "pred_dir": r["direction"],
                "conviction_pp": float(r["conviction_pp"]), "bucket": r["bucket"],
                "real_d_own": own, "real_d_bbk": bbk, "real_d_gdp_q": gq,
                "hit_own": float(np.sign(r["d_yoy_pp"]) == np.sign(own)) if not np.isnan(own) else np.nan,
                "hit_bbk": float(np.sign(r["d_yoy_pp"]) == np.sign(bbk)) if not np.isnan(bbk) else np.nan,
                "hit_gdp_q": float(np.sign(r["d_yoy_pp"]) == np.sign(gq)) if not np.isnan(gq) else np.nan,
                "covid": t in COVID_MONTHS,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no monthly growth calls in the window")
    return df


def summarize_monthly_growth(df: pd.DataFrame, ex_covid: bool = True) -> pd.DataFrame:
    """Hit rates by horizon and by conviction bucket, for each truth."""
    d = df[~df["covid"]] if ex_covid else df
    order = [b[2] for b in mg.BUCKETS]
    rows = []
    for h, sub in d.groupby("horizon"):
        rows.append({"horizon": int(h), "bucket": "ALL", "n": len(sub),
                     "hit_own": sub["hit_own"].mean(), "hit_bbk": sub["hit_bbk"].mean(),
                     "hit_gdp_q": sub["hit_gdp_q"].mean(),
                     "avg_conviction_pp": sub["conviction_pp"].mean()})
        for b in order:
            s = sub[sub["bucket"] == b]
            if s.empty:
                continue
            rows.append({"horizon": int(h), "bucket": b, "n": len(s),
                         "hit_own": s["hit_own"].mean(), "hit_bbk": s["hit_bbk"].mean(),
                         "hit_gdp_q": s["hit_gdp_q"].mean(),
                         "avg_conviction_pp": s["conviction_pp"].mean()})
    return pd.DataFrame(rows).set_index(["horizon", "bucket"])
