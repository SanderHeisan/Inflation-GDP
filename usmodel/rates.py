"""
The rate channel: changes in the policy rate -- past ones, and the
market-implied path ahead -- as a drag on the growth path past the nowcast
quarter.

Why it exists. Monetary policy works with a lag: a hike lowers real GDP
growth over the following one to two years, then fades. The quad's growth
call at two to four quarters out is "will next year's QoQ land above or
below the year-ago QoQ we already know", and a path held flat at trend
knows nothing about a Fed that is tightening into it. This module makes
the path know, and states what that is worth.

What the data say (real GDP 1960-2026 ex-COVID, quarterly means of the
effective fed funds rate):

  * A 1pp rise in the policy rate over four quarters lowers quarterly real
    GDP growth by ~0.10pp per quarter across the four quarters that start
    two quarters later (corr -0.27 at h=2..5; a free distributed lag sums
    to -0.70pp per 1pp step over lags 2-8). That is a level effect of
    roughly -0.4% to -0.7% after two years -- the low end of the FRB/US
    range, but the same sign and timing.
  * It is a 1960-1984 fact. From 1985 on the bivariate slope is zero to
    POSITIVE (+0.05 to +0.16): the Fed hikes into strength and the strength
    outlasts the hikes; 2022-23 is the loudest example. Mortgage and 10y
    windows behave the same way.
  * Walk-forward 2005-2026 on final data, a mechanical drag of this form
    LOWERS the growth-direction hit rate by 4-5pp at every horizon; the
    repo's own vintage backtest is quoted in the README.

So the channel ships as asked -- priced in, with the long-history
sensitivity re-estimated point-in-time on every vintage -- and its cost
is measured and printed rather than hidden. `config.RATE_CHANNEL_ENABLED`
turns it off; `config.RATE_SENSITIVITY_OVERRIDE` pins the size.

Mechanics. For each projected quarter q past the nowcast quarter,

    qoq(q) = trend + beta * [ff(q-L1) - ff(q-L2)],      (L1, L2) = (2, 6)

with beta <= 0 fitted on winsorized QoQ against that same window since
1960, and ff the quarterly mean of the effective rate: observed where
published, then the dated market-implied steps (config.POLICY_RATE_PATH,
live runs only), then flat. The nowcast quarter is left to the indicator
fit, which already sees the rate environment. The sign is imposed and the
size estimated: the prior that a hike never adds to growth is the reason
the channel exists, and the modern sample would otherwise hand back a
positive number that means "the Fed hikes when growth is strong".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MIN_FIT_QUARTERS = 80


def quarterly_mean(series: pd.Series) -> pd.Series:
    """Monthly (or daily-collapsed monthly) series -> quarterly means."""
    s = series.dropna()
    return s.groupby(s.index.asfreq("Q")).mean()


def expected_policy_path(observed_monthly: pd.Series,
                         steps: dict | None,
                         end: pd.Period) -> pd.Series:
    """Monthly expected policy rate through `end`: the observed monthly
    means (the last of which may be a partial month in progress), then the
    dated `steps` ({'YYYY-MM-DD': rate}) day-weighted within each month,
    then flat at the last level.

    A step dated inside an observed month is blended by calendar days: the
    days before it keep the observed mean, the days from it take the step.
    That is exact for a partial month observed up to the step date and a
    close approximation once the month is complete."""
    obs = observed_monthly.dropna()
    if obs.empty:
        raise ValueError("no observed policy rate")
    end = pd.Period(end, freq="M")
    full = pd.period_range(obs.index[0], max(end, obs.index[-1]), freq="M")
    out = obs.reindex(full).ffill()
    if not steps:
        return out
    st = sorted((pd.Timestamp(k).normalize(), float(v))
                for k, v in steps.items())
    start_m = pd.Period(st[0][0], freq="M")
    days = pd.date_range(start_m.start_time.normalize(),
                         max(end, start_m).end_time.normalize(), freq="D")
    before = float(out[start_m - 1]) if (start_m - 1) in out.index \
        else float(obs.iloc[0])
    lvl = pd.Series(before, index=days)
    if start_m in obs.index:
        lvl[days < st[0][0]] = float(obs[start_m])
    for d, v in st:
        lvl[days >= d] = v
    monthly = lvl.groupby(lvl.index.to_period("M")).mean()
    monthly = monthly[monthly.index <= full[-1]]
    out.loc[monthly.index] = monthly.to_numpy()
    return out


def fit_rate_sensitivity(gdp_level: pd.Series, policy_q: pd.Series,
                         lags: tuple[int, int] = config.RATE_LAG_QUARTERS,
                         fit_start: str = config.RATE_FIT_START,
                         exclude: tuple = config.RATE_FIT_EXCLUDE,
                         clip: tuple[float, float] = config.RATE_SENSITIVITY_CLIP,
                         override: float | None = config.RATE_SENSITIVITY_OVERRIDE,
                         winsor_mad: float = 3.0,
                         min_quarters: int = MIN_FIT_QUARTERS) -> dict | None:
    """beta (pp of quarterly growth per 1pp change in the policy rate over
    the lag window), fitted point-in-time on whatever is passed in. Returns
    None with too little history. `beta_raw` is the unclipped estimate, so
    a live run can show when the sign had to be imposed."""
    qoq = (gdp_level.pct_change() * 100).dropna()
    p = policy_q.dropna()
    x = p.shift(lags[0]) - p.shift(lags[1])
    df = pd.concat([qoq.rename("y"), x.rename("x")], axis=1).dropna()
    df = df[df.index >= pd.Period(fit_start, freq="Q")]
    df = df[~df.index.isin(pd.PeriodIndex(list(exclude), freq="Q"))]
    if len(df) < min_quarters or float(df["x"].var()) < 1e-12:
        return None
    y = df["y"]
    med = float(y.median())
    mad = float((y - med).abs().median()) * 1.4826
    if mad > 0:
        y = y.clip(med - winsor_mad * mad, med + winsor_mad * mad)
    beta_raw, _ = np.polyfit(df["x"].to_numpy(float), y.to_numpy(float), 1)
    beta = float(np.clip(beta_raw, clip[0], clip[1]))
    if override is not None:
        beta = float(override)
    return {"beta": beta, "beta_raw": float(beta_raw),
            "corr": float(df["x"].corr(y)), "n": int(len(df)),
            "fit_start": str(df.index[0]), "fit_end": str(df.index[-1]),
            "lags": tuple(lags), "overridden": override is not None}


def rate_drag(policy_q: pd.Series, quarters: pd.PeriodIndex, beta: float,
              lags: tuple[int, int] = config.RATE_LAG_QUARTERS) -> pd.Series:
    """Drag on QoQ growth (DECIMAL per quarter) for each target quarter:
    beta * [ff(q-L1) - ff(q-L2)] / 100, the path held flat past its last
    point. Zero where the window reaches before the data start."""
    p = policy_q.dropna()
    idx = pd.period_range(p.index[0], max(quarters.max(), p.index[-1]),
                          freq="Q")
    ext = p.reindex(idx).ffill()
    vals = []
    for q in quarters:
        a, b = q - lags[0], q - lags[1]
        if a in ext.index and b in ext.index:
            vals.append(beta * float(ext[a] - ext[b]) / 100.0)
        else:
            vals.append(0.0)
    return pd.Series(vals, index=quarters, name="rate_drag_qoq")


def rate_state(fed_funds_monthly: pd.Series, policy_q: pd.Series,
               fit: dict | None, drag: pd.Series | None,
               steps: dict | None = None,
               lags: tuple[int, int] = config.RATE_LAG_QUARTERS) -> dict:
    """The rate block as a reader wants it: where the policy rate is and
    how far it has moved, the path the projection assumes quarter by
    quarter and where each number comes from, the drag it puts on growth
    (pp annualized) and the cumulative level effect, plus the fit behind
    the sensitivity."""
    ff = fed_funds_monthly.dropna()
    obs_q = quarterly_mean(ff)
    first_step = (min(pd.Timestamp(k) for k in steps) if steps else None)
    last_obs_q = obs_q.index[-1]
    path = []
    for q, r in policy_q.items():
        if q < last_obs_q or (q == last_obs_q and first_step is None):
            src = "observed"
        elif q == last_obs_q:
            src = "observed + implied"
        elif first_step is not None and q.start_time <= \
                max(pd.Timestamp(k) for k in steps) + pd.Timedelta(days=95):
            src = "market-implied"
        else:
            src = "flat"
        path.append({"quarter": str(q), "rate_pct": float(r), "source": src})
    out = {
        "latest_month": str(ff.index[-1]), "level_pct": float(ff.iloc[-1]),
        "chg_4q_pp": float(obs_q.iloc[-1] - obs_q.iloc[-5]) if len(obs_q) > 5 else np.nan,
        "chg_8q_pp": float(obs_q.iloc[-1] - obs_q.iloc[-9]) if len(obs_q) > 9 else np.nan,
        "lags": list(lags), "path": path,
        "path_source": ("market-implied steps supplied" if steps
                        else "flat at the last observation"),
        "fit": fit,
    }
    if drag is not None and fit is not None:
        cum = 0.0
        rows = []
        for q, d in drag.items():
            cum += float(d) * 100
            rows.append({"quarter": str(q), "drag_qoq_pp": float(d) * 100,
                         "drag_ann_pp": float(((1 + d) ** 4 - 1) * 100),
                         "cum_level_pct": cum})
        out["drag"] = rows
    return out
