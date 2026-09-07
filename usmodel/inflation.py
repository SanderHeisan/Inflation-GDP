"""
Bottom-up CPI model for the United States.

Same architecture as the Norwegian model (quadmap/inflation.py): forecast
each block's monthly index momentum from its own driver, aggregate with
basket weights into a monthly index path, and let known year-ago levels
turn the level path into a YoY path mechanically (base effects).

Block -> driver map:
  gasoline          -> WTI crude via the oil->headline pass-through
                       (config.OIL_HEADLINE_BPS_PER_DOLLAR; Hedgeye's
                       ~3 bps per $1/bbl rule)
  electricity /
    energy_other    -> mild trend + seasonality
  shelter (OER+rent)-> market ("new-lease") rents lagged ~12 months. The
                       single most forecastable large block: last year's
                       observed rent momentum is most of next year's shelter.
  food              -> grocery/away pipeline toward trend
  core_goods        -> dollar (import-price) distributed-lag pass-through;
                       a stronger dollar is disinflationary for goods
  supercore         -> wage growth (AHE/ECI) with a productivity haircut
  medical           -> sticky trend

External market inputs (WTI strip, dollar path, wage growth, market-rent
momentum) come in via an assumptions dict / aux series -- the levers you
update each month, and where market views enter. Realized future prices
must never leak into these (the backtest's vintage builder enforces it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _wti_path(assumptions: dict, horizon: pd.PeriodIndex) -> pd.Series:
    """WTI level over the horizon: spot-carry (flat at last observed) unless
    the forward strip supplies a month. assumptions['wti_forward'] is a
    dict {'YYYY-MM': price}; anchor is assumptions['wti_recent']."""
    fwd = assumptions.get("wti_forward", {})
    last = float(assumptions["wti_recent"])
    out, prev = {}, last
    for p in horizon:
        prev = float(fwd.get(str(p), prev))
        out[p] = prev
    anchor = pd.Series({horizon[0] - 1: last})
    return pd.concat([anchor, pd.Series(out)])


def project_energy(assumptions: dict, horizon: pd.PeriodIndex,
                   weights: dict) -> dict[str, pd.Series]:
    """Gasoline from WTI via the oil pass-through, plus a mild electricity /
    other-energy drift. The gasoline block MoM is set so the *headline*
    contribution equals OIL_HEADLINE_BPS_PER_DOLLAR bps per $1/bbl move."""
    total_w = sum(weights.values())
    gas_share = weights["gasoline"] / total_w

    wti = _wti_path(assumptions, horizon)
    dwti = wti.diff().reindex(horizon)
    bps = config.OIL_HEADLINE_BPS_PER_DOLLAR
    # headline contribution (fraction of index) = bps/1e4 * dWTI; carried
    # into the gasoline block by dividing out its weight share.
    headline_frac = (bps / 1e4) * dwti
    same, carry = config.OIL_SAME_MONTH_SHARE, 1 - config.OIL_SAME_MONTH_SHARE
    gas_headline = headline_frac * same + headline_frac.shift(1).fillna(0) * carry
    gasoline = gas_headline / gas_share

    # Electricity + other energy: small positive drift with winter tilt.
    elec = pd.Series([0.002 / 12 + (0.004 if p.month in (1, 2, 7, 8) else 0)
                      * 0 for p in horizon], index=horizon)
    return {"gasoline": gasoline.fillna(0.0), "electricity": elec,
            "energy_other": elec.copy()}


def project_shelter(cpi_hist: pd.Series, assumptions: dict,
                    horizon: pd.PeriodIndex,
                    market_rent: pd.Series | None) -> pd.Series:
    """CPI shelter from market rents lagged ~12 months. For each projected
    month, the target shelter YoY blends the lagged market-rent YoY (mostly
    already observed) with the long-run trend; converted to monthly."""
    lag = config.SHELTER_MARKET_RENT_LAG_M
    pt, trend = config.SHELTER_PASSTHROUGH, config.SHELTER_TREND_YOY

    if market_rent is not None and len(market_rent) > lag + 12:
        mr_yoy = (market_rent / market_rent.shift(12) - 1.0) * 100.0
    else:
        mr_yoy = None
    fallback = assumptions.get("market_rent_yoy_recent", trend)

    out = {}
    for p in horizon:
        src = p - lag
        lagged = (float(mr_yoy.get(src, np.nan)) if mr_yoy is not None
                  else np.nan)
        if lagged != lagged:            # NaN -> use the recent fallback
            lagged = fallback
        shelter_yoy = pt * lagged + (1 - pt) * trend
        out[p] = shelter_yoy / 12.0 / 100.0
    return pd.Series(out)


def project_core_goods(assumptions: dict, horizon: pd.PeriodIndex,
                       dollar: pd.Series | None) -> pd.Series:
    """Import-price pass-through: build projected 12m core-goods inflation
    from already-observed dollar moves (the lags mean most of next year is
    baked in) plus the assumed dollar path, then spread to monthly."""
    coeffs = config.DOLLAR_PASSTHROUGH_LAGS
    base = assumptions.get("core_goods_baseline_yoy",
                           config.CORE_GOODS_BASELINE_YOY)
    if dollar is None or len(dollar) < 13:
        return pd.Series(base / 12.0 / 100.0, index=horizon)

    dxy = dollar.copy()
    path = assumptions.get("dollar_path", {})
    for p in horizon:
        dxy.loc[p] = float(path.get(str(p), dxy.iloc[-1]))
    dxy_12m = dxy.pct_change(12) * 100.0    # % change, positive = stronger $

    out = {}
    for p in horizon:
        contrib = sum(c * dxy_12m.get(p - l, 0.0) for l, c in coeffs.items())
        out[p] = (base + contrib) / 12.0 / 100.0
    return pd.Series(out)


def project_supercore(assumptions: dict, horizon: pd.PeriodIndex) -> pd.Series:
    wage = assumptions.get("wage_growth_pct", 4.0)
    annual = wage * config.SUPERCORE_WAGE_PASSTHROUGH
    return pd.Series(annual / 12.0 / 100.0, index=horizon)


def project_food(assumptions: dict, horizon: pd.PeriodIndex) -> pd.Series:
    annual = assumptions.get("food_pipeline_yoy", config.FOOD_TREND_YOY)
    return pd.Series(annual / 12.0 / 100.0, index=horizon)


def project_medical(horizon: pd.PeriodIndex) -> pd.Series:
    return pd.Series(config.MEDICAL_TREND_YOY / 12.0 / 100.0, index=horizon)


def build_cpi_projection(cpi_hist: pd.Series, horizon_months: int,
                         assumptions: dict, aux: dict | None = None,
                         weights: dict | None = None) -> pd.DataFrame:
    """Projected total CPI index path + per-block contributions.

    cpi_hist : observed monthly CPI-U index level.
    aux      : optional {'market_rent': Series, 'dollar': Series} of observed
               driver series (Period[M]).
    Returns DataFrame with cpi_index, mom_pct, yoy_pct, and contrib_* columns
    (the auditable decomposition).
    """
    w = weights or config.CPI_WEIGHTS
    total_w = sum(w.values())
    aux = aux or {}
    last = cpi_hist.index[-1]
    horizon = pd.period_range(last + 1, periods=horizon_months, freq="M")

    energy = project_energy(assumptions, horizon, w)
    blocks = {
        "gasoline":     energy["gasoline"],
        "electricity":  energy["electricity"],
        "energy_other": energy["energy_other"],
        "shelter":      project_shelter(cpi_hist, assumptions, horizon,
                                        aux.get("market_rent")),
        "food_at_home": project_food(assumptions, horizon),
        "food_away":    project_food(assumptions, horizon),
        "core_goods":   project_core_goods(assumptions, horizon,
                                           aux.get("dollar")),
        "supercore":    project_supercore(assumptions, horizon),
        "medical":      project_medical(horizon),
    }

    rows = []
    level = cpi_hist.iloc[-1]
    for p in horizon:
        contribs = {b: w[b] / total_w * float(blocks[b][p]) for b in blocks}
        mom = sum(contribs.values())
        level = level * (1.0 + mom)
        rows.append({"period": p, "cpi_index": level, "mom_pct": mom * 100,
                     **{f"contrib_{b}": v * 100 for b, v in contribs.items()}})

    proj = pd.DataFrame(rows).set_index("period")
    full = pd.concat([cpi_hist.rename("cpi_index").to_frame(), proj])
    full["yoy_pct"] = (full["cpi_index"] / full["cpi_index"].shift(12) - 1) * 100
    if "mom_pct" not in full:
        full["mom_pct"] = full["cpi_index"].pct_change() * 100
    return full
