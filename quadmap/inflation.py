"""
Bottom-up CPI model for Norway.

Architecture: forecast each COICOP block's *monthly index level* with its own
driver, aggregate with basket weights, and let known year-ago index levels
turn the level path into a YoY path mechanically (base effects).

Component -> driver map:
  electricity        -> Nord Pool forward curve, run through the stromstotte
                        subsidy formula (spot exposure is what makes Norwegian
                        CPI electricity so volatile)
  other_energy       -> Brent forward curve converted to NOK
  imported goods     -> distributed-lag pass-through from I-44 (weaker NOK ->
                        higher imported-goods inflation 3-18 months later)
  housing_ex_energy  -> rent indexation: last year's CPI + wage growth blend
  food               -> world food prices in NOK, applied in the Feb/Jul
                        repricing windows Norwegian grocery chains use
  domestic services  -> the annual wage settlement norm (frontfagsrammen)
                        with a lag

External market inputs (power forwards, Brent, NOK path, wage norm) come in
via an assumptions dict / assumptions.yaml -- these are the levers you update
each month, and where your own market views enter the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Component projections. Each takes the historical monthly index for that
# component and returns projected monthly % changes for the horizon.
# ---------------------------------------------------------------------------

def project_electricity(hist_index: pd.Series, horizon: pd.PeriodIndex,
                        assumptions: dict) -> pd.Series:
    """Map the Nord Pool forward curve into the CPI electricity index.

    assumptions['power_forward_ore_kwh'] : dict {'YYYY-MM': spot ore/kWh}
    Applies the stromstotte formula so the *consumer-facing* price is what
    drives the index.
    """
    sub = config.STROMSTOTTE
    fwd = assumptions["power_forward_ore_kwh"]

    def consumer_price(spot_ore: float) -> float:
        excess = max(spot_ore - sub["threshold_ore_kwh"], 0.0)
        return spot_ore - sub["coverage_share"] * excess

    # Anchor: infer the consumer price consistent with the last observed
    # index level using the last known spot in the assumptions history.
    last_spot = assumptions["power_recent_ore_kwh"]
    last_cp = consumer_price(last_spot)
    # Grid rent + retail margin ~ stable adder; consumer price moves the
    # variable part. Assume variable part is VARIABLE_SHARE of the bill.
    variable_share = assumptions.get("power_variable_share", 0.55)

    changes = []
    prev_cp = last_cp
    for p in horizon:
        spot = fwd.get(str(p), last_spot)
        cp = consumer_price(spot)
        mom = variable_share * (cp / prev_cp - 1.0)
        changes.append(mom)
        prev_cp = cp
    return pd.Series(changes, index=horizon)


def project_fuel(hist_index: pd.Series, horizon: pd.PeriodIndex,
                 assumptions: dict) -> pd.Series:
    """Brent forward in NOK -> pump prices. Taxes are a large fixed component,
    so pass only ~40% of the crude move through to the CPI fuel index."""
    brent = assumptions["brent_forward_usd"]          # {'YYYY-MM': usd/bbl}
    usdnok = assumptions["usdnok_path"]               # {'YYYY-MM': rate}
    passthrough = assumptions.get("fuel_passthrough", 0.40)

    last_nok = assumptions["brent_recent_usd"] * assumptions["usdnok_recent"]
    changes = []
    prev = last_nok
    for p in horizon:
        nok_price = brent.get(str(p), assumptions["brent_recent_usd"]) * \
            usdnok.get(str(p), assumptions["usdnok_recent"])
        changes.append(passthrough * (nok_price / prev - 1.0))
        prev = nok_price
    return pd.Series(changes, index=horizon)


def estimate_fx_passthrough(imported_cpi_12m: pd.Series,
                            i44_12m: pd.Series) -> dict[int, float]:
    """Re-estimate the distributed-lag pass-through on live history.

    Regress 12m imported-goods inflation on lagged 12m I-44 changes.
    Falls back to config coefficients if the sample is too short.
    """
    lags = sorted(config.FX_PASSTHROUGH_LAGS)
    X = pd.concat({f"lag{l}": i44_12m.shift(l) for l in lags}, axis=1)
    df = pd.concat([imported_cpi_12m.rename("y"), X], axis=1).dropna()
    if len(df) < 60:
        return dict(config.FX_PASSTHROUGH_LAGS)
    beta, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(df)), df[X.columns].values]),
        df["y"].values, rcond=None)
    coeffs = {l: max(b, 0.0) for l, b in zip(lags, beta[1:])}
    return coeffs


def project_imported_goods(horizon: pd.PeriodIndex, i44_hist: pd.Series,
                           assumptions: dict,
                           coeffs: dict[int, float] | None = None) -> pd.Series:
    """FX pass-through: build the projected 12m imported-goods inflation from
    *already observed* I-44 changes (the lags mean most of the next year's
    pass-through is baked in) plus the assumed NOK path, then convert the YoY
    path to monthly changes."""
    coeffs = coeffs or dict(config.FX_PASSTHROUGH_LAGS)
    i44 = i44_hist.copy()
    for p in horizon:  # extend I-44 with the assumed path
        i44.loc[p] = assumptions["i44_path"].get(str(p), i44.iloc[-1])
    i44_12m = i44.pct_change(12) * 100.0

    base = assumptions.get("imported_goods_baseline_yoy", 1.0)
    yoy = pd.Series(index=horizon, dtype=float)
    for p in horizon:
        contrib = sum(c * i44_12m.get(p - l, 0.0) for l, c in coeffs.items())
        yoy[p] = base + contrib
    # YoY -> approximate monthly changes (spread evenly)
    return yoy / 12.0 / 100.0


def project_rent(cpi_yoy_last_year: float, wage_norm: float,
                 horizon: pd.PeriodIndex) -> pd.Series:
    """Rent: CPI-indexed leases reprice to last year's CPI; the rest drifts
    with wages. Extremely forecastable -- this is the anchor of the sticky
    part of the basket."""
    annual = (config.RENT_CPI_INDEXATION_SHARE * cpi_yoy_last_year
              + config.RENT_WAGE_SHARE * wage_norm)
    return pd.Series(annual / 12.0 / 100.0, index=horizon)


def project_food(horizon: pd.PeriodIndex, assumptions: dict) -> pd.Series:
    """World food prices in NOK, released into CPI in the Feb/Jul repricing
    windows plus a small monthly drift."""
    annual = assumptions.get("food_pipeline_yoy", 3.0)
    drift = annual * 0.4 / 12.0 / 100.0
    window_jump = annual * 0.6 / 2.0 / 100.0
    out = pd.Series(drift, index=horizon)
    for p in horizon:
        if p.month in config.FOOD_ADJUSTMENT_MONTHS:
            out[p] += window_jump
    return out


def project_services(wage_norm: float, horizon: pd.PeriodIndex,
                     assumptions: dict) -> pd.Series:
    """Domestic services track the wage settlement with a margin haircut."""
    haircut = assumptions.get("services_wage_haircut", 0.75)
    return pd.Series(wage_norm * haircut / 12.0 / 100.0, index=horizon)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

COMPONENT_PROJECTORS = {
    "electricity": "energy",
    "other_energy": "energy",
}

STICKY_COMPONENTS = ["housing_ex_energy", "health", "education",
                     "restaurants_hotels", "misc", "communication",
                     "recreation", "transport_ex_fuel", "furnishings",
                     "clothing", "alcohol_tobacco"]


def build_cpi_projection(cpi_hist: pd.Series, horizon_months: int,
                         assumptions: dict, i44_hist: pd.Series,
                         weights: dict | None = None) -> pd.DataFrame:
    """Produce the projected total CPI index path + component contributions.

    cpi_hist: monthly total CPI index (observed).
    Returns DataFrame with columns: index level, yoy, and per-block
    contribution to monthly change.
    """
    w = weights or config.CPI_WEIGHTS_FALLBACK
    total_w = sum(w.values())
    last = cpi_hist.index[-1]
    horizon = pd.period_range(last + 1, periods=horizon_months, freq="M")

    wage_norm = assumptions["wage_norm_pct"]
    cpi_yoy_last = (cpi_hist.iloc[-1] / cpi_hist.iloc[-13] - 1) * 100.0

    blocks = {
        "electricity": project_electricity(None, horizon, assumptions),
        "other_energy": project_fuel(None, horizon, assumptions),
        "imported": project_imported_goods(horizon, i44_hist, assumptions),
        "rent_housing": project_rent(cpi_yoy_last, wage_norm, horizon),
        "food": project_food(horizon, assumptions),
        "services_other": project_services(wage_norm, horizon, assumptions),
    }
    block_weights = {
        "electricity": w["electricity"],
        "other_energy": w["other_energy"],
        "imported": w["clothing"] + w["furnishings"] + 0.5 * w["recreation"]
                    + 0.5 * w["transport_ex_fuel"],
        "rent_housing": w["housing_ex_energy"],
        "food": w["food"] + w["alcohol_tobacco"],
    }
    block_weights["services_other"] = total_w - sum(block_weights.values())

    rows = []
    level = cpi_hist.iloc[-1]
    for p in horizon:
        contribs = {b: block_weights[b] / total_w * blocks[b][p]
                    for b in blocks}
        mom = sum(contribs.values())
        level = level * (1.0 + mom)
        rows.append({"period": p, "cpi_index": level, "mom_pct": mom * 100,
                     **{f"contrib_{b}": v * 100 for b, v in contribs.items()}})

    proj = pd.DataFrame(rows).set_index("period")
    full = pd.concat([cpi_hist.rename("cpi_index").to_frame(), proj])
    full["yoy_pct"] = (full["cpi_index"] / full["cpi_index"].shift(12) - 1) * 100
    return full
