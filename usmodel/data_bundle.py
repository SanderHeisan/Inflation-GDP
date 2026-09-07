"""
US data bundle: synthetic-but-plausible series so the whole US pipeline runs
and is testable offline (FRED/BLS are unreachable from the sandbox, exactly
like SSB was). Real data drops in via fetch_data_us.py on a networked machine.

The synthetic CPI is *coupled* to its drivers (oil, market rents, the dollar,
wages) the way real US CPI is, so the model sees genuine signal rather than
noise -- otherwise a backtest would be meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class USDataBundle:
    cpi: pd.Series           # Period[M] CPI-U index level (current vintage)
    gdp: pd.Series           # Period[Q] real GDP level
    wti: pd.Series           # Period[M] WTI $/bbl
    dollar: pd.Series        # Period[M] broad USD index (higher = stronger)
    market_rent: pd.Series   # Period[M] market/new-lease rent index
    wages: pd.Series         # Period[M] avg hourly earnings index
    source: str = "unknown"


def make_demo_bundle(seed: int = 11, start: str = "2004-01",
                     end: str = "2026-08") -> USDataBundle:
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    t = np.arange(len(months), dtype=float)

    def bump(center: str, width: float, height: float) -> np.ndarray:
        c = months.get_loc(pd.Period(center, freq="M"))
        return height * np.exp(-((t - c) / width) ** 2)

    # WTI: 2008 spike, 2014-16 crash, 2020 COVID crash, 2022 war spike.
    wti = (68.0 + bump("2008-06", 6, 70.0) - bump("2015-12", 12, 35.0)
           - bump("2020-04", 3, 45.0) + bump("2022-05", 7, 45.0)
           + np.cumsum(rng.normal(0, 1.6, len(months))) * 0.25).clip(16.0)
    wti = pd.Series(wti, index=months)

    # Broad USD: stronger through 2015 and 2022.
    dollar = pd.Series(100.0 + 0.03 * t + bump("2015-03", 10, 6.0)
                       + bump("2022-09", 8, 9.0)
                       + np.cumsum(rng.normal(0, 0.15, len(months))) * 0.3,
                       index=months)

    # Market ("new-lease") rents: the US rent boom of 2021-22 then cooling.
    mr_yoy = (3.2 + bump("2021-11", 8, 13.0) - bump("2024-02", 10, 2.0)
              + 0.6 * np.sin(t / 15.0) + rng.normal(0, 0.4, len(months)))
    market_rent = pd.Series(100.0 * np.cumprod(1 + mr_yoy / 100.0 / 12.0),
                            index=months)

    # Wages (AHE index): ~3% pre-2021, ~5.5% 2022 peak, ~4% now.
    wage_yoy = (3.0 + bump("2022-03", 10, 2.6) + rng.normal(0, 0.2, len(months)))
    wages = pd.Series(100.0 * np.cumprod(1 + wage_yoy / 100.0 / 12.0),
                      index=months)

    # ---- Build CPI by coupling to the drivers (so the model has signal) ----
    w = config.CPI_WEIGHTS
    tw = sum(w.values())
    dwti = wti.diff().fillna(0.0)
    gas_mom = (config.OIL_HEADLINE_BPS_PER_DOLLAR / 1e4) * dwti / (w["gasoline"] / tw)
    shelter_yoy = (config.SHELTER_PASSTHROUGH
                   * (market_rent / market_rent.shift(12) - 1.0) * 100.0
                   + (1 - config.SHELTER_PASSTHROUGH) * config.SHELTER_TREND_YOY)
    shelter_mom = (shelter_yoy / 12.0 / 100.0).fillna(config.SHELTER_TREND_YOY
                                                      / 12.0 / 100.0)
    dxy_12 = (dollar.pct_change(12) * 100.0).fillna(0.0)
    core_goods_mom = (config.CORE_GOODS_BASELINE_YOY / 100.0
                      + sum(c * dxy_12.shift(l) for l, c in
                            config.DOLLAR_PASSTHROUGH_LAGS.items()).fillna(0)
                      / 100.0) / 12.0
    supercore_mom = (wage_yoy * config.SUPERCORE_WAGE_PASSTHROUGH
                     / 100.0 / 12.0)
    food_mom = pd.Series(config.FOOD_TREND_YOY / 100.0 / 12.0, index=months)
    med_mom = pd.Series(config.MEDICAL_TREND_YOY / 100.0 / 12.0, index=months)
    elec_mom = pd.Series(0.002 / 12.0, index=months)

    block_mom = {
        "gasoline": gas_mom, "electricity": elec_mom, "energy_other": elec_mom,
        "shelter": shelter_mom, "food_at_home": food_mom, "food_away": food_mom,
        "core_goods": core_goods_mom, "supercore": supercore_mom,
        "medical": med_mom,
    }
    mom = sum(w[b] / tw * block_mom[b] for b in block_mom)
    mom = mom + rng.normal(0, 0.0006, len(months))   # idiosyncratic noise
    cpi = pd.Series(190.0 * np.cumprod(1 + mom.fillna(0.0)), index=months,
                    name="cpi")

    # Real GDP (quarterly): recessions 2008-09, 2020; cycle.
    quarters = pd.period_range(pd.Period(start, "M").asfreq("Q"),
                               pd.Period(end, "M").asfreq("Q"), freq="Q")
    tq = np.arange(len(quarters), dtype=float)

    def qbump(center: str, width: float, height: float) -> np.ndarray:
        c = quarters.get_loc(pd.Period(center, freq="Q"))
        return height * np.exp(-((tq - c) / width) ** 2)

    qoq = (0.0045 + qbump("2009Q1", 2.0, -0.022) + qbump("2020Q2", 1.1, -0.085)
           + qbump("2021Q2", 2.5, 0.020) + 0.003 * np.sin(tq / 4.0)
           + rng.normal(0, 0.002, len(quarters)))
    gdp = pd.Series(15000.0 * np.cumprod(1 + qoq), index=quarters, name="gdp")

    return USDataBundle(cpi=cpi, gdp=gdp, wti=wti, dollar=dollar,
                        market_rent=market_rent, wages=wages, source="demo")


def assumptions_from_bundle(bundle: USDataBundle,
                            asof_cpi_month: pd.Period | None = None) -> dict:
    """Spot-carry assumptions as of the last observed month: WTI/dollar
    frozen at last spot, wage growth and market-rent momentum read off the
    observed series. Realized-future leakage is the vintage builder's job
    (this helper is for the demo runner)."""
    m = asof_cpi_month or bundle.cpi.index[-1]
    wage_yoy = (bundle.wages.loc[m] / bundle.wages.shift(12).loc[m] - 1) * 100
    mr_yoy = (bundle.market_rent.loc[m]
              / bundle.market_rent.shift(12).loc[m] - 1) * 100
    return {
        "wti_recent": float(bundle.wti.loc[m]),
        "wti_forward": {},                       # flat strip in the demo
        "dollar_recent": float(bundle.dollar.loc[m]),
        "dollar_path": {},
        "wage_growth_pct": float(wage_yoy),
        "market_rent_yoy_recent": float(mr_yoy),
        "food_pipeline_yoy": config.FOOD_TREND_YOY,
        "core_goods_baseline_yoy": config.CORE_GOODS_BASELINE_YOY,
    }
