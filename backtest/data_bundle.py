"""
Raw data for the backtest, current-vintage ("final") plus whatever true
vintage material is available. The bundle is the *superset* of information;
backtest/vintage.py is the only module allowed to slice it down to what was
knowable at an as-of date, and the engine only ever sees those slices.

Two loaders:
  load_bundle(data_dir)  - cached CSVs written by backtest/fetch_data.py
  make_demo_bundle(seed) - synthetic but economically plausible series so the
                           whole harness runs and is testable offline
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import btconfig


@dataclass
class RawDataBundle:
    gdp_final: pd.Series           # Period[Q]; mainland GDP level, current vintage
    cpi: pd.Series                 # Period[M]; total CPI index (SSB CPI is unrevised)
    i44: pd.Series                 # Period[M]; import-weighted krone index
    market: pd.DataFrame           # Period[M]; power_ore_kwh, brent_usd, usdnok
    gdp_vintages: pd.DataFrame | None = None  # rows Period[Q] x cols vintage Timestamp
    wage_norms: pd.Series | None = None       # int year -> settlement norm (pct)
    # Optional CPI sub-indices: observable momentum for the two blocks whose
    # static assumptions caused the June-2026 level miss (food, imported
    # goods/electronics). When present, the vintage builder anchors the
    # forward assumptions on their trailing trends instead of constants.
    cpi_food: pd.Series | None = None          # Period[M] food sub-index
    cpi_imported: pd.Series | None = None      # Period[M] imported-goods proxy
    source: str = "unknown"

    def has_vintage_panel(self) -> bool:
        return self.gdp_vintages is not None and not self.gdp_vintages.empty


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _read_period_series(path: Path, freq: str) -> pd.Series:
    df = pd.read_csv(path)
    s = pd.Series(df["value"].to_numpy(dtype=float),
                  index=pd.PeriodIndex(df["period"], freq=freq))
    return s.sort_index()


def load_bundle(data_dir: str | Path = btconfig.DATA_DIR) -> RawDataBundle:
    d = Path(data_dir)
    f = btconfig.DATA_FILES
    for key in ("gdp", "cpi", "i44", "market"):
        if not (d / f[key]).exists():
            raise FileNotFoundError(
                f"{d / f[key]} missing. Run backtest/fetch_data.py on a "
                "machine with network access, or use --demo.")

    market = pd.read_csv(d / f["market"])
    market.index = pd.PeriodIndex(market.pop("period"), freq="M")

    vintages = None
    if (d / f["gdp_vintages"]).exists():
        vintages = pd.read_csv(d / f["gdp_vintages"], index_col=0)
        vintages.index = pd.PeriodIndex(vintages.index, freq="Q")
        vintages.columns = pd.to_datetime(vintages.columns)
        vintages = vintages.sort_index().sort_index(axis=1)

    norms = None
    if (d / f["wage_norms"]).exists():
        wdf = pd.read_csv(d / f["wage_norms"])
        norms = pd.Series(wdf["norm_pct"].to_numpy(dtype=float),
                          index=wdf["year"].astype(int)).sort_index()

    def _optional(key):
        p = d / f[key]
        return _read_period_series(p, "M") if p.exists() else None

    return RawDataBundle(
        gdp_final=_read_period_series(d / f["gdp"], "Q"),
        cpi=_read_period_series(d / f["cpi"], "M"),
        i44=_read_period_series(d / f["i44"], "M"),
        market=market.sort_index(),
        gdp_vintages=vintages,
        wage_norms=norms,
        cpi_food=_optional("cpi_food"),
        cpi_imported=_optional("cpi_imported"),
        source="csv",
    )


# ---------------------------------------------------------------------------
# Synthetic demo bundle. Not Norway's actual history, but reproduces the
# dynamics that matter for the harness: trend growth with two recessions,
# an inflation wave, an energy price spike, a persistently weakening krone.
# ---------------------------------------------------------------------------

def make_demo_bundle(seed: int = 7, start: str = "2004-01",
                     end: str = "2026-06") -> RawDataBundle:
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    t = np.arange(len(months), dtype=float)

    def bump(center_m: str, width: float, height: float) -> np.ndarray:
        c = months.get_loc(pd.Period(center_m, freq="M"))
        return height * np.exp(-((t - c) / width) ** 2)

    # CPI: 2% trend, a 2008-ish blip, a large 2022 wave, noise.
    mom = (0.021 / 12
           + bump("2008-07", 8, 0.02 / 12)
           + bump("2022-09", 14, 0.045 / 12)
           + 0.008 / 12 * np.sin(t / 9.0)
           + rng.normal(0, 0.0012, len(months)))
    cpi = pd.Series(85.0 * np.cumprod(1 + mom), index=months, name="cpi")

    # GDP: quarterly, trend ~1.6% ann., recessions in 2009 and 2020, cycle.
    quarters = pd.period_range(pd.Period(start, "M").asfreq("Q"),
                               pd.Period(end, "M").asfreq("Q"), freq="Q")
    tq = np.arange(len(quarters), dtype=float)

    def qbump(center_q: str, width: float, height: float) -> np.ndarray:
        c = quarters.get_loc(pd.Period(center_q, freq="Q"))
        return height * np.exp(-((tq - c) / width) ** 2)

    qoq = (0.004
           + qbump("2009Q1", 2.0, -0.020)
           + qbump("2020Q2", 1.2, -0.045)
           + qbump("2021Q2", 2.5, 0.012)
           + 0.003 * np.sin(tq / 4.0)
           + rng.normal(0, 0.0020, len(quarters)))
    gdp = pd.Series(600.0 * np.cumprod(1 + qoq), index=quarters, name="gdp")

    # I-44: drifting weaker with shocks (higher = weaker NOK).
    i44 = pd.Series(90.0 + 0.045 * t + np.cumsum(rng.normal(0, 0.55, len(months)))
                    + bump("2020-03", 3, 8.0),
                    index=months, name="I44")

    # Market spots: power spike 2021-22, oil crash 2015 + 2020, NOK weakening.
    power = (55.0 + bump("2022-08", 10, 160.0) + bump("2010-02", 6, 30.0)
             + 12.0 * np.sin(2 * np.pi * (t % 12) / 12.0)   # winter seasonality
             + rng.normal(0, 6.0, len(months))).clip(15.0)
    brent = (70.0 + bump("2008-07", 6, 60.0) - bump("2015-12", 10, 35.0)
             - bump("2020-04", 4, 45.0) + bump("2022-06", 8, 40.0)
             + np.cumsum(rng.normal(0, 1.4, len(months))) * 0.3).clip(18.0)
    usdnok = 6.2 + 0.020 * t + bump("2020-03", 3, 1.6) \
        + np.cumsum(rng.normal(0, 0.05, len(months))) * 0.3

    market = pd.DataFrame({"power_ore_kwh": power, "brent_usd": brent,
                           "usdnok": usdnok}, index=months)

    years = range(quarters[0].year, quarters[-1].year + 1)
    cpi_yoy = cpi.pct_change(12) * 100
    norms = {}
    for y in years:
        p = pd.Period(f"{y - 1}-12", freq="M")
        base = cpi_yoy.get(p)
        if base is None or base != base:
            base = 3.0
        norms[y] = float(np.clip(base + 1.3, 2.0, 6.0))
    wage_norms = pd.Series(norms)

    # Sub-index demo series: food noisier around the total, imported goods
    # softer (mirrors Norway: imported deflation through strong-NOK spells).
    food = pd.Series(
        cpi.to_numpy() * (1 + 0.02 * np.sin(t / 7.0)
                          + rng.normal(0, 0.004, len(months))),
        index=months)
    imported = pd.Series(
        cpi.to_numpy() * (1 - 0.0005 * t / 12
                          + rng.normal(0, 0.003, len(months))),
        index=months)

    return RawDataBundle(gdp_final=gdp, cpi=cpi, i44=i44, market=market,
                         gdp_vintages=None, wage_norms=wage_norms,
                         cpi_food=food, cpi_imported=imported,
                         source="demo")


def synthesize_vintage_panel(gdp_final: pd.Series, sigma_pp: float = 0.25,
                             decay: float = 0.6, maturity_quarters: int = 8,
                             pub_lag_days: int = btconfig.GDP_PUB_LAG_DAYS,
                             seed: int = 0) -> pd.DataFrame:
    """Build a synthetic true-vintage panel from a final series (for tests and
    demo of 'realtime' mode). Each quarter gets a persistent revision error
    that decays across successive vintages, mimicking how first releases
    converge to final."""
    from . import vintage as vt   # local import to avoid a cycle

    quarters = gdp_final.index
    vintage_dates = [vt.first_release_date(q, pub_lag_days) for q in quarters]
    qoq = gdp_final.pct_change()

    cols = {}
    for vdate, last_q in zip(vintage_dates, quarters):
        known = quarters[quarters <= last_q]
        levels, level = [], None
        for q in known:
            age = (last_q - q).n
            err = 0.0
            if age < maturity_quarters:
                err = vt.revision_draw(q, seed, sigma_pp) * decay ** age / 100.0
            if level is None:
                level = gdp_final.loc[q]
            else:
                level = level * (1.0 + qoq.loc[q] + err)
            levels.append(level)
        cols[vdate] = pd.Series(levels, index=known)
    panel = pd.DataFrame(cols)
    panel.columns = pd.to_datetime(panel.columns)
    return panel
