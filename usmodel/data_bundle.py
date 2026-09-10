"""
US data bundle.

Two ways to fill it:

  load_bundle()      REAL data cached by `python -m usmodel.fetch_data_us`
                     (FRED + Zillow ZORI). Both hosts are reachable from
                     the sandbox, so this is the default for the backtest
                     and the accuracy numbers quoted in the README.
  make_demo_bundle() synthetic-but-plausible series, for offline tests and
                     for validating the pipeline without a network. The
                     synthetic CPI is *coupled* to its drivers (oil, market
                     rents, the dollar, wages) the way real US CPI is, so
                     the plumbing sees signal rather than noise -- but no
                     accuracy claim may ever be quoted off it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "us"

# Activity indicators the GDP nowcast may use, if cached.
INDICATOR_NAMES = ("payrolls", "indpro", "retail", "claims", "cfnai",
                   "sentiment", "yield_curve", "hours",
                   "nfci", "credit_spread", "permits", "capex_orders",
                   "stocks", "real_m2", "housing_starts",
                   "real_pce", "real_income", "saving_rate")


@dataclass
class USDataBundle:
    cpi: pd.Series           # Period[M] CPI-U index level (current vintage)
    gdp: pd.Series           # Period[Q] real GDP level
    wti: pd.Series           # Period[M] WTI $/bbl
    dollar: pd.Series        # Period[M] broad USD index (higher = stronger)
    market_rent: pd.Series   # Period[M] market/new-lease rent index
    wages: pd.Series         # Period[M] avg hourly earnings index
    # Optional extras: used for diagnostics and for the NSA robustness
    # check (CPIAUCNS is never revised; CPIAUCSL's seasonal factors are).
    cpi_nsa: pd.Series | None = None
    cpi_core: pd.Series | None = None
    cpi_shelter: pd.Series | None = None
    cpi_supercore: pd.Series | None = None
    cpi_food: pd.Series | None = None
    cpi_energy: pd.Series | None = None
    gdp_vintages: pd.DataFrame | None = None
    # Growth-side activity indicators (Period[M]), keyed by the names in
    # fetch_data_us.INDICATOR_PUB_LAG_DAYS.
    indicators: dict[str, pd.Series] = field(default_factory=dict)
    # {series name: [months filled by interpolation]} -- see
    # fill_single_month_gaps; empty when the data had no holes.
    filled: dict[str, list[str]] = field(default_factory=dict)
    source: str = "unknown"

    def has_vintage_panel(self) -> bool:
        return self.gdp_vintages is not None and not self.gdp_vintages.empty


def _read_series(path: Path, freq: str) -> pd.Series:
    df = pd.read_csv(path)
    return pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                     index=pd.PeriodIndex(df["period"].astype(str),
                                          freq=freq)).dropna()


def fill_single_month_gaps(series: pd.Series,
                           max_gap: int = 1) -> tuple[pd.Series, list[str]]:
    """Complete a monthly index and fill isolated holes of up to `max_gap`
    months by geometric interpolation, returning the filled months.

    Why this exists: BLS never published the October 2025 CPI (the autumn
    2025 shutdown), and FRED carries the month as missing. Every positional
    12-row shift downstream then silently turned into a 13-month change for
    any window spanning the hole -- the live YoY read 3.54% where the true
    12-month rate was 3.30%. A geometric fill (the standard analyst
    treatment, and what the two-month change BLS did publish implies) keeps
    base effects on a calendar footing; the filled months are recorded on
    the bundle so any output can flag them. Longer gaps are left as NaN
    rather than invented."""
    if series.empty or series.index.freqstr not in ("M", "ME"):
        return series, []
    full = pd.period_range(series.index[0], series.index[-1], freq="M")
    if len(full) == len(series):
        return series, []
    out = series.reindex(full)
    missing = out.index[out.isna()]
    filled = []
    for m in missing:
        prev, nxt = m - 1, m + 1
        if prev in series.index and nxt in series.index and max_gap >= 1:
            out[m] = float(np.sqrt(series[prev] * series[nxt]))
            filled.append(str(m))
    return out.dropna(), filled


def load_bundle(data_dir: Path | str = DATA_DIR,
                start: str | None = "2004-01") -> USDataBundle:
    """Load the real cached US series. Raises if the required files are
    missing -- run `python -m usmodel.fetch_data_us` first."""
    d = Path(data_dir)
    required = ["cpi", "gdp", "wti", "dollar", "wages", "market_rent"]
    missing = [f for f in required if not (d / f"{f}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {d} - run `python -m usmodel.fetch_data_us`")

    filled: dict[str, list[str]] = {}

    def monthly(name: str) -> pd.Series | None:
        p = d / f"{name}.csv"
        if not p.exists():
            return None
        series, gaps = fill_single_month_gaps(_read_series(p, "M"))
        if gaps:
            filled[name] = gaps
        return series

    opt = monthly
    cpi = monthly("cpi")
    gdp = _read_series(d / "gdp.csv", "Q")
    if start:
        cpi = cpi[cpi.index >= pd.Period(start, "M")]
        gdp = gdp[gdp.index >= pd.Period(start, "M").asfreq("Q")]

    panel = None
    vpath = d / "gdp_vintages.csv"
    if vpath.exists():
        raw = pd.read_csv(vpath, index_col=0)
        raw.index = pd.PeriodIndex(raw.index.astype(str), freq="Q")
        raw.columns = pd.to_datetime(raw.columns)
        panel = raw

    return USDataBundle(
        cpi=cpi, gdp=gdp,
        wti=monthly("wti"),
        dollar=monthly("dollar"),
        wages=monthly("wages"),
        market_rent=monthly("market_rent"),
        cpi_nsa=opt("cpi_nsa"), cpi_core=opt("cpi_core"),
        cpi_shelter=opt("cpi_shelter"), cpi_supercore=opt("cpi_supercore"),
        cpi_food=opt("cpi_food"),
        cpi_energy=opt("cpi_energy"),
        gdp_vintages=panel,
        indicators={k: s for k in INDICATOR_NAMES
                    if (s := opt(k)) is not None and len(s)},
        filled=filled,
        source=f"real ({d})")


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
