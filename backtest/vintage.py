"""
Point-in-time reconstruction of the information set at an as-of date.

This module is the leakage firewall of the backtest: everything the model is
allowed to see at a given as-of date must come out of build_vintage(), and
nothing else from the raw bundle may reach the projection code.

Rules implemented here:
  * Publication lags: quarter Q's GDP exists only from ~40 days after the
    quarter ends; month M's CPI only from ~10 days after the month ends.
  * GDP revisions: 'realtime' replays a true vintage panel (Norges Bank
    real-time database); 'noise' simulates first releases by adding a
    persistent, deterministic N(0, sigma) error to current-vintage QoQ that
    decays as the quarter matures; 'none' uses current-vintage values.
    SSB CPI is never revised, so truncation alone is exact for inflation.
  * Market inputs: spot-carry. Forward curves are frozen at the last
    observed spot; realized future prices never enter the assumptions dict.
  * Wage norm: year Y's settlement norm only counts as known from mid-April
    of year Y; before that the prior year's norm applies.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quadmap import config as qconfig

from . import btconfig
from .btconfig import VintageConfig
from .data_bundle import RawDataBundle

# Electricity's approximate share of the CPI basket (per-mille weights).
W_ELEC = qconfig.CPI_WEIGHTS_FALLBACK["electricity"] / 1000.0


def consumer_power_price(power_ore: pd.Series) -> pd.Series:
    """Spot -> consumer-facing price through the stromstotte formula."""
    sub = qconfig.STROMSTOTTE
    excess = (power_ore - sub["threshold_ore_kwh"]).clip(lower=0.0)
    return power_ore - sub["coverage_share"] * excess


def _month_demeaned(s: pd.Series) -> pd.Series:
    med = s.groupby(s.index.month).transform("median")
    return s - med


def estimate_power_passthrough(cpi_vintage: pd.Series,
                               power_observed: pd.Series,
                               default: float = 0.35,
                               window_months: int = 96) -> float:
    """Effective monthly spot->CPI pass-through, estimated from this
    vintage's own history. Feeding raw Nord Pool swings through the naive
    0.55 'variable share' overshot badly in the backtest (a +2.4pp
    predicted MoM against +0.2 realized): fixed-price contracts, hedging
    and the subsidy damp what a spot move does to the index within the
    month. Regressing calendar-demeaned CPI MoM on calendar-demeaned
    consumer-price MoM (so seasonal winter power is not double counted
    with the seasonal overlay) lets each vintage set its own coefficient."""
    cp = consumer_power_price(power_observed)
    df = pd.concat({"x": cp.pct_change(), "y": cpi_vintage.pct_change()},
                   axis=1).dropna().iloc[-window_months:]
    if len(df) < 36:
        return default
    xd = _month_demeaned(df["x"])
    yd = _month_demeaned(df["y"])
    denom = float((xd ** 2).sum())
    if denom < 1e-6:          # e.g. flat pre-2021 proxy: no signal to fit
        return default
    beta = float((xd * yd).sum()) / denom
    return float(np.clip(beta / W_ELEC, 0.05, 0.55))


def ex_energy_seasonal_profile(cpi_vintage: pd.Series,
                               power_observed: pd.Series,
                               variable_share: float,
                               years: int = 10) -> dict[int, float]:
    """Calendar seasonality of CPI MoM with the modeled electricity
    contribution removed - the energy block carries the actual power path,
    so the overlay must only carry the NON-energy seasonal pattern."""
    cp = consumer_power_price(power_observed)
    df = pd.concat({"x": cp.pct_change(), "y": cpi_vintage.pct_change()},
                   axis=1).dropna().iloc[-years * 12:]
    ex = df["y"] - W_ELEC * variable_share * df["x"]
    overall = float(ex.median())
    return {m: float(ex[ex.index.month == m].median() - overall)
            if (ex.index.month == m).any() else 0.0
            for m in range(1, 13)}


# ---------------------------------------------------------------------------
# Availability rules
# ---------------------------------------------------------------------------

def first_release_date(period: pd.Period, lag_days: int) -> pd.Timestamp:
    """First calendar date on which data for `period` is public."""
    return period.end_time.normalize() + pd.Timedelta(days=lag_days)


def truncate(series: pd.Series, asof: pd.Timestamp, lag_days: int) -> pd.Series:
    """Drop every observation not yet published at `asof`."""
    keep = [p for p in series.index if first_release_date(p, lag_days) <= asof]
    return series.loc[keep].dropna()


# ---------------------------------------------------------------------------
# GDP revision models
# ---------------------------------------------------------------------------

def revision_draw(quarter: pd.Period, seed: int, sigma_pp: float) -> float:
    """Deterministic 'total revision' draw for a quarter, in percentage
    points of QoQ growth. Seeded from (seed, quarter) with a stable hash so
    the same quarter is mis-measured the same way at every as-of date and
    across runs (builtin hash() is randomized per process; sha256 is not)."""
    digest = hashlib.sha256(f"{seed}:{quarter}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return float(rng.normal(0.0, sigma_pp))


def apply_revision_noise(gdp_truncated: pd.Series,
                         cfg: VintageConfig) -> pd.Series:
    """Simulate the vintage as seen at the time: quarters younger than
    maturity get their QoQ growth perturbed by their persistent revision
    error, scaled down by decay**age (estimates converge to final as SSB
    revises). Levels are rebuilt by compounding from the last mature level,
    so the perturbation is internally consistent."""
    if gdp_truncated.empty:
        return gdp_truncated
    quarters = gdp_truncated.index
    last_q = quarters[-1]
    qoq = gdp_truncated.pct_change()

    out = gdp_truncated.copy()
    level = None
    for q in quarters:
        age = (last_q - q).n
        if age >= cfg.revision_maturity_quarters or q == quarters[0]:
            level = out.loc[q]
            continue
        err_pp = revision_draw(q, cfg.seed, cfg.revision_sigma_pp) \
            * cfg.revision_decay ** age
        level = level * (1.0 + qoq.loc[q] + err_pp / 100.0)
        out.loc[q] = level
    return out


def gdp_from_vintage_panel(panel: pd.DataFrame, asof: pd.Timestamp,
                           cfg: VintageConfig) -> pd.Series:
    """Latest true vintage published on or before `asof`. The publication-lag
    truncation is applied on top as a guard: a panel column mislabelled with
    an early date must still never reveal an unpublished quarter."""
    cols = [c for c in panel.columns if c <= asof]
    if not cols:
        return pd.Series(dtype=float)
    vint = panel[max(cols)].dropna()
    return truncate(vint, asof, cfg.gdp_pub_lag_days)


# ---------------------------------------------------------------------------
# Point-in-time market assumptions (spot-carry)
# ---------------------------------------------------------------------------

def spot_carry_assumptions(bundle: RawDataBundle, asof: pd.Timestamp,
                           cfg: VintageConfig, cpi_vintage: pd.Series,
                           horizon_months: int = 18) -> dict:
    """Assumptions dict as it could have been written on `asof`: every
    forward curve is flat at the last fully-observed monthly spot. This is
    the conservative point-in-time stance where no historical forward-curve
    archive exists — realized future prices must never appear here."""
    market = bundle.market
    keep = [p for p in market.index
            if first_release_date(p, cfg.market_pub_lag_days) <= asof]
    if not keep:
        raise ValueError(f"no market data observed before {asof}")
    observed = market.loc[keep]

    # Anchor at the last CPI month: that is the price level embedded in the
    # observed index. Months after it that are ALREADY OBSERVED in the
    # market data (spot prices publish daily; CPI lags ~40 days behind
    # them) enter the forward path at their actual values - this is the
    # June-2026 lesson: the power slide into the print month was public
    # information, and pure spot-carry (flat-at-anchor) threw it away.
    m0 = cpi_vintage.index[-1]
    anchor = observed.loc[m0] if m0 in observed.index else observed.iloc[-1]
    horizon = pd.period_range(m0 + 1, periods=horizon_months, freq="M")

    def fwd(col: str) -> dict:
        path, last_val = {}, float(anchor[col])
        for p in horizon:
            if p in observed.index:
                last_val = float(observed.loc[p, col])
            path[str(p)] = last_val   # observed where known, then carried
        return path
    spot = anchor

    # Wage norm: settlement for year Y known only from spring of year Y.
    norm_cutoff_year = asof.year if asof >= pd.Timestamp(
        asof.year, btconfig.WAGE_NORM_KNOWN_MONTH,
        btconfig.WAGE_NORM_KNOWN_DAY) else asof.year - 1
    if bundle.wage_norms is not None:
        known = bundle.wage_norms[bundle.wage_norms.index <= norm_cutoff_year]
        wage_norm = float(known.iloc[-1]) if len(known) else 3.5
    else:
        # Proxy: last observed CPI YoY + typical real-wage margin.
        yoy = (cpi_vintage.iloc[-1] / cpi_vintage.iloc[-13] - 1) * 100.0
        wage_norm = float(np.clip(yoy + 1.3, 2.0, 6.0))

    cpi_yoy_last = (cpi_vintage.iloc[-1] / cpi_vintage.iloc[-13] - 1) * 100.0

    def _subindex_yoy(series: pd.Series | None, default: float,
                      lo: float, hi: float) -> float:
        """Trailing YoY of a CPI sub-index, truncated like the CPI itself.
        This is how observable food / imported-goods momentum (the causes
        SSB named for the June-2026 miss) enters the forward assumptions
        instead of a static constant."""
        if series is None:
            return default
        s = truncate(series, asof, cfg.cpi_pub_lag_days)
        if len(s) < 13:
            return default
        return float(np.clip((s.iloc[-1] / s.iloc[-13] - 1) * 100.0, lo, hi))

    food_yoy = _subindex_yoy(bundle.cpi_food,
                             float(np.clip(cpi_yoy_last, 0.0, 6.0)),
                             -2.0, 8.0)
    imported_yoy = _subindex_yoy(bundle.cpi_imported, 1.0, -3.0, 6.0)

    vs_est = estimate_power_passthrough(cpi_vintage,
                                        observed["power_ore_kwh"])
    seasonal_profile = ex_energy_seasonal_profile(
        cpi_vintage, observed["power_ore_kwh"], vs_est)

    return {
        "power_recent_ore_kwh": float(spot["power_ore_kwh"]),
        "power_forward_ore_kwh": fwd("power_ore_kwh"),
        "power_variable_share": vs_est,
        "brent_recent_usd": float(spot["brent_usd"]),
        "brent_forward_usd": fwd("brent_usd"),
        "usdnok_recent": float(spot["usdnok"]),
        "usdnok_path": fwd("usdnok"),
        "i44_path": {},                       # flat at last observed I-44
        "imported_goods_baseline_yoy": imported_yoy,
        "wage_norm_pct": wage_norm,
        "food_pipeline_yoy": food_yoy,
        "fuel_passthrough": 0.40,
        "services_wage_haircut": 0.75,
        # Calendar seasonality of MoM prints: the ex-energy profile, since
        # the electricity block already carries the actual power path.
        "cpi_seasonality": True,
        "cpi_seasonal_profile": seasonal_profile,
    }


# ---------------------------------------------------------------------------
# The vintage dataset
# ---------------------------------------------------------------------------

@dataclass
class VintageDataset:
    asof: pd.Timestamp
    gdp_level: pd.Series
    cpi_index: pd.Series
    i44: pd.Series
    assumptions: dict
    indicators: dict
    revision_mode: str            # 'realtime' | 'noise' | 'none'
    last_gdp_quarter: pd.Period = field(init=False)
    last_cpi_month: pd.Period = field(init=False)

    def __post_init__(self):
        self.last_gdp_quarter = self.gdp_level.index[-1]
        self.last_cpi_month = self.cpi_index.index[-1]


def resolve_revision_mode(bundle: RawDataBundle, cfg: VintageConfig) -> str:
    if cfg.revision_mode == "auto":
        return "realtime" if bundle.has_vintage_panel() else "noise"
    if cfg.revision_mode == "realtime" and not bundle.has_vintage_panel():
        raise ValueError("revision_mode='realtime' requires a GDP vintage "
                         "panel (data/gdp_vintages.csv) in the bundle")
    return cfg.revision_mode


def build_vintage(bundle: RawDataBundle, asof: pd.Timestamp | str,
                  cfg: VintageConfig | None = None,
                  horizon_months: int = 18) -> VintageDataset:
    """Reconstruct the information set at `asof`."""
    cfg = cfg or VintageConfig()
    asof = pd.Timestamp(asof)
    mode = resolve_revision_mode(bundle, cfg)

    if mode == "realtime":
        gdp = gdp_from_vintage_panel(bundle.gdp_vintages, asof, cfg)
    else:
        gdp = truncate(bundle.gdp_final, asof, cfg.gdp_pub_lag_days)
        if mode == "noise":
            gdp = apply_revision_noise(gdp, cfg)

    cpi = truncate(bundle.cpi, asof, cfg.cpi_pub_lag_days)
    i44 = truncate(bundle.i44, asof, cfg.market_pub_lag_days)

    if gdp.empty or len(cpi) < 14:
        raise ValueError(f"not enough published history at {asof.date()}")

    assumptions = spot_carry_assumptions(bundle, asof, cfg, cpi,
                                         horizon_months=horizon_months)

    # No point-in-time archive of Regional Network / PMI / retail prints is
    # wired in, so the nowcast blend degrades gracefully to its momentum
    # leg (nowcast_qoq renormalizes over the indicators present).
    indicators: dict = {}

    return VintageDataset(asof=asof, gdp_level=gdp, cpi_index=cpi, i44=i44,
                          assumptions=assumptions, indicators=indicators,
                          revision_mode=mode)
