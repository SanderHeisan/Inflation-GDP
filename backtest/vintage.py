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

from . import btconfig
from .btconfig import VintageConfig
from .data_bundle import RawDataBundle


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
    spot = market.loc[keep].iloc[-1]
    last_m = keep[-1]
    horizon = pd.period_range(last_m + 1, periods=horizon_months, freq="M")

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

    return {
        "power_recent_ore_kwh": float(spot["power_ore_kwh"]),
        "power_forward_ore_kwh": {str(p): float(spot["power_ore_kwh"])
                                  for p in horizon},
        "power_variable_share": 0.55,
        "brent_recent_usd": float(spot["brent_usd"]),
        "brent_forward_usd": {str(p): float(spot["brent_usd"])
                              for p in horizon},
        "usdnok_recent": float(spot["usdnok"]),
        "usdnok_path": {str(p): float(spot["usdnok"]) for p in horizon},
        "i44_path": {},                       # flat at last observed I-44
        "imported_goods_baseline_yoy": imported_yoy,
        "wage_norm_pct": wage_norm,
        "food_pipeline_yoy": food_yoy,
        "fuel_passthrough": 0.40,
        "services_wage_haircut": 0.75,
        # Calendar seasonality of MoM prints, estimated point-in-time from
        # the vintage CPI history inside the projection.
        "cpi_seasonality": True,
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
