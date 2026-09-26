"""
Point-in-time reconstruction of the US information set at an as-of date.

This is the leakage firewall: everything the model may see at a given as-of
date comes out of build_vintage(), and nothing else from the raw bundle
reaches the projection code.

Rules implemented here:
  * Publication lags (usbacktest.btconfig): CPI ~13 days after month end,
    BEA advance GDP ~28 days after quarter end, average hourly earnings ~8
    days, Zillow ZORI ~20 days, monthly market averages only once the month
    is over.
  * GDP revisions: 'realtime' replays a true vintage panel if one is
    supplied; 'noise' (the default here, because ALFRED and the Philly Fed
    real-time set are unreachable from this sandbox) simulates first
    releases with a persistent, deterministic, BEA-calibrated revision error
    per quarter that decays as the quarter matures; 'none' hands over final
    data and so measures an upper bound rather than a forecast.
  * Market inputs: observed-then-carried. Months already observed at the
    as-of date enter at their actual values -- WTI and the dollar print
    daily while CPI lags ~6 weeks behind them, so a month of oil that the
    forecaster genuinely saw is legitimate information -- and everything
    beyond the last observation is carried flat. Realized *future* prices
    never enter.
  * Self-calibration: the shelter pass-through and the supercore wage
    haircut are re-estimated on each vintage's own published history, so
    the coefficients are point-in-time rather than fitted on the full
    sample. Set USVintageConfig(self_calibrate=False) to run the static
    config constants instead.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quadmap import quads
from usmodel import config as uconfig, gdp as gdp_mod, nowcast, rates
from usmodel.data_bundle import USDataBundle
from usmodel.fetch_data_us import INDICATOR_PUB_LAG_DAYS

from . import btconfig
from .btconfig import USVintageConfig


# ---------------------------------------------------------------------------
# Availability rules
# ---------------------------------------------------------------------------

def first_release_date(period: pd.Period, lag_days: int) -> pd.Timestamp:
    """First calendar date on which data for `period` is public."""
    return period.end_time.normalize() + pd.Timedelta(days=lag_days)


def truncate(series: pd.Series | None, asof: pd.Timestamp,
             lag_days: int) -> pd.Series:
    """Drop every observation not yet published at `asof`. Vectorized over
    the index: the walk-forward calls this for every series at every as-of
    date, and a Python loop over periods dominated the runtime."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    released = series.index.end_time.normalize() + pd.Timedelta(days=lag_days)
    return series[released <= pd.Timestamp(asof)].dropna()


# ---------------------------------------------------------------------------
# GDP revision models
# ---------------------------------------------------------------------------

def revision_draw(quarter: pd.Period, seed: int, sigma_pp: float) -> float:
    """Deterministic 'total revision' draw for a quarter, in percentage
    points of QoQ growth. Seeded from (seed, quarter) with a stable hash so
    a quarter is mis-measured identically at every as-of date and across
    runs (builtin hash() is salted per process; sha256 is not)."""
    digest = hashlib.sha256(f"{seed}:{quarter}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return float(rng.normal(0.0, sigma_pp))


def apply_revision_noise(gdp_truncated: pd.Series,
                         cfg: USVintageConfig) -> pd.Series:
    """Simulate the vintage as published at the time: quarters younger than
    maturity get their QoQ growth perturbed by their persistent revision
    error, scaled by decay**age (BEA estimates converge toward final).
    Levels are rebuilt by compounding from the last mature level, so the
    perturbation stays internally consistent."""
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
        err_pp = (revision_draw(q, cfg.seed, cfg.revision_sigma_pp)
                  * cfg.revision_decay ** age)
        level = level * (1.0 + qoq.loc[q] + err_pp / 100.0)
        out.loc[q] = level
    return out


def gdp_from_vintage_panel(panel: pd.DataFrame, asof: pd.Timestamp,
                           cfg: USVintageConfig) -> pd.Series:
    """Latest true vintage published on or before `asof`, with the
    publication-lag truncation applied on top as a guard."""
    cols = [c for c in panel.columns if c <= asof]
    if not cols:
        return pd.Series(dtype=float)
    return truncate(panel[max(cols)].dropna(), asof, cfg.gdp_pub_lag_days)


# ---------------------------------------------------------------------------
# Point-in-time self-calibration
# ---------------------------------------------------------------------------

def estimate_shelter_passthrough(cpi_shelter: pd.Series,
                                 market_rent: pd.Series,
                                 default: float = uconfig.SHELTER_PASSTHROUGH,
                                 lag: int = uconfig.SHELTER_MARKET_RENT_LAG_M
                                 ) -> tuple[float, float]:
    """Regress observed CPI-shelter YoY on market-rent YoY lagged `lag`
    months, using only this vintage's published history:

        shelter_yoy_t = a + b * market_rent_yoy_{t-lag}

    Returns (b, a) -- the pass-through and the intercept that plays the role
    of config.SHELTER_TREND_YOY. Falls back to the config constants when the
    vintage is too short to fit (ZORI history is what binds early on)."""
    if cpi_shelter is None or market_rent is None:
        return default, uconfig.SHELTER_TREND_YOY
    sh = quads.calendar_pct_change(cpi_shelter, 12)
    mr = quads.calendar_pct_change(market_rent, 12)
    mr.index = mr.index + lag                       # align driver to target
    df = pd.concat({"y": sh, "x": mr}, axis=1).dropna()
    if len(df) < 24 or float(df["x"].var()) < 1e-6:
        return default, uconfig.SHELTER_TREND_YOY
    b, a = np.polyfit(df["x"].to_numpy(float), df["y"].to_numpy(float), 1)
    # Guard against a degenerate fit on a short window.
    b = float(np.clip(b, 0.05, 1.2))
    a = float(np.clip(a, -2.0, 6.0))
    return b, a


def estimate_supercore_passthrough(cpi_supercore: pd.Series,
                                   wage_growth: pd.Series,
                                   default: float
                                   = uconfig.SUPERCORE_WAGE_PASSTHROUGH
                                   ) -> tuple[float, float]:
    """Same idea for core services ex shelter against wage growth:
    supercore_yoy = a + b * wage_yoy, fitted on published history only.

    `wage_growth` is a series ALREADY IN PERCENT, so the caller decides what
    "wage growth" means - a 12-month change of the average-hourly-earnings
    index, or the Atlanta Fed tracker, which is published as a rate. The
    coefficient is refitted per vintage either way, so swapping the series
    swaps the fit with it rather than carrying a constant tuned to the
    other one."""
    if cpi_supercore is None or wage_growth is None:
        return default, 0.0
    sc = quads.calendar_pct_change(cpi_supercore, 12)
    wg = pd.Series(wage_growth).astype(float)
    df = pd.concat({"y": sc, "x": wg}, axis=1).dropna()
    if len(df) < 36 or float(df["x"].var()) < 1e-6:
        return default, 0.0
    b, a = np.polyfit(df["x"].to_numpy(float), df["y"].to_numpy(float), 1)
    return float(np.clip(b, 0.0, 2.0)), float(np.clip(a, -3.0, 6.0))


def estimate_gasoline_seasonal(cpi_gasoline: pd.Series | None,
                               pump: pd.Series | None, years: int = 15
                               ) -> tuple[dict[int, float], float]:
    """BLS's seasonal factor on gasoline, read off published data: the
    median gap by calendar month between the SA gasoline index's MoM and
    the retail pump price's MoM (pp), and the slope of the former on the
    latter after that seasonal is removed. Point-in-time: both inputs are
    vintage-truncated. Returns ({month: pp}, slope); empty/1.0 when there
    is not enough history."""
    if cpi_gasoline is None or pump is None or cpi_gasoline.empty or pump.empty:
        return {}, 1.0
    g = quads.calendar_pct_change(cpi_gasoline, 1)
    pmom = quads.calendar_pct_change(pump, 1)
    df = pd.concat({"g": g, "p": pmom}, axis=1).dropna().iloc[-years * 12:]
    df = df[~df.index.isin(pd.period_range("2020-03", "2020-06", freq="M"))]
    if len(df) < 48:
        return {}, 1.0
    gap = df["g"] - df["p"]
    seasonal = {m: float(gap[df.index.month == m].median())
                for m in range(1, 13) if (df.index.month == m).any()}
    adj = df["g"] - pd.Series([seasonal.get(m, 0.0) for m in df.index.month],
                              index=df.index)
    denom = float((df["p"] ** 2).sum())
    slope = float((adj * df["p"]).sum() / denom) if denom > 1e-9 else 1.0
    return seasonal, float(np.clip(slope, 0.5, 1.2))


# ---------------------------------------------------------------------------
# Point-in-time assumptions (observed-then-carried)
# ---------------------------------------------------------------------------

def _observed_then_carried(series: pd.Series, anchor_month: pd.Period,
                           horizon: pd.PeriodIndex) -> dict[str, float]:
    """Forward path for a market series: actual values for months already
    observed at the as-of date, then flat at the last observation."""
    path, last_val = {}, float(series.loc[anchor_month]) \
        if anchor_month in series.index else float(series.iloc[-1])
    for p in horizon:
        if p in series.index:
            last_val = float(series.loc[p])
        path[str(p)] = last_val
    return path


def wage_growth_vintage(bundle: USDataBundle, asof: pd.Timestamp,
                        cfg: USVintageConfig) -> tuple[pd.Series, str]:
    """The wage-growth series in percent as published at `asof`, and which
    source it came from.

    'ahe' is the published basis: the 12-month change of average hourly
    earnings, a mean across whoever is on payrolls, so it moves when the
    COMPOSITION of employment changes (April 2020: +8.1% with nobody given a
    raise). 'tracker' is the Atlanta Fed Wage Growth Tracker, the median
    12-month growth of individuals observed in both periods, which is
    already a rate and needs no differencing. Falling back to AHE when the
    tracker is absent keeps a vintage before the tracker's history, or a
    checkout without the file, on the published basis rather than failing.
    """
    want = str(getattr(cfg, "wage_source", "ahe") or "ahe").lower()
    if want == "tracker":
        tracker = truncate(getattr(bundle, "wage_tracker", None), asof,
                           getattr(cfg, "wage_tracker_pub_lag_days",
                                   btconfig.WAGE_TRACKER_PUB_LAG_DAYS))
        if tracker is not None and len(tracker) >= 36:
            return tracker.astype(float), "tracker"
    ahe = truncate(bundle.wages, asof, cfg.wage_pub_lag_days)
    return quads.calendar_pct_change(ahe, 12).dropna(), "ahe"


def _last_clipped(series: pd.Series, default: float,
                  lo: float, hi: float) -> float:
    """The last published value of a series that is already a rate."""
    if series is None or not len(series):
        return default
    return float(np.clip(float(series.iloc[-1]), lo, hi))


def _yoy_last(series: pd.Series, default: float,
              lo: float, hi: float) -> float:
    if series is None or len(series) < 13:
        return default
    last = series.index[-1]
    if (last - 12) not in series.index:
        return default
    return float(np.clip((series.iloc[-1] / series[last - 12] - 1) * 100.0,
                         lo, hi))


def vintage_assumptions(bundle: USDataBundle, asof: pd.Timestamp,
                        cfg: USVintageConfig, cpi_vintage: pd.Series,
                        horizon_months: int = 20) -> tuple[dict, dict]:
    """Assumptions dict as it could have been written on `asof`, plus the
    calibration diagnostics that produced it."""
    # A live run may see the month in progress for daily/weekly series.
    market_lag = -31 if cfg.live_partial_month else cfg.market_pub_lag_days
    wti = truncate(bundle.wti, asof, market_lag)
    dollar = truncate(bundle.dollar, asof, market_lag)
    pump = truncate((bundle.indicators or {}).get("gasoline_retail"), asof,
                    market_lag)
    cpi_gasoline = truncate(getattr(bundle, "cpi_gasoline", None), asof,
                            cfg.cpi_pub_lag_days)
    wage_growth, wage_source = wage_growth_vintage(bundle, asof, cfg)
    rent = truncate(bundle.market_rent, asof, cfg.rent_pub_lag_days)
    cpi_food = truncate(bundle.cpi_food, asof, cfg.cpi_pub_lag_days)
    cpi_shelter = truncate(bundle.cpi_shelter, asof, cfg.cpi_pub_lag_days)
    cpi_supercore = truncate(getattr(bundle, "cpi_supercore", None), asof,
                             cfg.cpi_pub_lag_days)

    if wti.empty or dollar.empty:
        raise ValueError(f"no market data observed before {asof.date()}")

    # Anchor at the last published CPI month: that is the price level
    # already embedded in the observed index. Months after it that the
    # forecaster HAD seen (oil and FX print daily) enter at their real
    # values; everything past the last observation is carried flat.
    m0 = cpi_vintage.index[-1]
    horizon = pd.period_range(m0 + 1, periods=horizon_months, freq="M")
    wti_path = _observed_then_carried(wti, m0, horizon)
    dollar_path = _observed_then_carried(dollar, m0, horizon)
    # observed months only (unlike the carried WTI/dollar paths): the block
    # treats the months in this dict as known and the rest as flat-SA.
    pump_path = ({str(p): float(pump.loc[p]) for p in horizon if p in pump.index}
                 if not pump.empty and m0 in pump.index else {})
    gas_seasonal, gas_slope = estimate_gasoline_seasonal(cpi_gasoline, pump)

    wage_yoy = _last_clipped(wage_growth, 3.5, 0.0, 9.0)
    rent_yoy = _yoy_last(rent, uconfig.SHELTER_TREND_YOY, -6.0, 20.0)
    food_yoy = _yoy_last(cpi_food, uconfig.FOOD_TREND_YOY, -2.0, 14.0)

    if cfg.self_calibrate:
        sh_b, sh_a = estimate_shelter_passthrough(cpi_shelter, rent)
        sc_b, sc_a = estimate_supercore_passthrough(cpi_supercore, wage_growth)
    else:
        sh_b, sh_a = uconfig.SHELTER_PASSTHROUGH, uconfig.SHELTER_TREND_YOY
        sc_b, sc_a = uconfig.SUPERCORE_WAGE_PASSTHROUGH, 0.0
    # Research override: a pass-through frozen outside the vintage, so the
    # refit cannot quietly absorb what the wage series gets wrong. Shelter is
    # untouched, which is the point - it isolates the supercore question.
    if (cfg.supercore_passthrough_fixed is not None
            and cfg.supercore_intercept_fixed is not None):
        sc_b = float(cfg.supercore_passthrough_fixed)
        sc_a = float(cfg.supercore_intercept_fixed)

    assumptions = {
        "wti_recent": float(wti.loc[m0]) if m0 in wti.index
        else float(wti.iloc[-1]),
        "wti_forward": wti_path,
        "dollar_recent": float(dollar.loc[m0]) if m0 in dollar.index
        else float(dollar.iloc[-1]),
        "dollar_path": dollar_path,
        "gasoline_pump_recent": (float(pump.loc[m0]) if not pump.empty
                                 and m0 in pump.index else None),
        "gasoline_pump_path": pump_path,
        "gasoline_seasonal": gas_seasonal,
        "gasoline_pump_slope": gas_slope,
        "wage_growth_pct": wage_yoy,
        "market_rent_yoy_recent": rent_yoy,
        "food_pipeline_yoy": food_yoy,
        "core_goods_baseline_yoy": uconfig.CORE_GOODS_BASELINE_YOY,
        # point-in-time coefficients consumed by usmodel.inflation
        "shelter_passthrough": sh_b,
        "shelter_trend_yoy": sh_a,
        "supercore_passthrough": sc_b,
        "supercore_intercept": sc_a,
    }
    diagnostics = {
        "gasoline_pump_months_observed": int((pump.index > m0).sum())
        if not pump.empty else 0,
        "gasoline_pump_slope": gas_slope,
        "shelter_b": sh_b, "shelter_a": sh_a,
        "supercore_b": sc_b, "supercore_a": sc_a,
        "wage_yoy": wage_yoy, "wage_source": wage_source,
        "market_rent_yoy": rent_yoy,
        "wti_anchor": assumptions["wti_recent"],
        "n_rent_obs": len(rent),
    }
    return assumptions, diagnostics


# Daily/weekly rate series a LIVE run may read for the month in progress,
# the way it reads oil, the dollar and the pump price: a policy move on the
# 17th is public on the 17th. None of these feed the nowcast regression, so
# the partial month cannot change how many months it counts as published.
LIVE_PARTIAL_INDICATORS = ("fed_funds", "treasury_10y", "mortgage_30y")


def published_indicators(bundle: USDataBundle, asof: pd.Timestamp,
                         cfg: USVintageConfig) -> dict[str, pd.Series]:
    """Each activity indicator truncated with its own release lag (payrolls
    ride the employment report, IP and retail sales land mid-month, CFNAI
    near month end, weekly/daily series almost immediately)."""
    out = {}
    for name, series in (bundle.indicators or {}).items():
        lag = cfg.indicator_pub_lag_days.get(
            name, INDICATOR_PUB_LAG_DAYS.get(name, 20))
        if cfg.live_partial_month and name in LIVE_PARTIAL_INDICATORS:
            lag = -31
        t = truncate(series, asof, lag)
        if len(t):
            out[name] = t
    return out


# ---------------------------------------------------------------------------
# The vintage dataset
# ---------------------------------------------------------------------------

@dataclass
class USVintage:
    asof: pd.Timestamp
    gdp_level: pd.Series
    cpi_index: pd.Series
    market_rent: pd.Series
    dollar: pd.Series
    indicator_panel: dict
    assumptions: dict
    indicators: dict
    diagnostics: dict
    revision_mode: str
    last_gdp_quarter: pd.Period = field(init=False)
    last_cpi_month: pd.Period = field(init=False)

    def __post_init__(self):
        self.last_gdp_quarter = self.gdp_level.index[-1]
        self.last_cpi_month = self.cpi_index.index[-1]

    @property
    def aux(self) -> dict:
        return {"market_rent": self.market_rent, "dollar": self.dollar}


def resolve_revision_mode(bundle: USDataBundle, cfg: USVintageConfig) -> str:
    if cfg.revision_mode == "auto":
        return "realtime" if bundle.has_vintage_panel() else "noise"
    if cfg.revision_mode == "realtime" and not bundle.has_vintage_panel():
        raise ValueError("revision_mode='realtime' needs a GDP vintage panel "
                         "(data/us/gdp_vintages.csv)")
    return cfg.revision_mode


def build_vintage(bundle: USDataBundle, asof: pd.Timestamp | str,
                  cfg: USVintageConfig | None = None,
                  horizon_months: int = 20) -> USVintage:
    """Reconstruct the US information set at `asof`."""
    cfg = cfg or USVintageConfig()
    asof = pd.Timestamp(asof)
    mode = resolve_revision_mode(bundle, cfg)

    if mode == "realtime":
        gdp = gdp_from_vintage_panel(bundle.gdp_vintages, asof, cfg)
    else:
        gdp = truncate(bundle.gdp, asof, cfg.gdp_pub_lag_days)
        if mode == "noise":
            gdp = apply_revision_noise(gdp, cfg)

    cpi = truncate(bundle.cpi, asof, cfg.cpi_pub_lag_days)
    rent = truncate(bundle.market_rent, asof, cfg.rent_pub_lag_days)
    dollar = truncate(bundle.dollar, asof, cfg.market_pub_lag_days)

    if len(gdp) < 5 or len(cpi) < 14:
        raise ValueError(f"not enough published history at {asof.date()}")

    assumptions, diagnostics = vintage_assumptions(
        bundle, asof, cfg, cpi, horizon_months=horizon_months)

    # Activity indicators, each truncated with its own release lag, then a
    # ridge nowcast refitted on this vintage's published history alone.
    panel = published_indicators(bundle, asof, cfg)
    indicators: dict = {}
    if cfg.fit_trend:
        trend = gdp_mod.estimate_trend_qoq(gdp)
        if trend is not None:
            indicators["fitted_trend_qoq"] = trend
            diagnostics["trend_qoq_pct"] = trend * 100
    if cfg.fit_convergence:
        conv = gdp_mod.estimate_convergence(gdp)
        if conv is not None:
            indicators["fitted_trend_qoq"] = conv[0]
            indicators["fitted_convergence"] = conv[1]
            diagnostics.update({"trend_qoq_pct": conv[0] * 100,
                                "convergence": conv[1]})
    if cfg.use_indicator_nowcast and panel:
        spec = nowcast.NOWCAST_SPECS.get(
            str(getattr(cfg, "nowcast_spec", "base")), nowcast.FEATURE_SPEC)
        fit = nowcast.fit_nowcast(gdp, panel, gdp.index[-1] + 1, spec=spec)
        if fit is not None:
            indicators["fitted_qoq_pct"] = fit["qoq_pct"]
            diagnostics.update({"nowcast_qoq_pct": fit["qoq_pct"],
                                "nowcast_k": fit["k"],
                                "nowcast_n_train": fit["n_train"]})
    diagnostics.setdefault("nowcast_qoq_pct", float("nan"))
    diagnostics.setdefault("nowcast_k", -1)
    diagnostics.setdefault("trend_qoq_pct", uconfig.GDP_TREND_QOQ * 100)
    diagnostics.setdefault("convergence", uconfig.GDP_CONVERGENCE)
    # The pace past the nowcast quarter, and what it resolves to here.
    indicators["pace_mode"] = cfg.pace_mode
    q1 = gdp_mod.nowcast_qoq(gdp, indicators)
    anchor, persistence, _ = gdp_mod.pace_past_nowcast(q1, indicators)
    diagnostics.update({"pace_mode": cfg.pace_mode, "pace_qoq_pct": anchor * 100,
                        "pace_persistence": persistence})

    # The rate channel: sensitivity fitted on this vintage's published
    # history (1960+ via the untruncated GDP series, rates as published),
    # then the drag for the quarters the projection will cover, off a
    # policy path that is observed, then market-implied (live only), then
    # flat. Every number here is knowable at the as-of date.
    if cfg.rate_channel and "fed_funds" in panel:
        long_gdp = (truncate(bundle.gdp_long, asof, cfg.gdp_pub_lag_days)
                    if bundle.gdp_long is not None else gdp)
        if len(long_gdp) < len(gdp):
            long_gdp = gdp
        ff_m = panel["fed_funds"]
        last_q = gdp.index[-1]
        targets = pd.period_range(last_q + 1, periods=12, freq="Q")
        path_m = rates.expected_policy_path(
            ff_m, cfg.policy_rate_path, targets[-1].asfreq("M", "end"))
        policy_q = rates.quarterly_mean(path_m)
        fit = rates.fit_rate_sensitivity(long_gdp,
                                         rates.quarterly_mean(ff_m))
        if fit is not None:
            drag = rates.rate_drag(policy_q, targets, fit["beta"])
            indicators["rate_drag_qoq"] = drag
            diagnostics.update({
                "rate_beta": fit["beta"], "rate_beta_raw": fit["beta_raw"],
                "rate_fit_n": fit["n"], "rate_fit_corr": fit["corr"],
                "rate_drag_pp": {str(q): float(d) * 100
                                 for q, d in drag.items()},
                "rate_state": rates.rate_state(ff_m, policy_q, fit, drag,
                                               cfg.policy_rate_path),
            })
    diagnostics.setdefault("rate_beta", float("nan"))

    return USVintage(asof=asof, gdp_level=gdp, cpi_index=cpi,
                     market_rent=rent, dollar=dollar, indicator_panel=panel,
                     assumptions=assumptions, indicators=indicators,
                     diagnostics=diagnostics, revision_mode=mode)
