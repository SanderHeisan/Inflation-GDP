"""
US backtest configuration: publication lags, the BEA revision model, and
where the cached real data lives. Everything encoding an assumption about
*what was knowable when* lives here, so the point-in-time rules are
auditable in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Publication lags (calendar days after the reference period ends). A value
# for period P is in the information set at as-of date D only if
#   D >= end_of(P) + lag.
# ---------------------------------------------------------------------------
# BLS releases CPI for month M around the 10th-15th of M+1.
CPI_PUB_LAG_DAYS = 13
# BEA's advance estimate for quarter Q lands ~28-30 days after quarter end.
GDP_PUB_LAG_DAYS = 28
# Average hourly earnings ride the employment report: first Friday of M+1.
WAGE_PUB_LAG_DAYS = 8
# Zillow publishes ZORI for month M around the middle of M+1.
RENT_PUB_LAG_DAYS = 20
# WTI and the broad dollar index are daily. The model consumes monthly
# means, which are only complete once the month is over.
MARKET_PUB_LAG_DAYS = 0

REVISION_MODES = ("auto", "realtime", "noise", "none")


@dataclass
class USVintageConfig:
    """Rules for reconstructing the US information set at an as-of date."""

    cpi_pub_lag_days: int = CPI_PUB_LAG_DAYS
    gdp_pub_lag_days: int = GDP_PUB_LAG_DAYS
    wage_pub_lag_days: int = WAGE_PUB_LAG_DAYS
    rent_pub_lag_days: int = RENT_PUB_LAG_DAYS
    market_pub_lag_days: int = MARKET_PUB_LAG_DAYS
    # Per-indicator release lags for the GDP nowcast; defaults come from
    # usmodel.fetch_data_us.INDICATOR_PUB_LAG_DAYS.
    indicator_pub_lag_days: dict = field(default_factory=dict)

    # GDP revision handling:
    #   'realtime' - replay a true vintage panel (data/us/gdp_vintages.csv)
    #   'noise'    - simulate first releases: current-vintage QoQ plus a
    #                persistent N(0, sigma) revision error per quarter that
    #                decays as the quarter matures
    #   'none'     - current vintage, truncation only (an upper bound on
    #                accuracy: it hands the model final GDP data)
    #   'auto'     - 'realtime' if a vintage panel is present, else 'noise'
    revision_mode: str = "auto"

    # sigma of the total advance-to-latest revision of real GDP growth. BEA's
    # own revision studies put the mean absolute revision of quarterly SAAR
    # growth from the advance estimate to the latest at roughly 1.1-1.3pp,
    # implying sigma ~1.4pp annualized == ~0.35pp at a quarterly rate.
    revision_sigma_pp: float = 0.35
    revision_decay: float = 0.7
    revision_maturity_quarters: int = 12
    seed: int = 0

    # Whether the CPI blocks may self-calibrate on the vintage's own history
    # (see usbacktest.vintage.calibrate_from_vintage). Estimation uses only
    # data published at the as-of date, so it is point-in-time clean; the
    # flag exists so the backtest can quote both the raw-config model and
    # the self-calibrating one.
    self_calibrate: bool = True

    # Whether the GDP nowcast may use the point-in-time ridge fit on real
    # activity indicators (usmodel.nowcast). False falls back to the
    # trailing-momentum blend, which is what the first cut of the model did
    # -- the flag is what makes the growth-side before/after measurable.
    use_indicator_nowcast: bool = True

    # Whether the trend the projection holds past the nowcast quarter is the
    # vintage's own trailing median QoQ rather than the static config value.
    # On by default: that constant is what every multi-quarter growth call is
    # measured against, so it should track the data.
    fit_trend: bool = True

    # Whether to also carry an AR(1) persistence past the nowcast quarter.
    # OFF by default because the backtest scores it as harmful -- past the
    # nowcast quarter US real GDP QoQ is not forecastable, so a moving path
    # only adds noise to the known base effect. Kept switchable so
    # `--growth-variants` can reproduce that measurement.
    fit_convergence: bool = False

    def __post_init__(self):
        if self.revision_mode not in REVISION_MODES:
            raise ValueError(f"revision_mode must be one of {REVISION_MODES}, "
                             f"got {self.revision_mode!r}")


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "us"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_us"

# The backtest window is bounded by the shortest driver series: Zillow ZORI
# starts 2015-01, needs 12 months to have a YoY, and CPI shelter is driven
# by that YoY lagged a further 12 months -- so the first as-of date whose
# shelter block is driven by observed market rents is 2017-01.
BACKTEST_START = "2017-01"
