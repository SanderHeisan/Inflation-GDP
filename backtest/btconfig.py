"""
Backtest configuration: publication lags, the GDP revision model, and file
locations. Everything that encodes an assumption about *what was knowable
when* lives here so the point-in-time rules are auditable in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Publication lags (calendar days after the reference period ends).
#
# GDP: SSB publishes the QNA roughly 40 days after quarter end (e.g. Q4 in
# early/mid February). CPI: roughly 10 days after month end. A series value
# for period P is in the information set at as-of date D only if
#   D >= end_of(P) + lag.
# ---------------------------------------------------------------------------
GDP_PUB_LAG_DAYS = 40
CPI_PUB_LAG_DAYS = 10
# I-44 / market spots are observable near-real-time, but the model consumes
# monthly averages; a month's average is only fully known once the month is
# over. Zero extra days beyond month end.
MARKET_PUB_LAG_DAYS = 0

# Wage settlement (frontfagsrammen): the TBU/frontfag norm for year Y is
# known once the spring settlement concludes; before that the prior year's
# norm is the best point-in-time anchor.
WAGE_NORM_KNOWN_MONTH = 4
WAGE_NORM_KNOWN_DAY = 15

REVISION_MODES = ("auto", "realtime", "noise", "none")


@dataclass
class VintageConfig:
    """Rules for reconstructing the information set at an as-of date."""

    gdp_pub_lag_days: int = GDP_PUB_LAG_DAYS
    cpi_pub_lag_days: int = CPI_PUB_LAG_DAYS
    market_pub_lag_days: int = MARKET_PUB_LAG_DAYS

    # GDP revision handling:
    #   'realtime' - use a true vintage panel (Norges Bank real-time database)
    #   'noise'    - simulate first releases: current-vintage QoQ + a
    #                persistent N(0, sigma) revision error per quarter that
    #                decays as the quarter matures
    #   'none'     - current vintage, truncation only (revisions ignored)
    #   'auto'     - 'realtime' if the bundle has a vintage panel, else 'noise'
    revision_mode: str = "auto"

    # Noise model. sigma is the stdev of the *total* first-release-to-final
    # revision of mainland QoQ growth, in percentage points. SSB's revision
    # analyses of the QNA put mean absolute revisions of mainland QoQ growth
    # around 0.2pp, consistent with sigma ~ 0.25pp. Override per run.
    revision_sigma_pp: float = 0.25
    # Estimate error decays by this factor per quarter of age; after
    # maturity_quarters the quarter is treated as final (annual accounts).
    revision_decay: float = 0.6
    revision_maturity_quarters: int = 8
    seed: int = 0

    def __post_init__(self):
        if self.revision_mode not in REVISION_MODES:
            raise ValueError(f"revision_mode must be one of {REVISION_MODES}, "
                             f"got {self.revision_mode!r}")


# ---------------------------------------------------------------------------
# Cached data files (populated by backtest/fetch_data.py on a machine with
# network access; this sandbox cannot reach data.ssb.no / norges-bank.no).
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATA_FILES = {
    "gdp": "gdp_mainland_q.csv",        # period (2012Q1), value
    "cpi": "cpi_monthly.csv",           # period (2012-01), value
    "i44": "i44_monthly.csv",           # period (2012-01), value
    "market": "market_monthly.csv",     # period, power_ore_kwh, brent_usd, usdnok
    "gdp_vintages": "gdp_vintages.csv", # quarter rows x vintage-date columns
    "wage_norms": "wage_norms.csv",     # year, norm_pct
}
