"""
Central configuration for the US GIP / Quad Map pipeline.

Same philosophy as the Norwegian model (quadmap/config.py): everything that
is a *judgment call* or *changes over time* lives here, so the model code
stays mechanical. Only the numbers are US-specific; the quad engine
(quadmap.quads) and the GDP nowcast+convergence (quadmap.gdp) are reused
unchanged.
"""

# ---------------------------------------------------------------------------
# FRED series ids (pulled keyless via fredgraph.csv, like the Brent fetch in
# the Norwegian model). BLS/BEA revise heavily; the Philadelphia Fed's
# Real-Time Data Set is the US vintage source (wired in fetch_data_us.py).
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "cpi":            "CPIAUCSL",     # CPI-U, all items, SA (index level)
    "cpi_core":       "CPILFESL",     # CPI ex food & energy
    "gdp":            "GDPC1",        # Real GDP, chained, SAAR (quarterly)
    "wti":            "DCOILWTICO",   # WTI crude, $/bbl (daily)
    "gasoline_cpi":   "CUSR0000SETB01",  # CPI motor fuel (SA)
    "shelter_cpi":    "CUSR0000SAH1",    # CPI shelter (SA)
    "rent_cpi":       "CUSR0000SEHA",    # CPI rent of primary residence
    "oer_cpi":        "CUSR0000SEHC",    # CPI owners' equivalent rent
    "food_cpi":       "CPIUFDSL",        # CPI food
    "energy_cpi":     "CPIENGSL",        # CPI energy
    "dollar":         "DTWEXBGS",        # Broad trade-weighted USD (higher = stronger)
    "wages":          "CES0500000003",   # Avg hourly earnings, total private
    "market_rent":    None,          # Zillow ZORI / Apartment List - see fetch script
}

# ---------------------------------------------------------------------------
# CPI-U component weights (relative importance, per-mille of the basket).
# Rough 2024-vintage CPI-U. BLS republishes annually; the pipeline should
# overwrite from the BLS relative-importance table, but these keep the demo
# and backtest running offline.
# ---------------------------------------------------------------------------
CPI_WEIGHTS = {
    "food_at_home":   82,    # groceries
    "food_away":      51,    # food away from home
    "gasoline":       34,    # motor fuel
    "electricity":    25,
    "energy_other":   11,    # utility gas, fuel oil
    "shelter":       345,    # OER (~267) + rent of primary residence (~78)
    "core_goods":    185,    # vehicles, apparel, household furnishings (ex food/energy)
    "medical":        74,    # medical care
    "supercore":     193,    # core services ex shelter ("supercore")
}

# ---------------------------------------------------------------------------
# Energy: the oil -> gasoline -> headline pass-through.
#
# Hedgeye rule of thumb: a sustained $1/bbl higher WTI adds ~3 bps to
# headline CPI (through the gasoline complex). Encoded as a headline
# sensitivity and routed through the energy blocks. RECALIBRATE by
# regressing headline energy contribution on WTI changes once live data is
# wired in -- this is a judgment call, hence it lives here.
# ---------------------------------------------------------------------------
OIL_HEADLINE_BPS_PER_DOLLAR = 3.0   # bps of headline CPI per $1/bbl WTI
# Gasoline responds to crude within ~2-4 weeks, so the move lands in the
# same or next month. Share of the oil effect that hits within the month:
OIL_SAME_MONTH_SHARE = 0.7

# ---------------------------------------------------------------------------
# Shelter: OER and rent lag market ("new-lease") rents by roughly a year.
# This is the most forecastable large block in US core CPI -- last year's
# observed market-rent momentum tells you where shelter is going. The US
# analog of the Norwegian model's rent-indexation anchor, but far bigger.
# ---------------------------------------------------------------------------
SHELTER_MARKET_RENT_LAG_M = 12   # months market rents lead CPI shelter
SHELTER_PASSTHROUGH = 0.55       # share of lagged market-rent YoY that shows
                                 # up in CPI shelter YoY (partial: new leases
                                 # are a fraction of the stock repricing)
SHELTER_TREND_YOY = 3.2          # long-run shelter YoY it converges toward

# ---------------------------------------------------------------------------
# Core goods: import-price / dollar pass-through (a stronger dollar is
# disinflationary for goods, with a lag). Analog of the Norwegian I-44
# distributed lag. Positive coeff on a WEAKER dollar.
# ---------------------------------------------------------------------------
DOLLAR_PASSTHROUGH_LAGS = {3: -0.02, 6: -0.04, 9: -0.03, 12: -0.02}  # per 1% dollar chg
CORE_GOODS_BASELINE_YOY = 0.0    # secular goods trend (roughly flat/deflationary)

# ---------------------------------------------------------------------------
# Supercore (core services ex shelter): wage-driven with a productivity
# haircut. Analog of the Norwegian domestic-services block.
# ---------------------------------------------------------------------------
SUPERCORE_WAGE_PASSTHROUGH = 0.55
MEDICAL_TREND_YOY = 2.8

# Food repricing has mild seasonality; grocery pipeline toward trend.
FOOD_TREND_YOY = 2.5

# ---------------------------------------------------------------------------
# GDP nowcast blend (reuses quadmap.gdp.project_gdp). US indicators map onto
# the same generic slots; calibrate the multipliers on realized QoQ history.
# ---------------------------------------------------------------------------
# NO `ism` SLOT. It carried 0.30 here from the day the model was written and
# never fired once: ISM restricted redistribution, FRED dropped the NAPM
# series, and nothing ever supplied one, so the blend renormalised over the
# three real slots and a reader of this file was told a third of the fallback
# rested on an input that did not exist. Removing it is numerically inert
# (tests/test_pmi_surveys.py proves it). The diffusion indices that ARE free -
# the regional Fed manufacturing surveys - were scored in their place on
# 2026-09-24 and made the growth DIRECTION call worse, so they are available
# to the nowcast panel as a research spec and are not in production. See the
# README's survey section.
GDP_NOWCAST_WEIGHTS = {
    "momentum":          0.35,   # trailing QoQ SAAR momentum
    "payrolls":          0.20,   # nonfarm payroll momentum
    "retail":            0.15,   # real retail sales momentum
}
GDP_TREND_QOQ = 0.0045           # ~1.8% annualized potential (per quarter);
                                 # only a fallback -- the trend is normally
                                 # re-estimated per vintage, see below.
# Trend is the trailing MEDIAN of published QoQ over this window. The median,
# not the mean: the 2020 crash-and-rebound pair drags a mean badly, and the
# constant is what every multi-quarter growth call is measured against.
GDP_TREND_WINDOW_Q = 24
# Persistence carried past the nowcast quarter. Measured to be worth ZERO:
# beyond the current quarter US real GDP QoQ is not forecastable (see the
# README's growth-axis section), so any non-constant path adds error that is
# uncorrelated with the truth to a call whose only real signal is the KNOWN
# year-ago QoQ. Backtested, direction accuracy falls monotonically as this
# rises: 0.653 at 0.0, 0.644 at 0.30, 0.639 at 0.50 (the old default).
GDP_CONVERGENCE = 0.0

# Publication lags (calendar days): US CPI ~13 days after month end, the
# BEA advance GDP estimate ~28 days after quarter end (heavily revised
# afterward -- the Philadelphia Fed real-time set captures the vintages).
CPI_PUB_LAG_DAYS = 13
GDP_PUB_LAG_DAYS = 28

# Quad deadband (percentage points of YoY acceleration): below this the
# quarter is flagged low-conviction. Shared convention with the NO model.
QUAD_DEADBAND_PP = 0.10

# ---------------------------------------------------------------------------
# The rate channel (usmodel.rates): policy-rate changes as a drag on the
# growth path past the nowcast quarter. Priced in on request; its measured
# cost to the growth-direction call is stated in the README rather than
# hidden -- turn it off here to get the flat path back.
# ---------------------------------------------------------------------------
RATE_CHANNEL_ENABLED = True
# Hikes over the year ending two quarters before the target quarter: the
# 1960-2026 distributed lag puts the drag at lags 2-8 with the largest
# single coefficient at lag 2, and this window is its parsimonious form.
RATE_LAG_QUARTERS = (2, 6)
# The sample the sensitivity is fitted on, point-in-time. 1985+ alone has
# NO measurable drag (the slope is zero to positive), so the fit must reach
# back to the Volcker era to find the effect the channel encodes.
RATE_FIT_START = "1960Q1"
RATE_FIT_EXCLUDE = ("2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2")
# pp of quarterly growth per 1pp of policy-rate change over the window. The
# sign is IMPOSED (a hike never adds to growth); the size is estimated and
# runs about -0.10 on the full history.
RATE_SENSITIVITY_CLIP = (-0.30, 0.0)
# A number here replaces the estimate (e.g. -0.20 for FRB/US-strength
# transmission, where 100bp costs ~0.8% of GDP over two years).
RATE_SENSITIVITY_OVERRIDE = None
# Market-implied policy path, LIVE runs only: dated steps in the effective
# rate, day-weighted into monthly means by usmodel.rates. Source: the
# meeting-date implied path read off a Hedgeye slide supplied on
# 2026-09-18 (0.57 hikes priced for 28 Oct 2026 rising to 3.00 hikes by
# 28 Jul 2027). The first entry is the post-September-meeting rate that
# slide implies (4.03 - 0.57 x 0.25); the daily effective rate confirms or
# corrects it as it is published. Replace when the curve moves -- the
# backtest never sees this (it holds the rate flat, having no futures
# history), so it cannot leak.
POLICY_RATE_PATH_ASOF = "2026-09-18"
POLICY_RATE_PATH = {
    "2026-09-18": 3.88,
    "2026-10-28": 4.03,
    "2026-12-09": 4.21,
    "2027-01-27": 4.31,
    "2027-03-17": 4.47,
    "2027-04-28": 4.55,
    "2027-06-09": 4.62,
    "2027-07-28": 4.63,
}

# ---------------------------------------------------------------------------
# The pace the growth path assumes past the nowcast quarter (usmodel.gdp).
#   "trailing_median"  the vintage's trailing 24-quarter median QoQ. Backward-
#                      looking: 3.1% annualized today because the window is the
#                      post-2020 boom. Scored best on 2017-2026, a period in
#                      which the past kept repeating.
#   "potential"        a long-run potential pace (GDP_POTENTIAL_ANN_PCT) from
#                      q+2 on: "this quarter from the data, then the economy's
#                      normal pace". Forward-looking in the sense that it does
#                      not assume the boom continues.
#   "nowcast"          this quarter's nowcast carried forward: the latest
#                      activity data set the pace.
#   "glide"            the nowcast fading toward potential with persistence
#                      GDP_GLIDE_PERSISTENCE per quarter.
# The rate channel's drag sits on top in every mode (relative to the nowcast
# quarter in "nowcast" mode, which already sees the rate environment). All
# four are measured by `python us_backtest.py` (us_pace_modes.csv).
# ---------------------------------------------------------------------------
GDP_PACE_MODE = "potential"
GDP_POTENTIAL_ANN_PCT = 2.0        # CBO-style potential; the bar the boom-era median overstates
GDP_GLIDE_PERSISTENCE = 0.5
GDP_PACE_MODES = ("trailing_median", "potential", "nowcast", "glide")
