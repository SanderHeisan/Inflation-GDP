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
GDP_NOWCAST_WEIGHTS = {
    "momentum":          0.35,   # trailing QoQ SAAR momentum
    "ism":               0.30,   # ISM manufacturing+services -> QoQ proxy
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
