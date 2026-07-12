"""
Central configuration for the Norway GIP / Quad Map pipeline.

Everything that is a *judgment call* or *changes over time* lives here,
so the model code itself stays mechanical.
"""

# ---------------------------------------------------------------------------
# SSB StatBank table IDs (PxWebApi v0, https://data.ssb.no/api/v0/en/table/{id})
# Verify with quadmap.data_sources.get_table_metadata(table_id) before first
# run -- SSB occasionally restructures tables.
# ---------------------------------------------------------------------------
SSB_TABLES = {
    # CPI: index level by COICOP consumption group. 'Kpi10' is the API id
    # of the 2025=100 rebase (StatBank catalog number 14710, history to
    # 1920) - the id verified working via POST; the numeric 14710 routes
    # to a differently-structured endpoint and 400s. Superseded 03013,
    # frozen at 2025M12. fetch_data falls back to the legacy table and
    # auto-discovers successors if this one freezes or breaks again.
    "cpi": "Kpi10",
    "cpi_legacy": "03013",
    # CPI by delivery sector (imported vs domestically produced split) --
    # this is the table that makes the FX pass-through model easy
    "cpi_delivery_sector": "05327",
    # CPI-ATE / KPI-JAE (core: adjusted for taxes, excluding energy)
    "cpi_core": "05327",
    # Quarterly national accounts, constant prices, seasonally adjusted.
    # Use "Mainland Norway" (Fastlands-Norge) market values.
    "gdp_qna": "09190",
    # Retail sales volume index (monthly nowcast input)
    "retail": "07129",
}

SSB_BASE = "https://data.ssb.no/api/v0/en/table/"

# Norges Bank open data (SDMX-JSON). I-44 = import-weighted krone index.
NORGES_BANK_I44 = (
    "https://data.norges-bank.no/api/data/EXR/M.I44.NOK.SP"
    "?format=sdmx-json&startPeriod={start}"
)

# ---------------------------------------------------------------------------
# CPI component model
# ---------------------------------------------------------------------------
# Fallback COICOP weights (per-mille of the total basket). SSB publishes new
# weights every year; the pipeline should overwrite these from the API, but
# these keep the demo/backtest running offline. Rough 2025-vintage values.
CPI_WEIGHTS_FALLBACK = {
    "food":               126,   # 01 Food and non-alcoholic beverages
    "alcohol_tobacco":     40,   # 02
    "clothing":            44,   # 03
    "housing_ex_energy":  183,   # 04 excl. electricity/fuels (incl. rent)
    "electricity":         45,   # 04.5.1 electricity incl. grid rent
    "other_energy":        35,   # heating fuels + 07.2.2 motor fuel
    "furnishings":         64,   # 05
    "health":              33,   # 06
    "transport_ex_fuel":  120,   # 07 excl. fuel
    "communication":       22,   # 08
    "recreation":         120,   # 09
    "education":            5,   # 10
    "restaurants_hotels":  63,   # 11
    "misc":                100,  # 12
}

# Electricity subsidy (stromstotte). The state covers COVERAGE_SHARE of the
# spot price above THRESHOLD (ore/kWh incl. VAT), settled hourly.
# >>> VERIFY current parameters at regjeringen.no before production use. <<<
STROMSTOTTE = {
    "threshold_ore_kwh": 93.75,   # 75 ore ex-VAT * 1.25
    "coverage_share": 0.90,
}

# FX pass-through: distributed lag from 12m change in I-44 (positive = weaker
# NOK) to 12m imported-goods inflation. Estimated on 2001-2024 monthly data;
# re-estimate from history once live data is wired in (see inflation.py).
FX_PASSTHROUGH_LAGS = {3: 0.02, 6: 0.05, 9: 0.06, 12: 0.05, 18: 0.02}

# Rent model: Norwegian leases are overwhelmingly CPI-indexed annually.
RENT_CPI_INDEXATION_SHARE = 0.75   # share of stock that reprices to last CPI
RENT_WAGE_SHARE = 0.25             # remainder tracks wage growth loosely

# Food price adjustment windows (grocery chains reprice Feb 1 / Jul 1)
FOOD_ADJUSTMENT_MONTHS = [2, 7]

# ---------------------------------------------------------------------------
# GDP nowcast blend weights (next quarter). Beyond the nowcast horizon the
# model converges geometrically to trend growth.
# ---------------------------------------------------------------------------
GDP_NOWCAST_WEIGHTS = {
    "momentum": 0.35,          # trailing 2-quarter QoQ average
    "regional_network": 0.35,  # Norges Bank Regional Network output index
    "pmi": 0.15,               # NIMA/DNB PMI, mapped to QoQ
    "retail": 0.15,            # retail volume, mapped to QoQ
}
GDP_TREND_QOQ = 0.004          # ~1.6% annualized mainland trend growth
GDP_CONVERGENCE = 0.5          # per-quarter geometric convergence to trend

# Quad classification: |acceleration| below this (percentage points of YoY)
# is flagged as low-conviction / "on the line".
QUAD_DEADBAND_PP = 0.10
