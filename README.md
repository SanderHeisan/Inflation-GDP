# Norway GIP Quad Map

A forward-looking growth/inflation quadrant model for Norway, in the style of
Hedgeye's GIP framework. Classifies each quarter — realized and projected —
into one of four regimes based on the *rate of change* of YoY real GDP growth
and YoY CPI inflation:

| | Inflation decelerating | Inflation accelerating |
|---|---|---|
| **Growth accelerating** | Quad 1 · Goldilocks | Quad 2 · Reflation |
| **Growth decelerating** | Quad 4 · Disinflation | Quad 3 · Stagflation |

## Quick start

```bash
pip install -r requirements.txt
python run.py --demo      # synthetic data, validates the pipeline offline
python run.py             # live: SSB + Norges Bank APIs + assumptions.yaml
```

Outputs: `quad_projections.csv` (the quad table), `cpi_decomposition.csv`
(monthly CPI path with per-component contributions), `quad_map.png`.

## How the forecast works

The whole edge of the framework is that **YoY rates are half-known in
advance**: next quarter's YoY compares a future level to a level already in
the books. So the pipeline forecasts *index levels*, and the YoY path — and
therefore the quad — falls out mechanically against known base periods.

### GDP (mainland Norway, table 09190)

1. **Nowcast next quarter's QoQ** as a weighted blend: trailing momentum,
   Norges Bank Regional Network output index, PMI, retail volume
   (weights in `config.GDP_NOWCAST_WEIGHTS` — calibrate the indicator
   multipliers by regressing each on realized QoQ history).
2. **Converge** later quarters geometrically toward trend (~0.4% QoQ),
   or override with Norges Bank's MPR forecast path.
3. Compound onto the last observed level → YoY path → quarterly deltas.

Mainland GDP, not total: total GDP is dominated by petroleum extraction
volumes and doesn't describe the domestic cycle Norges Bank reacts to.

### Inflation (bottom-up by component)

Each block gets its own forward driver, then blocks aggregate with SSB
basket weights into a monthly index path:

| Block | Driver | Forward-looking input |
|---|---|---|
| Electricity | Nord Pool spot via strømstøtte formula | Power forward curve |
| Fuel | Brent in NOK, ~40% pass-through (taxes) | Brent futures + NOK path |
| Imported goods | I-44 distributed-lag pass-through (3–18m) | Mostly *already observed* FX moves |
| Rent / housing | CPI-indexation of leases + wages | Last year's CPI (known) + wage norm |
| Food | World food prices in NOK | Feb/Jul repricing windows |
| Domestic services | Wage settlement (frontfagsrammen) | TBU norm, announced each spring |

Run it for both headline CPI and KPI-JAE (core): the gap is your energy/tax
call, and Norges Bank steers on JAE.

### Quads

`quads.classify()` takes the two quarterly YoY series, computes QoQ deltas,
assigns the quad, and flags quarters inside a ±0.10pp deadband as
low-conviction. `quads.quad_flips()` extracts regime changes — the events a
subscriber briefing should lead with.

## Backtesting

The walk-forward harness answers one question: how often would this model
have called the correct quad 1-4 quarters ahead, using only information
available at the time?

### Easiest way to run it: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SanderHeisan/Inflation-GDP/blob/claude/norway-gip-backtest-harness-nln7uw/backtest_colab.ipynb)

Open `backtest_colab.ipynb` in Colab and hit **Runtime -> Run all**. It
clones the repo, installs dependencies, fetches **real data** (SSB GDP +
CPI, Norges Bank I-44/USDNOK, Brent from FRED, plus a documented power
proxy), runs the backtest, and renders the summary tables and charts
inline. Because this repo is private you must authorize GitHub in Colab
(tick "Include private repositories" in the Colab GitHub dialog) and paste
a personal access token when the clone step asks. If any live fetch fails
the notebook falls back to the synthetic demo bundle, clearly labelled.

### Locally

```bash
python backtest.py --demo --start 2012 --end 2025 --horizon 4   # synthetic
python backtest.py --start 2012 --end 2025 --horizon 4          # cached data
python -m pytest tests/                                          # leakage & scoring tests
```

Outputs: `backtest_results.parquet` (every prediction: as-of date, target
quarter, horizon, quad, deltas, conviction, benchmarks, vintage
bookkeeping), `summary.csv` (per-horizon hit rates, direction hit rates,
flip precision/recall, and the model's edge over persistence /
base-effects / random benchmarks — scored against both final-vintage and
first-release realized quads), plus hit-rate, confusion-matrix and
timeline plots.

### Point-in-time discipline

`backtest/vintage.py` is the leakage firewall. At each as-of date it
rebuilds the information set: QNA truncated to a ~40-day publication lag,
CPI to ~10 days (SSB CPI is never revised, so truncation is exact there),
market forwards frozen at the last observed spot (spot-carry — realized
future prices never enter the assumptions dict), and the wage norm only
visible from mid-April of its settlement year. `tests/test_vintage.py`
asserts all of it, including a mutate-the-future test proving predictions
cannot change when post-as-of data changes.

GDP revisions run in one of three flagged modes (recorded in every output
row):

* `realtime` — replay true vintages from `data/gdp_vintages.csv`. Norges
  Bank maintains a real-time database of Norwegian QNA vintages (the
  dataset behind their nowcasting research); it is not fetchable from this
  sandbox, so download it yourself and convert to the CSV format described
  in `backtest/fetch_data.py`. The harness switches over automatically.
* `noise` (default fallback) — simulate first releases: current-vintage
  QoQ plus a persistent, per-quarter N(0, sigma) revision error that
  decays as the quarter matures. `--sigma` defaults to 0.25pp, in line
  with SSB's published revision statistics for mainland QoQ growth.
* `none` — truncation only, revisions ignored (an upper bound).

Run `python -m backtest.fetch_data` on a networked machine to populate
`data/` with current-vintage SSB GDP/CPI and Norges Bank I-44; market and
vintage files are documented there too.

## Before production

- Run `data_sources.get_table_metadata()` on each SSB table and confirm the
  variable codes in the queries (SSB restructures tables occasionally).
- Verify current strømstøtte parameters (`config.STROMSTOTTE`).
- Re-estimate the FX pass-through coefficients on live delivery-sector data
  (`inflation.estimate_fx_passthrough`) instead of the config fallback.
- Calibrate the Regional Network / PMI multipliers on realized QoQ history.
- **Backtest**: reconstruct what the model would have projected each quarter
  using only vintage data (SSB revises GDP substantially — first releases,
  not current values) and score quad hit-rates 1–2 quarters out. This is the
  number that makes the service credible to subscribers.
- SSB data is free under NLOD/CC BY 4.0 — attribution required if you
  redistribute; check terms if selling derived products commercially.
- If this feeds subscriber positioning advice, mind the line between
  publishing analysis and giving personal investment advice (concession/
  regulatory requirements differ; worth a legal check for Norway).

## Extending to a multi-country map

The quad engine is country-agnostic — only `data_sources` is
Norway-specific. For a cross-country map, add fetchers against the OECD
data API (consistent quarterly GDP + CPI for all members in one format)
and reuse `quads.classify()` per country. The bottom-up inflation model is
the Norway-specific value-add; other countries can start with the simpler
momentum + base-effects projection.
