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
