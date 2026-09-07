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

The headline output is `quad_probabilities_quarterly.png` /
`_monthly.png` (+ `.csv`): the probability of each quad in each of the
next four quarters. Default method (`residual`): the model's per-horizon
backtest error cloud on (d_growth, d_inflation) is placed around today's
predicted deltas and P(quad) is the share landing in each quadrant — so
the probabilities carry the model's measured error rate *and* respond
continuously to the inputs. The alternative `calibration` method reads
the point call through the per-horizon confusion matrix instead.

The backtest also scores the sharpest monthly call: the **direction of
the next CPI print's YoY rate** (accelerating vs decelerating), one month
ahead, bucketed by conviction (`monthly_direction_summary.csv`). Next
month's YoY change is half-known — it equals next month's MoM versus the
already-published MoM from a year ago — so months with an extreme known
hurdle are near-certain calls, and the conviction buckets quantify
exactly how much to trust each live call.

## Live daily forecast

```bash
python forecast.py            # after a backtest run has produced calibration
```

Re-runs today's forecast with **live market prices** fetched at runtime:
Brent (FRED daily), USDNOK and I-44 (Norges Bank daily), and Nord Pool
area power prices (hvakosterstrommen.no). Live spots replace the
*forward* paths while the recent anchors stay at the level embedded in
the last CPI print — the gap between them is the price impulse the
projection reacts to. A power or oil spike today therefore raises the
projected inflation deltas and P(quad 2/3) today, without waiting for
the next CPI release. Base effects from already-published index levels
remain the backbone of the YoY arithmetic throughout. Each run appends
to `forecast_history.csv` so call drift is auditable.

`.github/workflows/daily-forecast.yml` runs the whole chain (fetch →
backtest → live forecast) every morning at 06:45 UTC and publishes the
outputs to the `live-forecast` branch plus a workflow artifact; it can
also be triggered manually from the Actions tab.

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

## Roadmap (in payoff order)

1. **Real Nord Pool power history** (needs sourcing) — the monthly misses
   cluster in energy months and the power column is a flagged proxy.
   Replace `data/market_monthly.csv` with real NO1/NO2/NO5 monthly
   averages (ore/kWh incl VAT) back to ~2010, plus the stromstotte
   parameter history, and delete `data/.market_is_proxy`.
2. **Live track record** (automatic) — the daily workflow appends every
   forecast to `forecast_history.csv` on the `live-forecast` branch;
   after a few months this is the out-of-sample proof no backtest can be.
3. **Growth-side indicators** (archive building automatically) — the
   nowcast blend has sockets for Regional Network / jobs / PMI / retail,
   but backtesting them honestly needs point-in-time data that public
   archives don't provide. `backtest/snapshots.py` therefore records the
   current value of each indicator daily; once the archive spans a few
   quarters, calibrate the nowcast weights on it without leakage.
4. **Norges Bank real-time GDP vintages** (needs a manual download) —
   drop into `data/gdp_vintages.csv` (format in `backtest/fetch_data.py`)
   and the quad backtest switches from simulated to true first releases.
5. **Split-sample validation** (automatic) — every backtest run now
   calibrates on pre-2019 data and verifies on 2019+
   (`calibration_validation.csv`); quote the out-of-sample column, not
   the in-sample one.

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


---

# United States GIP Quad Map

The same framework applied to the US, and — unlike the Norwegian side — it
runs on **real data end to end**, because FRED and Zillow are both reachable
from this sandbox. Every number below is measured, not assumed.

```bash
python -m usmodel.fetch_data_us    # cache real FRED + Zillow series (once)
python us_backtest.py              # walk-forward backtest -> results_us/
python run_us.py --demo            # synthetic pipeline smoke test
```

## Data

| Series | Source | Feeds |
|---|---|---|
| CPI-U SA / NSA / core, plus shelter, gasoline, food, energy, supercore | FRED (BLS) | the CPI blocks; NSA is the unrevised scoring check |
| Real GDP (GDPC1) | FRED (BEA) | the growth axis |
| WTI, broad trade-weighted dollar | FRED, daily → monthly mean | gasoline and core-goods blocks |
| Average hourly earnings | FRED (BLS) | supercore block |
| **ZORI market rents** | Zillow research CSVs | the shelter block — the model's biggest edge |
| Payrolls, industrial production, retail sales, initial claims, CFNAI, U. Mich sentiment, 10y–3m spread, manufacturing hours | FRED | the GDP nowcast |

ZORI starts 2015-01 and CPI shelter follows it with a ~12 month lag, so the
first as-of date whose shelter block is driven by *observed* market rents is
**2017-01**. That bounds the backtest: **115 monthly as-of dates, 2017-01 to
2026-07, 575 quad predictions.**

## The stats

Walk-forward and point-in-time. At each as-of date the information set is
rebuilt (CPI ~13 days after month end, BEA advance GDP ~28 days, payrolls
~8, ZORI ~20, each nowcast indicator on its own release lag), the model runs
on that vintage, and nothing published later can reach it.
`tests/test_us_vintage.py` proves that with a mutate-the-future test:
corrupt every series after the as-of date and the predicted quad table must
come out bit-identical.

### 1. Next CPI print — the sharpest recurring call

Direction of the next YoY print (accelerating vs decelerating), bucketed by
conviction = |predicted change in YoY|:

| Conviction bucket | Share of months | Hit rate |
|---|---|---|
| ALL months | 100% | **83.2%** |
| coin-flip (<0.05pp) | 19% | 59.1% |
| lean (0.05–0.15pp) | 26% | 82.8% |
| call (0.15–0.30pp) | 23% | **92.3%** |
| high conviction (>0.30pp) | 32% | **91.7%** |
| callable (≥0.05pp) | 81% | **89.0%** |

The buckets are the product: on the 81% of months where the model leans at
all, it is right 89% of the time, and the coin-flip bucket is honestly
labelled as such. No calendar year in the sample scores below 70%.

Why this works is arithmetic, not magic: next month's YoY change is roughly
next month's MoM minus the *already published* MoM from a year ago, so only
one monthly number needs forecasting and the hurdle is known in advance.

### 2. CPI level accuracy

Mean absolute error of the CPI YoY forecast, in percentage points, against
two naive benchmarks built from the same vintage:

| Horizon | Model MAE | Re-scored vs NSA | Random walk | Seasonal naive | Skill vs RW |
|---|---|---|---|---|---|
| 1 month | **0.17** | 0.17 | 0.30 | 0.21 | 44% |
| 3 months | **0.46** | 0.46 | 0.66 | 0.51 | 30% |
| 6 months | **0.82** | 0.83 | 1.03 | 0.88 | 21% |
| 12 months | **1.53** | 1.54 | 1.78 | 1.78 | 14% |

MoM error is a flat ~0.16–0.21pp at every horizon, so the growing YoY error
is accumulated monthly error rather than a model that decays. The NSA column
matters for honesty: the model projects the seasonally adjusted index, whose
factors BLS revises every February, so each YoY figure is re-scored against
CPIAUCNS, which is never revised. The two agree to 0.01pp — no result here
rests on revised seasonal factors.

By calendar year the one-print-ahead YoY error is 0.09–0.15pp in normal
years and peaks at **0.38pp in 2021**. That shows up as a negative bias
growing with horizon (−0.05pp at 1 month, −0.77pp at 12): across 2017–2026
the model under-forecasts US inflation, because a mean-reverting component
model cannot anticipate a regime break. It was late to the surge, like
everyone else.

### 3. Quad hit rates

Against **first-release** realized quads — what a real-time reader actually
saw, and the fair test for a real-time product:

| Horizon | Model | High conviction | Persistence | Base effects | Random |
|---|---|---|---|---|---|
| 0q (nowcast) | **60.5%** | **73.6%** | 31.6% | 57.0% | 25% |
| +1q | 45.0% | 49.4% | 27.0% | 45.9% | 25% |
| +2q | 47.2% | 47.6% | 25.0% | 41.7% | 25% |
| +3q | 41.0% | 40.0% | 22.9% | 40.0% | 25% |
| +4q | 30.4% | 32.8% | 26.5% | 33.3% | 25% |

Against final-vintage quads: 56.1 / 43.2 / 47.2 / 43.8 / 35.3%. Both bases,
plus flip precision/recall and confusion matrices, are in `results_us/`.

**Read this honestly.** The model beats persistence by 4–29pp and random by
5–35pp at every horizon. It does **not** meaningfully beat the base-effects
benchmark past the nowcast quarter: the edge runs −0.9 to +5.6pp on the
first-release basis and +1.0 to +8.3pp on the final basis, on ~110
observations per horizon — well inside sampling noise. Beyond one quarter the US
quad call is base-effect arithmetic on known year-ago levels, and the
component model adds little on top of it. The nowcast quarter is where the
model earns its keep.

### 4. Why the quad hit rate sits below the direction hit rates

The quad is the AND of two calls, so it can only be as good as the product
of the two axes:

| Horizon | Growth direction | Inflation direction | Product | Actual quad hit |
|---|---|---|---|---|
| 0q | 72.8% | 79.8% | 58.1% | 60.5% |
| +1q | 67.6% | 68.5% | 46.3% | 45.0% |
| +2q | 68.5% | 64.8% | 44.4% | 47.2% |
| +3q | 68.6% | 61.0% | 41.8% | 41.0% |
| +4q | 55.9% | 63.7% | 35.6% | 30.4% |

(first-release basis; the two axes are near-independent, so the quad lands
close to the product each time.)

The second reason is that many quarters are decided by a move smaller than
the model's own error bar: **29% of target quarters move their YoY
inflation rate by less than 0.10pp**. Those are flagged `low_conviction`, and
excluding them lifts the nowcast hit rate from 60.5% to 73.6%.

Growth is the binding axis — inflation calls direction 61–80% of the time
while growth started near a coin flip — which is what the next section
responds to.

## What the backtest changed in the model

The first cut of the US model blended trailing GDP momentum with indicator
*slots* that had no data behind them, so in practice it was momentum-only.
`usmodel/nowcast.py` replaces that with a ridge regression from published
monthly activity data to the current quarter's real GDP QoQ, **refitted at
every as-of date on only the history published at that date**. Partial
quarters are explicit: `k` is the number of months of the target quarter
already published, and a separate fit runs per `k`, so training and
prediction features always match in construction. At `k=0` the features are
read one quarter back and it becomes a genuine one-quarter-ahead model.

The nowcast error falls as the quarter fills in, exactly as it should
(MAE of QoQ growth, pp):

| Months of the target quarter published | n | Fitted nowcast | Momentum |
|---|---|---|---|
| k=0 (one quarter ahead) | 38 | 0.89 | 1.13 |
| k=1 | 38 | 0.59 | 1.13 |
| k=2 | 38 | 0.47 | 1.13 |

The convergence path beyond the nowcast quarter is fitted too — an AR(1) on
published QoQ growth, clipped at zero persistence because any window
containing 2020 fits a *negative* rho that would make the projection
oscillate instead of converge.

`python us_backtest.py --growth-variants` runs the full 2×2 and writes
`results_us/us_growth_variants.csv`. The shipped setting was picked off that
table, so the whole selection surface is published rather than just the
winner:

| Variant | Quad hit h0 | Quad hit mean | Growth dir h0 | Growth YoY MAE h0 | MAE mean |
|---|---|---|---|---|---|
| momentum + static convergence (original) | 56.1% | 44.7% | 65.8% | 1.38 | 1.80 |
| indicators + static convergence | 60.5% | 42.4% | 72.8% | 0.89 | 1.58 |
| momentum + fitted convergence | 56.1% | 45.3% | 65.8% | 1.38 | 1.80 |
| **indicators + fitted convergence (shipped)** | **60.5%** | 44.8% | **72.8%** | **0.89** | **1.54** |

The growth *level* forecast improves 36% at the nowcast quarter and 15% on
average, and the nowcast-quarter quad call gains 4.4pp. The multi-quarter
quad hit rate does not move outside noise in any of the four — consistent
with the base-effects finding above, and the reason the mean column is not
the thing to optimize.

## Caveats that bound every number here

1. **Simulated GDP vintages.** BEA revises real GDP heavily, and both
   archives of what a quarter looked like on release day — ALFRED and the
   Philadelphia Fed Real-Time Data Set — sit behind bot protection this
   sandbox cannot clear. The default `noise` mode simulates first releases
   with a persistent per-quarter error, sigma 0.35pp of QoQ (≈1.4pp
   annualized, in line with BEA's published advance-to-latest revision
   statistics). Every output row records the mode that produced it. Drop a
   real vintage panel at `data/us/gdp_vintages.csv` and the harness switches
   to `realtime` automatically; `--revision-mode none` hands the model final
   GDP and gives the upper bound.
2. **Indicator revisions in the nowcast training set.** Prediction features
   and the regression target are strictly point-in-time, but the training
   *features* use current-vintage payrolls and industrial production, which
   are revised. The growth numbers are therefore a modest upper bound.
   `backtest/snapshots.py` is accumulating a real indicator archive to close
   this.
3. **One inflation cycle.** 115 as-of dates, but essentially one regime
   break. The direction call held through it; the level forecast did not.
   Do not read 0.17pp one-month MAE as a promise for the next shock.
4. **No live track record.** Everything above is a backtest. The Norwegian
   side appends to `forecast_history.csv` daily; the US model has no
   equivalent running yet.

## Next, in payoff order

1. **Real GDP vintages** — the biggest credibility upgrade available, and it
   is one manual download away.
2. **Growth beyond the nowcast quarter** — where the model is near a coin
   flip and where base effects are not beaten. A quarter-ahead factor model
   on the same indicator panel is the obvious next attempt.
3. **A live US forecast loop** mirroring `forecast.py`, appending every call
   to a history file — the proof no backtest can supply.
4. **Core CPI as a second target.** The Fed steers on core; `CPILFESL` is
   already cached and the block structure supports it.
