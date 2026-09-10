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
| ALL months | 100% | **84.3%** |
| coin-flip (<0.05pp) | 19% | 59.1% |
| lean (0.05–0.15pp) | 24% | 85.7% |
| call (0.15–0.30pp) | 25% | **93.1%** |
| high conviction (>0.30pp) | 31% | **91.7%** |
| callable (≥0.05pp) | 81% | **90.3%** |

The buckets are the product: on the 81% of months where the model leans at
all, it is right 90% of the time, and the coin-flip bucket is honestly
labelled as such. No calendar year in the sample scores below 70%.

Why this works is arithmetic, not magic: next month's YoY change is roughly
next month's MoM minus the *already published* MoM from a year ago, so only
one monthly number needs forecasting and the hurdle is known in advance.

### 2. CPI level accuracy

Mean absolute error of the CPI YoY forecast, in percentage points, against
two naive benchmarks built from the same vintage:

| Horizon | Model MAE | Re-scored vs NSA | Random walk | Seasonal naive | Skill vs RW |
|---|---|---|---|---|---|
| 1 month | **0.16** | 0.17 | 0.30 | 0.21 | 45% |
| 3 months | **0.45** | 0.45 | 0.67 | 0.50 | 33% |
| 6 months | **0.81** | 0.81 | 1.04 | 0.87 | 22% |
| 12 months | **1.50** | 1.51 | 1.76 | 1.77 | 15% |

MoM error is a flat ~0.16–0.21pp at every horizon, so the growing YoY error
is accumulated monthly error rather than a model that decays. The NSA column
matters for honesty: the model projects the seasonally adjusted index, whose
factors BLS revises every February, so each YoY figure is re-scored against
CPIAUCNS, which is never revised. The two agree to 0.01pp — no result here
rests on revised seasonal factors.

By calendar year the one-print-ahead YoY error is 0.09–0.15pp in normal
years and peaks at **0.38pp in 2021**. That shows up as a negative bias
growing with horizon (−0.05pp at 1 month, −0.75pp at 12): across 2017–2026
the model under-forecasts US inflation, because a mean-reverting component
model cannot anticipate a regime break. It was late to the surge, like
everyone else.

### 3. Quad hit rates

Against **first-release** realized quads — what a real-time reader actually
saw, and the fair test for a real-time product:

| Horizon | Model | High conviction | Persistence | Base effects | Random |
|---|---|---|---|---|---|
| 0q (nowcast) | **58.8%** | **71.8%** | 31.6% | 57.0% | 25% |
| +1q | 45.9% | 52.1% | 27.0% | 45.9% | 25% |
| +2q | 41.7% | 47.0% | 25.0% | 41.7% | 25% |
| +3q | 38.1% | 42.3% | 22.9% | 40.0% | 25% |
| +4q | 32.4% | 39.1% | 26.5% | 33.3% | 25% |

Against final-vintage quads: 54.4 / 45.9 / 47.2 / 46.7 / 37.3%, where the
edge over base effects is +0.0 / +8.1 / +8.3 / +3.8 / +8.8pp. The two bases
disagree by more than the effect being measured, which is the honest summary:
**the model beats persistence by 6–27pp and random by 7–34pp at every
horizon, and ties the base-effects benchmark past the nowcast quarter.**
Full tables in `results_us/`.

### 4. Why the quad hit rate sits below the direction hit rates

The quad is the AND of two calls, so it can only be as good as the product
of the two axes:

| Horizon | Growth direction | Inflation direction | Product | Actual quad hit |
|---|---|---|---|---|
| 0q | 71.9% | 79.8% | 57.4% | 58.8% |
| +1q | 70.3% | 68.5% | 48.2% | 45.9% |
| +2q | 66.7% | 64.8% | 43.2% | 41.7% |
| +3q | 65.7% | 61.0% | 40.1% | 38.1% |
| +4q | 53.9% | 63.7% | 34.3% | 32.4% |

(first-release basis; the axes are near-independent, so the quad lands close
to the product each time.) The second reason is that many quarters are
decided by a move smaller than the model's own error bar: **29% of target
quarters move their YoY inflation rate by less than 0.10pp**. Those are
flagged `low_conviction`, and excluding them lifts the nowcast hit rate from
58.8% to 71.8%.

## Direction calls — the product

Hitting the level is not the job; calling the *direction* is. So the four
calls a subscriber actually uses are backtested as calls, each with its
conviction, so you know when to trust one:

| Call | Question it answers |
|---|---|
| growth, YoY basis | is YoY GDP growth accelerating or decelerating? (the quad's growth axis) |
| growth, QoQ basis | will this quarter's sequential growth print above or below last quarter's? (two reversal votes: GDP and real PCE) |
| inflation, YoY basis | is the quarterly-average CPI YoY rate accelerating or decelerating? (the quad's inflation axis) |
| inflation, QoQ basis | is the sequential quarterly CPI pace picking up or slowing? |

Hit rate by horizon, 2017–2026, final-vintage truth (`results_us/us_direction.csv`;
ex-COVID targets in brackets):

| Horizon | growth YoY | growth QoQ | inflation YoY | inflation QoQ |
|---|---|---|---|---|
| 0q (current) | 64.9% (63.5) | **71%** blended; **73–79% when votes agree** | **81.7%** | **87.0%** |
| +1q | 64.9% (64.5) | 51–57% *weak lean* | 71.9% | 63.2% |
| +2q | 66.7% (66.7) | no call | 68.5% | 60.4% |
| +3q | 68.6% (69.0) | no call | 64.8% | 52.8% |
| +4q | 56.9% (54.8) | no call | 65.7% | 51.4% |

Hit rate by conviction, all horizons pooled (`us_direction_conviction.csv`):

| Conviction | growth YoY | inflation YoY | inflation QoQ |
|---|---|---|---|
| <0.10pp | 64.4% | 47.5% | 53.6% |
| 0.10–0.25pp | 60.0% | 77.6% | 76.5% |
| 0.25–0.50pp | 49.3% | 66.7% | **86.2%** |
| >0.50pp | **78.2%** | **82.4%** | 75.6% |

How to read that:

* **Inflation grades cleanly.** Both inflation calls get better as conviction
  rises, so the conviction number is a usable trust dial — the same
  property the monthly CPI print call has (84% overall, 90% when it leans).
* **Growth YoY is two-regime, not graded.** Above 0.50pp it is right ~80%
  of the time (81% ex-COVID); below that it is 50–64% regardless of size.
  So the flag for growth is a single threshold, `YOY_HIGH_CONVICTION_PP =
  0.50`, and `run_us.py` labels growth-YoY calls *strong* or *weak* on it.
  The ceiling analysis below is why: the call's only real signal is a known
  base effect, and either that base is far from trend or it isn't.
* **Growth QoQ is a different model, graded by agreement, and it abstains.** See next.

### Growth on a QoQ basis: two reversal votes

This was the "try something else" that worked. The activity-indicator
nowcast, which cuts the QoQ *level* error substantially, turns out to be a
coin flip — **50%** — on the *sign* of this quarter's deviation from trend.
It knows roughly where the quarter lands, not which side of trend. So every
sequential-direction call routed through the nowcast collapses toward 50%.

What does work is the oldest fact about US GDP: quarterly growth
mean-reverts hard (autocorrelation of ΔQoQ ≈ −0.43), so **a quarter that
printed above trend is followed by a lower print ~72% of the time, from
published data alone.** `usmodel/growth_direction.py` fits a point-in-time
AR(1) on published QoQ growth (winsorized at 3 MADs so 2020Q2/Q3 cannot own
it) and calls the current quarter as reversal toward the AR mean. Blending
the nowcast in only dilutes it.

A second vote comes from the consumer. Real personal consumption is ~68% of
GDP, its quarterly growth mean-reverts the same way, and it is published
monthly — so the last full quarter of real PCE growth against its own
trailing median is an independent reversal signal, known before GDP is. The
call is graded by whether the two votes agree:

| Current-quarter sequential-growth call | final GDP | simulated revisions | ex-COVID |
|---|---|---|---|
| nowcast vs last print (what the level path implies) | 68% | 68% | 68% |
| GDP reversal vote alone | 63% | 63–76% across seeds | 66–81% |
| real-PCE reversal vote alone | 68% | 68% | 75% |
| **both votes agree (~2 in 3 months) — graded "call"** | **73%** | **79%** | **81–91%** |
| votes split — graded "coin flip" | 67% | 30% | — |

Two honesty notes. The single GDP-vote number depends on the revision
assumption — 63% on final data, 63–76% across simulated-revision seeds, and
an earlier version of this README quoted the top of that range. The
agreement call does not: 73–79% whichever way revisions are simulated, which
is what makes it the call. And the split grade is unreliable in *both*
directions on a dozen distinct quarters, so it is labelled a coin flip rather
than given a number it has not earned.

One quarter further out the same structure gives a *zigzag* lean — ΔQoQ
alternates in sign, so the lean for q+1 is the opposite of the call for q —
measured at **51% on final data** (57% under simulated revisions): close to
nothing, labelled weak. Past that, nothing tested beats a coin flip (direct
regressions, AR(2)–AR(4), iterated glides, all 37–58% across horizons and
windows), so **the model abstains rather than manufacture a number**.
Inflation on a QoQ basis was checked for the same reversal structure; the
bottom-up CPI path already beats it, so inflation is unchanged.

### The consumer: what extremes actually predict

The natural next question is whether a *stretched* consumer — spending or
confidence at a high — is a warning. Measured on the realized record
(1995–2026, ex-COVID, quarterly, each series against the top and bottom
decile of its own trailing decade), the answer splits:

| Series at an extreme | Next-quarter GDP QoQ lower | Next-quarter YoY growth decelerating |
|---|---|---|
| base rate | 51% | 55% |
| **real PCE growth, top decile** | **70–86%** | **70%** |
| consumer sentiment, top decile | 47% (no signal) | 53% |
| consumer sentiment, **bottom** decile | **37%** (i.e. growth picks up) | 63% |
| retail sales YoY, top decile | 50% | 62% |
| real disposable income, top decile | 33% (growth picks up) | 33% |
| saving rate, top decile | 47% | 53% |

So the intuition is right for **spending** and wrong for **confidence**:
stretched spending growth reverts and drags GDP with it; a confidence high
predicts nothing, and it is a confidence *low* that carries signal, in the
other direction. Strong income growth is a *tailwind* (income leads
spending). Walk-forward, the stretched-spending flag does not help the
sequential call as an override (63%, worse than the reversal), but on the
**YoY axis** it is a real upgrade: when real PCE growth is in its top decile
*and* the model already says decelerating, the call is right **79–84%** of
the time against 64–66% otherwise (`results_us/us_consumer_stretch.csv`).
Adding PCE and the saving rate to the nowcast regression makes it *worse*
(PCE only starts in 2007, so the fit window is short and COVID-heavy), so
they stay out of it.

`growth_direction.consumer_state` reports the block as a reader wants it —
spending, sentiment, income and saving against their own trailing decade,
each flag carrying its base rate — and `run_us.py` prints it above the
direction calls.

The live block, `python run_us.py`:

```
=== Direction calls (accelerating / decelerating) ===
quarter              growth YoY             growth QoQ          inflation YoY          inflation QoQ
2026Q3   down  0.49pp      weak   up  0.15pp      call down  0.58pp    strong down  1.29pp    strong
2026Q4     up  0.66pp    strong down  0.41pp      lean down  0.00pp      weak   up  0.42pp    strong
2027Q1     up  0.26pp      weak                no call down  0.32pp    strong down  0.06pp      weak
```

## The growth axis, and where its ceiling is

The first backtest showed growth was the binding axis, so it got a second
pass. The useful output is not a bigger number — it is knowing exactly how
big the number can get.

### The call is half-known, like the inflation one

```
d_growth(q) = yoy(q) - yoy(q-1)  ~=  qoq(q) - qoq(q-4)
```

and `qoq(q-4)` is **already published** at every horizon the quad table
covers except +4q. So the growth call is not "what will GDP do" — it is
"will next year's quarterly growth land above or below a number we already
have". Exactly the structure that makes the CPI direction call work.

### Past the nowcast quarter, the other half is not forecastable

Everything reasonable was tried, walk-forward and point-in-time, targeting
`qoq(q)` at 1–5 quarters out:

* geometric convergence from the nowcast (the original design)
* a point-in-time AR(1) on published QoQ
* direct ridge regressions fitted separately at each horizon on the
  coincident panel (payrolls, IP, retail sales, claims, CFNAI, hours)
* the same on a **leading** panel added for this purpose — Chicago Fed
  financial conditions, Baa credit spread, building permits, core capital
  goods orders, equities, real M2, housing starts
* both panels together

Past the nowcast quarter every one of them had **~zero correlation with
realized QoQ** (−0.29 to +0.36 across horizons, straddling zero) and a
**worse MAE than simply predicting a constant**. The leading panel is kept
in `usmodel/nowcast.py` as `LEADING_SPEC` so the negative result stays
reproducible rather than being a claim.

That has a direct consequence: a projected path that *moves* without
carrying information adds error uncorrelated with the truth on top of the
one real signal, the known base. So the projection now **steps** to trend
the moment the nowcast quarter is past, instead of gliding. Measured, growth
direction falls monotonically as persistence rises:

| Persistence carried past the nowcast quarter | Growth direction (mean) |
|---|---|
| **0.0 — flat at trend (shipped)** | **65.3%** |
| 0.30 (the sample AR(1)) | 64.4% |
| 0.50 (the original config) | 63.9% |

The constant it steps to is the vintage's own **trailing 24-quarter median**
QoQ — median, not mean, because 2020Q2/Q3 are a crash-and-rebound pair that
drags a mean for years, and this constant is what every multi-quarter growth
call is measured against. Using the *static* config constant instead costs
5pp of growth direction, which is the same sensitivity the arithmetic
predicts (a 0.2pp error in the constant costs ~1.8pp of accuracy).

One thing to know before reading a live growth *level*: this estimator is
backward-looking, and on today's data the 24-quarter median is 0.77% QoQ —
**3.1% annualized**, well above any estimate of US potential, because the
window is dominated by the post-2020 expansion. Windows from 12 to 40
quarters all score within noise of each other in the backtest (63.3–65.3%
growth direction) while implying trends of 2.7–3.1% annualized, so the
window choice is not doing real work; but the projected growth level inherits
whichever one is used. The quad only consumes the *direction*, and across
that whole 0.10pp spread of constants the direction call moves by well under
a point — which is why the level caveat does not propagate into the hit
rates. `run_us.py` prints the trend it is using on every live run.

### The ceiling

If `qoq(q)` is unforecastable and the constant is well calibrated, the call
"trend vs known base" is right `E[Φ(|Z|)] = 75%` of the time. That is a
reference point, not a hard bound — an adaptive constant can beat it — but
it is the right thing to measure against. Scored on the same rows as the
model (`results_us/us_growth_ceiling.csv`):

| Horizon | Reference ceiling | Model | Gap | Year-ago base published? |
|---|---|---|---|---|
| 0q | 73.7% | 71.9% | −1.8pp | always |
| +1q | 73.0% | 70.3% | −2.7pp | always |
| +2q | 69.4% | 66.7% | −2.8pp | always |
| +3q | 74.3% | 65.7% | −8.6pp | always |
| +4q | 70.6% | 53.9% | −16.7pp | **never** |

Two readings, and they are the point of this section:

1. **At 0–2 quarters the model is within 3pp of the ceiling.** There is
   almost nothing left on the table there, and no amount of extra GDP
   modelling will find it.
2. **+4q is structurally different.** Its year-ago base is the as-of quarter
   itself, which BEA has not published — so instead of a known number the
   model must use its own nowcast, and inherits that error. The lever for
   +4q is therefore the *nowcast*, the same lever as 0q; it is not a
   longer-horizon modelling problem.

The +3q gap is mostly an artefact of the backtest, not the model: re-run
with `--revision-mode none` (final GDP instead of simulated first releases)
and growth direction jumps to 78.4 / 75.0 / 71.4% at +1/+2/+3q, above the
reference. **Simulated GDP revisions cost 5–8pp of growth-direction accuracy
at those horizons** — which is a strong argument for getting real vintages
(caveat 1 below) and a reason to read the growth numbers as a lower bound.
+4q is unmoved by revisions (53.9% vs 52.9%), confirming its gap is nowcast
error and nothing else.

### What actually changed

`python us_backtest.py --growth-variants` writes the whole surface to
`results_us/us_growth_variants.csv`:

| Variant | Quad hit h0 | Quad hit mean | Growth dir h0 | Growth dir h≥1 | Growth YoY MAE (mean) |
|---|---|---|---|---|---|
| momentum nowcast + glide (original) | 56.1% | 41.9% | 65.8% | 61.1% | 1.81 |
| fitted nowcast + glide | 58.8% | 43.4% | 71.9% | 63.2% | 1.57 |
| fitted nowcast + static trend, flat | 58.8% | 41.7% | 71.9% | 59.1% | 1.43 |
| **fitted nowcast + fitted trend, flat (shipped)** | **58.8%** | **43.4%** | **71.9%** | **64.1%** | **1.38** |

The shipped setting is best or tied on every column. Be precise about what
moved, though:

* **Error metrics improved materially and consistently.** Growth YoY MAE is
  down 24% on average versus the original (1.81 → 1.38pp) and 21% at +4q;
  the MAE of `d_growth` itself is down 18% at +1q. These are continuous
  measures over 102–111 observations, so they are real.
* **Hit rates did not move outside noise.** The effective sample is only
  ~38 distinct target quarters — as-of dates overlap heavily — so the ±1–3pp
  swings in quad hit rate between variants, and between the final and
  first-release bases, are not evidence of anything. The honest claim is
  that the growth axis is now *at* its ceiling for 0–2 quarters, not that
  it got dramatically more accurate.

## Caveats that bound every number here

0. **October 2025 does not exist.** BLS never published that month's CPI
   (the autumn 2025 shutdown) and FRED carries it as missing. A positional
   12-row shift then silently reads every 12-month rate spanning the hole
   as a 13-month one — the live July 2026 YoY read 3.54% where the true
   rate is 3.30%. Two defenses now hold: the loader fills an isolated
   missing month by geometric interpolation and records it on the bundle
   (`bundle.filled`), and every YoY in the model and the scorer is computed
   by calendar alignment (`quads.calendar_pct_change`), so a hole can only
   ever yield NaN, never a wrong number. October 2026's YoY rests on the
   filled base and is flagged wherever it is printed.
1. **Simulated GDP vintages.** BEA revises real GDP heavily, and both
   archives of what a quarter looked like on release day — ALFRED and the
   Philadelphia Fed Real-Time Data Set — sit behind bot protection this
   sandbox cannot clear. The default `noise` mode simulates first releases
   with a persistent per-quarter error, sigma 0.35pp of QoQ (≈1.4pp
   annualized, in line with BEA's published advance-to-latest revision
   statistics). Every output row records the mode that produced it. As the
   growth section shows, this costs 5–8pp of growth-direction accuracy at
   +1 to +3q, so it is now the single biggest measurable distortion in the
   backtest. Drop a real vintage panel at `data/us/gdp_vintages.csv` and the
   harness switches to `realtime` automatically; `--revision-mode none`
   gives the upper bound.
2. **Indicator revisions in the nowcast training set.** Prediction features
   and the regression target are strictly point-in-time, but the training
   *features* use current-vintage payrolls and industrial production, which
   are revised. The growth numbers are therefore a modest upper bound in
   that one respect. `backtest/snapshots.py` is accumulating a real
   indicator archive to close this.
3. **One inflation cycle.** 115 as-of dates, but essentially one regime
   break, and only ~38 distinct target quarters for the quarterly metrics.
   The direction call held through it; the level forecast did not. Do not
   read 0.17pp one-month MAE as a promise for the next shock.
4. **No live track record.** Everything above is a backtest. The Norwegian
   side appends to `forecast_history.csv` daily; the US model has no
   equivalent running yet.

## Next, in payoff order

1. **Real GDP vintages** — the biggest credibility upgrade available, and it
   is one manual download away. The sequential-growth call's single-vote
   number swings 63–76% with the simulated-revision seed; real vintages
   would settle it.
2. **A better nowcast of the current quarter** — it is now the lever for
   every growth call that is still short of its ceiling (+4q YoY, and the
   sign of the current quarter's deviation from trend, which the present
   nowcast gets right only 50% of the time). Higher-frequency inputs
   (weekly claims, daily financial conditions, card-spend proxies) are the
   obvious next attempt; longer-horizon GDP modelling is not.
3. **A live US forecast loop** mirroring `forecast.py`, appending every call
   to a history file — the proof no backtest can supply.
4. **Core CPI as a second target.** The Fed steers on core; `CPILFESL` is
   already cached and the block structure supports it.
