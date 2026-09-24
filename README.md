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
| WTI, broad trade-weighted dollar | FRED, daily → monthly mean | the oil rule, and the core-goods block |
| **Retail gasoline, regular (EIA weekly)** | FRED | the gasoline block, directly |
| Real PCE (total, durables, nondurables, services), real retail sales, real income, saving rate, consumer credit | FRED | the consumer block and the second growth vote |
| Average hourly earnings | FRED (BLS) | supercore block |
| **ZORI market rents** | Zillow research CSVs | the shelter block — the model's biggest edge |
| Payrolls, industrial production, retail sales, initial claims, CFNAI, U. Mich sentiment, 10y–3m spread, manufacturing hours | FRED | the GDP nowcast |

ZORI starts 2015-01 and CPI shelter follows it with a ~12 month lag, so the
first as-of date whose shelter block is driven by *observed* market rents is
**2017-01**. That bounds the backtest: **115 monthly as-of dates, 2017-01 to
2026-07, 575 quad predictions.**

## The sheet

Naming: everything a reader sees says **regime** and uses the four names
Sweet spot, Heating, Squeeze and Cooling. "Quad" and "GIP" are Hedgeye's
product names and appear nowhere public; the code keeps "quad" as its
generic identifier for the four-cell classification.

`python us_sheet.py` writes `results_us/us_regime_sheet.html`, a plain-language
page for a reader rather than a modeller: the quads ahead as cards, both
series on one chart with the quad of each quarter shaded, a month-by-month
table (inflation monthly, GDP beside its three months), and what is behind
the numbers — oil and the pump, the bar each quarter has to beat, the
consumer, interest rates and the subscriber's own view against the model's
— with every hit rate read from `results_us/` so the page cannot quote a
number the backtest did not produce. Regenerate it after each data release.

`python us_feed.py` then writes `results_us/latest.json`, the compact feed a
website reads: the quarters and months with their regimes, directions,
conviction and hit rates; `hit_rates`, the backtest's scorecard in one block
(the regime by distance, next month's inflation and the growth measure by
conviction); and `notes`, the story behind the numbers as plain text with no
markup (the next inflation print, the inflation path, oil, rents, this
quarter's growth, the pace assumed after it, rates, the consumer), every
sentence computed from the sheet. The "US macro regime" workflow publishes
it to the `live-us-regime` branch on weekdays. `tests/test_us_feed.py` holds
the contract and that no other vendor's product names are in the public
blocks.

## Sectors by regime

`python us_sectors.py` answers "which sectors have done best in which R" with the SPDR
sector funds (XLE, XLF, XLK, XLV, XLI, XLP, XLY, XLU, XLB, XLRE, XLC), SPY, long and
intermediate Treasuries (TLT, IEF) and gold (GLD), from dividend-adjusted month-end
closes (Yahoo Finance, cached in `data/us/sector_etf_monthly.csv`). It writes
`results_us/us_sectors_by_regime.csv` (every cell: count, mean, median, share
positive, excess over SPY, t-statistic) and `results_us/us_sectors_feed.json`, which
`us_feed.py` folds into `latest.json` as `sectors`. Three bases, kept apart:

- **Realized, monthly** (Feb 2008 to Jul 2026, 222 months): the regime each month turned
  out to be, from the published data (the monthly growth measure's year-over-year change
  and the CPI's, signs only). Known only after the fact.
- **Called, monthly** (Jan 2017 to Sep 2026, 117 months): the model's point-in-time call for
  month M made at the end of M-1 (the backtest's vintage builder). The version a reader
  could have acted on; the call agrees with the outcome 57% of the time.
- **Realized, quarterly** (Q4 1998 to Q2 2026, 111 quarters): real GDP year-over-year
  direction x CPI year-over-year direction.

Average month per regime (realized), SPY's average and share of positive months, and the
funds with the best and worst average excess over SPY in that regime:

| R | months | SPY | best vs SPY (pp a month) | worst vs SPY |
|---|---|---|---|---|
| R1 | 58 | +1.38% (64%) | Consumer discretionary (+0.91), Industrials (+0.57), Technology (+0.48) | Energy (-1.34), Treasuries (7 to 10 years) (-0.75), Consumer staples (-0.64) |
| R2 | 55 | +1.69% (69%) | Technology (+0.96), Industrials (+0.09), Consumer discretionary (+0.05) | Treasuries (7 to 10 years) (-1.63), Long Treasuries (20+ years) (-1.54), Gold (-1.06) |
| R3 | 58 | +0.84% (71%) | Energy (+1.16), Technology (+0.42), Communication services (+0.40) | Long Treasuries (20+ years) (-1.43), Treasuries (7 to 10 years) (-0.98), Utilities (-0.53) |
| R4 | 51 | +0.10% (63%) | Health care (+0.90), Consumer staples (+0.74), Long Treasuries (20+ years) (+0.41) | Real estate (-0.65), Consumer discretionary (-0.52), Financials (-0.32) |

On the model's own calls (Jan 2017 to Sep 2026):

| R | months | SPY | best vs SPY | worst vs SPY |
|---|---|---|---|---|
| R1 | 36 | +1.93% (69%) | Technology (+1.18), Consumer discretionary (+0.64), Energy (+0.44) | Long Treasuries (20+ years) (-1.91), Treasuries (7 to 10 years) (-1.80), Consumer staples (-1.64) |
| R2 | 20 | +2.83% (85%) | Technology (+1.41), Financials (+0.08), Energy (-0.24) | Treasuries (7 to 10 years) (-2.73), Long Treasuries (20+ years) (-2.58), Consumer staples (-2.08) |
| R3 | 24 | +1.26% (71%) | Technology (+1.11), Consumer discretionary (+0.39), Communication services (-0.10) | Health care (-1.68), Long Treasuries (20+ years) (-1.48), Treasuries (7 to 10 years) (-1.14) |
| R4 | 37 | -0.17% (62%) | Gold (+1.31), Health care (+1.25), Consumer staples (+1.15) | Consumer discretionary (-1.06), Communication services (-0.33), Materials (-0.33) |

The pattern is the textbook one and it holds on the model's own calls: R2 (growth up,
inflation up) is the strong regime for stocks and the weak one for bonds; R4 (both down)
is the weak regime for stocks, where health care, staples, Treasuries and gold lead; R3
(growth down, inflation up) is where energy and gold earn their keep. Plain averages over
a handful of dozens of months each, so read the counts: nothing here is fitted, and none
of it is a recommendation.

## The wage input, and the round that tested it

Core services ex shelter is the one CPI block built from wages:
`supercore_yoy = a + b * wage_yoy`, with `a` and `b` refitted on each vintage's own
published history (`usbacktest.vintage.estimate_supercore_passthrough`). The series
behind `wage_yoy` is average hourly earnings (`CES0500000003`), a **mean** across
whoever is on payrolls, so it moves when the composition of employment changes rather
than when anyone is paid more. April 2020 printed +8.1% while the Atlanta Fed's
matched-person tracker printed +3.7%: low-wage workers had been laid off, not given
raises. Through 2022-24 the bias ran the other way, the tracker up to 2.5 points
hotter. On the raw series the case for switching looks overwhelming: against services
inflation on the 2015+ sample the tracker correlates 0.87 at a three to six month
lead where average hourly earnings manages 0.48-0.51, and the two series' month-to-month
**changes** correlate 0.05.

So the swap was built and scored, point-in-time, 2017-01 to 2026-08, 116 month-end
as-of dates. `USVintageConfig(wage_source="tracker")` feeds the block the Atlanta Fed
Wage Growth Tracker (`FRBATLWGTUMHWGO`, published lag set to a deliberately
conservative 20 days) instead, and the pass-through refits on whichever series is
chosen. Pre-registered bar: the tracker replaces average hourly earnings only if the
one-print-ahead CPI direction hit rate improves by at least 1.0 point on n >= 100, the
projection error is no worse at every horizon, the quarterly regime call is not worse
by more than 1.0 point, and neither half of the window is worse.

**Verdict: 2 of 4. NOT A PROMOTION. The model stays on average hourly earnings.**

| | average hourly earnings | tracker |
|---|---|---|
| CPI direction, one print ahead | 89.7% | 90.5% |
| CPI error, MAE at 1 / 3 / 6 / 12 months | 0.131 / 0.451 / 0.795 / 1.495 | 0.130 / 0.452 / 0.801 / 1.510 |
| Regime hit rate, horizons 0-4 | 0.596 / 0.378 / 0.389 / 0.381 / 0.333 | 0.596 / 0.378 / 0.389 / 0.390 / 0.333 |
| Fitted pass-through across the 116 vintages | mean 0.581, sd 0.248, range 0.101-0.852 | mean 0.558, sd 0.120, range 0.455-0.733 |

The direction figure fails the bar at +0.9 points, and the reason matters more than the
number: **the two arms disagree on exactly one call in 116**, January 2022, where
average hourly earnings predicted -0.013pp and the tracker +0.003pp. Both are zero to
any honest reading. The median absolute change in the predicted month-ahead move across
all 116 dates is 0.0044pp, and 10 of 580 quarterly regime calls differ.

Why so little, when the raw series are so different? **The self-calibration had already
absorbed it.** A series whose level is distorted by composition gets a compensating
intercept and slope every time the block is refitted, so the distortion never reaches
the projection. The last row of the table is the evidence: the tracker is clearly the
better-behaved regressor, its fitted pass-through varying across a 0.28 band where
average hourly earnings swings across 0.75 — and it still changes nothing, because the
refit was doing that work already. The second reason is dilution: supercore is one of
six CPI blocks, and shelter, which is larger, already carries new-lease rents at a
year's lag.

What would change the verdict: a fixed pass-through (then the series choice would matter
a great deal, and the tracker would be the right one), a supercore-only scoring target
rather than headline CPI, or true vintages for both series. Both arms here use the
current vintage with truncation only, so neither gets revision-realistic treatment; that
is an equal handicap, not a tilt.

### Round 2: the fixed pass-through, which was supposed to rescue it

Round 1's own explanation named the test that would overturn it, so it was run. Freeze
the coefficient (`supercore_passthrough_fixed` / `supercore_intercept_fixed`, research
only, off by default and leaving shelter untouched) and the wage series has to carry its
own weight. Both coefficients are fitted once on the SAME pre-window months and then
frozen, so neither series gets more history than the other, and nothing after the cut
enters the fit.

**It does not rescue the tracker. It makes the case against it stronger.**

| frozen arms, 2017-01 to 2026-08 | average hourly earnings | tracker |
|---|---|---|
| CPI direction, one print ahead | 90.5% | 90.5% |
| CPI error, MAE at 1 / 3 / 6 / 12 months | 0.128 / 0.437 / 0.765 / 1.444 | 0.130 / 0.451 / 0.799 / 1.515 |
| Regime hit rate, horizons 0-4 | 0.596 / 0.387 / 0.407 / 0.391 / 0.353 | 0.588 / 0.378 / 0.389 / 0.391 / 0.333 |

The direction call ties exactly: each series is right where the other is wrong on one
month apiece. Average hourly earnings is better on error at every horizon and on the
regime call at four of five. The tracker also fails the stability clause, 1.7 points
worse in the first half of the window. **0 of 4. Still not a promotion.**

Worth recording, because it is a trap this repo can fall into again: on the supercore
block ALONE the frozen tracker looked clearly better, MAE 0.863 against 1.002. Being
better at one of six CPI blocks did not survive into headline CPI. Block-level evidence
is not evidence about the published call.

### What the control arm found instead

The same run compared frozen against refitted for each series, and turned up something
nobody asked for: **freezing the supercore pass-through beats refitting it per vintage,
on the series the model already uses.** Repeated at seven cut dates, each fitting before
the cut and testing only after it:

| fit through | AHE refit, MAE 12m | AHE frozen | AHE refit dir | AHE frozen dir |
|---|---|---|---|---|
| 2013 | 1.4629 | **1.4199** | 88.8% | **90.1%** |
| 2014 | 1.3756 | **1.3348** | 90.0% | **91.4%** |
| 2015 | 1.3890 | **1.3459** | 89.1% | **90.6%** |
| 2016 | 1.4948 | **1.4440** | 89.7% | **90.5%** |
| 2017 | 1.6517 | **1.5938** | 88.5% | **89.4%** |
| 2018 | 1.7306 | **1.6669** | 88.0% | **89.1%** |
| 2019 | 1.8607 | **1.7968** | 87.5% | **90.0%** |

Seven of seven on error and seven of seven on direction. For the tracker the same test
goes the other way, refit winning six of seven, and its direction call does not move at
all. That resolves round 1's puzzle: average hourly earnings' LONG-RUN relationship with
supercore is stable (the frozen b sits in 0.749-0.879 across every cut date), but the
PER-VINTAGE refit was swinging across 0.101-0.852, fitting noise in short recent windows.
Freezing it at the long-run value removes that noise. The tracker never had the problem,
which is why freezing does nothing for it.

**This is a lead, not a result, and it is NOT PROMOTED.** It came out of a control arm
rather than a pre-registered hypothesis, and the cut dates were chosen as a round of
years rather than by any rule. Promoting it needs its own round with its own bar: a
rule for choosing and re-estimating the frozen coefficient (a long rolling window is the
obvious candidate rather than a literal constant), the quarterly regime call scored as
the target rather than watched, and the same treatment asked of the shelter pass-through,
which is refitted the same way and may have the same problem.


### Where it leaves the model

The tracker is kept and **shown, not used**. `latest.json`'s `wages` driver row carries
the matched-person number with average hourly earnings beside it and the job
switcher/stayer split (`FRBATLWGT12MMUMHWGJSW` / `...JST`), because it is the better
read for a person even though it is not a better input for this model.
`tests/test_wage_source.py` pins the switch, the publication lag, the fact that the
tracker is a rate and is never differenced again, the fallback when it is not cached,
and that the default is still average hourly earnings.

## Surveys, the dead ISM weight, and the peak that was not one

`GDP_NOWCAST_WEIGHTS` carried an `ism` slot at 0.30 from the day the model was written
and it never fired once. ISM restricted redistribution, FRED dropped the NAPM series,
and nothing ever supplied one, so `gdp.nowcast_qoq` renormalised over its three real
slots and a reader of the config was told a third of the growth fallback rested on an
input that did not exist. It is gone. Removing it is numerically inert, and
`tests/test_pmi_surveys.py` pins that rather than asking anyone to take it on trust. The
slot was doubly dead: the live path is the ridge nowcast, and the weighted blend it sat
in is only the fallback for vintages too early to fit.

The free substitutes are the regional Fed manufacturing surveys, the same family of
diffusion index and what forecasters use to nowcast the ISM itself. They are also the
**timeliest** input available, published during the month they describe where payrolls
land eight days after it and industrial production seventeen:

| series | FRED id | from | publication lag used |
|---|---|---|---|
| Philadelphia, general activity | `GACDFSA066MSFRBPHI` | May 1968 | 0 |
| Philadelphia, new orders | `NOCDFSA066MSFRBPHI` | May 1968 | 0 |
| Empire State, general activity | `GACDISA066MSFRBNY` | Jul 2001 | 0 |
| Dallas, business activity | `BACTSAMFRBDAL` | Jun 2004 | 0 |

A lag of zero means "known once the month is over", which at the backtest's month-end
as-of dates is exact and for a live run understates their timeliness. They enter the
nowcast as LEVELS (`nowcast.SURVEY_SPEC`): a diffusion index is already a rate of change
in disguise, so differencing it would ask for the change in the change.
`USVintageConfig(nowcast_spec=...)` picks the panel: `base` is production, `surveys` adds
all four, `philly` adds only the longest.

### They make the size better and the direction worse

Scored point-in-time, 2017-01 to 2026-08, 114 nowcasts. Pre-registered bar: the nowcast
error improves, growth direction at horizon 0 improves by at least 2.0 points, the regime
call is not worse by more than 1.0 point at any horizon, and neither half of the window
is worse.

| | base | + Philadelphia | + all four |
|---|---|---|---|
| Nowcast QoQ, MAE | 0.6609 | 0.6521 | 0.6526 |
| Nowcast QoQ, RMSE | 1.3189 | 1.2480 | **1.1652** |
| Growth direction, horizon 0 | **0.6667** | 0.6491 | 0.6053 |
| Regime hit rate, horizon 0 | **0.5965** | 0.5789 | 0.5263 |
| Growth direction, first / second half | 61.4 / 71.9 | 57.9 / 71.9 | 57.9 / 63.2 |

**0 of 4. NOT A PROMOTION.** The surveys cut the nowcast's RMSE by 12%, so they genuinely
help with the SIZE of a quarter, mostly by shrinking the large misses. They make the SIGN
worse at every variant, and the sign is the only half the regime call uses. A more
accurate level with a worse direction is worth nothing here. Horizons 1 to 4 barely move,
which is expected: past the nowcast quarter this model does not forecast GDP at all.

### The peak claim, tested

The prompt for all this was a cycle-high PMI print and the inference that growth has
therefore peaked and the cycle turns down. That is testable. Three point-in-time readings
of "peaked", each against the question the model cares about: does the monthly growth
measure's year-over-year rate come in LOWER k months later?

Unconditionally it does 48.9% of the time at one month, 49.3% at three and six, 52.0% at
twelve. A coin flip. Conditional on a survey standing at a twelve-month high:

| survey (edge over base rate) | k=1 | k=3 | k=6 | k=12 | fires |
|---|---|---|---|---|---|
| Philadelphia, general activity | -12.9 | -21.3 | -9.3 | -12.0 | 25 |
| Philadelphia, new orders | -5.5 | -19.3 | -12.7 | -15.4 | 30 |
| Empire State | -19.2 | -30.8 | -27.1 | -18.7 | 27 |
| Dallas | -11.4 | -30.6 | -27.5 | -14.5 | 32 |

**Every survey, every horizon, the wrong sign for the thesis.** Eleven of twelve cells for
"rolled over" (fell after a local high) go the same way. A survey at a twelve-month high
has been followed by growth CONTINUING to accelerate more often than the base rate, not
less: three months after an Empire State local high, growth was lower only 18.5% of the
time against 49.3% unconditionally. At these horizons these indices carry momentum, not
mean reversion into a growth decline.

One cell supports the thesis and is worth naming rather than burying: Empire State in the
top quintile of its trailing ten years is followed by lower growth twelve months later
75.8% of the time against 52.0% base, on 33 observations. It is one survey at one horizon,
it disagrees with the same survey's local-high result, and it is not enough to act on.

Bounds on all of it: the sample is 2008-2026, 223 months, containing two downturns, so
genuine pre-recession survey peaks are nearly absent from it; the windows overlap, so the
effective sample is far smaller than 223; and a twelve-month high is a weak proxy for a
cycle peak, which is only identifiable afterwards. What the test does establish is the
narrower and more useful claim: **on this data, a high or newly-peaking survey is not by
itself evidence that growth is about to turn down.**

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
| ALL months | 100% | **89.7%** |
| coin-flip (<0.05pp) | 13% | 66.7% |
| lean (0.05–0.15pp) | 27% | 90.3% |
| call (0.15–0.30pp) | 33% | **89.5%** |
| high conviction (>0.30pp) | 28% | **100%** (32 of 32) |
| callable (≥0.05pp) | 87% | **93.1%** |

The buckets are the product: on the 87% of months where the model leans at
all, it is right 93% of the time, and the coin-flip bucket is honestly
labelled as such.

Why this works is arithmetic, not magic: next month's YoY change is roughly
next month's MoM minus the *already published* MoM from a year ago, so only
one monthly number needs forecasting and the hurdle is known in advance.

### Oil, the pump, and the month in progress

Crude to headline is the Hedgeye rule — **3 bps of headline CPI per $1/bbl**
move in monthly-average WTI — and the data confirms it: 3.1 bps same-month
on 2000–2026 (2.2 bps once the prior month's move is included). But the
gasoline index follows the *pump* price, which lags crude by weeks and is
public every Monday (EIA regular, `GASREGW`), and the two can diverge for a
month: in August 2026 WTI's monthly average rose $3 while the BLS gasoline
index rose 3.9%. So the gasoline block now runs off the retail pump price
for every month with an observed price, with BLS's seasonal factor added
back — that factor is large, about −4.5pp in March/April and +2 to +4pp in
September–December (`usbacktest.vintage.estimate_gasoline_seasonal`, fitted
point-in-time). Past the observed months the SA block is flat: an unobserved
pump price is expected to follow its own seasonal, which the SA index
removes. The WTI rule stays as the fallback and as the cross-check printed
on the sheet.

That change is worth a lot on the call that matters most: one-month YoY MAE
**0.164 → 0.131pp**, next-print direction **84% → 90%** overall and 93% when
the model leans, and **32 of 32** on high-conviction months. It costs
nothing at longer horizons.

Live runs also now see the month in progress. A live forecast's whole job is
"what is happening now", and a $14 September oil move is public
information on the day, not a forecast — so `USVintageConfig(live_partial_month=True)`
lets the daily and weekly market series (WTI, the dollar, the pump price)
contribute their partial-month mean. The backtest keeps this off: its as-of
dates are month-ends, where the month is complete anyway. The published
10 September call carried August's oil flat and missed September's rise
entirely; that is the gap this closes.

### 2. CPI level accuracy

Mean absolute error of the CPI YoY forecast, in percentage points, against
two naive benchmarks built from the same vintage:

| Horizon | Model MAE | Re-scored vs NSA | Random walk | Seasonal naive | Skill vs RW |
|---|---|---|---|---|---|
| 1 month | **0.13** | 0.13 | 0.30 | 0.21 | 56% |
| 3 months | **0.45** | 0.45 | 0.67 | 0.51 | 32% |
| 6 months | **0.80** | 0.80 | 1.04 | 0.87 | 23% |
| 12 months | **1.50** | 1.51 | 1.75 | 1.75 | 15% |

MoM error is a flat ~0.13–0.21pp at every horizon, so the growing YoY error
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

| Horizon | Shipped: potential pace + rate channel | Potential pace, rate channel off | Trailing-median pace + rate channel | Trailing-median pace, flat | High conviction (shipped) | Persistence | Base effects | Random |
|---|---|---|---|---|---|---|---|---|
| 0q (nowcast) | **64.0%** | 64.0% | 64.0% | 64.0% | **70.4%** | 28.9% | 57.0% | 25% |
| +1q | 37.8% | 41.4% | 35.1% | 45.9% | 40.9% | 27.0% | 45.9% | 25% |
| +2q | 38.9% | 47.2% | 36.1% | 41.7% | 34.6% | 25.0% | 41.7% | 25% |
| +3q | 35.2% | 41.0% | 35.2% | 38.1% | 36.0% | 22.9% | 40.0% | 25% |
| +4q | 25.5% | 31.4% | 29.4% | 29.4% | 25.0% | 26.5% | 30.4% | 25% |

Four model columns because two settings are choices rather than
measurements, and the backtest measures each on identical vintages: the
**pace** assumed past the nowcast quarter (a forward-looking potential pace
of 2.0%, shipped, or the backward-looking trailing median — see
[The pace past the nowcast quarter](#the-pace-past-the-nowcast-quarter)) and
the **rate channel** (policy-rate hikes as a drag on that path, on by
request — see [The rate channel](#the-rate-channel)). Mean over horizons on
this first-release basis: 40.3% shipped, 45.0% with the
channel off, 40.0% / 43.8% on the trailing-median pace; against
final-vintage quads 40.6% / 45.3% / 43.7% / 46.4%.
The honest summary: **every setting beats persistence by 6–27pp and random
by 7–34pp at every horizon; the potential pace ties or beats the base-effects
benchmark past the nowcast quarter with the rate channel off, and the rate
channel gives back 3–10pp of that at +1 to +3q.** Full tables in
`results_us/`.

### 4. Why the quad hit rate sits below the direction hit rates

The quad is the AND of two calls, so it can only be as good as the product
of the two axes:

| Horizon | Growth direction (shipped / trailing-median pace, flat) | Inflation direction | Product (shipped) | Actual quad hit (shipped) |
|---|---|---|---|---|
| 0q | 71.9% / 71.9% | 86.8% | 62.5% | 64.0% |
| +1q | 56.8% / 70.3% | 70.3% | 39.9% | 37.8% |
| +2q | 58.3% / 66.7% | 67.6% | 39.4% | 38.9% |
| +3q | 61.0% / 65.7% | 63.8% | 38.9% | 35.2% |
| +4q | 48.0% / 53.9% | 63.7% | 30.6% | 25.5% |

(first-release basis; the axes are near-independent, so the quad lands close
to the product each time — which is also why the rate channel's growth cost
shows up almost one for one in the quad.) The second reason is that many
quarters are decided by a move smaller than the model's own error bar: **29%
of target quarters move their YoY inflation rate by less than 0.10pp**.
Those are flagged `low_conviction`, and excluding them lifts the nowcast hit
rate from 64.0% to 70.4%.

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

| Horizon | growth YoY (shipped) | growth YoY (trailing-median pace, flat) | growth QoQ | inflation YoY | inflation QoQ |
|---|---|---|---|---|---|
| 0q (current) | 64.9% (63.5) | 64.9% (63.5) | **66–71%** blended; **73–79% when votes agree** | **87.1%** | **86.2%** |
| +1q | 56.8% (54.8) | 64.9% (64.5) | 51–57% *weak lean* | 71.1% | 66.7% |
| +2q | 58.3% (56.7) | 66.7% (66.7) | no call | 68.5% | 55.0% |
| +3q | 58.1% (56.3) | 68.6% (69.0) | no call | 64.8% | 52.8% |
| +4q | 51.0% (48.8) | 56.9% (54.8) | no call | 63.8% | 50.5% |

(Shipped = potential pace with the rate channel on; the second column is the
backward-looking pace with the channel off, `--pace-mode trailing_median
--no-rate-channel`. Only the growth-YoY column moves between settings.)

Hit rate by conviction, all horizons pooled (`us_direction_conviction.csv`):

| Conviction | growth YoY (shipped) | growth YoY (trailing-median pace, flat) | inflation YoY | inflation QoQ |
|---|---|---|---|---|
| <0.10pp | 51.1% | 64.4% | 47.5% | 53.6% |
| 0.10–0.25pp | 48.1% | 60.0% | 77.6% | 76.5% |
| 0.25–0.50pp | 42.9% | 49.3% | 66.7% | **86.2%** |
| >0.50pp | **73.3%** | **78.2%** | **82.4%** | 75.6% |

How to read that:

* **Inflation grades cleanly.** Both inflation calls get better as conviction
  rises, so the conviction number is a usable trust dial — the same
  property the monthly CPI print call has (90% overall, 93% when it leans).
* **Growth YoY is two-regime, not graded.** Above 0.50pp it is right
  73.3% of the time as shipped and ~78% on the trailing-median
  pace — the pace and the rate channel barely touch a strong call; below
  that it is 40–64% regardless of size.
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

**Wealth and rates.** Household net worth YoY is the one financial driver
with a measured lead on growth: in the top quintile of its trailing decade
it is followed by growth YoY *decelerating* 3–4 quarters later **61–68%** of
the time (1990–2026 ex-COVID, base ~50%), in the bottom quintile only
32–36%. A falling 30-year mortgage rate (bottom quintile of its 4-quarter
change) precedes growth *accelerating* 2–3 quarters later 75–79%; a rising
one carries little (54–57%), and the fed funds rate's 4-quarter change
carries less. Walk-forward 2017–2026 the net-worth flag does not beat the
base-effect growth call as an override (that call sits at its ceiling at
2–3 quarters), so it is reported next to the call with its base rate rather
than applied. The sheet also shows the quad each quarter would fall into
under alternative growth paths — the year-ago QoQ is a known *hurdle*, so
"what growth does the Quad 4 case need" is a number, not a narrative.

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

The constant it steps to is a choice, and the section
[The pace past the nowcast quarter](#the-pace-past-the-nowcast-quarter)
measures the options. The one that scored best on 2017–2026 is the
vintage's own **trailing 24-quarter median** QoQ (median, not mean, because
2020Q2/Q3 are a crash-and-rebound pair that drags a mean for years). It is
also backward-looking by construction: on today's data it is 0.77% QoQ,
**3.1% annualized**, well above any estimate of US potential, because the
window is the post-2020 expansion — and windows from 12 to 40 quarters all
score within noise of each other (63.3–65.3% growth direction) while
implying 2.7–3.1%, so the sample cannot tell which constant is right. The
shipped pace is therefore a **forward-looking potential pace of 2.0% a
year** (`config.GDP_PACE_MODE = "potential"`): this quarter from the data,
then the economy's normal pace, then the rate channel's drag on top. That
choice costs accuracy on the sample it can be measured on, and the numbers
are below. `run_us.py` prints the pace it is using on every live run.

### The ceiling

If `qoq(q)` is unforecastable and the constant is well calibrated, the call
"trend vs known base" is right `E[Φ(|Z|)] = 75%` of the time. That is a
reference point, not a hard bound — an adaptive constant can beat it — but
it is the right thing to measure against. Scored on the same rows as the
model (`results_us/us_growth_ceiling.csv`):

| Horizon | Reference ceiling | Trailing-median pace, flat | Gap | Model as shipped (potential pace + rate channel) | Year-ago base published? |
|---|---|---|---|---|---|
| 0q | 73.7% | 71.9% | −1.8pp | 71.9% | always |
| +1q | 73.0% | 70.3% | −2.7pp | 56.8% | always |
| +2q | 69.4% | 66.7% | −2.8pp | 58.3% | always |
| +3q | 74.3% | 65.7% | −8.6pp | 61.0% | always |
| +4q | 70.6% | 53.9% | −16.7pp | 48.0% | **never** |

Three readings, and they are the point of this section:

1. **At 0–2 quarters the backward-looking flat path is within 3pp of the
   ceiling.** There is almost nothing left on the table there, and no
   amount of extra GDP modelling will find it. The shipped model sits
   16–13pp below it at +1 to +3q — that gap is the two
   forward-looking choices (the potential pace and the rate channel),
   measured in the two sections after next, and it is there because they
   were asked for, not because they scored.
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
| momentum nowcast + glide (original) | 58.8% | 43.0% | 65.8% | 61.1% | 1.81 |
| fitted nowcast + glide | 64.0% | 43.8% | 71.9% | 63.2% | 1.57 |
| fitted nowcast + static trend, flat | 64.0% | 43.1% | 71.9% | 59.1% | 1.43 |
| fitted nowcast + fitted trend, flat | 64.0% | 43.8% | 71.9% | 64.1% | 1.38 |
| fitted nowcast + fitted trend + rate channel | 64.0% | 40.0% | 71.9% | 58.2% | 1.45 |
| fitted nowcast + potential pace + rate channel (shipped) | 64.0% | 40.3% | 71.9% | 56.0% | 1.48 |
| fitted nowcast carried forward + rate channel | 64.0% | 40.6% | 71.9% | 55.9% | 2.38 |
| fitted nowcast gliding to potential + rate channel | 64.0% | 38.8% | 71.9% | 54.1% | 1.67 |

The flat fitted-trend path is best or tied on every column of the
growth-side work itself; the shipped setting replaces its backward-looking
constant with a potential pace and adds the rate channel's drag, both at the
subscriber's request, and gives back 4pp of quad hit and 6pp of growth
direction past the nowcast quarter for it. Be precise about what moved in
the growth-side work itself, though:

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

### What has driven US growth, and what leads it

Before building anything monthly, the question the subscriber asked: what
has actually moved US growth up and down? Real GDP by component (BEA
contributions and real levels, 1960–2026 ex-COVID; `scratchpad` study,
numbers reproducible from the FRED component series):

* **Quarter-to-quarter growth is mostly noise from three places.** Since
  2010 the variance shares of QoQ SAAR growth are inventories 33%, consumer
  spending 27%, net exports 21%, government 13%, fixed investment 6%. Two
  of the top three carry no information about where the economy is going.
* **The quad axis — the change in YoY growth — is driven by the consumer
  and fixed investment, and they lead by about one quarter.** Over 1960–2026
  the change in fixed investment's contribution correlates +0.51 with the
  change in YoY growth in the same quarter and +0.42 one quarter ahead;
  consumer spending +0.40 and +0.35. Nothing carries a lead of two quarters
  or more. Since 1990 the same leads are +0.35 and +0.25.
* **Growth peaks are led by investment.** In the 2000, 2006, 2015, 2022 and
  2024 slowdowns, fixed investment's contribution peaked 3–6 quarters before
  YoY growth did; consumer spending 1–3 quarters before in 1984, 1989, 1994,
  2000, 2006 and 2022.
* **Monthly leads are modest and short.** Against the change in YoY growth
  one to two quarters ahead (1990–2026): Chicago Fed national activity
  +0.32 / 58% sign agreement, Baa credit spread +0.24 / 60–63%, financial
  conditions (NFCI) +0.22–0.30 / 55–66%, new home sales +0.28 / 58%,
  building permits +0.25 / 58%, equities +0.21 / 55%. Coincident, not
  leading: manufacturing hours (+0.46), real retail sales, industrial
  production, capacity utilization, housing starts. At three quarters
  nothing beats a coin flip. That is the same negative result the direct
  forecasting experiments found, now with the reason attached: the things
  that move the quad axis lead it by one quarter, and the rest is noise.

### The monthly growth measure

The subscriber's brief is a growth call one or two months ahead with a
good hit rate and strong direction, the same shape as the CPI call, and
the ability to score a monthly quad map (Hedgeye's is monthly). Quarterly
GDP cannot supply that, so `usmodel/monthly_growth.py` builds a monthly
measure of real activity from the NBER-style coincident set, weighted
toward the consumer because that is what GDP is made of: real PCE 0.55,
industrial production 0.15, real personal income ex transfers 0.10, real
retail sales 0.10, payrolls × hours 0.10 (geometric index). Against real
GDP on final data (2008–2026) its quarterly-average YoY correlates 0.97
with GDP YoY, and the direction of its change agrees with the quad axis
two quarters in three — the third is inventories, imports and government.

Its next month can be called exactly like next month's CPI: the change in
YoY = this month's MoM minus the MoM that drops out of the 12-month window,
and only the first term is a forecast (each component's trailing 6-month
mean where it has not printed, its actual value where it has). Walk-forward
2010–2026 ex-COVID, point-in-time with each component's release lag, as-of
the 20th and the end of every month (`results_us/us_growth_monthly.csv`):

| Call | n | Right on the measure's own path | Right vs BBK monthly GDP | Right vs the quarter's GDP direction |
|---|---|---|---|---|
| **+1 month, all calls** | 367 | 85% | 59% | 54% |
| +1 month, target month already has 3 of 5 components in (as-of the 20th) | 199 | 90% | 62% | 55% |
| +1 month, strong (move over 0.30pp) | 104 | 100% | 69% | 69% |
| +1 month, call (0.15–0.30pp) | 108 | 94% | 57% | 47% |
| +1 month, lean (0.05–0.15pp) | 107 | 71% | 53% | 51% |
| +1 month, toss-up (under 0.05pp) | 48 | 65% | 54% | 48% |
| **+2 months, all calls** | 366 | 79% | 55% | 54% |
| +2 months, strong | 95 | 100% | 61% | 62% |
| **+3 months, all calls** | 366 | 81% | 56% | 51% |
| +3 months, strong | 107 | 100% | 61% | 60% |

Read it the way it is meant: **the measure's own direction one month out is
called 85% of the time, 91% once most of the month's components are in,
and 100% of the time when the move is larger than 0.30pp** — that is the
monthly quad map's growth axis, and the sheet shows the monthly quads with
Hedgeye's path beside them. But it is the consumer-and-production trend,
not the GDP print: month by month it agrees with the *quarter's* GDP
direction only 54% of the time (59% with the Chicago Fed's monthly GDP),
because a quarter's YoY change is decided by the noise components above.
The quarterly quad therefore still comes from GDP, and the monthly quad is
the read underneath it. `python us_sheet.py` puts both on the page.

### The pace past the nowcast quarter

Past the nowcast quarter the path is a constant, and the constant is a
choice. Four are implemented (`config.GDP_PACE_MODE`) and every run of
`python us_backtest.py` measures all four on identical vintages
(`results_us/us_pace_modes.csv`; the rate-channel-off column from the same
code with `--no-rate-channel`):

| Pace past the nowcast quarter (rate channel on / off) | Quad hit, mean, first release | Quad hit, mean, final | Growth YoY direction, +1 to +4q | Growth YoY MAE |
|---|---|---|---|---|
| **potential 2.0% a year (shipped)** | **40.3%** / 45.0% | **40.6%** / 45.3% | **56.0%** / 61.2% | **1.48** / 1.39 |
| trailing 24-quarter median (backward-looking) | 40.0% / 43.8% | 43.7% / 46.4% | 61.0% / 64.2% | 1.45 / 1.38 |
| this quarter's nowcast carried forward | 40.6% / 42.1% | 42.5% / 42.9% | 57.0% / 57.3% | 2.38 / 2.33 |
| nowcast gliding to potential (persistence 0.5) | 38.8% / 44.8% | 39.5% / 44.1% | 53.7% / 59.2% | 1.67 / 1.57 |

What it says. The backward-looking median wins on growth direction by 3–5pp
and the quad hit rate is a wash: on the first-release basis the potential
pace is level or a point better, on the final basis a few points worse. The
reason is the sample, not the idea: 2017–2026 is a period in which US growth
ran above 2% almost throughout, so a constant that assumed the recent past
would repeat was right more often than one that assumed a return to normal.
That is exactly the property the subscriber does not want in a forecast, and
the switch was made on that ground — **as a judgment about 2027, not as a
measured improvement, and the sample cannot vindicate it either way.**
Carrying the nowcast forward is the worst forward-looking option on the level
(a noisy quarter compounds for a year); the glide is between the two.

What it does to the calls today (18 September 2026): the trailing median
assumed 3.1% a year and put every 2027 quarter above its bar; the potential
pace at 2.0% puts 2027Q1 a hair above its 2.1% bar (inside the deadband),
2027Q2 above its 1.5% bar, and 2027Q3–Q4 below theirs — quads 4, 2, 1
(deadband), 1, 3, 4 from 2026Q3. The Hedgeye-style Quad 4 in 2027Q2 still
needs growth under 1.5% annualized, which no constant supplies; it is a
forecast of a slowdown, and the sheet now shows exactly how far the rate
channel gets toward it.

### The rate channel

Rate hikes slow the economy with a lag, and the growth path now carries
that: `usmodel/rates.py`, on by request (`config.RATE_CHANNEL_ENABLED`).
For every quarter past the nowcast quarter

```
qoq(q) = trend + beta * [ff(q-2) - ff(q-6)]
```

where `ff` is the quarterly mean of the effective fed funds rate — observed
where published, then the market-implied path (`config.POLICY_RATE_PATH`,
dated steps read off the curve; **live runs only**, the backtest holds the
rate flat because there is no futures history in the repo, so a live path
can never leak into it), then flat — and `beta` is re-fitted point-in-time
at every vintage on 1960+ with the **sign imposed** (a hike never adds to
growth) and the **size estimated**: about −0.10pp of quarterly growth per
1pp of hikes over the window. The nowcast quarter is left to the indicator
fit, which already sees the rate environment.

**What the data say** (real GDP 1960–2026 ex-COVID, fed funds quarterly
means; `usmodel/rates.py` docstring):

| Sample | Slope of QoQ on the 2–6q window, pp per 1pp | Correlation |
|---|---|---|
| 1960+ | **−0.10** | −0.26 |
| 1985+ | +0.05 | +0.15 |
| 1995+ | +0.07 | +0.20 |
| 2005+ | +0.14 | +0.39 |

A free distributed lag on the full sample puts the drag at lags 2–8 (sum
−0.70pp per 1pp step, largest coefficient at lag 2, t = −4.4): a level
effect of roughly −0.4% to −0.7% two years out, the same sign and timing as
FRB/US at the low end of its size. But it is a **1960–1984 fact**. From 1985
on the slope is zero to positive — the Fed hikes into strength and the
strength outlasts the hikes; 2022–23 is the loudest example — and mortgage
and 10-year windows behave the same way. After a top-quintile 8-quarter
rise in the policy rate, growth ran below its trailing trend 54–59% of the
time over the following 1–5 quarters (1965–2026): a lean, not a law.

**Walk-forward, in the repo's own backtest** (`results_us/us_rate_channel.csv`,
on vs off on identical vintages, simulated first releases):

| Potential pace (shipped) | Rate channel on (shipped) | Off |
|---|---|---|
| Growth YoY direction, 0 / +1 / +2 / +3 / +4q | 64.9 / 56.8 / 58.3 / 58.1 / 51.0% | 64.9 / 59.5 / 63.9 / 65.7 / 55.9% |
| Quad hit, mean over horizons (first release) | 40.3% | 45.0% |
| Growth YoY MAE, mean over horizons | 1.48pp | 1.39pp |
| Growth calls the drag reversed | 42 of 540 | — |
| Hit rate on those rows | **23.8%** | **76.2%** |

On the trailing-median pace the same comparison reads 40.0% vs 43.8% on the
quad and 62.2 / 61.1 / 65.7% vs 64.9 / 66.7 / 68.6% on growth direction at
+1/+2/+3q, with 34 reversed calls right 29.4% of the time against 70.6%
without the drag; strong (≥0.50pp) growth calls are identical either way.
The channel touches only marginal calls, and when it flips one it is wrong
more than twice as often as right. Where: 2018 and 2023, both hiking cycles
into expansions that kept going; as of September 2023 the drag projected
2024 growth at 1.9–2.2% YoY against a realized 2.8–3.4%. On final GDP
instead of simulated first releases the picture is the same.

**What it says today** (18 September 2026: fed funds 3.63% through
mid-September, then the September hike and the curve's path to 4.63% by
July 2027): the 2024–26 cuts are still a tailwind of about +0.3pp
annualized through 2027Q1; the drag turns negative in 2027Q2 (−0.1pp) and
reaches −0.3 to −0.4pp annualized in 2027Q3–Q4. On the potential pace that
tailwind is what lifts 2027Q1 a hair above its 2.1% bar (Quad 1 inside the
deadband; Quad 4 inside the deadband without it), and the drag is what
takes 2027Q4 clearly below its bar. A Quad 4 in 2027Q2 needs growth under
1.5% annualized; the sheet shows how close a hike at every meeting gets
with the fastest response history records. `run_us.py` prints the path,
the drag and the quad with and without it every run.

So the channel is in, sized by the data, with its cost stated: it is the
mechanism that was asked for, and on the shipped pace it costs about 5pp of
growth-direction accuracy at 1–3 quarters and 5pp of quad hit rate. `config.RATE_CHANNEL_ENABLED = False` restores
the flat path; `config.RATE_SENSITIVITY_OVERRIDE` pins the size (−0.20 for
FRB/US-strength transmission); `python us_backtest.py` measures both
settings on every run, and `tests/test_us_rates.py` pins the mechanics.

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
5. **The rate channel's forward path is an input, not data.** The
   market-implied steps in `config.POLICY_RATE_PATH` were read off a curve
   dated 2026-09-18; they are replaced by the daily effective rate as it
   prints and go stale as the curve moves. And the channel itself is on
   because it was asked for: measured, it lowers the growth-direction and
   quad hit rates at +1 to +3q (section above). So is the potential pace,
   the forward-looking constant the path holds past the nowcast quarter:
   the backward-looking median scored better on 2017–2026 because that
   sample's past kept repeating. Every number in this README is quoted for
   the model as shipped, with the alternative beside it where they differ.

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
5. **A rate channel that scores.** The mechanical drag does not. The
   candidates are conditional forms — the drag only once the real policy
   rate is restrictive, or routed through credit conditions (NFCI, spreads)
   rather than the rate itself — but each is one modern cycle of evidence,
   and real GDP vintages should come first.
