# US macro regime feed

Rebuilt by the "US macro regime" workflow from fresh FRED and Zillow data.

- `latest.json` -- the regimes: each quarter (from GDP) and each month
  (from the monthly growth measure), with directions, conviction and the
  backtested hit rate of that kind of call; `hit_rates`, the backtest's
  scorecard in one block; and `notes`, the story behind the numbers as
  plain text (the next inflation print, the inflation path, oil, rents,
  this quarter's growth, the pace assumed after it, rates, the consumer)
- `us_sheet.json` -- the full data behind the page
- `us_regime_sheet.html` -- the page
- `us_run.txt` -- the console output of the live run

Read them by raw URL, e.g.
`https://raw.githubusercontent.com/SanderHeisan/Inflation-GDP/live-us-regime/latest.json`.
