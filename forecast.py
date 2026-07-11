"""
Live daily quad forecast.

    python forecast.py [--data-dir data] [--outdir results]

Re-run this every day (or let the GitHub Actions cron do it). It rebuilds
today's forecast from the freshest information available:

  * monthly macro data from data/ (CPI, GDP, I-44 - update with
    `python -m backtest.fetch_data` when new releases land)
  * LIVE market spots fetched at runtime, best-effort per source:
      Brent   - FRED daily series
      USDNOK  - Norges Bank EXR API, daily
      I-44    - Norges Bank EXR API, daily (patched into the FX history so
                today's krone move flows into the imported-goods blocks)
      power   - hvakosterstrommen.no hourly area prices (NO1/NO2/NO5 mean)
  * quad probabilities via the model's backtest error cloud
    (method='residual'), so every move in oil, the krone, or power prices
    shifts the probabilities the same day - no regime flip needed.

Requires a prior backtest run (backtest_results.parquet) for calibration;
run `python backtest.py` first. Appends one row per run to
forecast_history.csv so drift in the calls is auditable over time.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from backtest import btconfig, data_bundle, monthly, plots, probability, scoring
from quadmap import inflation
from backtest.vintage import build_vintage
from backtest.btconfig import VintageConfig
from backtest.fetch_data import FRED_BRENT

POWER_API = "https://www.hvakosterstrommen.no/api/v1/prices/{y}/{m:02d}-{d:02d}_{area}.json"
POWER_AREAS = ("NO1", "NO2", "NO5")   # southern areas, most CPI-relevant


# ---------------------------------------------------------------------------
# Live spot fetchers - each returns (value, source_note) or raises.
# ---------------------------------------------------------------------------

def live_brent() -> tuple[float, str]:
    df = pd.read_csv(FRED_BRENT)
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    last = vals.dropna().iloc[-1]
    return float(last), f"FRED daily ({df.iloc[:, 0].iloc[-1]})"


def _norges_bank_daily(base: str) -> tuple[float, str]:
    import requests
    url = (f"https://data.norges-bank.no/api/data/EXR/B.{base}.NOK.SP"
           "?format=sdmx-json&lastNObservations=5")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    series = js["data"]["dataSets"][0]["series"]
    obs = next(iter(series.values()))["observations"]
    periods = js["data"]["structure"]["dimensions"]["observation"][0]["values"]
    i = max(int(k) for k in obs)
    return float(obs[str(i)][0]), f"Norges Bank daily ({periods[i]['id']})"


def live_usdnok() -> tuple[float, str]:
    return _norges_bank_daily("USD")


def live_i44() -> tuple[float, str]:
    return _norges_bank_daily("I44")


def _area_day_mean_ore(d: dt.date) -> float | None:
    """Mean hourly spot across the southern areas for one day, in ore/kWh
    incl VAT, or None if not available."""
    import requests
    prices = []
    for area in POWER_AREAS:
        url = POWER_API.format(y=d.year, m=d.month, d=d.day, area=area)
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        prices.extend(h["NOK_per_kWh"] for h in r.json())
    return (sum(prices) / len(prices)) * 100.0 * 1.25 if prices else None


def live_power_ore_kwh(day: dt.date | None = None) -> tuple[float, str]:
    """Today's spot in ore/kWh incl VAT. Falls back one day if today's
    prices are not posted yet."""
    day = day or dt.date.today()
    for offset in (0, 1):
        d = day - dt.timedelta(days=offset)
        v = _area_day_mean_ore(d)
        if v is not None:
            return float(v), f"hvakosterstrommen.no ({d})"
    raise RuntimeError("no area prices available for today or yesterday")


def power_proxy_scale(anchor_month: pd.Period, proxy_anchor_value: float,
                      day_mean_fn=_area_day_mean_ore) -> float:
    """Bridge between proxy units and real ore/kWh.

    When market_monthly.csv is the CPI-subindex PROXY, its power column is
    on an arbitrary rescaled level - feeding a real Nord Pool spot into a
    proxy-anchored projection fabricates a giant price move (observed: a
    fake +1.3pp MoM impulse). Estimate the real ore level of the anchor
    month by sampling a few of its days, and return the factor that maps
    real ore into proxy units. Raises if the month cannot be sampled."""
    samples = []
    for dom in (4, 11, 18, 25):
        v = day_mean_fn(dt.date(anchor_month.year, anchor_month.month, dom))
        if v is not None:
            samples.append(v)
    if not samples:
        raise RuntimeError(f"could not sample real power prices for "
                           f"{anchor_month}")
    real_est = sum(samples) / len(samples)
    return float(proxy_anchor_value / real_est)


def gather_live_spots() -> tuple[dict, dict, list[str]]:
    """Best-effort live overrides. Whatever fails just keeps the monthly
    spot-carry anchor from the data bundle - the run never dies on a feed."""
    overrides: dict = {}
    live_notes: list[str] = []
    extras: dict = {}
    horizon = pd.period_range(pd.Period(dt.date.today(), freq="M"),
                              periods=18, freq="M")

    def attempt(name, fn, apply):
        try:
            val, note = fn()
            apply(val)
            live_notes.append(f"{name}: {val:.2f} [{note}]")
        except Exception as e:
            live_notes.append(f"{name}: LIVE FEED UNAVAILABLE ({e}) - "
                              "using last monthly value")

    # IMPORTANT: only the *forward* paths move to today's spot. The
    # *_recent anchors must stay at the last monthly value embedded in the
    # observed CPI level - that gap between anchor and live spot is exactly
    # the price impulse the projection should react to. Overriding both
    # would flatten the impulse to zero.
    attempt("brent_usd", live_brent, lambda v: overrides.update(
        brent_forward_usd={str(p): v for p in horizon}))
    attempt("usdnok", live_usdnok, lambda v: overrides.update(
        usdnok_path={str(p): v for p in horizon}))
    attempt("power_ore_kwh", live_power_ore_kwh, lambda v: overrides.update(
        power_forward_ore_kwh={str(p): v for p in horizon}))
    attempt("i44", live_i44, lambda v: extras.update(i44_today=v) or
            overrides.update(i44_path={str(p): v for p in horizon}))
    return overrides, extras, live_notes


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(btconfig.DATA_DIR))
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--results", default=None,
                    help="backtest_results.parquet for calibration "
                         "(default: <outdir>/backtest_results.parquet)")
    ap.add_argument("--method", choices=["residual", "calibration"],
                    default="residual")
    ap.add_argument("--demo", action="store_true",
                    help="use the synthetic bundle (harness check only)")
    ap.add_argument("--no-live", action="store_true",
                    help="skip live spot fetches (monthly anchors only)")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results) if args.results \
        else outdir / "backtest_results.parquet"
    if not results_path.exists():
        raise SystemExit(f"{results_path} not found - run `python "
                         "backtest.py` first (the backtest supplies the "
                         "probability calibration)")
    preds = pd.read_parquet(results_path)

    bundle = data_bundle.make_demo_bundle() if args.demo \
        else data_bundle.load_bundle(args.data_dir)
    realized = scoring.realized_quads_final(bundle)
    cfg = VintageConfig()

    overrides: dict = {}
    if not args.no_live and not args.demo:
        overrides, _, notes = gather_live_spots()
        print("live market inputs:")
        for n in notes:
            print("  " + n)

        # Proxy-unit bridge: if the market file is the CPI-subindex proxy,
        # a real ore/kWh spot must be rescaled into proxy units before it
        # touches the projection, or the unit gap masquerades as a shock.
        if "power_forward_ore_kwh" in overrides and \
                (Path(args.data_dir) / ".market_is_proxy").exists():
            try:
                anchor_m = bundle.market.index[-1]
                scale = power_proxy_scale(
                    anchor_m, float(bundle.market["power_ore_kwh"].iloc[-1]))
                overrides["power_forward_ore_kwh"] = {
                    k: v * scale
                    for k, v in overrides["power_forward_ore_kwh"].items()}
                print(f"  power rescaled to proxy units (x{scale:.3f}, "
                      f"anchored on real {anchor_m} prices)")
            except Exception as e:
                overrides.pop("power_forward_ore_kwh", None)
                print(f"  power override DROPPED (proxy-unit bridge failed: "
                      f"{e}) - power stays at the monthly proxy anchor")

    # As-of is *today*: the vintage truncation applies the publication lags
    # automatically, exactly as the backtest's as-of dates did. Horizon 0 is
    # the quarter we are standing in right now.
    prob, calls, asof = probability.current_quad_probabilities(
        bundle, preds, realized, cfg, method=args.method,
        asof=pd.Timestamp(dt.date.today()),
        horizons=(0, 1, 2, 3, 4),
        assumption_overrides=overrides or None)

    print(f"\n=== Quad probabilities, current + next 4 quarters "
          f"(as of {asof.date()}, method={args.method}) ===")
    print(probability.format_probability_table(prob, calls).to_string())

    # The monthly money-call: direction of the NEXT CPI print's YoY rate,
    # with the empirical accuracy of past calls of this conviction.
    try:
        fcfg = VintageConfig(revision_mode="none")
        v = build_vintage(bundle, asof, fcfg)
        if overrides:
            v.assumptions.update(overrides)
        proj = inflation.build_cpi_projection(v.cpi_index, 3,
                                              v.assumptions, v.i44)
        m0, m1 = v.last_cpi_month, v.last_cpi_month + 1
        staleness = (asof - m0.end_time).days
        if staleness > 45:
            print(f"\nWARNING: newest CPI in the dataset is {m0} - "
                  f"{staleness} days before the as-of date. The 'next "
                  "print' below is NOT the upcoming release; refresh data/ "
                  "(python -m backtest.fetch_data) and check its staleness "
                  "diagnostics.")
        yoy = proj["yoy_pct"]
        delta = float(yoy[m1] - yoy[m0])
        direction = "ACCELERATING" if delta > 0 else "DECELERATING"
        line = (f"{m1}: YoY {direction} - {yoy[m0]:.2f}% -> "
                f"{yoy[m1]:.2f}% ({delta:+.2f}pp, "
                f"{monthly.bucket_label(abs(delta))})")
        mdir_path = outdir / "monthly_direction.parquet"
        if mdir_path.exists():
            hist = pd.read_parquet(mdir_path)
            acc, n = monthly.direction_hit_rate_for(abs(delta), hist)
            line += (f"\n  historical accuracy of calls this size: "
                     f"{acc * 100:.0f}% ({n} backtested months)")
        print(f"\n=== Next CPI print ===\n{line}")
    except Exception as e:
        print(f"\nnext-print card skipped: {e}")

    prob.to_csv(outdir / "quad_probabilities.csv")
    plots.plot_quad_probability_heatmap(
        prob, calls, asof, str(outdir / "quad_probabilities_quarterly.png"),
        method=args.method)

    # Monthly-resolution inflation call: every print gets its own base
    # effect and its own odds - months within a quarter do NOT repeat here.
    mpath_file = outdir / "monthly_path_errors.parquet"
    if mpath_file.exists():
        mpath = pd.read_parquet(mpath_file)
        minfl = monthly.monthly_inflation_probabilities(
            bundle, mpath, cfg, asof=asof,
            assumption_overrides=overrides or None)
        minfl.to_csv(outdir / "monthly_inflation_probabilities.csv")
        plots.plot_monthly_inflation_direction(
            minfl, str(outdir / "monthly_inflation_direction.png"))
        print("\n=== P(YoY inflation accelerating), print by print ===")
        print(minfl.round(3).to_string())

    # Append to the run history so day-to-day drift is auditable.
    hist_path = outdir / "forecast_history.csv"
    row = {"run_date": dt.date.today().isoformat(),
           "asof": asof.date().isoformat(), "method": args.method,
           "live_overrides": ";".join(sorted(overrides)) or "none"}
    for tq in prob.columns:
        row[f"call_{tq}"] = calls.get(tq)
        for q in (1, 2, 3, 4):
            row[f"p{q}_{tq}"] = round(float(prob.loc[q, tq]), 4)
    hist = pd.DataFrame([row])
    if hist_path.exists():
        hist = pd.concat([pd.read_csv(hist_path), hist], ignore_index=True)
    hist.to_csv(hist_path, index=False)
    print(f"\nwrote quad_probabilities.[csv|*.png] and appended "
          f"{hist_path.name} ({len(hist)} runs)")
    return prob


if __name__ == "__main__":
    main()
