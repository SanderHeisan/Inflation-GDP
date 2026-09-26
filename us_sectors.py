"""
Sector returns by regime: what each R has meant for the SPDR sector funds
(and SPY, long Treasuries, gold), month by month and quarter by quarter.

    python us_sectors.py            # writes results_us/us_sectors_by_regime.csv
                                    #        results_us/us_sectors_feed.json

Two bases, kept apart because they answer different questions:

1. REALIZED (monthly, 2008+): the regime each month turned out to be, from
   the published data - the monthly growth measure's year-over-year change
   and the CPI's, signs only, with a "clear" flag where both moved by at
   least the deadband. Answers "how did each fund do in months that were
   R1..R4". Known only after the fact: the growth measure's last component
   publishes about a month after month end.
2. CALLED (monthly, 2017+): the model's point-in-time call for month M,
   made at the end of month M-1 from what was published by then (the same
   vintage builder the backtest uses, no partial month). Answers "how did
   each fund do in months the model said would be R1..R4 before they
   began" - the version a subscriber could have acted on.

And REALIZED QUARTERLY (1999+): real GDP year-over-year direction x CPI
year-over-year direction, against each fund's quarterly return, which
reaches back to the funds' launch.

Returns are monthly (or quarterly) total returns from dividend-adjusted
closes. Every statistic is a plain average over the months in the cell,
with its count, the share of positive months, the excess over SPY and a
t-statistic on that excess; nothing is fitted. The output is a description
of the past, not a recommendation.

Prices come from data/us/sector_etf_monthly.csv (month-end adjusted
closes). If the file is missing and yfinance is importable, it is fetched
and written; the workflow does that once a day.
"""
from __future__ import annotations

import datetime
import json
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quadmap import quads
from usmodel import data_bundle, inflation, monthly_growth as mg
from usbacktest.btconfig import RESULTS_DIR, USVintageConfig
from usbacktest.vintage import build_vintage

PRICES = data_bundle.DATA_DIR / "sector_etf_monthly.csv" if hasattr(data_bundle, "DATA_DIR") else None
OUT_CSV = RESULTS_DIR / "us_sectors_by_regime.csv"
OUT_JSON = RESULTS_DIR / "us_sectors_feed.json"

FUNDS: Dict[str, str] = {
    "SPY": "S&P 500", "XLE": "Energy", "XLF": "Financials", "XLK": "Technology", "XLV": "Health care",
    "XLI": "Industrials", "XLP": "Consumer staples", "XLY": "Consumer discretionary", "XLU": "Utilities",
    "XLB": "Materials", "XLRE": "Real estate", "XLC": "Communication services",
    "TLT": "Long Treasuries (20+ years)", "IEF": "Treasuries (7 to 10 years)", "GLD": "Gold",
}
SECTORS = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
BENCH = "SPY"
DEADBAND_PP = 0.05          # a "clear" month: both sides moved by at least this
CALLED_START = "2016-12"    # first month-end as-of for the called basis (the CPI backtest's span)
REGIME_NAME = {1: "R1", 2: "R2", 3: "R3", 4: "R4"}


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------

def _prices_path():
    return PRICES if PRICES is not None else (RESULTS_DIR.parent / "data" / "us" / "sector_etf_monthly.csv")


def load_monthly_closes() -> pd.DataFrame:
    """Month-end dividend-adjusted closes, one column per fund, Period[M] index."""
    path = _prices_path()
    if not path.exists():
        import yfinance as yf                     # optional: only when the file is missing
        raw = yf.download(list(FUNDS), start="1998-12-01", auto_adjust=False, progress=False, threads=False)
        adj = raw["Adj Close"]
        path.parent.mkdir(parents=True, exist_ok=True)
        adj.groupby(adj.index.to_period("M")).last().to_timestamp(how="end").to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[[c for c in FUNDS if c in df.columns]]
    df.index = pd.DatetimeIndex(df.index).to_period("M")
    return df.groupby(level=0).last()


def monthly_returns(closes: pd.DataFrame) -> pd.DataFrame:
    return closes.pct_change() * 100.0


def quarterly_returns(closes: pd.DataFrame) -> pd.DataFrame:
    q = closes.groupby(closes.index.asfreq("Q")).last()
    return q.pct_change() * 100.0


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------

def realized_monthly(bundle) -> pd.DataFrame:
    """quad / close per month from the published data (the model's own
    monthly growth measure and the CPI index), 2008 onward."""
    cfg = USVintageConfig(revision_mode="none", live_partial_month=True)
    v = build_vintage(bundle, pd.Timestamp.today().normalize(), cfg, horizon_months=3)
    mgi = mg.activity_index(v.indicator_panel, horizon_months=1)
    if mgi is None:
        raise RuntimeError("the monthly growth measure could not be built")
    actual = mgi[mgi["status"] == "actual"]
    cpi_yoy = quads.calendar_pct_change(bundle.cpi, 12)
    mq = mg.monthly_quads(actual["yoy_pct"], cpi_yoy, deadband_pp=DEADBAND_PP)
    return mq[["quad", "close", "growth_yoy", "inflation_yoy"]]


def called_monthly(bundle, start: str = CALLED_START, end: Optional[pd.Period] = None) -> pd.DataFrame:
    """The model's call for month M made at the end of M-1: one vintage per
    month-end, the CPI path and the growth measure projected, the cell read
    off month M. Columns: quad, close, horizon (months past the growth
    measure's last complete month), status."""
    cfg = USVintageConfig(revision_mode="none", live_partial_month=False)
    first = pd.Period(start, "M")
    last = end if end is not None else pd.Period(datetime.date.today(), "M") - 1
    rows = []
    skipped = 0
    for asof_m in pd.period_range(first, last, freq="M"):
        asof = asof_m.to_timestamp(how="end").normalize()
        target = asof_m + 1
        try:
            v = build_vintage(bundle, asof, cfg, horizon_months=8)
            cpi = inflation.build_cpi_projection(v.cpi_index, 6, v.assumptions, v.aux)
            yoy_p = quads.calendar_pct_change(cpi["cpi_index"], 12)
            mgi = mg.activity_index(v.indicator_panel, horizon_months=6)
            if mgi is None or target not in mgi.index:
                skipped += 1
                continue
            mq = mg.monthly_quads(mgi["yoy_pct"], yoy_p, deadband_pp=DEADBAND_PP)
            if target not in mq.index:
                skipped += 1
                continue
            r = mq.loc[target]
            rows.append({"month": target, "asof": asof.date(), "quad": int(r["quad"]), "close": bool(r["close"]),
                         "horizon": int((target - mgi.attrs["last_complete"]).n) if mgi.attrs.get("last_complete") is not None else None,
                         "status": str(mgi.loc[target, "status"])})
        except Exception as e:                    # noqa: BLE001 - one bad vintage must not stop the table
            skipped += 1
            print(f"[us_sectors] {asof.date()} skipped: {type(e).__name__}: {e}")
    out = pd.DataFrame(rows).set_index("month") if rows else pd.DataFrame(columns=["quad", "close", "horizon", "status"])
    out.attrs["skipped"] = skipped
    return out


def realized_quarterly(bundle) -> pd.DataFrame:
    gdp = bundle.gdp_long if bundle.gdp_long is not None else bundle.gdp
    gdp_yoy = gdp.pct_change(4) * 100.0
    cpi_q = quads.monthly_to_quarterly_yoy(quads.calendar_pct_change(bundle.cpi, 12).dropna())
    tbl = quads.classify(gdp_yoy.dropna(), cpi_q)
    return pd.DataFrame({"quad": tbl["quad"].astype(int), "close": tbl["low_conviction"].astype(bool)})


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def cell_stats(rets: pd.DataFrame, regime: pd.Series, clear: Optional[pd.Series] = None,
               basis: str = "", clear_only: bool = False) -> pd.DataFrame:
    """One row per (regime, fund): count, mean, median, share positive,
    excess over SPY and its t-statistic. Regime 0 is every period."""
    idx = rets.index.intersection(regime.index)
    if clear_only and clear is not None:
        idx = idx.intersection(clear[~clear].index)
    rets = rets.reindex(idx)
    reg = regime.reindex(idx)
    rows = []
    for r in (0, 1, 2, 3, 4):
        sel = rets if r == 0 else rets[reg == r]
        spy = sel[BENCH] if BENCH in sel.columns else None
        for t in rets.columns:
            s = sel[t].dropna()
            if len(s) < 3:
                continue
            ex = (s - spy.reindex(s.index)).dropna() if spy is not None and t != BENCH else None
            rows.append({
                "basis": basis, "regime": r, "fund": t, "name": FUNDS.get(t, t), "n": int(len(s)),
                "mean": float(s.mean()), "median": float(s.median()), "hit": float((s > 0).mean()),
                "excess": (float(ex.mean()) if ex is not None and len(ex) else None),
                "tstat": (float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))) if ex is not None and len(ex) > 2 and ex.std(ddof=1) > 0 else None),
                "first": str(s.index[0]), "last": str(s.index[-1]),
            })
    return pd.DataFrame(rows)


def _label(p: pd.Period) -> str:
    return p.strftime("%b %Y") if p.freqstr.startswith("M") else f"Q{p.quarter} {p.year}"


def _block(stats: pd.DataFrame, regime_series: pd.Series, funds: List[str], top: int = 3) -> dict:
    """The compact shape the website reads: per regime the count, SPY's
    average month, the best and worst funds by excess over SPY; and one
    table of average returns per fund and regime."""
    out = {"regimes": {}, "table": []}
    counts = regime_series.value_counts()
    for r in (1, 2, 3, 4):
        sub = stats[(stats["regime"] == r) & (stats["fund"].isin(funds))].dropna(subset=["excess"])
        spy = stats[(stats["regime"] == r) & (stats["fund"] == BENCH)]
        ranked = sub.sort_values("excess", ascending=False)
        pick = lambda df: [{"fund": x.fund, "name": x.name, "mean": round(x.mean, 2), "excess": round(x.excess, 2),
                            "hit": round(x.hit, 2), "n": int(x.n)} for x in df.itertuples()]
        out["regimes"][str(r)] = {
            "n": int(counts.get(r, 0)),
            "spy_mean": (round(float(spy["mean"].iloc[0]), 2) if len(spy) else None),
            "spy_hit": (round(float(spy["hit"].iloc[0]), 2) if len(spy) else None),
            "best": pick(ranked.head(top)), "worst": pick(ranked.tail(top).iloc[::-1]),
        }
    for t in [BENCH] + funds:
        row = {"fund": t, "name": FUNDS.get(t, t)}
        for r in (0, 1, 2, 3, 4):
            cell = stats[(stats["regime"] == r) & (stats["fund"] == t)]
            key = "all" if r == 0 else f"r{r}"
            row[key] = (round(float(cell["mean"].iloc[0]), 2) if len(cell) else None)
            row[key + "_n"] = (int(cell["n"].iloc[0]) if len(cell) else 0)
        out["table"].append(row)
    return out


def build(bundle=None, closes: Optional[pd.DataFrame] = None, called_end: Optional[pd.Period] = None) -> dict:
    # A longer CPI history than the modelling window: the quarterly basis reaches back to the funds' launch.
    bundle = bundle or data_bundle.load_bundle(start="1995-01")
    closes = closes if closes is not None else load_monthly_closes()
    mret = monthly_returns(closes)
    qret = quarterly_returns(closes)
    funds = [f for f in SECTORS + ["TLT", "IEF", "GLD"] if f in closes.columns]

    real_m = realized_monthly(bundle)
    real_q = realized_quarterly(bundle)
    called = called_monthly(bundle, end=called_end)

    frames = [
        cell_stats(mret, real_m["quad"], real_m["close"], basis="realized_monthly"),
        cell_stats(mret, real_m["quad"], real_m["close"], basis="realized_monthly_clear", clear_only=True),
        cell_stats(qret, real_q["quad"], real_q["close"], basis="realized_quarterly"),
    ]
    if len(called):
        frames.append(cell_stats(mret, called["quad"], called["close"], basis="called_monthly"))
    stats = pd.concat(frames, ignore_index=True)

    def span(idx) -> str:
        idx = idx.intersection(mret.index) if idx.freqstr.startswith("M") else idx.intersection(qret.index)
        return f"{_label(idx.min())} to {_label(idx.max())}" if len(idx) else ""

    feed = {
        "generated": str(datetime.date.today()),
        "funds": {t: FUNDS[t] for t in [BENCH] + funds},
        "realized_monthly": {"span": span(real_m.index), "n": int(len(real_m.index.intersection(mret.index))),
                             **_block(stats[stats.basis == "realized_monthly"], real_m["quad"].reindex(real_m.index.intersection(mret.index)), funds)},
        "realized_quarterly": {"span": span(real_q.index), "n": int(len(real_q.index.intersection(qret.index))),
                               **_block(stats[stats.basis == "realized_quarterly"], real_q["quad"].reindex(real_q.index.intersection(qret.index)), funds)},
    }
    if len(called):
        idx = called.index.intersection(mret.index)
        feed["called_monthly"] = {"span": span(called.index), "n": int(len(idx)), "skipped": int(called.attrs.get("skipped", 0)),
                                  **_block(stats[stats.basis == "called_monthly"], called["quad"].reindex(idx), funds)}
        # how often the call and the outcome agree, on the months both exist
        both = called[["quad"]].join(real_m[["quad"]], how="inner", lsuffix="_called", rsuffix="_real")
        feed["called_monthly"]["agreement"] = (round(float((both["quad_called"] == both["quad_real"]).mean()), 3) if len(both) else None)
        feed["called_monthly"]["agreement_n"] = int(len(both))
    return {"stats": stats, "feed": feed, "realized_monthly": real_m, "called_monthly": called, "realized_quarterly": real_q}


def print_summary(res: dict) -> None:
    for basis in ("realized_monthly", "called_monthly", "realized_quarterly"):
        blk = res["feed"].get(basis)
        if not blk:
            continue
        print(f"\n== {basis}: {blk['span']} ({blk['n']} periods)" + (f", call agrees with outcome {blk['agreement']:.0%} of {blk['agreement_n']}" if blk.get("agreement") is not None else ""))
        for r in ("1", "2", "3", "4"):
            g = blk["regimes"][r]
            best = ", ".join(f"{b['fund']} {b['excess']:+.2f}" for b in g["best"])
            worst = ", ".join(f"{b['fund']} {b['excess']:+.2f}" for b in g["worst"])
            print(f"  R{r}: n={g['n']:3d}  SPY avg {g['spy_mean']:+.2f}% ({g['spy_hit']:.0%} positive)  best vs SPY: {best}  |  worst: {worst}")


def main() -> None:
    res = build()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res["stats"].to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(res["feed"], indent=1))
    res["realized_monthly"].to_csv(RESULTS_DIR / "us_regime_monthly_realized.csv")
    res["called_monthly"].to_csv(RESULTS_DIR / "us_regime_monthly_called.csv")
    res["realized_quarterly"].to_csv(RESULTS_DIR / "us_regime_quarterly_realized.csv")
    print_summary(res)
    print(f"\nwrote {OUT_CSV} ({len(res['stats'])} rows) and {OUT_JSON}")


if __name__ == "__main__":
    main()
