"""
Populate data/ with the cached series the backtest needs. Run this on a
machine with network access (the hosted sandbox blocks data.ssb.no and
norges-bank.no):

    python -m backtest.fetch_data [--data-dir data]

What it writes automatically:
    gdp_mainland_q.csv   SSB 09190 mainland GDP, constant prices, SA (current vintage)
    cpi_monthly.csv      SSB 03013 total CPI index (unrevised by construction)
    i44_monthly.csv      Norges Bank I-44 krone index

What it cannot fetch for you (documented formats below):

  gdp_vintages.csv - TRUE GDP VINTAGES (Norges Bank real-time database).
    Norges Bank maintains real-time vintages of the Norwegian QNA (the
    dataset behind their nowcasting/forecast-evaluation research; see
    norges-bank.no -> Statistics, or contact their statistics unit).
    Convert whatever export you obtain to a CSV shaped as:
        first column: quarter ('2012Q1'), one row per quarter
        remaining columns: vintage/publication dates ('2012-05-15'),
        values: mainland GDP level as published in that vintage (blank where
        the quarter did not exist yet).
    If the file is present the backtest switches to revision_mode='realtime'
    automatically; otherwise it runs the flagged revision-noise model.

  market_monthly.csv - columns: period, power_ore_kwh, brent_usd, usdnok.
    Monthly averages of Nord Pool system/NO-area spot (ore/kWh incl VAT),
    Brent front price (USD/bbl) and USDNOK. Nord Pool and Brent history need
    a data vendor; USDNOK can come from Norges Bank's EXR API.

  wage_norms.csv - columns: year, norm_pct. The frontfag settlement norm
    per year (TBU reports). A dozen rows typed in by hand.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import btconfig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(btconfig.DATA_DIR))
    args = ap.parse_args(argv)
    d = Path(args.data_dir)
    d.mkdir(parents=True, exist_ok=True)

    from quadmap import data_sources as ds

    print("fetching SSB mainland GDP (09190)...")
    gdp = ds.fetch_mainland_gdp()
    gdp.rename("value").rename_axis("period").to_csv(
        d / btconfig.DATA_FILES["gdp"])

    print("fetching SSB CPI (03013)...")
    cpi = ds.fetch_cpi_by_group().iloc[:, 0].dropna()
    cpi.rename("value").rename_axis("period").to_csv(
        d / btconfig.DATA_FILES["cpi"])

    print("fetching Norges Bank I-44...")
    i44 = ds.fetch_i44()
    i44.rename("value").rename_axis("period").to_csv(
        d / btconfig.DATA_FILES["i44"])

    print(f"wrote GDP/CPI/I-44 to {d.resolve()}")
    for manual in ("market", "gdp_vintages", "wage_norms"):
        f = d / btconfig.DATA_FILES[manual]
        if not f.exists():
            print(f"MANUAL: {f.name} not present - see module docstring "
                  "for the expected format")


if __name__ == "__main__":
    main()
