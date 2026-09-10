"""
Populate data/us/ with the real series the US backtest needs.

    python -m usmodel.fetch_data_us [--data-dir data/us]

Unlike SSB/Norges Bank, both sources here are reachable from the hosted
sandbox, so the US model runs on real data end to end:

    FRED (keyless fredgraph.csv endpoint)
        cpi.csv            CPIAUCSL    CPI-U all items, SA (index)
        cpi_nsa.csv        CPIAUCNS    CPI-U all items, NSA (never revised)
        cpi_core.csv       CPILFESL    CPI ex food & energy, SA
        gdp.csv            GDPC1       real GDP, chained 2017$, SAAR
        wti.csv            DCOILWTICO  WTI spot, monthly mean of daily
        dollar.csv         DTWEXBGS    broad trade-weighted USD, monthly mean
        wages.csv          CES0500000003 average hourly earnings, total private
        cpi_shelter.csv    CUSR0000SAH1
        cpi_gasoline.csv   CUSR0000SETB01
        cpi_food.csv       CPIUFDSL
        cpi_energy.csv     CPIENGSL
        cpi_supercore.csv  CUSR0000SASLE  (services less energy services)
        payrolls.csv       PAYEMS      nonfarm payrolls
        indpro.csv         INDPRO      industrial production
        retail.csv         RSAFS       retail & food services sales
        claims.csv         ICSA        initial jobless claims (weekly -> monthly)
        cfnai.csv          CFNAI       Chicago Fed national activity index
        sentiment.csv      UMCSENT     U. Michigan consumer sentiment
        yield_curve.csv    T10Y3M      10y-3m spread (daily -> monthly)
        hours.csv          AWHMAN      average weekly hours, manufacturing
        nfci.csv           NFCI        Chicago Fed financial conditions (weekly)
        credit_spread.csv  BAA10Y      Baa corporate spread over 10y Treasury
        permits.csv        PERMIT      building permits
        capex_orders.csv   NEWORDER    core capital goods new orders
        stocks.csv         NASDAQCOM   broad equity index (daily -> monthly)
        real_m2.csv        M2REAL      real M2 money stock
        housing_starts.csv HOUST       housing starts

    Zillow (public research CSVs)
        market_rent.csv    ZORI, national, smoothed, all homes+condos
                           -- the new-lease rent series CPI shelter follows
                           with a ~12 month lag. History starts 2015-01,
                           which is what bounds the backtest window.

WHAT CANNOT BE FETCHED HERE: true real-time GDP vintages. BEA revises real
GDP heavily, and the two archives that record what each quarter looked like
on its release day -- ALFRED (alfred.stlouisfed.org) and the Philadelphia
Fed Real-Time Data Set -- both sit behind bot protection that this sandbox
cannot clear (ALFRED's fredgraph endpoint returns a JS challenge; so does
the Philly Fed file host). The backtest therefore runs BEA-calibrated
*simulated* first releases by default (usbacktest.btconfig), and every
output row is flagged with the revision mode that produced it. If you can
download either archive, drop it in as data/us/gdp_vintages.csv (quarter
rows x vintage-date columns) and the harness switches to true vintages
automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
ZORI_CSV = ("https://files.zillowstatic.com/research/public_csvs/zori/"
            "Metro_zori_uc_sfrcondomfr_sm_month.csv")

# series key -> (FRED id, native frequency)
FRED_SERIES = {
    "cpi":           ("CPIAUCSL", "M"),
    "cpi_nsa":       ("CPIAUCNS", "M"),
    "cpi_core":      ("CPILFESL", "M"),
    "gdp":           ("GDPC1", "Q"),
    "wti":           ("DCOILWTICO", "D"),
    "dollar":        ("DTWEXBGS", "D"),
    "wages":         ("CES0500000003", "M"),
    "cpi_shelter":   ("CUSR0000SAH1", "M"),
    "cpi_gasoline":  ("CUSR0000SETB01", "M"),
    "cpi_food":      ("CPIUFDSL", "M"),
    "cpi_energy":    ("CPIENGSL", "M"),
    "cpi_supercore": ("CUSR0000SASLE", "M"),
    # Growth-side activity indicators for the GDP nowcast. The quad's
    # binding constraint is the growth axis, and a momentum-only nowcast
    # calls the direction of the change in YoY growth barely better than a
    # coin flip; these are the real-time series that carry the signal.
    "payrolls":      ("PAYEMS", "M"),      # nonfarm payrolls, level (k)
    "indpro":        ("INDPRO", "M"),      # industrial production index
    "retail":        ("RSAFS", "M"),       # retail & food services sales ($m)
    "claims":        ("ICSA", "W"),        # initial jobless claims, weekly
    "cfnai":         ("CFNAI", "M"),       # Chicago Fed national activity index
    "sentiment":     ("UMCSENT", "M"),     # U. Michigan consumer sentiment
    "yield_curve":   ("T10Y3M", "D"),      # 10y-3m spread, daily
    "hours":         ("AWHMAN", "M"),      # avg weekly hours, manufacturing
    # Genuinely LEADING series. The coincident block above nowcasts the
    # current quarter well but says nothing about growth two to four
    # quarters out; these are the series with a documented lead on the
    # business cycle, and they are what the multi-horizon fits lean on.
    "nfci":          ("NFCI", "W"),        # Chicago Fed financial conditions
    "credit_spread": ("BAA10Y", "D"),      # Baa corporate over 10y Treasury
    "permits":       ("PERMIT", "M"),      # building permits
    "capex_orders":  ("NEWORDER", "M"),    # core capital goods new orders
    "stocks":        ("NASDAQCOM", "D"),   # broad equity index
    "real_m2":       ("M2REAL", "M"),      # real M2 money stock
    "housing_starts": ("HOUST", "M"),      # housing starts
    # Consumer block: the direct spending series (real PCE is ~68% of GDP;
    # retail sales above is a nominal, goods-heavy proxy), real disposable
    # income and the saving rate. Used both as nowcast inputs and for the
    # 'stretched consumer' extremity signal.
    "real_pce":      ("PCEC96", "M"),      # real personal consumption expenditures
    "real_income":   ("DSPIC96", "M"),     # real disposable personal income
    "saving_rate":   ("PSAVERT", "M"),     # personal saving rate, %
}

# Publication lag in days for each indicator, used by the backtest's vintage
# builder. Employment-report series land first Friday; IP and retail sales
# mid-month; CFNAI near month end; weekly/daily series almost immediately.
INDICATOR_PUB_LAG_DAYS = {
    "payrolls": 8, "indpro": 17, "retail": 17, "claims": 5,
    "cfnai": 26, "sentiment": 2, "yield_curve": 0, "hours": 8,
    # leading block
    "nfci": 5, "credit_spread": 0, "permits": 18, "capex_orders": 26,
    "stocks": 0, "real_m2": 30, "housing_starts": 18,
    # personal income & outlays lands ~30 days after month end
    "real_pce": 30, "real_income": 30, "saving_rate": 30,
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "us"


def fetch_fred(series_id: str, freq: str = "M") -> pd.Series:
    """One FRED series as a Period-indexed Series. Daily series are
    collapsed to monthly means (the CPI blocks consume monthly averages);
    '.' is FRED's missing-value marker."""
    import requests
    r = requests.get(FRED_CSV.format(sid=series_id), timeout=90)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col])).dropna()
    if freq in ("D", "W"):
        return s.groupby(s.index.to_period("M")).mean()
    return pd.Series(s.values, index=s.index.to_period(freq)).dropna()


def fetch_zori_national() -> pd.Series:
    """Zillow Observed Rent Index, United States, monthly. Columns after the
    5 id columns are month-end dates."""
    import requests
    r = requests.get(ZORI_CSV, timeout=120)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    row = df[df["RegionName"].astype(str).str.strip() == "United States"]
    if row.empty:
        raise ValueError("no 'United States' row in the Zillow ZORI file")
    wide = row.iloc[0, 5:].astype(float).dropna()
    return pd.Series(wide.values,
                     index=pd.PeriodIndex(pd.to_datetime(wide.index), freq="M"))


def _write(series: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("value").to_frame().rename_axis("period").to_csv(path)


def main(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    data_dir = Path(data_dir)
    for key, (sid, freq) in FRED_SERIES.items():
        try:
            s = fetch_fred(sid, freq)
            _write(s, data_dir / f"{key}.csv")
            print(f"  {key:15s} {sid:15s} {len(s):5d} obs  "
                  f"{s.index[0]}..{s.index[-1]}")
        except Exception as exc:                      # noqa: BLE001
            print(f"  {key:15s} {sid:15s} FAILED: {exc}")

    try:
        z = fetch_zori_national()
        _write(z, data_dir / "market_rent.csv")
        print(f"  {'market_rent':15s} {'ZORI (US)':15s} {len(z):5d} obs  "
              f"{z.index[0]}..{z.index[-1]}")
    except Exception as exc:                          # noqa: BLE001
        print(f"  {'market_rent':15s} {'ZORI (US)':15s} FAILED: {exc}")

    print(f"\nWrote {data_dir}. Real-time GDP vintages are NOT fetchable "
          f"here - see the module docstring.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()
    print("Fetching US series...")
    main(Path(args.data_dir))
