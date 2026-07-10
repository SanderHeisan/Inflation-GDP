"""
The quad engine. Everything upstream exists to feed two quarterly series
into this file: YoY real GDP growth and YoY CPI inflation.

Definitions (Hedgeye GIP convention):
  Quad 1: growth accelerating, inflation decelerating   (Goldilocks)
  Quad 2: growth accelerating, inflation accelerating   (Reflation)
  Quad 3: growth decelerating, inflation accelerating   (Stagflation)
  Quad 4: growth decelerating, inflation decelerating   (Disinflation)

'Accelerating' = this quarter's YoY minus last quarter's YoY > 0.
The magnitude of that delta is the conviction measure; deltas inside the
config deadband are flagged low-conviction.
"""
from __future__ import annotations

import pandas as pd

from . import config

QUAD_LABELS = {1: "Goldilocks", 2: "Reflation",
               3: "Stagflation", 4: "Disinflation"}


def monthly_to_quarterly_yoy(cpi_yoy_monthly: pd.Series) -> pd.Series:
    """Quarterly CPI YoY = average of the three monthly YoY prints."""
    q = cpi_yoy_monthly.copy()
    q.index = q.index.asfreq("Q")
    return q.groupby(level=0).mean()


def classify(growth_yoy: pd.Series, inflation_yoy: pd.Series) -> pd.DataFrame:
    """Build the quad table.

    Both inputs: quarterly YoY (%), aligned Period[Q] index, including the
    projected quarters. Returns one row per quarter with deltas, quad number,
    label, and conviction flag.
    """
    df = pd.DataFrame({"growth_yoy": growth_yoy,
                       "inflation_yoy": inflation_yoy}).dropna()
    df["d_growth"] = df["growth_yoy"].diff()
    df["d_inflation"] = df["inflation_yoy"].diff()
    df = df.dropna()

    def _quad(row) -> int:
        g_up = row["d_growth"] > 0
        i_up = row["d_inflation"] > 0
        if g_up and not i_up:
            return 1
        if g_up and i_up:
            return 2
        if not g_up and i_up:
            return 3
        return 4

    df["quad"] = df.apply(_quad, axis=1)
    df["label"] = df["quad"].map(QUAD_LABELS)
    db = config.QUAD_DEADBAND_PP
    df["low_conviction"] = (df["d_growth"].abs() < db) | \
                           (df["d_inflation"].abs() < db)
    return df


def quad_flips(quad_table: pd.DataFrame) -> pd.DataFrame:
    """Quarters where the quad changes vs the prior quarter -- these are the
    signals the framework claims edge on, so they're what a subscriber
    briefing should lead with."""
    changed = quad_table["quad"] != quad_table["quad"].shift()
    return quad_table[changed & quad_table["quad"].shift().notna()]
