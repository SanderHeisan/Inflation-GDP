"""
Scoring: realized quads (final-vintage and first-release), per-horizon hit
rates, confusion matrices, flip precision/recall, and benchmark edges.

Two realized truths matter and are reported side by side:
  * final          - quads computed from today's (current-vintage) data;
                     "what actually happened" after all revisions
  * first_release  - quads as they printed in real time, from each quarter's
                     first published estimate (true vintages in 'realtime'
                     mode, the simulated ones in 'noise' mode)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quadmap import quads

from .btconfig import VintageConfig
from .data_bundle import RawDataBundle
from . import vintage as vt

RANDOM_QUAD_HIT = 0.25       # uniform over 4 quads
RANDOM_DIRECTION_HIT = 0.50  # coin flip per axis

BENCHMARK_COLS = {"persistence": "bench_persistence_quad",
                  "base_effects": "bench_base_quad"}


# ---------------------------------------------------------------------------
# Realized quads
# ---------------------------------------------------------------------------

def realized_quads_final(bundle: RawDataBundle) -> pd.DataFrame:
    """Quad per quarter from current-vintage (fully revised) data."""
    g_yoy = (bundle.gdp_final / bundle.gdp_final.shift(4) - 1) * 100
    cpi_yoy = (bundle.cpi.pct_change(12) * 100).dropna()
    i_yoy_q = quads.monthly_to_quarterly_yoy(cpi_yoy)
    return quads.classify(g_yoy.dropna(), i_yoy_q)


def realized_quads_first_release(bundle: RawDataBundle,
                                 cfg: VintageConfig | None = None
                                 ) -> pd.DataFrame:
    """Quad per quarter as it printed the day its GDP was first released.

    For each quarter q, rebuild the vintage at q's first release date and
    classify; q's row in that table is the first-release quad (its YoY delta
    is computed against the previous quarter *as estimated in that same
    vintage*, exactly as a real-time reader would have seen it).
    """
    cfg = cfg or VintageConfig()
    rows = {}
    for q in bundle.gdp_final.index:
        asof = vt.first_release_date(q, cfg.gdp_pub_lag_days)
        try:
            v = vt.build_vintage(bundle, asof, cfg)
        except ValueError:      # not enough history at the start of sample
            continue
        g_yoy = ((v.gdp_level / v.gdp_level.shift(4) - 1) * 100).dropna()
        cpi_yoy = (v.cpi_index.pct_change(12) * 100).dropna()
        table = quads.classify(g_yoy, quads.monthly_to_quarterly_yoy(cpi_yoy))
        if q in table.index:
            rows[q] = table.loc[q]
    out = pd.DataFrame(rows).T
    out.index = pd.PeriodIndex(out.index, freq="Q")
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _attach_realized(preds: pd.DataFrame,
                     realized: pd.DataFrame) -> pd.DataFrame:
    """Join realized quad/deltas for target quarter and its predecessor."""
    r = realized[["quad", "d_growth", "d_inflation"]].astype(float)
    r.index = r.index.astype(str)
    df = preds.copy()
    df = df.join(r.add_prefix("real_"), on="target_quarter")
    prev = r.copy()
    prev.index = (pd.PeriodIndex(prev.index, freq="Q") + 1).astype(str)
    df = df.join(prev[["quad"]].add_prefix("real_prev_"), on="target_quarter")
    return df.dropna(subset=["real_quad"])


def flip_metrics(scored: pd.DataFrame) -> dict:
    """Regime-change calls. A predicted flip: the call for the target
    quarter differs from the realized quad of the quarter before it. Of
    those, precision counts how many quarters actually changed regime
    (exact: changed *to the predicted quad*); recall counts how many actual
    flips were called ahead of time."""
    df = scored.dropna(subset=["real_prev_quad"])
    pred_flip = df["pred_quad"] != df["real_prev_quad"]
    real_flip = df["real_quad"] != df["real_prev_quad"]

    n_pred = int(pred_flip.sum())
    n_real = int(real_flip.sum())
    return {
        "n_pred_flips": n_pred,
        "n_real_flips": n_real,
        "flip_precision": float((pred_flip & real_flip).sum() / n_pred)
        if n_pred else np.nan,
        "flip_precision_exact": float(
            (pred_flip & real_flip
             & (df["pred_quad"] == df["real_quad"])).sum() / n_pred)
        if n_pred else np.nan,
        "flip_recall": float((pred_flip & real_flip).sum() / n_real)
        if n_real else np.nan,
    }


def confusion_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    """4x4: rows = realized quad, columns = predicted quad."""
    m = pd.crosstab(scored["real_quad"].astype(int),
                    scored["pred_quad"].astype(int))
    return m.reindex(index=range(1, 5), columns=range(1, 5), fill_value=0)


def _hit_rates(scored: pd.DataFrame) -> dict:
    hc = scored[~scored["low_conviction"]]
    out = {
        "n": len(scored),
        "hit_rate": float((scored["pred_quad"] == scored["real_quad"]).mean()),
        "growth_dir_hit": float(
            ((scored["pred_d_growth"] > 0)
             == (scored["real_d_growth"] > 0)).mean()),
        "inflation_dir_hit": float(
            ((scored["pred_d_inflation"] > 0)
             == (scored["real_d_inflation"] > 0)).mean()),
        "n_high_conviction": len(hc),
        "hit_rate_high_conviction": float(
            (hc["pred_quad"] == hc["real_quad"]).mean()) if len(hc) else np.nan,
    }
    out.update(flip_metrics(scored))
    return out


def score(preds: pd.DataFrame, realized: pd.DataFrame, basis: str,
          horizons: range | list | None = None) -> pd.DataFrame:
    """Per-horizon summary for the model and each benchmark against one
    realized-quad table. Returns tidy rows keyed by (basis, strategy,
    horizon) with the model's edge over every benchmark attached."""
    horizons = list(horizons) if horizons is not None \
        else sorted(preds["horizon"].unique())
    joined = _attach_realized(preds, realized)

    rows = []
    for h in horizons:
        sub = joined[joined["horizon"] == h]
        if sub.empty:
            continue
        model_row = {"basis": basis, "strategy": "model", "horizon": h,
                     **_hit_rates(sub)}

        bench_hits = {}
        for name, col in BENCHMARK_COLS.items():
            b = sub.dropna(subset=[col])
            hit = float((b[col] == b["real_quad"]).mean()) if len(b) else np.nan
            bench_hits[name] = hit
            bflip = b[col] != b["real_prev_quad"]
            rflip = b["real_quad"] != b["real_prev_quad"]
            rows.append({
                "basis": basis, "strategy": name, "horizon": h, "n": len(b),
                "hit_rate": hit,
                "flip_precision": float((bflip & rflip).sum() / bflip.sum())
                if bflip.sum() else np.nan,
                "flip_recall": float((bflip & rflip).sum() / rflip.sum())
                if rflip.sum() else np.nan,
            })
        rows.append({
            "basis": basis, "strategy": "random", "horizon": h, "n": len(sub),
            "hit_rate": RANDOM_QUAD_HIT,
            "growth_dir_hit": RANDOM_DIRECTION_HIT,
            "inflation_dir_hit": RANDOM_DIRECTION_HIT,
        })

        model_row["edge_vs_persistence"] = \
            model_row["hit_rate"] - bench_hits.get("persistence", np.nan)
        model_row["edge_vs_base_effects"] = \
            model_row["hit_rate"] - bench_hits.get("base_effects", np.nan)
        model_row["edge_vs_random"] = model_row["hit_rate"] - RANDOM_QUAD_HIT
        rows.append(model_row)

    df = pd.DataFrame(rows).set_index(["basis", "strategy", "horizon"])
    return df.sort_index()


def confusion_by_horizon(preds: pd.DataFrame, realized: pd.DataFrame,
                         horizons: range | list) -> dict[int, pd.DataFrame]:
    joined = _attach_realized(preds, realized)
    return {h: confusion_matrix(joined[joined["horizon"] == h])
            for h in horizons}
