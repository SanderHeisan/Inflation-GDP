"""
Scoring the US quad calls: realized quads, per-horizon hit rates, confusion
matrices, flip precision/recall, and the edge over each benchmark.

Two realized truths are reported side by side:
  final          - quads from today's (fully revised) GDP and CPI; "what
                   actually happened"
  first_release  - quads as they printed in real time, from each quarter's
                   first published estimate. With BEA this matters a lot:
                   a quarter can be a Quad 1 on the advance estimate and a
                   Quad 4 two years later. The model is a real-time product,
                   so the first-release column is the one it should be
                   judged on; the final column is what a subscriber sees in
                   hindsight.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quadmap import quads
from usmodel.data_bundle import USDataBundle

from .btconfig import USVintageConfig
from . import vintage as vt

RANDOM_QUAD_HIT = 0.25       # uniform over 4 quads
RANDOM_DIRECTION_HIT = 0.50  # coin flip per axis

BENCHMARK_COLS = {"persistence": "bench_persistence_quad",
                  "base_effects": "bench_base_quad"}


def realized_quads_final(bundle: USDataBundle) -> pd.DataFrame:
    """Quad per quarter from current-vintage data."""
    g_yoy = (bundle.gdp.pct_change(4) * 100).dropna()
    i_yoy_q = quads.monthly_to_quarterly_yoy(
        quads.calendar_pct_change(bundle.cpi, 12).dropna())
    return quads.classify(g_yoy, i_yoy_q)


def realized_quads_first_release(bundle: USDataBundle,
                                 cfg: USVintageConfig | None = None
                                 ) -> pd.DataFrame:
    """Quad per quarter as it printed on its GDP release day: rebuild the
    vintage at that date and read the quarter's row, so its YoY delta is
    measured against the previous quarter *as estimated in that same
    vintage* -- exactly what a real-time reader saw."""
    # Realized quads read published levels only - no projection, so the GDP
    # nowcast fit would be wasted work.
    cfg = replace(cfg or USVintageConfig(), use_indicator_nowcast=False)
    rows = {}
    for q in bundle.gdp.index:
        asof = vt.first_release_date(q, cfg.gdp_pub_lag_days)
        try:
            v = vt.build_vintage(bundle, asof, cfg)
        except ValueError:
            continue
        g_yoy = (v.gdp_level.pct_change(4) * 100).dropna()
        i_yoy_q = quads.monthly_to_quarterly_yoy(
            quads.calendar_pct_change(v.cpi_index, 12).dropna())
        table = quads.classify(g_yoy, i_yoy_q)
        if q in table.index:
            rows[q] = table.loc[q]
    out = pd.DataFrame(rows).T
    out.index = pd.PeriodIndex(out.index, freq="Q")
    return out


def _attach_realized(preds: pd.DataFrame,
                     realized: pd.DataFrame) -> pd.DataFrame:
    """Join the realized quad/deltas for the target quarter and the quarter
    before it (the latter is what a 'flip' is measured against)."""
    r = realized[["quad", "d_growth", "d_inflation"]].astype(float)
    r.index = r.index.astype(str)
    df = preds.join(r.add_prefix("real_"), on="target_quarter")
    prev = r[["quad"]].copy()
    prev.index = (pd.PeriodIndex(prev.index, freq="Q") + 1).astype(str)
    df = df.join(prev.add_prefix("real_prev_"), on="target_quarter")
    return df.dropna(subset=["real_quad"])


def flip_metrics(scored: pd.DataFrame) -> dict:
    """Regime-change calls. A predicted flip: the call for the target
    quarter differs from the realized quad of the quarter before it.
    Precision counts how many of those quarters actually changed regime
    (and, 'exact', changed to the predicted quad); recall counts how many
    real flips were called ahead of time."""
    df = scored.dropna(subset=["real_prev_quad"])
    pred_flip = df["pred_quad"] != df["real_prev_quad"]
    real_flip = df["real_quad"] != df["real_prev_quad"]
    n_pred, n_real = int(pred_flip.sum()), int(real_flip.sum())
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
        "growth_dir_hit": float(((scored["pred_d_growth"] > 0)
                                 == (scored["real_d_growth"] > 0)).mean()),
        "inflation_dir_hit": float(((scored["pred_d_inflation"] > 0)
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
    realized-quad table, keyed by (basis, strategy, horizon)."""
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
        rows.append({"basis": basis, "strategy": "random", "horizon": h,
                     "n": len(sub), "hit_rate": RANDOM_QUAD_HIT,
                     "growth_dir_hit": RANDOM_DIRECTION_HIT,
                     "inflation_dir_hit": RANDOM_DIRECTION_HIT})

        model_row["edge_vs_persistence"] = \
            model_row["hit_rate"] - bench_hits.get("persistence", np.nan)
        model_row["edge_vs_base_effects"] = \
            model_row["hit_rate"] - bench_hits.get("base_effects", np.nan)
        model_row["edge_vs_random"] = model_row["hit_rate"] - RANDOM_QUAD_HIT
        rows.append(model_row)

    return pd.DataFrame(rows).set_index(
        ["basis", "strategy", "horizon"]).sort_index()


def confusion_by_horizon(preds: pd.DataFrame, realized: pd.DataFrame,
                         horizons: range | list) -> dict[int, pd.DataFrame]:
    joined = _attach_realized(preds, realized)
    return {h: confusion_matrix(joined[joined["horizon"] == h])
            for h in horizons}


def growth_direction_ceiling(bundle: USDataBundle, preds: pd.DataFrame,
                             window: int = 24) -> pd.DataFrame:
    """How good the growth-direction call could possibly be, per horizon.

    Past the nowcast quarter, d_growth(q) = yoy(q) - yoy(q-1) ~= qoq(q) -
    qoq(q-4), and qoq(q-4) is already published for most horizons. So the
    call is "will next year's QoQ land above or below a number we already
    know". If qoq(q) is genuinely unforecastable -- which is what every
    experiment in this repo found past the nowcast quarter -- the best
    available prediction is `trend - qoq(q-4)`, and how often THAT is right
    is the ceiling. For iid qoq the closed form is E[Phi(|Z|)] = 75%.

    Scored on the same (as-of, horizon, target) rows the model was scored
    on, using the true year-ago QoQ, so the model's gap to this column is
    the part of the growth axis that is actually still on the table.
    `base_published_share` says how often that year-ago quarter was really
    in the vintage: where it is 0 the base is itself a model estimate, and
    the ceiling is not reachable without a better nowcast.
    """
    qoq = bundle.gdp.pct_change().dropna()
    rows = []
    for _, r in preds.iterrows():
        target = pd.Period(r["target_quarter"], freq="Q")
        base, last = target - 4, pd.Period(r["last_gdp_quarter"], freq="Q")
        if base not in qoq.index or target not in qoq.index:
            continue
        trend = float(qoq[qoq.index <= last].tail(window).median())
        rows.append({
            "horizon": int(r["horizon"]),
            "hit": np.sign(trend - float(qoq[base]))
            == np.sign(float(qoq[target] - qoq[base])),
            "base_published": base <= last,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    out = df.groupby("horizon").agg(n=("hit", "size"),
                                    ceiling=("hit", "mean"),
                                    base_published_share=("base_published",
                                                          "mean"))
    out["closed_form_ceiling"] = 0.75
    return out


def axis_accuracy(preds: pd.DataFrame, realized: pd.DataFrame
                  ) -> pd.DataFrame:
    """Per-horizon accuracy of the two YoY *levels* the quad is built from,
    in percentage points. A quad can be wrong while both levels are close
    (a near-zero delta flipping sign), so these separate the model's
    forecasting error from the classification's knife edge."""
    joined = _attach_realized(preds, realized)
    g_yoy = realized["growth_yoy"].astype(float)
    g_yoy.index = g_yoy.index.astype(str)
    i_yoy = realized["inflation_yoy"].astype(float)
    i_yoy.index = i_yoy.index.astype(str)
    joined = joined.join(g_yoy.rename("real_growth_yoy"), on="target_quarter")
    joined = joined.join(i_yoy.rename("real_inflation_yoy"),
                         on="target_quarter")

    rows = []
    for h, sub in joined.groupby("horizon"):
        rows.append({
            "horizon": int(h), "n": len(sub),
            "mae_growth_yoy_pp":
                float((sub["pred_growth_yoy"]
                       - sub["real_growth_yoy"]).abs().mean()),
            "mae_inflation_yoy_pp":
                float((sub["pred_inflation_yoy"]
                       - sub["real_inflation_yoy"]).abs().mean()),
            "mae_d_growth_pp":
                float((sub["pred_d_growth"] - sub["real_d_growth"]).abs().mean()),
            "mae_d_inflation_pp":
                float((sub["pred_d_inflation"]
                       - sub["real_d_inflation"]).abs().mean()),
        })
    return pd.DataFrame(rows).set_index("horizon").sort_index()
