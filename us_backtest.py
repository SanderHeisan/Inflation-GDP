"""
US GIP quad model -- walk-forward backtest CLI.

    python -m usmodel.fetch_data_us          # once, to cache real data
    python us_backtest.py                    # full run, real FRED+Zillow data
    python us_backtest.py --start 2017 --end 2026 --horizon 4
    python us_backtest.py --revision-mode none      # upper bound (final GDP)
    python us_backtest.py --no-self-calibrate       # static config constants

Writes to results_us/:
    us_backtest_results.parquet   every quad prediction, with vintage bookkeeping
    us_summary.csv                per-horizon hit rates + benchmark edges
    us_axis_accuracy.csv          MAE of the growth/inflation YoY levels
    us_cpi_errors.csv             CPI YoY/MoM MAE vs naive benchmarks
    us_cpi_direction.csv          next-print direction hit rate by conviction
    us_confusion_h*.csv           confusion matrices per horizon
    us_growth_variants.csv        --growth-variants: the growth-side 2x2
    us_growth_ceiling.csv         per-horizon ceiling for the growth call
    us_direction.csv              the four direction calls: hit by call x horizon
    us_direction_conviction.csv   the same by conviction bucket
    us_direction_calls.csv        every direction call made (or abstained)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dataclasses import replace

from usbacktest import direction, monthly, scoring
from usbacktest.btconfig import BACKTEST_START, RESULTS_DIR, USVintageConfig
from usbacktest.engine import run_backtest
from usmodel import data_bundle


def growth_variants(bundle, start, end, horizon, cfg, real_first, real_final
                    ) -> pd.DataFrame:
    """The growth-side selection surface, published rather than just its
    winner. Two independent switches: whether the nowcast quarter uses the
    fitted activity-indicator regression, and whether the path past it
    glides (geometric convergence from the nowcast) or steps straight to a
    trend re-estimated per vintage. Effective sample is only ~38 distinct
    target quarters, so treat differences of a couple of points as noise."""
    variants = {
        # the original: momentum-only nowcast, geometric glide to a static trend
        "momentum nowcast + glide (original)":
            replace(cfg, use_indicator_nowcast=False, fit_trend=False,
                    fit_convergence=True),
        "fitted nowcast + glide":
            replace(cfg, use_indicator_nowcast=True, fit_trend=False,
                    fit_convergence=True),
        "fitted nowcast + static trend, flat":
            replace(cfg, use_indicator_nowcast=True, fit_trend=False,
                    fit_convergence=False),
        "fitted nowcast + fitted trend, flat (shipped)":
            replace(cfg, use_indicator_nowcast=True, fit_trend=True,
                    fit_convergence=False),
    }
    rows = {}
    for name, vcfg in variants.items():
        preds = run_backtest(bundle, start, end, max_horizon=horizon,
                             cfg=vcfg)
        sc = scoring.score(preds, real_first, "fr", range(horizon + 1)).loc[
            ("fr", "model")]
        ax = scoring.axis_accuracy(preds, real_final)
        rows[name] = pd.Series({
            "quad_hit_h0": sc["hit_rate"][0],
            "quad_hit_mean": sc["hit_rate"].mean(),
            "growth_dir_h0": sc["growth_dir_hit"][0],
            "growth_dir_h1plus": sc["growth_dir_hit"][1:].mean(),
            "growth_dir_mean": sc["growth_dir_hit"].mean(),
            "mae_growth_yoy_h0": ax["mae_growth_yoy_pp"][0],
            "mae_growth_yoy_mean": ax["mae_growth_yoy_pp"].mean(),
        })
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=BACKTEST_START)
    ap.add_argument("--end", default=None,
                    help="default: the last month with published CPI")
    ap.add_argument("--horizon", type=int, default=4,
                    help="max quad horizon in quarters")
    ap.add_argument("--freq", default="M", choices=["M", "Q"],
                    help="as-of date frequency")
    ap.add_argument("--revision-mode", default="auto",
                    choices=["auto", "realtime", "noise", "none"])
    ap.add_argument("--sigma", type=float, default=None,
                    help="GDP revision sigma in pp of QoQ (noise mode)")
    ap.add_argument("--no-indicator-nowcast", action="store_true",
                    help="disable the fitted activity-indicator GDP nowcast "
                         "and fall back to trailing momentum (the growth-side "
                         "before/after)")
    ap.add_argument("--no-self-calibrate", action="store_true",
                    help="use the static config coefficients instead of "
                         "re-fitting them on each vintage's own history")
    ap.add_argument("--growth-variants", action="store_true",
                    help="also run the 2x2 of growth-side settings "
                         "(momentum vs fitted nowcast) x (static vs fitted "
                         "convergence) and write us_growth_variants.csv")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    bundle = data_bundle.load_bundle()
    end = args.end or str(bundle.cpi.index[-1])
    cfg = USVintageConfig(
        revision_mode=args.revision_mode,
        self_calibrate=not args.no_self_calibrate,
        use_indicator_nowcast=not args.no_indicator_nowcast)
    if args.sigma is not None:
        cfg.revision_sigma_pp = args.sigma

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"US GIP backtest | data: {bundle.source}")
    print(f"  CPI  {bundle.cpi.index[0]}..{bundle.cpi.index[-1]}   "
          f"GDP {bundle.gdp.index[0]}..{bundle.gdp.index[-1]}   "
          f"ZORI {bundle.market_rent.index[0]}..{bundle.market_rent.index[-1]}")
    print(f"  as-of {args.start}..{end} ({args.freq})   horizon "
          f"0..{args.horizon}q   revisions={cfg.revision_mode}"
          f"{'  self-calibrating' if cfg.self_calibrate else ''}"
          f"{'  indicator-nowcast' if cfg.use_indicator_nowcast else '  momentum-nowcast'}")

    # ---- Quad backtest -----------------------------------------------------
    preds = run_backtest(bundle, args.start, end, freq=args.freq,
                         max_horizon=args.horizon, cfg=cfg)
    preds.to_parquet(out / "us_backtest_results.parquet")
    print(f"\n{len(preds)} quad predictions over "
          f"{preds['asof'].nunique()} as-of dates "
          f"(revision mode: {preds['revision_mode'].unique().tolist()})")

    real_final = scoring.realized_quads_final(bundle)
    real_first = scoring.realized_quads_first_release(bundle, cfg)
    summary = pd.concat([
        scoring.score(preds, real_final, "final", range(args.horizon + 1)),
        scoring.score(preds, real_first, "first_release",
                      range(args.horizon + 1)),
    ])
    summary.to_csv(out / "us_summary.csv")

    show = ["n", "hit_rate", "hit_rate_high_conviction", "growth_dir_hit",
            "inflation_dir_hit", "edge_vs_persistence", "edge_vs_base_effects"]
    print("\n=== Quad hit rate by horizon (vs FINAL-vintage realized quads) ===")
    print(summary.loc["final"].reindex(columns=show).round(3).to_string())
    print("\n=== Same, vs FIRST-RELEASE realized quads (the real-time truth) ===")
    print(summary.loc["first_release"].reindex(columns=show).round(3)
          .to_string())

    ceil = scoring.growth_direction_ceiling(bundle, preds)
    gdir = summary.loc[("first_release", "model"), "growth_dir_hit"]
    ceil["model"] = gdir
    ceil["gap"] = ceil["model"] - ceil["ceiling"]
    ceil.to_csv(out / "us_growth_ceiling.csv")
    print("\n=== Growth-direction call vs its structural ceiling ===")
    print("  d_growth(q) ~= qoq(q) - qoq(q-4), and the year-ago term is")
    print("  already published at most horizons, so the call is 'trend vs a")
    print("  known base'. With no skill on qoq(q) -- which is what the")
    print("  experiments found past the nowcast quarter -- that IS the best")
    print("  available call, and how often it is right is the ceiling.")
    print(ceil.round(3).to_string())

    axis = scoring.axis_accuracy(preds, real_final)
    axis.to_csv(out / "us_axis_accuracy.csv")
    print("\n=== Forecast error of the two axes (pp, vs final vintage) ===")
    print(axis.round(3).to_string())

    for h, cm in scoring.confusion_by_horizon(
            preds, real_final, range(args.horizon + 1)).items():
        cm.to_csv(out / f"us_confusion_h{h}.csv")

    # ---- Monthly CPI backtest ---------------------------------------------
    errs = monthly.monthly_backtest(bundle, args.start, end, cfg)
    err_summary = monthly.summarize_errors(errs)
    err_summary.to_csv(out / "us_cpi_errors.csv")
    print("\n=== CPI forecast error, pp (MAE of YoY at h months ahead) ===")
    print(err_summary.round(3).to_string())

    cpi_dir = monthly.direction_backtest(bundle, args.start, end, cfg)
    dir_summary = monthly.summarize_direction(cpi_dir)
    dir_summary.to_csv(out / "us_cpi_direction.csv")
    cpi_dir.to_csv(out / "us_cpi_direction_calls.csv", index=False)
    print("\n=== Next CPI print: YoY direction call by conviction ===")
    print(dir_summary.round(3).to_string())

    # ---- Direction calls: growth and inflation, QoQ and YoY ---------------
    dcalls = direction.direction_backtest(bundle, args.start, end,
                                          args.horizon, cfg)
    dcalls.to_csv(out / "us_direction_calls.csv", index=False)
    by_h = direction.summarize_by_horizon(dcalls)
    by_h.to_csv(out / "us_direction.csv")
    by_c = direction.summarize_by_conviction(dcalls)
    by_c.to_csv(out / "us_direction_conviction.csv")
    by_g = direction.summarize_by_grade(dcalls)
    by_g.to_csv(out / "us_direction_growth_qoq_grade.csv")
    stretch = direction.summarize_stretch(dcalls)
    stretch.to_csv(out / "us_consumer_stretch.csv")
    print("\n=== Sequential growth call by grade (two votes: GDP + real PCE) ===")
    print(by_g.round(3).to_string())
    print("\n=== Stretched consumer as a supporting flag on growth YoY ===")
    print(stretch.round(3).to_string())
    print("\n=== Direction calls: hit rate by horizon (call share in brackets) ===")
    wide = by_h["hit"].unstack("horizon")
    share = by_h["call_share"].unstack("horizon")
    disp = (wide * 100).round(1).astype(str) + " (" + \
        (share * 100).round(0).astype(int).astype(str) + "%)"
    disp = disp.mask(wide.isna(), "no call")
    disp.index = [direction.CALL_LABELS[c] for c in disp.index]
    print(disp.to_string())
    print("\n=== Direction calls: hit rate by conviction (all horizons) ===")
    print(by_c.round(3).to_string())

    if args.growth_variants:
        print("\n=== Growth-side variants (first-release truth) ===")
        var = growth_variants(bundle, args.start, end, args.horizon, cfg,
                              real_first, real_final)
        var.to_csv(out / "us_growth_variants.csv")
        print(var.round(3).to_string())

    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
