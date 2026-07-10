"""
Walk-forward backtest of the Norway GIP quad map.

    python backtest.py --start 2012 --end 2025 --horizon 4

At each as-of date the harness rebuilds the point-in-time information set
(publication lags, GDP revision handling, spot-carry market assumptions),
runs the existing quadmap projection code, and scores the predicted quads
1..H quarters ahead against realized quads - both final-vintage and
first-release truth - versus persistence / base-effects / random benchmarks.

Outputs (in --outdir):
    backtest_results.parquet   every prediction, tidy
    summary.csv                per-horizon metrics per strategy and basis
    hit_rate_final.png / hit_rate_first_release.png
    confusion_h*.png
    timeline.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest import btconfig, data_bundle, engine, plots, probability, scoring
from backtest.vintage import resolve_revision_mode


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2012", help="first as-of year or date")
    ap.add_argument("--end", default="2025", help="last as-of year or date")
    ap.add_argument("--horizon", type=int, default=4,
                    help="max prediction horizon in quarters")
    ap.add_argument("--freq", choices=["M", "Q"], default="M",
                    help="as-of date frequency (month-end or quarter-end)")
    ap.add_argument("--data-dir", default=str(btconfig.DATA_DIR),
                    help="directory of cached CSVs from fetch_data.py")
    ap.add_argument("--demo", action="store_true",
                    help="run on the synthetic demo bundle (no data files)")
    ap.add_argument("--revision-mode", choices=list(btconfig.REVISION_MODES),
                    default="auto",
                    help="GDP vintages: realtime panel, noise model, or none")
    ap.add_argument("--sigma", type=float, default=0.25,
                    help="noise mode: stdev of first-release QoQ revision (pp)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the revision-noise draws")
    ap.add_argument("--outdir", default=".", help="output directory")
    return ap.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = data_bundle.make_demo_bundle() if args.demo \
        else data_bundle.load_bundle(args.data_dir)
    cfg = btconfig.VintageConfig(revision_mode=args.revision_mode,
                                 revision_sigma_pp=args.sigma, seed=args.seed)
    mode = resolve_revision_mode(bundle, cfg)
    print(f"data source: {bundle.source} | GDP revision mode: {mode}"
          + (f" (sigma={args.sigma}pp, seed={args.seed})"
             if mode == "noise" else ""))
    if mode == "noise":
        print("NOTE: results use SIMULATED first releases, not true vintages. "
              "Drop Norges Bank real-time data into data/gdp_vintages.csv "
              "for revision_mode='realtime'.")
    elif mode == "none":
        print("NOTE: GDP revisions are IGNORED (first release treated as "
              "final) - an upper bound on real-time performance.")

    print(f"running walk-forward {args.start}..{args.end} "
          f"({args.freq} as-of dates, horizons 0..{args.horizon})...")
    preds = engine.run_backtest(bundle, args.start, args.end, freq=args.freq,
                                max_horizon=args.horizon, cfg=cfg)
    preds.to_parquet(outdir / "backtest_results.parquet", index=False)
    print(f"{len(preds)} predictions from "
          f"{preds['asof'].nunique()} as-of dates "
          f"-> {outdir / 'backtest_results.parquet'}")

    horizons = list(range(1, args.horizon + 1))
    realized_final = scoring.realized_quads_final(bundle)
    realized_first = scoring.realized_quads_first_release(bundle, cfg)

    summary = pd.concat([
        scoring.score(preds, realized_final, "final", horizons),
        scoring.score(preds, realized_first, "first_release", horizons),
    ])
    summary.insert(0, "revision_mode", mode)
    summary.to_csv(outdir / "summary.csv")

    show = summary.reset_index()
    show = show[show["strategy"].isin(["model", "persistence",
                                       "base_effects"])]
    cols = ["basis", "strategy", "horizon", "n", "hit_rate",
            "hit_rate_high_conviction", "growth_dir_hit",
            "inflation_dir_hit", "flip_precision", "flip_recall",
            "edge_vs_persistence", "edge_vs_base_effects", "edge_vs_random"]
    pd.set_option("display.width", 200)
    print("\n=== Summary (see summary.csv for all columns) ===")
    print(show[[c for c in cols if c in show.columns]].round(3)
          .to_string(index=False))

    for basis, realized in (("final", realized_final),
                            ("first_release", realized_first)):
        plots.plot_hit_rate_by_horizon(summary, basis,
                                       str(outdir / f"hit_rate_{basis}.png"))
    for h, m in scoring.confusion_by_horizon(preds, realized_final,
                                             horizons).items():
        plots.plot_confusion_matrix(m, h, str(outdir / f"confusion_h{h}.png"))
    plots.plot_timeline(preds, realized_final, horizons,
                        str(outdir / "timeline.png"))

    # Headline forecast: today's calls read through the backtest's own
    # calibration -> P(quad) per target quarter for the year ahead.
    try:
        prob, calls, asof = probability.current_quad_probabilities(
            bundle, preds, realized_final, cfg, horizons=tuple(horizons),
            method="residual")
        prob.to_csv(outdir / "quad_probabilities.csv")
        plots.plot_quad_probability_heatmap(
            prob, calls, asof,
            str(outdir / "quad_probabilities_quarterly.png"),
            method="residual")
        plots.plot_quad_probability_monthly(
            probability.monthly_probabilities(prob),
            str(outdir / "quad_probabilities_monthly.png"))
        print(f"\n=== Quad probabilities for the year ahead "
              f"(forecast as of {asof.date()}, calibrated on the backtest) ===")
        print(probability.format_probability_table(prob, calls).to_string())
    except Exception as e:
        print(f"\nforward probability view skipped: {e}")

    print(f"\nwrote summary.csv, quad-probability/hit-rate/confusion/"
          f"timeline plots to {outdir.resolve()}")
    return summary


if __name__ == "__main__":
    main()
