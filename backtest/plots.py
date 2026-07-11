"""
Static report figures for the backtest (matplotlib, light surface).

Quad colors are a fixed CVD-validated categorical mapping (worst adjacent
CVD deltaE 24; the yellow slot is below 3:1 contrast on the surface, so every
figure that uses it also carries direct labels).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

QUAD_COLORS = {1: "#008300",   # Goldilocks
               2: "#eda100",   # Reflation
               3: "#e34948",   # Stagflation
               4: "#2a78d6"}   # Disinflation
QUAD_NAMES = {1: "Q1 Goldilocks", 2: "Q2 Reflation",
              3: "Q3 Stagflation", 4: "Q4 Disinflation"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

STRATEGY_COLORS = {"model": "#2a78d6", "persistence": "#eda100",
                   "base_effects": "#4a3aa7"}
STRATEGY_NAMES = {"model": "Model", "persistence": "Persistence",
                  "base_effects": "Base effects only"}

SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec",
             "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_hit_rate_by_horizon(summary: pd.DataFrame, basis: str,
                             path: str) -> None:
    """Quad hit rate per horizon: model line vs each benchmark, random as a
    dashed reference."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    sub = summary.loc[basis]
    for strat, color in STRATEGY_COLORS.items():
        if strat not in sub.index.get_level_values(0):
            continue
        s = sub.loc[strat]["hit_rate"].sort_index()
        ax.plot(s.index, s.values * 100, color=color, linewidth=2,
                marker="o", markersize=5, label=STRATEGY_NAMES[strat])
        ax.annotate(STRATEGY_NAMES[strat],
                    (s.index[-1], s.values[-1] * 100),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", va="center")
    ax.axhline(25, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.annotate("Random (25%)", (ax.get_xlim()[0], 25), xytext=(4, 5),
                textcoords="offset points", color=MUTED, fontsize=9)

    ax.set_xlabel("Horizon (quarters ahead)", color=INK_2, fontsize=10)
    ax.set_ylabel("Quad hit rate (%)", color=INK_2, fontsize=10)
    ax.set_xticks(sorted(sub.index.get_level_values("horizon").unique()))
    ax.set_ylim(0, 100)
    ax.set_title(f"Quad hit rate by horizon - scored vs {basis} vintages",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrix(matrix: pd.DataFrame, horizon: int,
                          path: str) -> None:
    """4x4 heatmap, rows = realized quad, cols = predicted. Sequential blue
    ramp on row-normalized shares; every cell carries its count."""
    shares = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    cmap = LinearSegmentedColormap.from_list("blues", SEQ_BLUES)

    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(shares.fillna(0).values, cmap=cmap, vmin=0, vmax=1)

    labels = [QUAD_NAMES[q] for q in range(1, 5)]
    ax.set_xticks(range(4), labels, fontsize=8.5, color=INK_2)
    ax.set_yticks(range(4), labels, fontsize=8.5, color=INK_2)
    ax.set_xlabel("Predicted", color=INK_2, fontsize=10)
    ax.set_ylabel("Realized", color=INK_2, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(4):
        for j in range(4):
            share = shares.iloc[i, j]
            dark = share == share and share > 0.55
            txt = f"{matrix.iloc[i, j]}"
            if share == share:
                txt += f"\n{share * 100:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK,
                    fontweight="bold" if i == j else "normal")

    ax.set_title(f"Confusion matrix - {horizon} quarter(s) ahead",
                 color=INK, fontsize=12, loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Share of realized quad", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


METHOD_NOTES = {
    "calibration": "P(quad) = how often the realized quad followed this "
                   "call at this horizon in the walk-forward backtest. "
                   "Outlined = model's point call.",
    "residual": "P(quad) = share of the model's backtest error cloud "
                "(per horizon) landing in each quadrant around today's "
                "predicted deltas. Outlined = model's point call.",
}


def plot_quad_probability_heatmap(prob: pd.DataFrame, calls: dict,
                                  asof, path: str,
                                  method: str = "residual") -> None:
    """Headline forecast view: P(quad) for each of the next quarters.
    Sequential blue on probability; the model's point call gets a ring."""
    cmap = LinearSegmentedColormap.from_list("blues", SEQ_BLUES)
    n = len(prob.columns)
    fig, ax = plt.subplots(figsize=(1.9 + 1.55 * n, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(prob.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n), [str(c) for c in prob.columns],
                  fontsize=10, color=INK_2)
    ax.set_yticks(range(4), [QUAD_NAMES[q] for q in prob.index],
                  fontsize=9.5, color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i, q in enumerate(prob.index):
        for j, tq in enumerate(prob.columns):
            p = prob.iloc[i, j]
            called = calls.get(tq) == q
            ax.text(j, i, f"{p * 100:.0f}%", ha="center", va="center",
                    fontsize=13 if called else 11,
                    fontweight="bold" if called else "normal",
                    color="#ffffff" if p > 0.55 else INK)
            if called:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94,
                                           fill=False, edgecolor=INK,
                                           linewidth=2))
    ax.set_title(f"Quad probabilities by quarter"
                 f" (forecast as of {pd.Timestamp(asof).date()})",
                 color=INK, fontsize=12.5, loc="left", pad=10)
    fig.text(0.01, 0.015, METHOD_NOTES.get(method, ""),
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_quad_probability_monthly(mprob: pd.DataFrame, path: str) -> None:
    """The same distributions across the next twelve months: one stacked
    probability bar per month (months of a quarter share its distribution)."""
    months = list(mprob.columns)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    x = np.arange(len(months))
    bottom = np.zeros(len(months))
    for q in mprob.index:
        vals = mprob.loc[q].to_numpy()
        ax.bar(x, vals * 100, bottom=bottom * 100, width=0.82,
               color=QUAD_COLORS[q], label=QUAD_NAMES[q],
               edgecolor=SURFACE, linewidth=2)
        for j, v in enumerate(vals):
            if v >= 0.18:   # direct-label the meaningful bands only
                ax.text(x[j], (bottom[j] + v / 2) * 100, f"{v * 100:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="#ffffff" if q in (1, 3, 4) else INK)
        bottom += vals

    ax.set_xticks(x, [str(m) for m in months], rotation=45,
                  fontsize=8.5, color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)", color=INK_2, fontsize=10)
    ax.set_title("Quad probabilities by month",
                 color=INK, fontsize=12.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.text(0.01, 0.015,
             "Quads are quarterly: the three months of a quarter share its "
             "distribution.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


ACCEL = "#eda100"   # inflation accelerating (Reflation/Stagflation side)
DECEL = "#2a78d6"   # inflation decelerating (Goldilocks/Disinflation side)


def plot_monthly_inflation_direction(minfl: pd.DataFrame, path: str) -> None:
    """Monthly-resolution inflation call: P(YoY accelerating) per print,
    each month with its own base effect. The predicted YoY level runs
    above the bars so the path and the odds read together."""
    months = list(minfl.index)
    x = np.arange(len(months))
    pa = minfl["p_accel"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.6, len(months) * 0.62), 4.9),
                           dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    ax.bar(x, pa * 100, width=0.82, color=ACCEL, label="Accelerating",
           edgecolor=SURFACE, linewidth=2)
    ax.bar(x, (1 - pa) * 100, bottom=pa * 100, width=0.82, color=DECEL,
           label="Decelerating", edgecolor=SURFACE, linewidth=2)
    for j, p in enumerate(pa):
        if p >= 0.15:
            ax.text(x[j], p * 50, f"{p * 100:.0f}", ha="center",
                    va="center", fontsize=8.5, color=INK)
        if (1 - p) >= 0.15:
            ax.text(x[j], (p + (1 - p) / 2) * 100, f"{(1 - p) * 100:.0f}",
                    ha="center", va="center", fontsize=8.5, color="#ffffff")
        ax.text(x[j], 104, f"{minfl['pred_yoy'].iloc[j]:.1f}", ha="center",
                va="bottom", fontsize=8, color=INK_2)
    ax.text(-0.7, 104, "YoY %", ha="right", va="bottom", fontsize=8,
            color=MUTED)

    ax.axhline(50, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_xticks(x, [str(m) for m in months], rotation=45, fontsize=8.5,
                  color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Probability (%)", color=INK_2, fontsize=10)
    ax.set_title("Will YoY inflation accelerate? - print by print",
                 color=INK, fontsize=12.5, loc="left", pad=22)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.text(0.01, 0.015,
             "Each month carries its own base effect (the known "
             "same-month-last-year MoM) and its own backtested error "
             "distribution.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_predictions(mdir: pd.DataFrame, path: str) -> None:
    """Every monthly print: the YoY number the model published one month
    earlier vs what SSB printed, with direction misses flagged below."""
    df = mdir.copy()
    df["m"] = pd.PeriodIndex(df["target_print"], freq="M")
    df = df.sort_values("m")
    x = np.arange(len(df))

    fig, (ax, axe) = plt.subplots(
        2, 1, figsize=(max(9.0, len(df) * 0.11), 6.4), dpi=150,
        sharex=True, height_ratios=[2.4, 1.0])
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axe):
        _style_axes(a)

    ax.plot(x, df["real_yoy"], color="#2a78d6", linewidth=2,
            label="Actual CPI YoY")
    ax.plot(x, df["pred_yoy"], color="#1baf7a", linewidth=2,
            linestyle=(0, (5, 2.5)), label="Model, 1 month earlier")
    ax.annotate("Actual", (x[-1], df["real_yoy"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points", color="#2a78d6",
                fontsize=9, fontweight="bold", va="center")
    ax.annotate("Model", (x[-1], df["pred_yoy"].iloc[-1]),
                xytext=(8, -12), textcoords="offset points", color="#1baf7a",
                fontsize=9, fontweight="bold", va="center")
    ax.set_ylabel("CPI YoY (%)", color=INK_2, fontsize=10)
    mae = df["yoy_error"].abs().mean()
    ax.set_title("Next-print inflation: model vs actual, every month "
                 f"(YoY MAE {mae:.2f}pp, direction hit "
                 f"{df['hit'].mean() * 100:.0f}%)",
                 color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    miss = ~df["hit"].to_numpy()
    axe.bar(x[~miss], df["yoy_error"].to_numpy()[~miss], width=0.8,
            color=BASELINE, label="Error (direction right)")
    axe.bar(x[miss], df["yoy_error"].to_numpy()[miss], width=0.8,
            color="#d03b3b", label="Error (direction MISSED)")
    axe.axhline(0, color=MUTED, linewidth=1)
    axe.set_ylabel("Error (pp)", color=INK_2, fontsize=10)
    axe.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=2,
               loc="upper left")

    step = max(1, len(df) // 14)
    axe.set_xticks(x[::step], [str(m) for m in df["m"]][::step],
                   rotation=45, fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_timeline(preds: pd.DataFrame, realized: pd.DataFrame,
                  horizons: list[int], path: str) -> None:
    """Predicted vs realized quads over time: one row per horizon plus the
    realized row, one column per target quarter, cells colored by quad."""
    targets = sorted(preds["target_quarter"].unique())
    targets = [t for t in targets
               if t in realized.index.astype(str)]
    rows = ["Realized"] + [f"Predicted h={h}" for h in horizons]

    grid = np.full((len(rows), len(targets)), np.nan)
    real_by_q = realized["quad"].copy()
    real_by_q.index = real_by_q.index.astype(str)
    for j, t in enumerate(targets):
        grid[0, j] = real_by_q.get(t, np.nan)
        for i, h in enumerate(horizons, start=1):
            sub = preds[(preds["target_quarter"] == t)
                        & (preds["horizon"] == h)]
            if len(sub):
                # latest as-of call for that (target, horizon)
                grid[i, j] = sub.sort_values("asof")["pred_quad"].iloc[-1]

    cmap = ListedColormap([QUAD_COLORS[q] for q in range(1, 5)])
    fig_w = max(8.0, len(targets) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, 1.1 + 0.5 * len(rows)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    masked = np.ma.masked_invalid(grid)
    ax.pcolormesh(np.arange(len(targets) + 1), np.arange(len(rows) + 1),
                  masked, cmap=cmap, vmin=0.5, vmax=4.5,
                  edgecolors=SURFACE, linewidth=2)
    ax.invert_yaxis()

    step = max(1, len(targets) // 16)
    ax.set_xticks(np.arange(0.5, len(targets), step),
                  targets[::step], fontsize=8, color=INK_2, rotation=45)
    ax.set_yticks(np.arange(0.5, len(rows)), rows, fontsize=9, color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    handles = [Patch(facecolor=QUAD_COLORS[q], label=QUAD_NAMES[q])
               for q in range(1, 5)]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2,
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.42))
    ax.set_title("Predicted vs realized quads (latest call per horizon)",
                 color=INK, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
