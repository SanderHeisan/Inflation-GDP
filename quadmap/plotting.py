"""Quad map chart: the Hedgeye-style scatter with the trajectory path."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

QUAD_COLORS = {1: "#1D9E75", 2: "#EF9F27", 3: "#D85A30", 4: "#378ADD"}


def plot_quad_map(quad_table: pd.DataFrame, out_path: str,
                  n_history: int = 6, title: str = "Norway GIP quad map"):
    """Scatter of (d_growth, d_inflation): last n_history realized quarters
    in gray, projected quarters colored by quad, connected in time order."""
    df = quad_table.tail(n_history + quad_table.get("projected",
                         pd.Series(False, index=quad_table.index)).sum())
    fig, ax = plt.subplots(figsize=(9, 7))

    lim = max(1.0, df[["d_growth", "d_inflation"]].abs().max().max() * 1.3)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    pad = lim * 0.9
    for (x, y, q) in [(pad, -pad, 1), (pad, pad, 2),
                      (-pad, pad, 3), (-pad, -pad, 4)]:
        ax.text(x, y, f"Quad {q}", ha="center", va="center",
                fontsize=16, color=QUAD_COLORS[q], alpha=0.35,
                fontweight="bold")

    xs, ys = df["d_growth"].values, df["d_inflation"].values
    ax.plot(xs, ys, color="#bbb", lw=1.2, zorder=1)
    is_proj = df.get("projected", pd.Series(False, index=df.index))
    for (p, row), proj in zip(df.iterrows(), is_proj):
        color = QUAD_COLORS[row["quad"]] if proj else "#999"
        size = 140 if proj else 70
        ax.scatter(row["d_growth"], row["d_inflation"], s=size, c=color,
                   zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(str(p), (row["d_growth"], row["d_inflation"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9,
                    color="#333" if proj else "#999")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Δ Real GDP growth YoY (pp, QoQ change)")
    ax.set_ylabel("Δ CPI inflation YoY (pp, QoQ change)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
