"""Summary figures for MEMO.md: headline AUROC comparison for each prior.
Matplotlib only (no interactive/web tooling needed for a static memo figure).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS_DIR / "figures"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    }
)


def sem(x: np.ndarray) -> float:
    return x.std(ddof=1) / np.sqrt(len(x))


def fig_headline(main_results: pd.DataFrame, prior_name: str, signed: bool, display_name: str) -> None:
    d = main_results[main_results["target"] == "disease_status"]
    d = d[(d["metric"] == "auroc") & d["condition"].str.startswith(prior_name)]

    groups = {
        "baseline_pca\n(no graph)": (d[d["condition"] == f"{prior_name}_baseline_pca"]["score"].to_numpy(), BLUE),
        f"{prior_name}_real\n({display_name})": (d[d["condition"] == f"{prior_name}_real"]["score"].to_numpy(), ORANGE),
        "C1\n(degree-preserving\nrewire, 5 seeds)": (
            d[d["condition"].str.startswith(f"{prior_name}_c1_")]["score"].to_numpy(), AQUA,
        ),
        "C2\n(fully random\nrewire, 5 seeds)": (
            d[d["condition"].str.startswith(f"{prior_name}_c2_")]["score"].to_numpy(), YELLOW,
        ),
    }
    if signed:
        groups["C3\n(signs\nscrambled)"] = (d[d["condition"] == f"{prior_name}_c3_sign_flipped"]["score"].to_numpy(), MAGENTA)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(groups))
    means = [v[0].mean() for v in groups.values()]
    sems = [sem(v[0]) for v in groups.values()]
    colors = [v[1] for v in groups.values()]

    ax.bar(xs, means, yerr=sems, capsize=4, color=colors, width=0.6, zorder=3)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.text(len(groups) - 0.4, 0.505, "chance", color=MUTED, fontsize=9, va="bottom")

    for x, m, e in zip(xs, means, sems):
        ax.text(x, m + e + 0.015, f"{m:.3f}", ha="center", fontsize=9, color=INK)

    ax.set_xticks(xs)
    ax.set_xticklabels(groups.keys(), fontsize=8.5)
    ax.set_ylabel("Donor-level AUROC (SLE vs normal)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{prior_name}: real graph vs baseline and structural controls\n(error bars: SEM over 30 seed x fold splits)", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"headline_auroc_comparison_{prior_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main_results = pd.read_csv(RESULTS_DIR / "tables" / "main_results.csv")

    fig_headline(main_results, "tf_target", signed=True, display_name="DoRothEA")
    fig_headline(main_results, "epigenomic", signed=False, display_name="ABC model")
    print(f"Wrote figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
