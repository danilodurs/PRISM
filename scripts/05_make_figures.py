"""Summary figures for MEMO.md: headline AUROC comparison for each prior,
plus (`--prior-mode combined`) the combined condition against baseline,
joint structural controls, both single-prior reals, and the exploratory
only-one-real ablations. Matplotlib only (no interactive/web tooling needed
for a static memo figure).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.datasets import DATASETS, DatasetConfig

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


def _bar_group_fig(groups: dict, ylabel: str, title: str, out_path: Path, figsize=(7.5, 4.5), fontsize=8.5) -> None:
    fig, ax = plt.subplots(figsize=figsize)
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
    ax.set_xticklabels(groups.keys(), fontsize=fontsize)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_headline(
    main_results: pd.DataFrame, prior_name: str, signed: bool, display_name: str, disease_label: str, prefix: str = ""
) -> None:
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

    _bar_group_fig(
        groups,
        ylabel=f"Donor-level AUROC ({disease_label})",
        title=f"{prior_name}: real graph vs baseline and structural controls\n(error bars: SEM over 30 seed x fold splits)",
        out_path=FIG_DIR / f"{prefix}headline_auroc_comparison_{prior_name}.png",
    )


def fig_headline_combined(main_results: pd.DataFrame, disease_label: str, prefix: str) -> None:
    """Combined condition vs baseline, joint structural controls, and both
    RA single-prior reals (does combining beat the better of the two
    priors alone?), plus the exploratory only-one-real ablations.
    """
    d = main_results[main_results["target"] == "disease_status"]
    d = d[d["metric"] == "auroc"]

    def scores(cond_prefix_or_name: str, startswith: bool = False) -> np.ndarray:
        if startswith:
            return d[d["condition"].str.startswith(cond_prefix_or_name)]["score"].to_numpy()
        return d[d["condition"] == cond_prefix_or_name]["score"].to_numpy()

    groups = {
        "baseline_pca\n(no graph)": (scores("combined_baseline_pca"), BLUE),
        "tf_target_real\n(single prior)": (scores("tf_target_real"), AQUA),
        "epigenomic_real\n(single prior)": (scores("epigenomic_real"), YELLOW),
        "combined_real\n(joint mask)": (scores("combined_real"), ORANGE),
        "joint C1\n(degree-preserving,\n5 seeds)": (scores("combined_c1_degree_preserving_seed", startswith=True), MAGENTA),
        "joint C2\n(fully random,\n5 seeds)": (scores("combined_c2_fully_random_seed", startswith=True), "#8a7ec8"),
        "joint C3\n(TF signs\nscrambled)": (scores("combined_c3_sign_flipped"), "#c85f8a"),
        "only_tf_real\n(ablation, 5 seeds)": (scores("combined_only_tf_real_seed", startswith=True), "#5fa3c8"),
        "only_epi_real\n(ablation, 5 seeds)": (scores("combined_only_epi_real_seed", startswith=True), "#c8a35f"),
    }
    groups = {k: v for k, v in groups.items() if len(v[0]) > 0}

    _bar_group_fig(
        groups,
        ylabel=f"Donor-level AUROC ({disease_label})",
        title="combined: joint TF-target + epigenomic mask vs baseline, joint controls,\nsingle priors, and ablations (error bars: SEM over 30 seed x fold splits)",
        out_path=FIG_DIR / f"{prefix}headline_auroc_comparison_combined.png",
        figsize=(11, 5),
        fontsize=7.5,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ra")
    parser.add_argument("--prior-mode", choices=["independent", "combined"], default="combined")
    args = parser.parse_args()
    config: DatasetConfig = DATASETS[args.dataset]
    prefix = config.file_prefix
    disease_label = config.disease_display_label

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main_results = pd.read_csv(RESULTS_DIR / "tables" / f"{prefix}main_results.csv")

    fig_headline(main_results, "tf_target", signed=True, display_name="DoRothEA", disease_label=disease_label, prefix=prefix)
    fig_headline(main_results, "epigenomic", signed=False, display_name="ABC model", disease_label=disease_label, prefix=prefix)

    if args.prior_mode == "combined" and (main_results["condition"] == "combined_real").any():
        fig_headline_combined(main_results, disease_label=disease_label, prefix=prefix)

    print(f"Wrote figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
