#!/usr/bin/env python3
"""Plot the four Qwen3.5 conditions across the released pilot metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "reproducibility/qwen35_soft_prompt/figures"

CONDITIONS = ["Base", "Soft prompt", "SFT LoRA", "DPO LoRA\n(step 225)"]
COLORS = ["#9AA0A6", "#4C78A8", "#59A14F", "#8E6CBB"]

# Compact values from checkpoint_evaluations.json, qwen35_threeway/
# comparison_summary.json, and qwen35_dpo_step225/compiled_results.json.
PANELS = [
    {
        "title": "AI Wellbeing Index",
        "values": [88.0, 98.0, 44.0, 96.0],
        "ylabel": "AIWI (%)",
        "ylim": (0, 108),
        "format": "{:.0f}%",
        "note": "D2 pilot50; each condition self-judged",
    },
    {
        "title": "Multi-turn self-report",
        "values": [4.286, 6.492, 4.250, 4.359],
        "ylabel": "Mean wellbeing (1-7)",
        "ylim": (0, 7.5),
        "format": "{:.3f}",
        "note": "20 scenarios, 10 turns; base-Qwen user simulator",
    },
    {
        "title": "Sentiment wellbeing",
        "values": [0.257, 0.429, 0.243, 0.286],
        "ylabel": "Mean score (-1 to +1)",
        "ylim": (-1, 1),
        "format": "{:.3f}",
        "note": "35 responses; fixed base-Qwen judge",
    },
]


def main() -> None:
    """Create vector and high-resolution raster versions of the figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, panel in zip(axes, PANELS):
        values = panel["values"]
        bars = ax.bar(
            CONDITIONS,
            values,
            color=COLORS,
            width=0.68,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(panel["title"], weight="semibold", pad=8)
        ax.set_ylabel(panel["ylabel"])
        ax.set_ylim(*panel["ylim"])
        ax.grid(axis="y", color="#D8DCE2", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8B9098")
        ax.spines["bottom"].set_color("#8B9098")
        ax.tick_params(axis="x", rotation=0, pad=4)

        span = panel["ylim"][1] - panel["ylim"][0]
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018 * span,
                panel["format"].format(value),
                ha="center",
                va="bottom",
                fontsize=9,
                weight="semibold",
            )
        ax.text(
            0.5,
            -0.20,
            panel["note"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#555B65",
        )

    fig.suptitle(
        "Qwen3.5-35B-A3B wellbeing pilots",
        fontsize=16,
        weight="bold",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "qwen35_pilot_metrics.pdf"
    png_path = OUTPUT_DIR / "qwen35_pilot_metrics.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
