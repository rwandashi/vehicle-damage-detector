"""Exploratory Data Analysis for the Car Damage Detection dataset.

Generates three figures and saves them to ``reports/figures/``:

* ``class_distribution.png`` — per-class instance counts in the training set
* ``bbox_analysis.png``      — bounding-box width/height heatmap and area histogram
* ``aspect_ratio.png``       — bounding-box aspect ratio distribution

Usage:
    python -m src.eda --data datasets/car-damage/data.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.plot_style import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    BG_DARK,
    BG_PANEL,
    apply_dark_theme,
    style_axis,
)


def _read_label_dir(label_dir: Path):
    """Return (class_counts, widths, heights, areas) parsed from YOLO labels."""
    class_counts: Counter[int] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []

    for label_file in label_dir.iterdir():
        if label_file.suffix != ".txt":
            continue
        with label_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                class_counts[class_id] += 1
                widths.append(w)
                heights.append(h)
                areas.append(w * h)

    return class_counts, widths, heights, areas


def plot_class_distribution(class_counts: Counter, class_names: list[str], out: Path) -> None:
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    sorted_idx = np.argsort(counts)[::-1]
    sorted_names = [class_names[i] for i in sorted_idx]
    sorted_counts = [counts[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_colors = [ACCENT_ORANGE if c < 100 else ACCENT_BLUE for c in sorted_counts]
    bars = ax.barh(sorted_names, sorted_counts, color=bar_colors, edgecolor="none", height=0.7)
    ax.set_xlabel("Number of Instances", fontsize=12, labelpad=10)
    ax.set_title("Class Distribution in Training Set", fontsize=16, fontweight="bold", pad=20)
    ax.axvline(
        np.mean(sorted_counts),
        color=ACCENT_GREEN,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(sorted_counts):.0f}",
    )

    legend_elements = [
        mpatches.Patch(facecolor=ACCENT_BLUE, label="Well represented"),
        mpatches.Patch(facecolor=ACCENT_ORANGE, label="Underrepresented (<100)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, facecolor=BG_PANEL, labelcolor="white")
    for bar, count in zip(bars, sorted_counts):
        ax.text(count + 10, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9, color="white")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)


def plot_bbox_analysis(widths, heights, areas, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)
    for ax in axes:
        style_axis(ax)

    axes[0].hist2d(widths, heights, bins=50, cmap="Blues")
    axes[0].set_xlabel("Box Width (normalized)", fontsize=11)
    axes[0].set_ylabel("Box Height (normalized)", fontsize=11)
    axes[0].set_title("Bounding Box Width vs Height", fontsize=13, fontweight="bold")

    axes[1].hist(areas, bins=50, color=ACCENT_BLUE, edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("Box Area (normalized)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Bounding Box Area Distribution", fontsize=13, fontweight="bold")
    axes[1].axvline(
        np.mean(areas),
        color=ACCENT_ORANGE,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(areas):.3f}",
    )
    axes[1].legend(fontsize=10, facecolor=BG_PANEL, labelcolor="white")

    fig.suptitle("Bounding Box Analysis", fontsize=15, fontweight="bold", color="white")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)


def plot_aspect_ratio(widths, heights, out: Path) -> None:
    aspect_ratios = [w / h for w, h in zip(widths, heights) if h > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_DARK)
    style_axis(ax)

    ax.hist(aspect_ratios, bins=60, color=ACCENT_PURPLE, edgecolor="none", alpha=0.85)
    ax.axvline(1.0, color=ACCENT_GREEN, linestyle="--", linewidth=1.5, label="Square (1:1)")
    ax.axvline(
        np.mean(aspect_ratios),
        color=ACCENT_ORANGE,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(aspect_ratios):.2f}",
    )
    ax.set_xlabel("Aspect Ratio (width / height)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Bounding Box Aspect Ratio Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, facecolor=BG_PANEL, labelcolor="white")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="datasets/car-damage/data.yaml",
        help="Path to the YOLO data.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/figures",
        help="Where to write the generated figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with data_yaml.open() as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]

    train_images = (data_yaml.parent / data_cfg["train"]).resolve()
    label_dir = train_images.parent / "labels"
    if not label_dir.is_dir():
        label_dir = train_images.with_name("labels")

    print(f"Reading labels from {label_dir}")
    apply_dark_theme()

    class_counts, widths, heights, areas = _read_label_dir(label_dir)

    plot_class_distribution(class_counts, class_names, out_dir / "class_distribution.png")
    plot_bbox_analysis(widths, heights, areas, out_dir / "bbox_analysis.png")
    plot_aspect_ratio(widths, heights, out_dir / "aspect_ratio.png")

    print(f"Wrote figures to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())