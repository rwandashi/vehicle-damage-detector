"""Evaluate a fine-tuned YOLOv8 checkpoint and produce result figures.

Outputs (saved to ``reports/figures/`` by default):

* ``overall_metrics.png``  — Precision / Recall / mAP / F1 bar chart
* ``map_per_class.png``    — per-class mAP@0.5 ranked bar chart
* ``confusion_matrix.png`` — normalized confusion matrix

Also prints the headline numbers to stdout.

Usage:
    python -m src.evaluate --weights checkpoints/best.pt \\
                           --data datasets/car-damage/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.plot_style import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    BG_DARK,
    apply_dark_theme,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True,
                        help="Path to a trained YOLOv8 .pt checkpoint.")
    parser.add_argument("--data", required=True,
                        help="Path to the dataset YAML.")
    parser.add_argument("--out-dir", default="reports/figures")
    return parser.parse_args()


def plot_overall(results, out: Path) -> dict:
    p = float(results.box.mp)
    r = float(results.box.mr)
    m50 = float(results.box.map50)
    f1 = 2 * p * r / (p + r + 1e-9)
    overall = {"mAP@0.5": m50, "Precision": p, "Recall": r, "F1": f1}

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE]
    bars = ax.bar(overall.keys(), overall.values(), color=bar_colors,
                  edgecolor="none", width=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_title("Overall Model Performance", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Score", fontsize=12)
    for bar, val in zip(bars, overall.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    return overall


def plot_map_per_class(results, class_names: list[str], out: Path) -> None:
    map_scores = results.box.ap50
    sorted_idx = np.argsort(map_scores)[::-1]
    sorted_names = [class_names[i] for i in sorted_idx]
    sorted_scores = map_scores[sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.RdYlGn(sorted_scores)
    bars = ax.barh(sorted_names, sorted_scores, color=colors, edgecolor="none", height=0.7)
    ax.set_xlabel("mAP@0.5", fontsize=12, labelpad=10)
    ax.set_title("mAP@0.5 per Damage Class", fontsize=16, fontweight="bold", pad=20)
    ax.axvline(sorted_scores.mean(), color=ACCENT_BLUE, linestyle="--",
               linewidth=1.5, label=f"Mean: {sorted_scores.mean():.2f}")
    ax.legend(fontsize=11, facecolor="#161b22", labelcolor="white")
    ax.set_xlim(0, 1.05)
    for bar, score in zip(bars, sorted_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=9, color="white")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)


def plot_confusion_matrix(results, class_names: list[str], out: Path) -> None:
    cm = results.confusion_matrix.matrix
    full_names = list(class_names) + ["background"]
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(26, 22))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Normalized Count", color="black", fontsize=12)
    cbar.ax.tick_params(colors="black", labelsize=11)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    n = len(full_names)
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#30363d", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(full_names, rotation=45, ha="right", fontsize=10, color="black")
    ax.set_yticklabels(full_names, fontsize=10, color="black")
    ax.set_xlabel("Actual", fontsize=14, labelpad=15, color="black")
    ax.set_ylabel("Predicted", fontsize=14, labelpad=15, color="black")
    ax.set_title("Confusion Matrix (Normalized)", fontsize=18,
                 fontweight="bold", pad=25, color="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_dark_theme()

    print(f"Loading model from {args.weights}")
    model = YOLO(args.weights)
    print(f"Validating against {args.data}")
    results = model.val(data=args.data, verbose=False)

    overall = plot_overall(results, out_dir / "overall_metrics.png")
    class_names = list(model.names.values())
    plot_map_per_class(results, class_names, out_dir / "map_per_class.png")
    plot_confusion_matrix(results, class_names, out_dir / "confusion_matrix.png")

    print("\nResults:")
    for k, v in overall.items():
        print(f"  {k:<10} {v:.3f}")
    print(f"\nFigures written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())