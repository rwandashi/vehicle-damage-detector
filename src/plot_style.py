"""Shared dark-theme matplotlib configuration.

Call ``apply_dark_theme()`` once at the start of any plotting script to get
the GitHub-style dark palette used throughout this project's figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# GitHub-dark inspired palette — kept as module-level constants so that other
# modules can reference exact colors when drawing custom artists.
BG_DARK = "#0d1117"          # figure background
BG_PANEL = "#161b22"         # axes background
EDGE = "#30363d"             # spines / gridlines
TEXT = "white"

ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE = "#f78166"
ACCENT_PURPLE = "#d2a8ff"


def apply_dark_theme() -> None:
    """Configure matplotlib rcParams for a dark, high-contrast theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG_DARK,
            "axes.facecolor": BG_PANEL,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.edgecolor": EDGE,
            "font.family": "DejaVu Sans",
        }
    )


def style_axis(ax) -> None:
    """Apply dark-theme styling to a single Axes object.

    Useful when an Axes was created before ``apply_dark_theme`` was called,
    or when you need to restyle a subplot grid produced by another library.
    """
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE)