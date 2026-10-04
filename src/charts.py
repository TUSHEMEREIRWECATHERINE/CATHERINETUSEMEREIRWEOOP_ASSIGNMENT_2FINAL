"""Shared chart styling and output locations for the notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA = PACKAGE_ROOT / "data"
FIGURES = PACKAGE_ROOT / "figures"

#: Series colours, chosen to stay distinguishable in greyscale print.
SERIES_COLOURS = ("#28506e", "#a8420f", "#3f7d20", "#6b3fa0", "#8a7014")

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.titlesize": 11,
    "font.size": 9,
    "legend.frameon": False,
    "lines.markersize": 4,
}


def use_house_style() -> None:
    """Apply the shared rcParams and make sure the output folders exist."""
    mpl.rcParams.update(STYLE)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(SERIES_COLOURS))
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)


def store(figure: plt.Figure, filename: str) -> Path:
    """Write a figure into ``figures/`` and report where it went."""
    FIGURES.mkdir(exist_ok=True)
    destination = FIGURES / filename
    figure.savefig(destination)
    return destination
