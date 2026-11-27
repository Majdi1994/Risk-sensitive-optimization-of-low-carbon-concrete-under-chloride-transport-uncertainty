#!/usr/bin/env python3
"""Shared matplotlib style for the journal figures.

Blue palette; Times New Roman; bold axis labels / legends / titles at 11–12 pt.
"""
from __future__ import annotations

from typing import Sequence

# Distinct blues (avoid default tab10 rainbow).
BLUE_PALETTE: Sequence[str] = (
    "#0B3D91",  # deep navy
    "#1F6AA5",  # medium blue
    "#4C9BD6",  # light blue
    "#7EB6D9",  # pale blue
    "#084C61",  # teal-navy
    "#2E86AB",  # steel
)


def apply_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
        "axes.prop_cycle": mpl.cycler(color=list(BLUE_PALETTE)),
        "axes.edgecolor": "#1A1A1A",
        "axes.linewidth": 0.8,
        "grid.color": "#D0D7DE",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_legend(ax) -> None:
    leg = ax.get_legend()
    if leg is None:
        return
    for text in leg.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontsize(11)
        text.set_fontweight("bold")
    title = leg.get_title()
    if title is not None:
        title.set_fontfamily("Times New Roman")
        title.set_fontsize(11)
        title.set_fontweight("bold")


def style_axes(ax) -> None:
    ax.title.set_fontfamily("Times New Roman")
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontfamily("Times New Roman")
    ax.xaxis.label.set_fontsize(12)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontfamily("Times New Roman")
    ax.yaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontsize(11)
    style_legend(ax)
