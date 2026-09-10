"""Render-level quality checks for figure builders.

These helpers measure what a figure actually renders rather than what its
specification declares. Plotly silently drops tick labels when their pitch
falls below roughly 1.2 label heights, and silently drops legend entries when
the plot area cannot seat them; neither loss is visible in the figure
specification. The rendered SVG carries one ``class="xtick"`` node per drawn
tick label on the base x axis (``x2tick``, ``x3tick``, … on subplot axes;
``ytick`` and friends likewise) and one ``class="legendtext"`` node per drawn
legend entry, so counting those nodes measures the render exactly.

``pixel_diff_count`` compares exported PNGs channel-wise on RGB. It exists
because the obvious alternative is a trap: ``ImageChops.difference`` on two
opaque RGBA images has a zero alpha channel everywhere, and ``getbbox()``
consults alpha, so that check reports any two same-size opaque images as
identical.
"""

import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

_LEGEND_TEXT_RE = re.compile(r'class="legendtext[^>]*>([^<]*)<')


def pixel_diff_count(a_path: Path | str, b_path: Path | str) -> int:
    """Number of pixels whose RGB values differ between two images.

    Raises ``ValueError`` on a size mismatch rather than returning a count,
    because a resized figure differing "everywhere" and a same-size figure
    differing everywhere are different findings.
    """
    a = np.asarray(Image.open(a_path).convert("RGB"))
    b = np.asarray(Image.open(b_path).convert("RGB"))
    if a.shape != b.shape:
        raise ValueError(
            f"image size mismatch: {a.shape[1]}x{a.shape[0]} vs "
            f"{b.shape[1]}x{b.shape[0]}"
        )
    return int((a != b).any(axis=-1).sum())


def rendered_svg(fig: go.Figure) -> str:
    """The figure as Kaleido renders it, in SVG form."""
    return fig.to_image(format="svg").decode()


def drawn_tick_count(fig: go.Figure, *, axis: str) -> int:
    """How many tick labels the render draws on the given axis kind.

    Counts across every subplot axis of that kind (``xtick`` plus ``x2tick``,
    ``x3tick``, …), which is the useful total for a faceted figure: a shared-x
    composite that should label only its bottom row draws exactly one leaf
    tick per configuration per bottom-row axis, and both a silent drop and a
    repeated upper-row tick band show up as a wrong total.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return len(re.findall(f'class="{axis}[0-9]*tick"', rendered_svg(fig)))


def drawn_legend_entries(fig: go.Figure) -> list[str]:
    """The legend entry labels the render actually draws, in drawn order."""
    return _LEGEND_TEXT_RE.findall(rendered_svg(fig))


def declared_legend_entries(fig: go.Figure) -> list[str]:
    """The legend entry labels the specification asks for, in trace order.

    A trace declares an entry when it carries a name and does not opt out
    with ``showlegend=False``. Compared against ``drawn_legend_entries``,
    this catches both silent drops (declared but not drawn) and the
    dropped-declaration class of defect, such as a baseline drawn with a
    bare ``add_hline`` that renders the line but never declares its
    legend entry.
    """
    return [
        trace.name
        for trace in fig.data
        if trace.name and trace.showlegend is not False
    ]
