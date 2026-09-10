"""Plotly figure construction for results communication.

Each function accepts pre-shaped data (list[dict] from visualizations_data.py)
and returns a plotly.graph_objects.Figure. This separation keeps Plotly as a
presentation concern only — data transformation lives in visualizations_data.py.
"""

import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.experiment_analysis.config_identity import (
    CANONICAL_REASONING_ORDER,
    config_tie_break_key,
    display_config_key,
    display_model_slug,
    display_provider,
    display_reasoning_effort_level,
    display_reasoning_transition,
    is_authoring_model,
    mark_authoring,
    parse_config_key,
)
from utils.experiment_analysis.stimulus_metadata import STIMULUS_METADATA_REGISTRY
from utils.formatting import format_cost_per_trial


# ── Shared styling constants ───────────────────────────────────────────

CONDITION_COLORS: dict[str, str] = {
    "transparent": "#009E73",  # Okabe-Ito bluish green (sensitivity — transparent)
    "opaque": "#E69F00",       # Okabe-Ito orange (sensitivity — opaque)
    "correct": "#0072B2",      # Okabe-Ito blue (specificity)
}

PROVIDER_COLORS: dict[str, str] = {
    "openai": "#10b981",      # emerald-500
    "anthropic": "#8b5cf6",   # violet-500
    "gemini": "#3b82f6",      # blue-500
    "fireworks": "#f97316",   # orange-500
}

# Per-model-group colors for line charts where sibling models from the same
# provider must be visually distinct.  Uses a sibling-shade approach: models
# from one provider share a hue family but differ in lightness so provider
# grouping remains apparent at a glance.
MODEL_GROUP_COLORS: dict[str, str] = {
    # Anthropic family (violet hues — darker = larger model)
    "anthropic--claude-opus-4-6": "#7c3aed",     # violet-600
    "anthropic--claude-haiku-4-5": "#a78bfa",     # violet-400
    # Gemini family (blue hues)
    "gemini--gemini-3-pro-preview": "#2563eb",    # blue-600
    "gemini--gemini-3-flash-preview": "#60a5fa",  # blue-400
    # OpenAI family (emerald hues)
    "openai--gpt-5.2": "#059669",                 # emerald-600
    "openai--gpt-5-mini": "#34d399",              # emerald-400
}

FAILURE_MODE_COLORS: dict[str, str] = {
    # Incorrect-draft (transparent/opaque): discriminated by articulation
    "capability_absent": "#6366f1",  # indigo-500
    "selection_failed": "#ec4899",   # pink-500
    # Correct-draft: discriminated by governing
    "field_label": "#a78bfa",        # violet-400
    "governed": "#f472b6",           # pink-400
}

CELL_STATUS_COLORS: dict[str, str] = {
    "rejected": "#22c55e",  # green-500
    "futile": "#94a3b8",    # slate-400
    "active": "#f59e0b",    # amber-500
}

# Display titles for per-example condition accuracy heatmaps, mapping
# value_column names to unabbreviated, publication-ready titles.
_CONDITION_ACC_TITLES: dict[str, str] = {
    "correct_acc": "Correct-Draft Condition Accuracy (Specificity)",
    "transparent_acc": "Transparent Condition Accuracy (Sensitivity)",
    "opaque_acc": "Opaque Condition Accuracy (Sensitivity)",
}

FIGURE_TEMPLATE = "plotly_white"

# Canonical figure width for every chart in the results deck. Picked to
# match the densest faceted chart (3 panels × ~26 configs) so every figure
# has enough horizontal room for -45° rotated tick labels and outside
# trace-text annotations without clipping at facet boundaries. Sharing
# one width across the deck also guarantees that vertically stacked
# paired panels align cleanly instead of producing ragged seams.
FIGURE_WIDTH: int = 1400

# Width for the three charts that force a label at every configuration tick
# (Figure 2 and Supplementary Figures S5 and S9b): 21 rotated labels per
# panel collide at FIGURE_WIDTH, and 2400 lifts adjacent labels' separation
# comfortably above the glyph height on the three-panel charts while every
# export stays under the 34-megapixel guard in
# tests/scripts/test_committed_png_dimensions.py (the widest lands at 30.8).
EVERY_TICK_FIGURE_WIDTH: int = 2400

# Width the font sizes below are calibrated against, in typographic points:
# the paper's true printed measure. ``build_paper``'s geometry pins
# ``\textwidth`` at 430 pt, and every portrait figure prints as a full-width
# ``width=\textwidth`` float at exactly that measure, so calibrating against
# it is what makes a derived size print at the size the model claims.
# (The deck was once calibrated against IEEEtran's native 516 pt block and
# printed about 17% under target; the recalibration regenerated every
# export.) Working in points keeps this in the same unit as the legibility
# target, so the two combine without a conversion factor.
FONT_CALIBRATION_WIDTH_POINTS: float = 430.0

# The landscape counterpart: a figure on a rotated pdflscape page prints
# LANDSCAPE_MEASURE_POINTS wide (the portrait \textheight geometry pins at
# 696 pt), and the wide-canvas composites registered for landscape pages
# derive their font from this measure instead.
LANDSCAPE_MEASURE_POINTS: float = 696.0

# Calibration target for printed figure text, in typographic points, at
# IEEEtran's \footnotesize; the paper's captions set one point larger, at
# \small (9 pt). Under the 430 pt printed measure the derived sizes land
# below this target (see FONT_CALIBRATION_WIDTH_POINTS above).
MINIMUM_LEGIBLE_POINT_SIZE: float = 8.0

# Every text size in the deck derives from this one number, so the typographic
# hierarchy cannot drift as individual charts are tuned. It is the smallest
# whole pixel size that prints at the target under the calibration model: a
# FIGURE_WIDTH-px canvas printed FONT_CALIBRATION_WIDTH_POINTS wide renders
# each logical pixel as ``FONT_CALIBRATION_WIDTH_POINTS / FIGURE_WIDTH``
# points. Export scale never enters — it multiplies pixel count and physical
# size alike, leaving printed type the same size.
BASE_FONT_SIZE: int = math.ceil(
    MINIMUM_LEGIBLE_POINT_SIZE * FIGURE_WIDTH
    / FONT_CALIBRATION_WIDTH_POINTS
)

# One 4:3 step above the base, for text that must read as a heading or a mark
# rather than as data: figure titles and the significance asterisks. Nothing
# steps *below* the base, because the base is the legibility floor — hierarchy
# in this deck is expressed by enlarging, never by shrinking.
EMPHASIS_FONT_SIZE: int = round(BASE_FONT_SIZE * 4 / 3)

# The landscape-page counterparts, for the EVERY_TICK_FIGURE_WIDTH canvases
# whose composites set on rotated pages: the same 8 pt target priced at the
# 696 pt landscape measure over the 2400 px canvas.
LANDSCAPE_BASE_FONT_SIZE: int = math.ceil(
    MINIMUM_LEGIBLE_POINT_SIZE * EVERY_TICK_FIGURE_WIDTH
    / LANDSCAPE_MEASURE_POINTS
)
LANDSCAPE_EMPHASIS_FONT_SIZE: int = round(LANDSCAPE_BASE_FONT_SIZE * 4 / 3)

# Canonical height for the two base-example heatmaps, fixed by the y tick
# labels. Plotly drops every other category label once an axis falls under
# roughly 1.2 label heights per category, and at BASE_FONT_SIZE the ten base
# examples go from ten labels to five between a 260px and a 240px plot area.
# These builders set ``CROWDED_CONFIG_BOTTOM_MARGIN`` against Plotly's default
# 100px top margin, so 680px leaves 340px of plot area: all ten labels, at a
# 34px row pitch rather than the 26px the drop boundary would give them. Both
# margins come out of the same canvas, so the depth the crowded tick angle
# claims below the plot is what this height has to make up above it. Label
# height is a property of ``EXPORT_FONT_FAMILY`` at ``BASE_FONT_SIZE``, so a
# change to either must re-derive this bound.
#
# The y-axis title imposes a second demand, but not a binding one. It is
# rotated 90° and centered on the plot area, so it fits only while
# ``height >= title_length + margin.bottom - margin.top``; at 150px of ink
# ``_BASE_EXAMPLE_AXIS_TITLE`` needs 290px, which any height clearing the
# label bound already provides.
HEATMAP_FIGURE_HEIGHT: int = 680


def _display_forced_order(
    levels_present: set[str],
) -> list[str]:
    """Canonical cross-model level ordering filtered to the levels present.

    Every provider's levels keep their own names ("off" and "none" are
    distinct provider settings and distinct categories), so the order is
    ``CANONICAL_REASONING_ORDER`` with absent levels dropped.
    """
    return [lvl for lvl in CANONICAL_REASONING_ORDER if lvl in levels_present]


# Typeface for every statically exported chart and table. Latin Modern's own
# sans, kin to the Computer Modern prose of the built paper and legible at the
# small sizes these figures print at. Named alone rather than at the head of a
# fallback chain: Kaleido substitutes a missing family silently, so a chain
# would let the exported typeface follow whatever fonts the rendering machine
# happens to have. A single name makes an absent font a condition the guard in
# ``tests/unit/experiment_analysis/test_font_resolution.py`` can detect.
EXPORT_FONT_FAMILY: str = "LMSans10"

# Consistent layout defaults for publication-quality figures. Tick labels, axis
# titles, legend entries, and colorbar labels carry no size of their own, so
# ``font_size`` is what they all inherit.
_BASE_LAYOUT = dict(
    template=FIGURE_TEMPLATE,
    font_family=EXPORT_FONT_FAMILY,
    font_size=BASE_FONT_SIZE,
)


def _figure_title(text: str) -> dict[str, Any]:
    """Title object carrying its own size, for ``update_layout(title=...)``.

    The size has to travel inside the title object rather than as a sibling
    ``title_font_size``: Plotly applies magic-underscore keywords in argument
    order, so a plain ``title="..."`` string replaces the whole title object
    and silently discards a size set beside it.
    """
    return dict(text=text, font=dict(size=EMPHASIS_FONT_SIZE))


def _faceted_figure(*, cols: int, panel_titles: list[str]) -> go.Figure:
    """One-row faceted figure with shared y-axes and legible panel titles.

    ``make_subplots`` writes panel titles as annotations at a hardcoded 16 px,
    ignoring the layout font entirely, so the size is restated here. Panel
    titles sit at the base size rather than the emphasis size because their
    width budget is a single panel, not the whole canvas.
    """
    fig = make_subplots(
        rows=1,
        cols=cols,
        subplot_titles=panel_titles,
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    return fig


# The disclosures that name an ordered axis's rule. A reader who cannot
# tell why one bar precedes another cannot read an ordered chart, so every
# ordered axis in the deck names its own rule — as an axis title on the
# categorical axes, and as the structural annotation on the two-tier
# numeric axes (the configuration and transition figures). Each is written
# once because several builders share it, and a divergence between two
# charts claiming one order would be invisible in review. Every
# configuration axis shares the structural order (provider, model,
# ascending reasoning effort), so one disclosure serves the whole deck.
_CONFIG_GROUPED_BY_MODEL_TITLE = (
    "Configuration (grouped by model, ordered by reasoning effort level)"
)
_REASONING_EFFORT_LEVEL_TITLE = (
    "Reasoning Effort Level (ordered within each model)"
)
_REASONING_EFFORT_TRANSITION_TITLE = (
    "Reasoning Effort Level Transition (grouped by model, "
    "ordered by reasoning effort level)"
)


# The heatmaps' shared y-axis title. The tick labels are base example ids, so
# the axis needs only to name what they are; a longer disclosure would repeat
# what the reader can already see.
_BASE_EXAMPLE_AXIS_TITLE = "Base Example"


# A y-axis title is rotated 90° and centered on the plot area, so the canvas
# *height* bounds its length rather than its width. At BASE_FONT_SIZE these two
# overrun their charts' bound on one line, so each is broken where its clause
# divides and the longest line becomes the binding one. The deck's shorter axis
# titles need no break.
_WRONG_VERDICT_TRIALS_AXIS_TITLE = "Number of<br>Wrong-Verdict Trials"
_MARGINAL_COST_AXIS_TITLE = "Cost per<br>Percentage Point<br>(USD)"

# The two ablation-effect charts name the quantity their bars measure rather
# than spelling out the subtraction, which their own titles already give in
# full. Naming the metric is also what keeps these titles short enough to clear
# the figure title: spelled out, the longer one runs past the top of its chart's
# plot area and strikes the title above it.
_ABLATION_BALANCED_ACCURACY_DELTA_AXIS_TITLE = "Δ Balanced Accuracy"
_ABLATION_CONDITION_ACCURACY_DELTA_AXIS_TITLE = "Δ Per-Condition<br>Accuracy"

# Rotation for a configuration tick label, with the bottom margin that rotation
# needs. Neighboring labels clear each other only while ``pitch × sin(angle)``
# reaches a label's ink height, so the angle is the lever when configurations
# crowd an axis: at BASE_FONT_SIZE a 23 px-tall label wants a 33 px pitch at 45°
# but only 27 px at 60°. What the steeper angle costs is depth, since a label
# reaches ``width × sin(angle)`` below the axis, and the margin has to hold it.
CONFIG_TICK_ANGLE: int = -45
CONFIG_TICK_BOTTOM_MARGIN: int = 180

# The steeper pair, for the axes whose 21 configurations share a faceted panel
# or a heatmap column strip rather than the full canvas width. Their pitch falls
# to about 30 px, under the 33 px that 45° needs, and no widening is available:
# the labels' own reading angle is the only room left.
CROWDED_CONFIG_TICK_ANGLE: int = -60
CROWDED_CONFIG_BOTTOM_MARGIN: int = 240

# Canvas height for the faceted per-condition accuracy chart, the one chart in
# the deck whose legend competes with its tick labels for the same canvas.
# Plotly bounds a legend by the plot area, and this chart's eight rows (two
# group titles over six entries) need ~265px of it at BASE_FONT_SIZE; below that
# the chance entry silently disappears and the row above it loses its descenders.
# The plot area is not the declared geometry: automargin grows the bottom margin
# past CROWDED_CONFIG_BOTTOM_MARGIN until the rotated tick labels and axis title
# fit, and the deeper of the two experiments' realized margins (the primary
# panel, whose printed labels are the longest) is what the legend must clear. At
# 620px that panel left the legend ~254px and clipped "Chance"; 680px restores
# the margin the eight rows need in both panels.
CONDITION_ACCURACY_FIGURE_HEIGHT: int = 680

# Plotly's own default top margin, which none of these builders overrides. It
# is named here because the plot area a chart ends up with, and therefore the
# room its axis has to reserve, is the declared height less this and the bottom
# margin.
_PLOTLY_DEFAULT_TOP_MARGIN: int = 100

# Depth an x-axis title claims below a rotated tick block. The renderer's
# automargin grows the bottom margin to seat the title, so a chart that gains
# one loses that much plot area unless its height grows to match. Measure this
# from an exported PNG, not from ``full_figure_for_development``: that runs no
# text-measurement pass and reports the declared margin unchanged, which reads
# as though the title were free. In the export the y=0 axis line moves up by
# exactly this much and returns when the height carries it. Measured at
# BASE_FONT_SIZE across three of the builders below, each of which returns to
# its baseline plot height at exactly this value. Added to ``height`` only by
# the charts that newly gain a title; it deliberately does not touch the shared
# margin constants, which feed ``_rate_axis_max`` and six other charts, so the
# pre-allowance geometry those expressions assume stays exact.
_AXIS_TITLE_ALLOWANCE: int = 33

# Canvas heights for the four failure mode charts: one for the single-panel
# correct-draft pair, and a taller one for the two-panel incorrect-draft pair,
# whose panel titles take a band the single-panel charts do not need.
_FAILURE_MODE_SINGLE_PANEL_HEIGHT: int = 500
_FAILURE_MODE_FACETED_HEIGHT: int = 550

# Canvas height for the token composition chart. A y-axis title is rotated onto
# the plot area's height rather than the canvas width, and what binds is the
# title's length against that height, not against the canvas: this chart spends
# 150px of a 500px canvas on rotated configuration labels, so its plot area is
# among the deck's shortest and its title overran it at both ends. The reasoning
# token scaling chart carries a longer title on the same canvas without trouble,
# because its horizontal tick labels leave the plot area tall. 600 is the
# smallest round canvas whose plot area seats this one clear of the figure title.
_TOKEN_COMPOSITION_HEIGHT: int = 600

# Room an upright mismatch-count annotation needs above a full-rate bar: its
# own width plus Plotly's standoff for outside text. Measured rather than
# derived, because the standoff is the renderer's constant and not this
# module's: at the 22 px font of the measurement era, 78 px lost the widest
# count's last digit to the axis boundary and 87 px rendered it whole. The
# ink is text, so the reserve scales linearly with the font in use; scaling
# the whole measured value also grows the constant standoff's share, which
# errs a few pixels toward more headroom.
_RATE_ANNOTATION_HEADROOM_MEASURED_PX: int = 90
_RATE_ANNOTATION_MEASURED_FONT_PX: int = 22


def _rate_annotation_headroom(font_px: int) -> int:
    """Pixels of headroom the outside count labels need at ``font_px``."""
    return math.ceil(
        _RATE_ANNOTATION_HEADROOM_MEASURED_PX
        * font_px
        / _RATE_ANNOTATION_MEASURED_FONT_PX
    )

# Tick positions for a failure mode rate axis. The axis top sits above 1.0
# (see ``_rate_axis_max``) purely to hold the outside count annotations, and
# Plotly's auto ticker would print a 1.5 label in that headroom. Rates live
# in [0, 1], so the ticks are pinned to the data domain: every printed label
# is a reachable rate while the range keeps the annotation room.
_RATE_AXIS_TICKVALS: list[float] = [0.0, 0.5, 1.0]


def _rate_axis_max(plot_height: int, *, font_px: int) -> float:
    """Top of a failure mode rate axis that clears its count annotations.

    A full-rate bar reaches ``plot_height / axis_max`` pixels, so reserving
    ``_rate_annotation_headroom(font_px)`` above it is what fixes the axis
    top. Reading the plot height per chart keeps each one's bars as tall as
    its own canvas allows, instead of pinning both to whichever chart is
    shorter. The space above 1.0 is annotation-only: tick labels stay pinned
    to ``_RATE_AXIS_TICKVALS``, so no tick prints beyond the [0, 1] rate
    domain.
    """
    return plot_height / (plot_height - _rate_annotation_headroom(font_px))


def _model_group_color(model_group: str, provider: str) -> str:
    """Resolve a distinct color for a model group.

    Looks up ``MODEL_GROUP_COLORS`` first, then falls back to the
    provider-level color for unknown model groups.
    """
    if model_group in MODEL_GROUP_COLORS:
        return MODEL_GROUP_COLORS[model_group]
    return PROVIDER_COLORS.get(provider, "#6b7280")


def _add_chance_baseline(
    fig: go.Figure,
    *,
    legendgroup: str | None,
    legendgrouptitle_text: str | None,
    y: float = 0.5,
) -> None:
    """Add a horizontal dashed chance baseline with a legend entry.

    Combines ``add_hline`` (full-width visual line) with an invisible
    ``go.Scatter`` trace so "Chance" appears in the legend rather than
    as a text annotation pinned to the line.

    ``legendgroup`` / ``legendgrouptitle_text`` place the Chance entry inside a
    named legend group (pass ``None`` for both to leave it ungrouped). Both are
    required so every caller states its grouping explicitly.
    """
    fig.add_hline(y=y, line_dash="dash", line_color="gray", line_width=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="gray", width=1, dash="dash"),
        showlegend=True,
        name="Chance",
        legendgroup=legendgroup,
        legendgrouptitle_text=legendgrouptitle_text,
        hoverinfo="skip",
    ))


def short_config_label(config_key: str) -> str:
    """Publication-ready configuration label, carrying the authoring dagger.

    ``paper/draft.md`` states that the authoring model "is therefore flagged
    with a dagger", and a figure legend is as much a place the reader meets a
    model name as a table cell is. Marking here rather than at each caller is
    what keeps a legend entry and a table row naming the same configuration
    identical, and it is why a caller must not append a dagger of its own.
    The dagger arrives inside ``display_config_key``, riding the model
    segment of the label.
    """
    return display_config_key(config_key)


def _compute_pareto_frontier(
    points: list[tuple[float, float]],
) -> list[int]:
    """Return indices of Pareto-optimal points (minimize x, maximize y).

    A point is Pareto-optimal when no other point has both lower cost (x)
    and higher score (y). Sorted by ascending x before the sweep so that
    only forward-looking domination is checked.
    """
    if not points:
        return []

    # Sort by ascending x; break ties by descending y
    indexed = sorted(enumerate(points), key=lambda t: (t[1][0], -t[1][1]))

    frontier_indices: list[int] = []
    max_y = float("-inf")
    for original_idx, (_, y) in indexed:
        if y > max_y:
            frontier_indices.append(original_idx)
            max_y = y

    return frontier_indices


# Candidate label placements for the cost scatter, expressed in label boxes
# rather than in fixed pixels so they track the type size instead of a size some
# earlier edit happened to leave behind. A configuration label is roughly eight
# times wider than it is tall, so lifting a label a whole row clear of its
# neighbor costs a fraction of what sliding it clear sideways costs: the row
# ladder is the primary axis, and the sidesteps only break ties where the dense
# upper-right cluster leaves no free row directly above or below.
_ANNOTATION_ROW_STEPS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
_ANNOTATION_COLUMN_STEPS: tuple[float, ...] = (
    0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05,
)

# Vertical pitch between stacked labels, as a multiple of the label box, and the
# per-character width used to size that box. The width is a deliberate
# overestimate of ``EXPORT_FONT_FAMILY``'s average advance, which biases the
# search toward spreading labels out rather than toward packing them.
_ANNOTATION_ROW_PITCH_RATIO: float = 1.15
_ANNOTATION_CHAR_WIDTH_RATIO: float = 0.55
_ANNOTATION_LABEL_HEIGHT_RATIO: float = 1.4

# Passes of re-siting after the initial placement. The arrangement stops moving
# well before this many rounds on the deck's data; the extra rounds cost a few
# thousand box comparisons and guarantee the sweep is not cut off mid-descent.
_ANNOTATION_REFINEMENT_ROUNDS: int = 6


def _place_annotations(
    points: list[tuple[float, float]],
    labels: list[str],
    *,
    chart_width: int = 1000,
    chart_height: int = 420,
    y_max: float = 1.05,
    font_size: float = BASE_FONT_SIZE,
    marker_radius: float = 8.0,
) -> list[tuple[int, int]]:
    """Compute (ax, ay) pixel offsets so annotation labels avoid overlap.

    A greedy pass sites each label at whichever candidate placement lands on
    the plot and collides least with what is already down, then refinement
    sweeps lift each label in turn and re-site it against the finished
    arrangement. Coordinates are projected into an approximate pixel space
    (log-scaled x, linear y) for the collision measure; the defaults describe
    the plot area ``build_cost_vs_balanced_accuracy_scatter`` declares, so the
    model's geometry matches the canvas the labels actually land on.
    """
    if not points:
        return []

    # Approximate pixel position from data coordinates (log x, linear y)
    log_xs = [math.log10(x) for x, _ in points]
    log_min, log_max = min(log_xs), max(log_xs)
    log_span = log_max - log_min if log_max > log_min else 1.0

    def _to_pixel(log_x: float, y: float) -> tuple[float, float]:
        px = (log_x - log_min) / log_span * chart_width
        py = (1 - y / y_max) * chart_height
        return (px, py)

    # Approximate label dimensions in pixels
    char_width = font_size * _ANNOTATION_CHAR_WIDTH_RATIO
    label_height = font_size * _ANNOTATION_LABEL_HEIGHT_RATIO
    row_pitch = label_height * _ANNOTATION_ROW_PITCH_RATIO

    anchors = [_to_pixel(log_x, y) for log_x, (_, y) in zip(log_xs, points)]
    half_sizes = [
        (len(label) * char_width / 2, label_height / 2) for label in labels
    ]

    # The markers are obstacles too, so a label is steered off the data as well
    # as off its neighbors: a configuration name laid across a point hides the
    # very value it names.
    marker_boxes = [
        (px, py, marker_radius, marker_radius) for px, py in anchors
    ]

    def _candidates(index: int) -> list[tuple[int, int]]:
        """Offsets to try for one label, in label-box units.

        Sidesteps scale with this label's own width, so a long label steps far
        enough to actually clear the column it is leaving.
        """
        half_w = half_sizes[index][0]
        return [
            (round(column * half_w * 2), round(direction * row * row_pitch))
            for row in _ANNOTATION_ROW_STEPS
            for direction in (-1, 1)
            for column in _ANNOTATION_COLUMN_STEPS
        ]

    def _clearance(
        box: tuple[float, float, float, float],
        obstacles: list[tuple[float, float, float, float]],
    ) -> tuple[float, float]:
        """How badly a box collides, and how much room it has when it does not.

        Returns ``(total_penetration, smallest_gap)``. A pair's gap is negative
        only while the boxes overlap on both axes, so summing the negatives
        measures the collision this box is party to; because that measure is
        symmetric, driving one label's total down drives the whole arrangement's
        down by the same amount, which is what makes the refinement sweep
        converge instead of cycling.
        """
        cx, cy, hw, hh = box
        penetration = 0.0
        smallest = float("inf")
        for pcx, pcy, phw, phh in obstacles:
            dx = abs(cx - pcx) - (hw + phw)
            dy = abs(cy - pcy) - (hh + phh)
            # Positive = separated on that axis; overlap requires BOTH negative
            gap = max(dx, dy)
            penetration += max(0.0, -gap)
            smallest = min(smallest, gap)
        return (penetration, smallest)

    def _fits_inside(box: tuple[float, float, float, float]) -> bool:
        """Whether a box lies wholly within the plot area.

        A label placed outside is worse than one that merely touches a
        neighbor: Plotly widens the axes to reach it, which rescales the data,
        and whatever still falls past the canvas is cut off outright.
        """
        cx, cy, hw, hh = box
        return (
            cx - hw >= 0 and cx + hw <= chart_width
            and cy - hh >= 0 and cy + hh <= chart_height
        )

    def _best_offset(
        index: int, obstacles: list[tuple[float, float, float, float]],
    ) -> tuple[int, int]:
        """Placement for one label: on the plot, then least collided, then roomiest."""
        px, py = anchors[index]
        half_w, half_h = half_sizes[index]

        def _rank(offset: tuple[int, int]) -> tuple[bool, float, float]:
            box = (px + offset[0], py + offset[1], half_w, half_h)
            penetration, smallest = _clearance(box, obstacles)
            return (_fits_inside(box), -penetration, smallest)

        return max(_candidates(index), key=_rank)

    def _box(index: int, offset: tuple[int, int]) -> tuple[float, float, float, float]:
        """The label box an offset produces for one point."""
        return (
            anchors[index][0] + offset[0],
            anchors[index][1] + offset[1],
            *half_sizes[index],
        )

    # First pass places each label against what is already down, which leaves
    # the early labels sited as though the plot were empty. The refinement
    # rounds lift each label in turn and re-site it against the finished
    # arrangement, which is what resolves the clusters a single pass strands.
    offsets: list[tuple[int, int]] = []
    for index in range(len(points)):
        placed = marker_boxes + [_box(i, off) for i, off in enumerate(offsets)]
        offsets.append(_best_offset(index, placed))

    for _ in range(_ANNOTATION_REFINEMENT_ROUNDS):
        for index in range(len(points)):
            others = marker_boxes + [
                _box(i, off) for i, off in enumerate(offsets) if i != index
            ]
            offsets[index] = _best_offset(index, others)

    return offsets


# ── Figure builders ────────────────────────────────────────────────────


# Shared mark specifications for the condition-accuracy family. The
# per-experiment builder and the paired composite must draw identical marks —
# same values, colors, symbols, and hover text — so the marks live in one
# place and each context supplies only its legend wiring. A composite that
# invented its own marks is exactly the divergence class review renders
# shipped once (wrong palette, dropped Chance entry).
_CONDITION_PANELS: tuple[tuple[str, str], ...] = (
    ("Correct-Draft (Specificity)", "correct"),
    ("Transparent (Sensitivity)", "transparent"),
    ("Opaque (Sensitivity)", "opaque"),
)

# Legend names for the condition bars. Correct-draft is spelled to match
# its panel title rather than the bare slug capitalization ("Correct").
_CONDITION_LEGEND_NAMES: dict[str, str] = {
    "correct": "Correct-Draft",
    "transparent": "Transparent",
    "opaque": "Opaque",
}

_OVERLAY_MARKERS: dict[str, dict[str, Any]] = {
    "Overall Accuracy": dict(symbol="diamond", size=10, color="#1e293b"),
    "Balanced Accuracy": dict(symbol="circle", size=8, color="#64748b"),
}


def _condition_accuracy_bar(
    condition: str,
    x: list[Any],
    values: list[float],
    config_labels: list[str],
    **legend: Any,
) -> go.Bar:
    """The condition bar trace: values, color, and hover in one place.

    ``config_labels`` ride ``customdata`` so the hover names the
    configuration even when ``x`` holds numeric grouped positions.
    """
    return go.Bar(
        name=_CONDITION_LEGEND_NAMES[condition],
        x=x,
        y=values,
        customdata=config_labels,
        marker_color=CONDITION_COLORS[condition],
        hovertemplate=(
            "Config: %{customdata}<br>"
            f"Condition: {condition}<br>"
            "Condition Accuracy: %{y:.3f}<extra></extra>"
        ),
        **legend,
    )


def _condition_overlay_marker(
    name: str,
    x: list[Any],
    values: list[float],
    config_labels: list[str],
    **legend: Any,
) -> go.Scatter:
    """An Overall or Balanced Accuracy overlay: symbol, size, ink shared."""
    return go.Scatter(
        name=name,
        x=x,
        y=values,
        mode="markers",
        customdata=config_labels,
        marker=dict(**_OVERLAY_MARKERS[name]),
        hovertemplate=(
            "Config: %{customdata}<br>"
            f"{name}: %{{y:.3f}}<extra></extra>"
        ),
        **legend,
    )


def build_condition_accuracy_chart(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Faceted 1x3 bar chart: one panel per condition, ordered by balanced accuracy.

    Three panels (Correct-Draft, Transparent, Opaque) with shared y-axis.
    Each panel shows one bar per configuration, colored by condition.
    Overall accuracy (diamond) and balanced accuracy (circle) overlay markers
    convey composite metrics alongside per-condition bars. Chance baseline
    at y=0.5 appears in the legend.
    """
    # Extract sorted unique config keys (data is pre-sorted by balanced_accuracy)
    seen: set[str] = set()
    config_keys: list[str] = []
    for row in data:
        if row["config_key"] not in seen:
            seen.add(row["config_key"])
            config_keys.append(row["config_key"])

    labels = [short_config_label(ck) for ck in config_keys]

    # Build condition → accuracy lookup
    acc_lookup: dict[str, dict[str, float]] = {}
    for row in data:
        acc_lookup.setdefault(row["condition"], {})[row["config_key"]] = row["accuracy"]

    # Build composite metric lookups (any row per config works)
    score_by_config = {r["config_key"]: r["overall_accuracy"] for r in data}
    bal_by_config = {r["config_key"]: r["balanced_accuracy"] for r in data}
    overall_values = [score_by_config.get(ck, 0.0) for ck in config_keys]
    bal_values = [bal_by_config.get(ck, 0.0) for ck in config_keys]

    fig = _faceted_figure(
        cols=3, panel_titles=[p[0] for p in _CONDITION_PANELS],
    )

    # Track legend entries already shown to avoid duplicates
    _legend_shown: set[str] = set()

    for col_idx, (_, condition) in enumerate(_CONDITION_PANELS, start=1):
        cond_acc = acc_lookup.get(condition, {})
        values = [cond_acc.get(ck, 0.0) for ck in config_keys]

        fig.add_trace(
            _condition_accuracy_bar(
                condition, labels, values, config_labels=labels,
                showlegend=condition not in _legend_shown,
                legendgroup="conditions",
                legendgrouptitle_text="Condition",
            ),
            row=1,
            col=col_idx,
        )
        _legend_shown.add(condition)

        # Composite metric overlays on each panel
        for overlay_name, overlay_values in (
            ("Overall Accuracy", overall_values),
            ("Balanced Accuracy", bal_values),
        ):
            fig.add_trace(
                _condition_overlay_marker(
                    overlay_name, labels, overlay_values,
                    config_labels=labels,
                    showlegend=overlay_name not in _legend_shown,
                    legendgroup="overlays",
                    legendgrouptitle_text="Overlays",
                ),
                row=1,
                col=col_idx,
            )
            _legend_shown.add(overlay_name)

    # Chance baseline on all subplots, grouped with the composite-metric overlays
    _add_chance_baseline(
        fig, legendgroup="overlays", legendgrouptitle_text="Overlays",
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Per-Condition Accuracy by Configuration [{experiment_label}]"),
        # Two per-group titles ("Condition" over the bars, "Overlays" over the
        # composite metrics + chance) replace the single layout-level legend
        # title that mislabeled the overlays as conditions. groupclick keeps
        # each legend entry independently toggleable.
        legend=dict(groupclick="toggleitem"),
        height=CONDITION_ACCURACY_FIGURE_HEIGHT + _AXIS_TITLE_ALLOWANCE,
        width=EVERY_TICK_FIGURE_WIDTH,
        margin=dict(b=CROWDED_CONFIG_BOTTOM_MARGIN),
    )
    fig.update_xaxes(tickangle=CROWDED_CONFIG_TICK_ANGLE, dtick=1)
    # Centered under the three facets, matching the ablation effect chart.
    fig.update_xaxes(
        title_text=_CONFIG_GROUPED_BY_MODEL_TITLE, row=1, col=2,
    )
    fig.update_yaxes(title_text="Condition Accuracy", range=[0, 1.05], row=1, col=1)

    return fig


# Flat legend ranks for the paired composites: bars in condition order, then
# the overlays, then the chance baseline. A flat horizontal legend renders in
# trace-addition order unless ranked, and a composite adds its traces
# column-by-column, which would interleave conditions with overlays.
_PAIRED_LEGEND_RANKS: dict[str, int] = {
    "Correct-Draft": 1,
    "Transparent": 2,
    "Opaque": 3,
    "Overall Accuracy": 4,
    "Balanced Accuracy": 5,
    "Chance": 6,
}

# Panel labels naming the experiments in a paired composite: columns on the
# landscape grids, rows on the portrait stacks.
_EXPERIMENT_PANEL_TITLES: tuple[str, str] = (
    "Primary Experiment", "Ablation Experiment",
)

# Composite chrome geometry, in canvas pixels. Plotly wants these as paper
# fractions, which scale with each figure's plot height, so the depths are
# fixed here in pixels and each composite converts against its own plot
# height (canvas height less the top and bottom margins): the legend
# baseline rises above the plot top, the model band labels drop below the
# leaf-tick band, and the structural axis title drops below the bands.
# Keeping the pixel depths shared is what makes the structural
# composites' chrome strata match even though their heights differ (the
# transition figures take the deeper drops below); the rise and drop depths
# serve both orientations because they are priced by the font, and the
# portrait font (27 px) is within one pixel of the landscape font (28 px).
# The portrait top margin is shallower because no portrait figure carries
# a column-title stratum (experiments label rows, on the right).
_LANDSCAPE_MARGINS_PX: dict[str, int] = {"t": 185, "b": 245, "l": 95, "r": 85}
_PORTRAIT_MARGINS_PX: dict[str, int] = {"t": 150, "b": 245, "l": 95, "r": 85}
_LEGEND_RISE_PX: int = 92
_BAND_DROP_PX: int = 100
# The transition axis's leaf ticks ("minimal to low") carry roughly twice
# the characters of a bare effort level, so their -45 degree ink runs about
# twice as deep and the 100 px corridor above would strike the rules and
# band names. These drops open a deeper corridor for the transition
# figures; probe renders at the calibrated font confirm clear air between
# tick ink, rule, band name, and structural title.
_TRANSITION_BAND_DROP_PX: int = 185
_TRANSITION_TITLE_DROP_PX: int = 325
# The flat dashes and regressed triangles sit exactly at $0, so the
# paired marginal cost composite's explicit dollar scale reserves this
# much of a row's height below zero — the marker ink clears the axis line
# instead of straddling it, matching the air autorange used to leave (and
# still leaves on the per-experiment chart, which keeps autorange).
_ZERO_MARKER_CLEARANCE_PX: int = 18
_STRUCTURAL_TITLE_DROP_PX: int = 228

# Horizontal clearance for a composite's rotated shared y title: the ink of
# a font-28 "0.5" tick label plus its standoff reaches about 54 px left of
# the axis, deeper than the 40 px make_subplots shifts its own y_title, so
# the deck places the title with its own clearance. The wide variant clears
# six-digit comma-grouped token ticks ("20,000" is about 100 px of ink).
_Y_TITLE_SHIFT_PX: int = 60
_WIDE_TICK_Y_TITLE_SHIFT_PX: int = 110

# Ink room a significance asterisk needs beyond a delta bar's tip: its 10 px
# shift plus roughly half the landscape emphasis glyph, with margin. Priced
# into an explicit delta-axis range so the glyph never crosses the plot edge.
_DELTA_ASTERISK_RESERVE_PX: int = 50


def build_paired_condition_accuracy(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
) -> go.Figure:
    """Per-condition accuracy for both experiments in one landscape composite.

    Three condition rows by two experiment columns, sharing one structural
    configuration axis: the transposition is what buys every one of the 21
    grouped bars a legible leaf tick on a rotated page, where three
    side-by-side panels per experiment could not reach the 8 pt target. The
    marks come from the same factories as the per-experiment builder, so a
    column here equals the standalone chart mark-for-mark; each trace carries
    ``meta={"experiment_column": n}`` so tests can assert that equivalence.

    Chrome follows the flat-legend convention: one horizontal ungrouped
    legend at the top left between the figure title and the column titles,
    ordered by ``_PAIRED_LEGEND_RANKS``; the right margin belongs to the
    two-line condition row labels alone.
    """
    order = structural_config_order(primary_data, ablation_data)
    positions, _ = grouped_config_positions(order)
    hover_labels = [short_config_label(ck) for ck in order]
    # 1150 px prints 334 pt tall, which shares the 430 pt rotated page with
    # this figure's nine-line caption; the budgeted 1230 px printed 357 pt
    # and pushed the caption onto its own near-empty rotated page.
    height = 1150
    plot_height = (
        height - _LANDSCAPE_MARGINS_PX["t"] - _LANDSCAPE_MARGINS_PX["b"]
    )

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.065,
        horizontal_spacing=0.035,
        column_titles=list(_EXPERIMENT_PANEL_TITLES),
        row_titles=[
            title.replace(" (", "<br>(") for title, _ in _CONDITION_PANELS
        ],
    )

    shown: set[str] = set()
    for col, data in ((1, primary_data), (2, ablation_data)):
        acc_lookup: dict[str, dict[str, float]] = {}
        for row_dict in data:
            acc_lookup.setdefault(
                row_dict["condition"], {},
            )[row_dict["config_key"]] = row_dict["accuracy"]
        score_by_config = {
            r["config_key"]: r["overall_accuracy"] for r in data
        }
        bal_by_config = {
            r["config_key"]: r["balanced_accuracy"] for r in data
        }

        for row, (_, condition) in enumerate(_CONDITION_PANELS, start=1):
            cond_acc = acc_lookup.get(condition, {})
            bar_name = _CONDITION_LEGEND_NAMES[condition]
            fig.add_trace(
                _condition_accuracy_bar(
                    condition,
                    positions,
                    [cond_acc.get(ck, 0.0) for ck in order],
                    config_labels=hover_labels,
                    width=BAR_SLOT_FRACTION,
                    showlegend=bar_name not in shown,
                    legendrank=_PAIRED_LEGEND_RANKS[bar_name],
                    meta={"experiment_column": col},
                ),
                row=row,
                col=col,
            )
            shown.add(bar_name)

            for overlay_name, lookup in (
                ("Overall Accuracy", score_by_config),
                ("Balanced Accuracy", bal_by_config),
            ):
                # Overlays declare their legend entries on the bottom row so
                # trace-declaration order matches the drawn (ranked) order:
                # bars first, then overlays, then the chance baseline.
                declare = (
                    row == len(_CONDITION_PANELS)
                    and overlay_name not in shown
                )
                fig.add_trace(
                    _condition_overlay_marker(
                        overlay_name,
                        positions,
                        [lookup.get(ck, 0.0) for ck in order],
                        config_labels=hover_labels,
                        showlegend=declare,
                        legendrank=_PAIRED_LEGEND_RANKS[overlay_name],
                        meta={"experiment_column": col},
                    ),
                    row=row,
                    col=col,
                )
                if declare:
                    shown.add(overlay_name)

    _add_chance_baseline(fig, legendgroup=None, legendgrouptitle_text=None)
    fig.data[-1].legendrank = _PAIRED_LEGEND_RANKS["Chance"]

    fig.update_layout(
        template=FIGURE_TEMPLATE,
        font=dict(family=EXPORT_FONT_FAMILY, size=LANDSCAPE_BASE_FONT_SIZE),
        width=EVERY_TICK_FIGURE_WIDTH,
        height=height,
        title=dict(
            text="Per-Condition Accuracy by Configuration [Primary + Ablation]",
            font=dict(size=LANDSCAPE_EMPHASIS_FONT_SIZE),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        legend=dict(
            orientation="h", x=0.033, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
        ),
        margin=dict(**_LANDSCAPE_MARGINS_PX),
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.update_yaxes(title_text="Condition Accuracy", row=2, col=1)
    fig.update_annotations(font_size=LANDSCAPE_BASE_FONT_SIZE)
    _finish_structural_axis(
        fig, order,
        band_axis_refs=("x5", "x6"),
        plot_height=plot_height,
        font_px=LANDSCAPE_BASE_FONT_SIZE,
    )
    return fig


# Grouped bar spacing for the model-grouped axes (configuration and
# transition alike). Within a model group, bars sit
# at unit pitch and fill BAR_SLOT_FRACTION of it, so neighbors nearly touch
# (the ~5% gap renders as a hairline at export sizes); between model groups an
# extra MODEL_GROUP_GAP slot of empty space shows membership. Heatmaps use
# HEATMAP_GROUP_GAP instead: Plotly sizes heatmap bricks from neighbor
# midpoints, so only a whole-slot gap — one blank gutter column — keeps
# every brick the same width (the fractional bar gap rendered the boundary
# bricks at roughly half width in probe renders).
BAR_SLOT_FRACTION: float = 0.95
MODEL_GROUP_GAP: float = 0.8
HEATMAP_GROUP_GAP: float = 1.0

# The group rule: one line under each model's leaf ticks, spanning that
# group's slots, binding the leaf ticks to the model name below.
# Spacing alone left the binding to proximity, and the -45 degree leaf
# labels lean left across group boundaries, so boundary-adjacent ticks read
# as the neighboring model's. The rule sits at the band-annotation drop —
# probe renders measured the free corridor there as 85..114 px below the
# axis at both calibrated fonts, between the deepest rotated tick ink and
# the model-name ink. The color is the deck's slate ink, shared with the
# marker for the overall accuracy overlay.
GROUP_RULE_COLOR: str = "#64748b"
GROUP_RULE_WIDTH: float = 2.5
# Heatmap rules stop short of the cell edges (cells span half a slot either
# side of their position) so adjacent groups' rules stay visibly separate
# across a single gutter column; bars use their slot half-width instead.
_HEATMAP_RULE_HALF_EXTENT: float = 0.42

# Two-line forms for band labels whose one-line ink would overrun a narrow
# model group's span. Keys are exact ``display_config_key`` model segments —
# the dagger included — because a key that drifts from that output
# silently stops wrapping. ``GPT-5-mini`` and ``GPT-5.2`` fit unwrapped.
MODEL_BAND_WRAP: dict[str, str] = {
    "Claude Haiku 4.5": "Claude<br>Haiku 4.5",
    "Claude Opus 4.6†": "Claude<br>Opus 4.6†",
    "Gemini 3 Flash": "Gemini 3<br>Flash",
    "Gemini 3 Pro": "Gemini 3<br>Pro",
}


def _grouped_model_positions(
    model_names: list[str],
    *,
    gap: float,
) -> tuple[list[float], list[tuple[str, float, float]]]:
    """Grouped x positions and model bands over a model-name sequence.

    One position per entry, at unit pitch within a contiguous model run
    and ``1 + gap`` across a run boundary, plus one ``(model display name,
    first x, last x)`` band per run. The configuration axis and the
    transition axis both reduce to this walk once their items are mapped
    to model display names.
    """
    positions: list[float] = []
    bands: list[tuple[str, float, float]] = []
    x = 0.0
    last_model: str | None = None
    for model in model_names:
        if last_model is not None and model != last_model:
            x += gap
        positions.append(x)
        if bands and bands[-1][0] == model:
            bands[-1] = (model, bands[-1][1], x)
        else:
            bands.append((model, x, x))
        last_model = model
        x += 1.0
    return positions, bands


def grouped_config_positions(
    config_order: list[str],
    *,
    gap: float = MODEL_GROUP_GAP,
) -> tuple[list[float], list[tuple[str, float, float]]]:
    """Numeric positions with a gap between model groups, plus the bands.

    Returns ``(positions, bands)``: one x position per configuration, at
    unit pitch within a model group and ``1 + gap`` across a group
    boundary, and one band per contiguous model run. Band names come from
    ``display_config_key``'s model segment, so they carry the authoring
    dagger. Bar axes use the default ``MODEL_GROUP_GAP``; heatmaps pass
    ``HEATMAP_GROUP_GAP`` so every position is an integer and each
    boundary's skipped integer becomes the blank gutter column.
    """
    model_names = [
        display_config_key(ck).split(" / ")[0] for ck in config_order
    ]
    return _grouped_model_positions(model_names, gap=gap)


def _transition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """The identity of one effort-level transition, for position lookups
    and cross-experiment equality checks."""
    return (row["model_slug"], row["from_level"], row["to_level"])


def _transition_leaf_label(row: dict[str, Any]) -> str:
    """The leaf-tick text for a transition axis: both endpoint levels in
    natural English ("low to medium"), the model half lifted into its
    band."""
    from_label = display_reasoning_effort_level(row["from_level"])
    to_label = display_reasoning_effort_level(row["to_level"])
    return f"{from_label} to {to_label}"


def grouped_transition_positions(
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[tuple[str, float, float]]]:
    """Grouped x positions and model bands over a transition sequence.

    The transition analogue of ``grouped_config_positions``: one position
    per transition at unit pitch within a model, ``1 + MODEL_GROUP_GAP``
    across a model boundary, and one band per contiguous model run. Band
    names derive from the row's ``model_slug`` through the same
    ``mark_authoring``/``display_model_slug`` pair as
    ``display_config_key``'s model segment, so ``MODEL_BAND_WRAP`` keys
    keep matching.
    """
    model_names = [
        mark_authoring(
            display_model_slug(row["model_slug"]),
            is_authoring=is_authoring_model(row["model_slug"]),
        )
        for row in rows
    ]
    return _grouped_model_positions(model_names, gap=MODEL_GROUP_GAP)


def _config_leaf_labels(config_order: list[str]) -> list[str]:
    """The leaf-tick text for a configuration axis: bare effort levels,
    since the model half of each config key lifts into its band."""
    return [
        display_reasoning_effort_level(parse_config_key(ck)[2])
        for ck in config_order
    ]


def _apply_two_tier_labels(
    fig: go.Figure,
    leaf_labels: list[str],
    positions: list[float],
    bands: list[tuple[str, float, float]],
    *,
    band_axis_refs: tuple[str, ...],
    band_y: float,
    font_px: int,
    x_range: list[float] | None,
    rule_half_extent: float,
) -> None:
    """The shared two-tier labeling core: leaf ticks, rules, band names.

    The leaf tier: every x axis gets the given leaf labels as tick text at
    the given positions, rotated -45 degrees. The band tier: one
    model-name annotation per group, centered under the group in
    **data coordinates** on each axis named in ``band_axis_refs``
    (paper-fraction anchoring drifts with the margins), at paper-fraction
    ``band_y`` below the plot area. Between the tiers, one group rule per
    band binds the ticks to their model name: a line at ``band_y`` (the
    free corridor between tick ink and name ink) spanning the group's
    slots plus ``rule_half_extent`` each side, drawn per band axis so a
    composite's every experiment panel carries its own copy.
    """
    axis_updates: dict[str, Any] = dict(
        tickmode="array",
        tickvals=positions,
        ticktext=leaf_labels,
        tickangle=-45,
        tickfont=dict(size=font_px),
    )
    if x_range is not None:
        axis_updates["range"] = x_range
    fig.update_xaxes(**axis_updates)
    for axis_ref in band_axis_refs:
        for model, x_first, x_last in bands:
            fig.add_shape(
                type="line",
                xref=axis_ref,
                yref="paper",
                x0=x_first - rule_half_extent,
                x1=x_last + rule_half_extent,
                y0=band_y,
                y1=band_y,
                line=dict(color=GROUP_RULE_COLOR, width=GROUP_RULE_WIDTH),
            )
            fig.add_annotation(
                x=(x_first + x_last) / 2,
                y=band_y,
                xref=axis_ref,
                yref="paper",
                text=MODEL_BAND_WRAP.get(model, model),
                showarrow=False,
                yanchor="top",
                align="center",
                # Full size, not a scaled-down tier: callers calibrate
                # font_px to the 8 pt print floor exactly, so any smaller
                # multiplier would print the model names below it.
                font=dict(size=font_px),
            )


def apply_two_tier_config_axis(
    fig: go.Figure,
    config_order: list[str],
    *,
    band_axis_refs: tuple[str, ...],
    band_y: float,
    font_px: int,
) -> None:
    """Two-tier labels over grouped bar positions.

    Callers place bars at ``grouped_config_positions`` x values with width
    ``BAR_SLOT_FRACTION``; this helper only labels the geometry, and pins
    the x range so the edge bars keep the same half-gap of air the group
    gaps provide inside. The group gap separates the groups; the group
    rules span each group's bar extent and bind its ticks to its name.
    """
    positions, bands = grouped_config_positions(config_order)
    _apply_two_tier_labels(
        fig, _config_leaf_labels(config_order), positions, bands,
        band_axis_refs=band_axis_refs,
        band_y=band_y,
        font_px=font_px,
        x_range=[positions[0] - 0.9, positions[-1] + 0.9],
        rule_half_extent=BAR_SLOT_FRACTION / 2,
    )


def apply_two_tier_heatmap_axis(
    fig: go.Figure,
    config_order: list[str],
    *,
    band_axis_refs: tuple[str, ...],
    band_y: float,
    font_px: int,
) -> None:
    """Two-tier labels for a heatmap's gutter-separated columns.

    Heatmap columns sit at unit-pitch grouped positions — one blank gutter
    column per model boundary, matching the trace cores' geometry — so the
    leaf ticks land on the real columns only, the model bands center over
    each gutter-separated run, and no explicit range is imposed (a numeric
    heatmap autorange stays flush to the outer cell edges).
    """
    positions, bands = grouped_config_positions(
        config_order, gap=HEATMAP_GROUP_GAP,
    )
    _apply_two_tier_labels(
        fig, _config_leaf_labels(config_order), positions, bands,
        band_axis_refs=band_axis_refs,
        band_y=band_y,
        font_px=font_px,
        x_range=None,
        rule_half_extent=_HEATMAP_RULE_HALF_EXTENT,
    )


def apply_two_tier_transition_axis(
    fig: go.Figure,
    rows: list[dict[str, Any]],
    *,
    band_axis_refs: tuple[str, ...],
    band_y: float,
    font_px: int,
) -> None:
    """Two-tier labels over grouped transition positions.

    The bar-axis treatment of ``apply_two_tier_config_axis`` applied to a
    transition sequence: callers place their marks at
    ``grouped_transition_positions`` x values with width
    ``BAR_SLOT_FRACTION``, the leaf ticks keep both endpoint levels
    ("off to enabled"), and the model names lift into the band tier.
    """
    positions, bands = grouped_transition_positions(rows)
    _apply_two_tier_labels(
        fig, [_transition_leaf_label(row) for row in rows], positions, bands,
        band_axis_refs=band_axis_refs,
        band_y=band_y,
        font_px=font_px,
        x_range=[positions[0] - 0.9, positions[-1] + 0.9],
        rule_half_extent=BAR_SLOT_FRACTION / 2,
    )


def _finish_structural_axis(
    fig: go.Figure,
    config_order: list[str],
    *,
    band_axis_refs: tuple[str, ...],
    plot_height: int,
    font_px: int,
    heatmap: bool = False,
) -> None:
    """Two-tier axis plus the centered structural ordering disclosure.

    Converts the shared chrome pixel depths (band drop, structural title
    drop) against this figure's plot height, so every structural-axis
    figure's bottom strata match regardless of its canvas. ``heatmap``
    selects the heatmap variant, whose columns sit at unit-pitch grouped
    positions with one blank gutter column per model boundary.
    """
    apply_axis = (
        apply_two_tier_heatmap_axis if heatmap
        else apply_two_tier_config_axis
    )
    apply_axis(
        fig, config_order,
        band_axis_refs=band_axis_refs,
        band_y=-_BAND_DROP_PX / plot_height,
        font_px=font_px,
    )
    fig.add_annotation(
        x=0.5, y=-_STRUCTURAL_TITLE_DROP_PX / plot_height,
        xref="paper", yref="paper", showarrow=False,
        text=_CONFIG_GROUPED_BY_MODEL_TITLE,
        font=dict(size=font_px),
    )


def _finish_transition_axis(
    fig: go.Figure,
    rows: list[dict[str, Any]],
    *,
    band_axis_refs: tuple[str, ...],
    plot_height: int,
    font_px: int,
) -> None:
    """Two-tier transition axis plus its centered ordering disclosure.

    The transition counterpart of ``_finish_structural_axis``: the same
    strata, converted from the deeper transition pixel drops (see
    ``_TRANSITION_BAND_DROP_PX``) against this figure's plot height, with
    ``_REASONING_EFFORT_TRANSITION_TITLE`` as the disclosure the axis
    title used to carry.
    """
    apply_two_tier_transition_axis(
        fig, rows,
        band_axis_refs=band_axis_refs,
        band_y=-_TRANSITION_BAND_DROP_PX / plot_height,
        font_px=font_px,
    )
    fig.add_annotation(
        x=0.5, y=-_TRANSITION_TITLE_DROP_PX / plot_height,
        xref="paper", yref="paper", showarrow=False,
        text=_REASONING_EFFORT_TRANSITION_TITLE,
        font=dict(size=font_px),
    )


def structural_config_order(*data_lists: list[dict[str, Any]]) -> list[str]:
    """The one structural order every configuration axis shares.

    Unions the ``config_key`` values across the given row lists and sorts them
    by :func:`config_tie_break_key` (provider, model, ascending reasoning
    effort). Passing both experiments' rows yields one shared order, so a
    paired figure stays column-aligned even when one experiment is missing a
    configuration the other carries.
    """
    configs: set[str] = set()
    for data in data_lists:
        configs.update(row["config_key"] for row in data)
    return sorted(configs, key=config_tie_break_key)


def validate_heatmap_alignment_inputs(
    data: list[dict[str, Any]],
    *,
    config_order: list[str],
) -> None:
    """Raise ``ValueError`` when ``config_order`` omits a key present in data.

    The opposite direction (order entries with no data rows) is intentionally
    permitted: heatmap builders render those as empty cells, which is the
    correct behavior when, e.g., an ablation snapshot is missing a config that
    appears in the primary anchor order.
    """
    data_configs = {row["config_key"] for row in data}
    missing = data_configs - set(config_order)
    if missing:
        raise ValueError(
            "config_order is missing config_key values present in "
            f"data: {sorted(missing)!r}"
        )


def sorted_base_examples() -> list[str]:
    """The heatmaps' fixed row order: every base example id, ascending.

    Both heatmap builders order rows by this one rule, so a given row is the
    same base example in every heatmap and readers can compare across
    figures. Ids are zero-padded, so a lexicographic sort is a numeric one,
    and ``yaxis_autorange="reversed"`` puts the first entry at the top.

    The rows come from the stimulus registry rather than from the data, so
    the set cannot go ragged: ``example_accuracy_matrix`` emits a cell only
    where at least one trial parsed, and letting that decide the rows would
    let one panel of a paired figure lose a row the other kept, silently
    misaligning every row below it.
    """
    return sorted(STIMULUS_METADATA_REGISTRY)


def _heatmap_column_layout(
    config_order: list[str],
) -> tuple[list[int], dict[int, str]]:
    """The gutter column geometry every heatmap trace and axis shares.

    Returns ``(column_xs, x_to_config)``: the full run of integer column
    positions — real columns and blank gutter columns alike — and the map
    from each real column's x to its config key. Positions come from
    :func:`grouped_config_positions` at ``HEATMAP_GROUP_GAP``, so each
    model boundary skips exactly one integer and that skipped column is
    the gutter.
    """
    positions, _ = grouped_config_positions(
        config_order, gap=HEATMAP_GROUP_GAP,
    )
    x_to_config = {
        int(p): ck for p, ck in zip(positions, config_order)
    }
    column_xs = list(range(int(positions[-1]) + 1)) if positions else []
    return column_xs, x_to_config


def heatmap_column_config(config_order: list[str]) -> dict[int, str]:
    """Column position → config key, for resolving heatmap click events.

    The inverse of the trace cores' gutter column layout, real columns
    only: a clicked heatmap point reports its numeric column position as
    ``x``, and a click on a blank gutter column resolves to no
    configuration.
    """
    _, x_to_config = _heatmap_column_layout(list(config_order))
    return x_to_config


def _condition_accuracy_heatmap_trace(
    data: list[dict[str, Any]],
    *,
    value_column: str,
    config_order: list[str],
    showscale: bool = True,
) -> go.Heatmap:
    """The per-example accuracy matrix trace: values, scale, and hover.

    Validates alignment, builds the ``sorted_base_examples`` × config
    matrix on the gutter column geometry (blank all-``None`` columns at
    model boundaries), and carries the fixed [0, 1] Viridis scale, so the
    paired composite and the per-experiment chart cannot diverge
    cell-for-cell. Config labels ride ``customdata`` because the numeric
    column positions would otherwise surface in hover through ``%{x}``.
    """
    validate_heatmap_alignment_inputs(data, config_order=config_order)

    examples_sorted = sorted_base_examples()
    column_xs, x_to_config = _heatmap_column_layout(config_order)

    data_lookup = {
        (r["base_example"], r["config_key"]): r.get(value_column, 0.0)
        for r in data
    }
    z_matrix = [
        [
            data_lookup.get((ex, x_to_config[x]), None)
            if x in x_to_config else None
            for x in column_xs
        ]
        for ex in examples_sorted
    ]
    label_row = [
        short_config_label(x_to_config[x]) if x in x_to_config else None
        for x in column_xs
    ]

    return go.Heatmap(
        z=z_matrix,
        x=column_xs,
        y=list(examples_sorted),
        customdata=[label_row for _ in examples_sorted],
        colorscale="Viridis",
        zmin=0,
        zmax=1,
        xgap=0,
        ygap=0,
        hoverongaps=False,
        showscale=showscale,
        hovertemplate=(
            "Config: %{customdata}<br>"
            "Example: %{y}<br>"
            "Condition Accuracy: %{z:.3f}<extra></extra>"
        ),
        colorbar_title="Condition Accuracy",
    )


def build_example_condition_accuracy_heatmap(
    data: list[dict[str, Any]],
    *,
    value_column: str = "correct_acc",
    config_order: list[str],
    experiment_label: str,
) -> go.Figure:
    """Heatmap: base examples (y-axis) x configs (x-axis), colored by accuracy.

    Rows run in the fixed order :func:`sorted_base_examples` returns, so the
    same row is the same example here and in the failure mode dominance
    heatmap, in both experiments.

    Parameters
    ----------
    config_order:
        The column order and universe, shared across primary and ablation
        renders so paired figures are axis-aligned cell-for-cell — pass
        :func:`structural_config_order` over both experiments' rows. Strict
        validation: every config key present in ``data`` must also be in the
        supplied order, otherwise :func:`validate_heatmap_alignment_inputs`
        raises ``ValueError``. Order entries with no data rows render as
        empty cells.
    """
    if not data:
        return go.Figure()

    fig = go.Figure(data=_condition_accuracy_heatmap_trace(
        data, value_column=value_column, config_order=config_order,
    ))

    _condition_title = _CONDITION_ACC_TITLES.get(value_column, value_column)
    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Per-Example {_condition_title} [{experiment_label}]"),
        xaxis_title=_CONFIG_GROUPED_BY_MODEL_TITLE,
        yaxis_title=_BASE_EXAMPLE_AXIS_TITLE,
        yaxis_autorange="reversed",
        xaxis_tickangle=CROWDED_CONFIG_TICK_ANGLE,
        height=HEATMAP_FIGURE_HEIGHT,
        width=FIGURE_WIDTH,
        margin=dict(b=CROWDED_CONFIG_BOTTOM_MARGIN),
    )
    _tick_heatmap_columns_with_config_labels(fig, config_order)

    return fig


def _tick_heatmap_columns_with_config_labels(
    fig: go.Figure,
    config_order: list[str],
) -> None:
    """Full configuration labels on a per-experiment heatmap's real columns.

    The trace cores put columns at numeric gutter positions, so the
    per-experiment views re-anchor their full config labels to the real
    columns; gutter columns carry no tick. The paired composites label the
    same geometry through the two-tier axis instead.
    """
    _, x_to_config = _heatmap_column_layout(list(config_order))
    tickvals = sorted(x_to_config)
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=[short_config_label(x_to_config[x]) for x in tickvals],
    )


def _paired_heatmap_scaffold(*, height: int) -> tuple[go.Figure, int]:
    """2 experiment-row scaffold for the paired heatmap composites.

    Experiment labels ride subplot titles above each matrix because the
    right margin belongs to the shared colorbar; the left and right margins
    stay on automargin for the example tick labels and the colorbar.
    Returns ``(fig, plot_height)``.
    """
    plot_height = (
        height - _PORTRAIT_MARGINS_PX["t"] - _PORTRAIT_MARGINS_PX["b"]
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=list(_EXPERIMENT_PANEL_TITLES),
    )
    return fig, plot_height


def build_paired_example_condition_accuracy_heatmap(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
    *,
    value_column: str = "correct_acc",
    config_order: list[str],
) -> go.Figure:
    """Per-example accuracy for both experiments in one portrait composite.

    Two experiment matrices stacked on a shared structural configuration
    axis (gutter-separated columns, two-tier labels on the bottom row)
    with one
    colorbar serving both, each matrix from the same trace core as the
    per-experiment chart; each trace carries ``meta={"experiment_row": n}``
    so tests can assert that equivalence. ``config_order`` is the shared
    column order and universe for both experiments.
    """
    height = 1150
    fig, plot_height = _paired_heatmap_scaffold(height=height)

    for row, data in ((1, primary_data), (2, ablation_data)):
        trace = _condition_accuracy_heatmap_trace(
            data, value_column=value_column, config_order=config_order,
            showscale=row == 1,
        )
        trace.meta = {"experiment_row": row}
        fig.add_trace(trace, row=row, col=1)

    _condition_title = _CONDITION_ACC_TITLES.get(value_column, value_column)
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            **_figure_title(
                f"Per-Example {_condition_title} [Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(
            t=_PORTRAIT_MARGINS_PX["t"], b=_PORTRAIT_MARGINS_PX["b"],
        ),
    )
    fig.update_yaxes(title_text=_BASE_EXAMPLE_AXIS_TITLE, autorange="reversed")
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    _finish_structural_axis(
        fig, list(config_order),
        band_axis_refs=("x2",),
        plot_height=plot_height,
        font_px=BASE_FONT_SIZE,
        heatmap=True,
    )
    return fig


def _model_effort_lines(
    data: list[dict[str, Any]],
    *,
    y_key: str,
    hover_label: str,
    hover_value_format: str,
) -> list[go.Scatter]:
    """One lines+markers trace per model group over the effort levels.

    Groups follow the structural order's leading elements (provider, then
    model); rows within a group are ordered by their reasoning effort
    ordinal, completing the structural key from parts every row carries.
    Colors come from ``MODEL_GROUP_COLORS`` (provider fallback) and labels
    carry the authoring dagger, so the paired composites and the
    per-experiment charts cannot diverge line-for-line.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in data:
        groups.setdefault(row["model_group"], []).append(row)

    traces: list[go.Scatter] = []
    for model_group, rows in sorted(
        groups.items(),
        key=lambda kv: (kv[1][0]["provider"], kv[1][0]["model_slug"]),
    ):
        rows_sorted = sorted(rows, key=lambda r: r["ordinal"])
        provider = rows_sorted[0]["provider"]
        color = _model_group_color(model_group, provider)
        model_slug = rows_sorted[0]["model_slug"]
        model_label = mark_authoring(
            display_model_slug(model_slug),
            is_authoring=is_authoring_model(model_slug),
        )
        traces.append(go.Scatter(
            name=model_label,
            x=[
                display_reasoning_effort_level(
                    r["reasoning_effort_level"],
                )
                for r in rows_sorted
            ],
            y=[r[y_key] for r in rows_sorted],
            mode="lines+markers",
            marker=dict(color=color, size=10),
            line=dict(color=color, width=2),
            hovertemplate=(
                f"Model: {model_label}<br>"
                "Reasoning Effort Level: %{x}<br>"
                f"{hover_label}: %{{y:{hover_value_format}}}<extra></extra>"
            ),
        ))
    return traces


def _display_levels_union(*data_lists: list[dict[str, Any]]) -> list[str]:
    """The canonical display order over every effort level present."""
    levels_present = {
        display_reasoning_effort_level(r["reasoning_effort_level"])
        for data in data_lists
        for r in data
    }
    return _display_forced_order(levels_present)


def build_reasoning_effort_chart(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Line chart: balanced accuracy vs reasoning effort level, one line per model group.

    Each model group gets a distinct color via ``MODEL_GROUP_COLORS`` so
    sibling models from the same provider are visually distinguishable.
    X-axis uses a canonical ordering of reasoning effort levels (lowest → highest
    compute) shared across all model families, so that heterogeneous level
    vocabularies align logically on the same axis.  Chance baseline at
    y=0.5 appears in the legend.
    """
    fig = go.Figure()

    # Canonical ordering with display aliases applied (e.g., "none" → "off")
    forced_order = _display_levels_union(data)

    for trace in _model_effort_lines(
        data, y_key="balanced_accuracy",
        hover_label="Balanced Accuracy", hover_value_format=".3f",
    ):
        fig.add_trace(trace)

    # This chart's legend is keyed by model, not grouped — leave Chance ungrouped.
    _add_chance_baseline(fig, legendgroup=None, legendgrouptitle_text=None)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Reasoning Effort vs. Balanced Accuracy [{experiment_label}]"),
        xaxis_title=_REASONING_EFFORT_LEVEL_TITLE,
        yaxis_title="Balanced Accuracy",
        yaxis_range=[0, 1.05],
        legend_title="Model",
        height=500,
        width=FIGURE_WIDTH,
        xaxis=dict(
            categoryorder="array",
            categoryarray=forced_order,
        ),
    )

    return fig


def _paired_effort_figure(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
    *,
    y_key: str,
    hover_label: str,
    hover_value_format: str,
    height: int,
    bottom_margin: int,
) -> tuple[go.Figure, int]:
    """2 experiment-row scaffold over a shared canonical effort axis.

    Adds each experiment's model lines (legend declared on row 1 only,
    ``meta={"experiment_row": n}`` on every trace) and pins both category
    axes to the union of the display levels. Returns ``(fig, plot_height)``
    for the caller's chrome placement.
    """
    plot_height = height - _PORTRAIT_MARGINS_PX["t"] - bottom_margin
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        row_titles=list(_EXPERIMENT_PANEL_TITLES),
    )
    for row, data in ((1, primary_data), (2, ablation_data)):
        for trace in _model_effort_lines(
            data, y_key=y_key,
            hover_label=hover_label,
            hover_value_format=hover_value_format,
        ):
            trace.meta = {"experiment_row": row}
            trace.showlegend = row == 1
            fig.add_trace(trace, row=row, col=1)
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=_display_levels_union(primary_data, ablation_data),
    )
    fig.update_xaxes(
        title_text=_REASONING_EFFORT_LEVEL_TITLE, row=2, col=1,
    )
    return fig, plot_height


def build_paired_reasoning_effort(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
) -> go.Figure:
    """Balanced accuracy vs reasoning effort for both experiments, stacked.

    Two experiment rows over one canonical effort-level axis, model lines
    from the same core as the per-experiment chart, one chance baseline
    entry covering both rows, and the flat horizontal legend keeping the
    per-experiment "Model" title. Both rows pin the [0, 1.05] accuracy
    domain.
    """
    height = 980
    fig, plot_height = _paired_effort_figure(
        primary_data, ablation_data,
        y_key="balanced_accuracy",
        hover_label="Balanced Accuracy",
        hover_value_format=".3f",
        height=height,
        bottom_margin=100,
    )
    _add_chance_baseline(fig, legendgroup=None, legendgrouptitle_text=None)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            **_figure_title(
                "Reasoning Effort vs. Balanced Accuracy [Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        legend=dict(
            orientation="h", x=0, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
            title=dict(text="Model"),
        ),
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(
            t=_PORTRAIT_MARGINS_PX["t"], b=100,
            l=_PORTRAIT_MARGINS_PX["l"],
        ),
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.add_annotation(
        x=0, xref="paper", xanchor="right", xshift=-_Y_TITLE_SHIFT_PX,
        y=0.5, yref="paper", textangle=-90, showarrow=False,
        text="Balanced Accuracy",
        font=dict(size=BASE_FONT_SIZE),
    )
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    return fig


def build_failure_mode_chart_correct(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Stacked bar chart: correct-draft failure modes (governing rate) by config.

    Single panel using the correct-draft discriminating flag
    (``operational_interpretation_governed_judgment``, the governing
    classification): Operational interpretation vs Field label matching.
    Separated from incorrect-draft because the two condition types use
    fundamentally different classification schemes.

    Configs render in the order the caller supplied, which the x-axis
    title names: balanced accuracy descending.
    """
    if not data:
        return go.Figure()

    labels = [short_config_label(r["config_key"]) for r in data]

    fig = go.Figure()

    # Bottom of stack: Field label matching (governing = false)
    fig.add_trace(_failure_mode_tally_bar(
        "Field label matching",
        labels,
        [r["correct_field_label"] for r in data],
        config_labels=labels,
        legendgroup="Field label matching",
    ))

    # Top of stack: Operational interpretation (governing = true)
    fig.add_trace(_failure_mode_tally_bar(
        "Operational interpretation",
        labels,
        [r["correct_governed"] for r in data],
        config_labels=labels,
        legendgroup="Operational interpretation",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Correct-Draft Failure Mode Breakdown by Configuration [{experiment_label}]"),
        barmode="stack",
        legend_title="Verdict Basis",
        xaxis_title=_CONFIG_GROUPED_BY_MODEL_TITLE,
        height=_FAILURE_MODE_SINGLE_PANEL_HEIGHT + _AXIS_TITLE_ALLOWANCE,
        width=FIGURE_WIDTH,
        margin=dict(b=CONFIG_TICK_BOTTOM_MARGIN),
    )
    fig.update_xaxes(tickangle=CONFIG_TICK_ANGLE)
    fig.update_yaxes(title_text=_WRONG_VERDICT_TRIALS_AXIS_TITLE)

    return fig


def build_failure_mode_chart_incorrect(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Faceted stacked bar chart: incorrect-draft failure modes by config.

    Two panels (Transparent, Opaque) with shared y-axis using the
    incorrect-draft discriminating flag
    (``articulated_operational_interpretation``, the articulation
    classification): Expressed vs Never expressed.
    Separated from correct-draft because the two condition types use
    fundamentally different classification schemes.

    Configs render in the order the caller supplied, which the x-axis
    title names: balanced accuracy descending.
    """
    if not data:
        return go.Figure()

    labels = [short_config_label(r["config_key"]) for r in data]

    fig = _faceted_figure(
        cols=2,
        panel_titles=[p[0] for p in _INCORRECT_FAILURE_MODE_COUNT_PANELS],
    )

    # Track legend entries already shown to avoid duplicates across panels
    _legend_shown: set[str] = set()

    for col_idx, (_, true_key, false_key) in enumerate(
        _INCORRECT_FAILURE_MODE_COUNT_PANELS, start=1,
    ):
        # Bottom of stack: Never expressed (articulation = false)
        fig.add_trace(
            _failure_mode_tally_bar(
                "Never expressed",
                labels,
                [r[false_key] for r in data],
                config_labels=labels,
                showlegend="Never expressed" not in _legend_shown,
                legendgroup="Never expressed",
            ),
            row=1,
            col=col_idx,
        )
        _legend_shown.add("Never expressed")

        # Top of stack: Expressed (articulation = true)
        fig.add_trace(
            _failure_mode_tally_bar(
                "Expressed",
                labels,
                [r[true_key] for r in data],
                config_labels=labels,
                showlegend="Expressed" not in _legend_shown,
                legendgroup="Expressed",
            ),
            row=1,
            col=col_idx,
        )
        _legend_shown.add("Expressed")

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Incorrect-Draft Failure Mode Breakdown by Configuration and Condition [{experiment_label}]"),
        barmode="stack",
        legend_title="Operational Interpretation",
        height=_FAILURE_MODE_FACETED_HEIGHT + _AXIS_TITLE_ALLOWANCE,
        width=EVERY_TICK_FIGURE_WIDTH,
        margin=dict(b=CONFIG_TICK_BOTTOM_MARGIN),
    )
    fig.update_xaxes(tickangle=CONFIG_TICK_ANGLE, dtick=1)
    # Set on one facet only: an unqualified update_xaxes writes the title
    # onto every facet, and the copies overprint each other.
    fig.update_xaxes(
        title_text=_CONFIG_GROUPED_BY_MODEL_TITLE, row=1, col=1,
    )
    fig.update_yaxes(title_text=_WRONG_VERDICT_TRIALS_AXIS_TITLE, row=1, col=1)

    return fig


# Stack-segment names → FAILURE_MODE_COLORS keys, across every failure mode
# chart: the correct-draft charts split on the governing classification, the
# incorrect-draft charts on the articulation classification.
_FAILURE_MODE_STACK_COLOR_KEYS: dict[str, str] = {
    "Field label matching": "field_label",
    "Operational interpretation": "governed",
    "Never expressed": "capability_absent",
    "Expressed": "selection_failed",
}

# Incorrect-draft rate panels: (display label, expressed-rate key,
# never-expressed rate key, mismatch total key), one per condition, in
# display order.
_INCORRECT_FAILURE_MODE_PANELS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Transparent",
        "transparent_selection_failed_rate",
        "transparent_capability_absent_rate",
        "transparent_total",
    ),
    (
        "Opaque",
        "opaque_selection_failed_rate",
        "opaque_capability_absent_rate",
        "opaque_total",
    ),
)

# Incorrect-draft count panels: (display label, expressed-count key,
# never-expressed count key), one per condition, in display order.
_INCORRECT_FAILURE_MODE_COUNT_PANELS: tuple[tuple[str, str, str], ...] = (
    (
        "Transparent",
        "transparent_selection_failed", "transparent_capability_absent",
    ),
    (
        "Opaque",
        "opaque_selection_failed", "opaque_capability_absent",
    ),
)


def _failure_mode_tally_bar(
    name: str,
    x: list[Any],
    counts: list[int],
    config_labels: list[str],
    **legend: Any,
) -> go.Bar:
    """A failure mode breakdown count bar: values, color, and hover.

    ``config_labels`` ride ``customdata`` so the hover names the
    configuration even when ``x`` holds numeric grouped positions.
    """
    return go.Bar(
        name=name,
        x=x,
        y=counts,
        customdata=config_labels,
        marker_color=FAILURE_MODE_COLORS[_FAILURE_MODE_STACK_COLOR_KEYS[name]],
        hovertemplate=(
            "Config: %{customdata}<br>"
            f"{name}: %{{y}}<extra></extra>"
        ),
        **legend,
    )


def _failure_mode_rate_bar(
    name: str,
    x: list[Any],
    rates: list[float],
    config_labels: list[str],
    **legend: Any,
) -> go.Bar:
    """A bottom-of-stack failure mode rate bar: values, color, hover.

    ``config_labels`` ride ``customdata`` so the hover names the
    configuration even when ``x`` holds numeric grouped positions.
    """
    return go.Bar(
        name=name,
        x=x,
        y=rates,
        customdata=config_labels,
        marker_color=FAILURE_MODE_COLORS[_FAILURE_MODE_STACK_COLOR_KEYS[name]],
        hovertemplate=(
            "Config: %{customdata}<br>"
            f"{name}: %{{y:.1%}}<extra></extra>"
        ),
        **legend,
    )


def _failure_mode_count_bar(
    name: str,
    x: list[Any],
    rates: list[float],
    counts: list[int],
    config_labels: list[str],
    *,
    count_font_px: int,
    **legend: Any,
) -> go.Bar:
    """The top-of-stack rate bar carrying the outside N mismatch counts.

    The counts render upright rather than slanted: a count set on the
    diagonal spreads its ink across neighboring bars, and at the deck's
    fonts that is wider than a bar's share of the axis. Turned vertical, a
    count occupies only its own line height of ink, which every bar pitch
    in the deck clears.
    """
    return go.Bar(
        name=name,
        x=x,
        y=rates,
        customdata=config_labels,
        marker_color=FAILURE_MODE_COLORS[_FAILURE_MODE_STACK_COLOR_KEYS[name]],
        text=[f"N={count}" for count in counts],
        textposition="outside",
        textangle=-90,
        textfont=dict(size=count_font_px),
        hovertemplate=(
            "Config: %{customdata}<br>"
            f"{name}: %{{y:.1%}}<br>"
            "%{text}<extra></extra>"
        ),
        **legend,
    )


def build_failure_mode_rate_chart_correct(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Stacked bar chart: correct-draft failure mode RATES (governing rate) by config.

    Single panel using normalized rates (0–1) for the correct-draft
    discriminating flag (``operational_interpretation_governed_judgment``,
    the governing classification): Operational interpretation vs Field
    label matching. Rates are directly comparable across configurations
    with different mismatch totals.

    The y-axis keeps headroom above 1.0 for the outside mismatch-count
    labels while its tick labels stay pinned to the [0, 1] rate domain.

    Configs render in the order the caller supplied, which the x-axis
    title names: balanced accuracy descending.
    """
    if not data:
        return go.Figure()

    labels = [short_config_label(r["config_key"]) for r in data]

    fig = go.Figure()

    # Bottom of stack: Field label matching rate (governing = false)
    fig.add_trace(_failure_mode_rate_bar(
        "Field label matching",
        labels,
        [r.get("correct_field_label_rate") or 0.0 for r in data],
        config_labels=labels,
        legendgroup="Field label matching",
    ))

    # Top of stack: Operational interpretation rate (governing = true),
    # annotated with the mismatch N
    fig.add_trace(_failure_mode_count_bar(
        "Operational interpretation",
        labels,
        [r.get("correct_governed_rate") or 0.0 for r in data],
        [r["correct_total"] for r in data],
        config_labels=labels,
        count_font_px=BASE_FONT_SIZE,
        legendgroup="Operational interpretation",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Correct-Draft Failure Mode Rate by Configuration [{experiment_label}]"),
        barmode="stack",
        legend_title="Verdict Basis",
        xaxis_title=_CONFIG_GROUPED_BY_MODEL_TITLE,
        height=_FAILURE_MODE_SINGLE_PANEL_HEIGHT + _AXIS_TITLE_ALLOWANCE,
        width=FIGURE_WIDTH,
        margin=dict(b=CONFIG_TICK_BOTTOM_MARGIN),
        uniformtext_minsize=BASE_FONT_SIZE,
        uniformtext_mode="show",
    )
    fig.update_xaxes(tickangle=CONFIG_TICK_ANGLE)
    fig.update_yaxes(
        title_text="Failure Mode Rate",
        range=[0, _rate_axis_max(
            _FAILURE_MODE_SINGLE_PANEL_HEIGHT
            - _PLOTLY_DEFAULT_TOP_MARGIN - CONFIG_TICK_BOTTOM_MARGIN,
            font_px=BASE_FONT_SIZE,
        )],
        tickvals=_RATE_AXIS_TICKVALS,
    )

    return fig


def build_failure_mode_rate_chart_incorrect(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Faceted stacked bar chart: incorrect-draft failure mode RATES by config.

    Two panels (Transparent, Opaque) with shared y-axis using normalized
    rates (0–1) for the incorrect-draft discriminating flag
    (``articulated_operational_interpretation``, the articulation
    classification): Expressed vs Never expressed. Rates
    are directly comparable across configurations with different mismatch
    totals.

    The y-axis keeps headroom above 1.0 for the outside mismatch-count
    labels while tick labels on both panels stay pinned to the [0, 1]
    rate domain.

    Configs render in the order the caller supplied, which the x-axis
    title names: balanced accuracy descending.

    A notebook analysis view; the paper's placed incorrect-draft failure
    mode figure is the count breakdown composite
    (``build_paired_failure_mode_incorrect``).
    """
    if not data:
        return go.Figure()

    labels = [short_config_label(r["config_key"]) for r in data]

    fig = _faceted_figure(
        cols=2, panel_titles=[p[0] for p in _INCORRECT_FAILURE_MODE_PANELS],
    )

    # Track legend entries already shown to avoid duplicates across panels
    _legend_shown: set[str] = set()

    for col_idx, (_, true_key, false_key, total_key) in enumerate(
        _INCORRECT_FAILURE_MODE_PANELS, start=1,
    ):
        # Bottom of stack: Never expressed rate (articulation = false)
        fig.add_trace(
            _failure_mode_rate_bar(
                "Never expressed",
                labels,
                [r.get(false_key) or 0.0 for r in data],
                config_labels=labels,
                showlegend="Never expressed" not in _legend_shown,
                legendgroup="Never expressed",
            ),
            row=1,
            col=col_idx,
        )
        _legend_shown.add("Never expressed")

        # Top of stack: Expressed rate (articulation = true), annotated with
        # the mismatch N
        fig.add_trace(
            _failure_mode_count_bar(
                "Expressed",
                labels,
                [r.get(true_key) or 0.0 for r in data],
                [r[total_key] for r in data],
                config_labels=labels,
                count_font_px=BASE_FONT_SIZE,
                showlegend="Expressed" not in _legend_shown,
                legendgroup="Expressed",
            ),
            row=1,
            col=col_idx,
        )
        _legend_shown.add("Expressed")

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Incorrect-Draft Failure Mode Rate by Configuration and Condition [{experiment_label}]"),
        barmode="stack",
        legend_title="Operational Interpretation",
        height=_FAILURE_MODE_FACETED_HEIGHT + _AXIS_TITLE_ALLOWANCE,
        # The every-tick canvas, like the count sibling: two 21-config
        # panels on the 1400 px canvas leave a tick pitch under Plotly's
        # label-drop threshold at the calibrated font.
        width=EVERY_TICK_FIGURE_WIDTH,
        margin=dict(b=CONFIG_TICK_BOTTOM_MARGIN),
        uniformtext_minsize=BASE_FONT_SIZE,
        uniformtext_mode="show",
    )
    fig.update_xaxes(tickangle=CONFIG_TICK_ANGLE)
    # Set on one facet only: an unqualified update_xaxes writes the title
    # onto every facet, and the copies overprint each other.
    fig.update_xaxes(
        title_text=_CONFIG_GROUPED_BY_MODEL_TITLE, row=1, col=1,
    )
    # Ticks are pinned on every y-axis, not just the label-bearing first one:
    # the second panel hides its labels but draws its own gridlines over the
    # matched range, and unpinned they would diverge from panel one's.
    fig.update_yaxes(tickvals=_RATE_AXIS_TICKVALS)
    fig.update_yaxes(
        title_text="Failure Mode Rate",
        range=[0, _rate_axis_max(
            _FAILURE_MODE_FACETED_HEIGHT
            - _PLOTLY_DEFAULT_TOP_MARGIN - CONFIG_TICK_BOTTOM_MARGIN,
            font_px=BASE_FONT_SIZE,
        )],
        row=1, col=1,
    )

    return fig


# Flat legend ranks for the paired failure mode composites, mirroring the
# per-experiment charts' declaration order: the bottom stack segment first.
_STACK_LEGEND_RANKS: dict[str, int] = {
    "Never expressed": 1,
    "Expressed": 2,
    "Field label matching": 1,
    "Operational interpretation": 2,
}


def build_paired_failure_mode_incorrect(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
) -> go.Figure:
    """Incorrect-draft failure mode counts for both experiments in one composite.

    Two condition rows (Transparent, Opaque) by two experiment columns on a
    shared structural configuration axis, replacing the PIL stitch of two
    per-experiment breakdown charts whose panels each carried their own
    order, legend, and tick band. The stacked bars come from the same
    factory as the per-experiment chart, so a column here equals the
    standalone chart mark-for-mark; each trace carries
    ``meta={"experiment_column": n}`` so tests can assert that equivalence.

    Chrome follows the flat-legend convention: one horizontal ungrouped
    legend at the top left, keeping the per-experiment chart's
    "Operational Interpretation" legend title; the right margin belongs to
    the condition row labels alone. The count scale is explicit and shared
    across every panel; counts are unbounded, so the ticks stay automatic.
    """
    order = structural_config_order(primary_data, ablation_data)
    positions, _ = grouped_config_positions(order)
    hover_labels = [short_config_label(ck) for ck in order]
    height = 1010
    vertical_spacing = 0.065
    plot_height = (
        height - _LANDSCAPE_MARGINS_PX["t"] - _LANDSCAPE_MARGINS_PX["b"]
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.035,
        column_titles=list(_EXPERIMENT_PANEL_TITLES),
        row_titles=[p[0] for p in _INCORRECT_FAILURE_MODE_COUNT_PANELS],
    )

    max_stack_total = 0
    shown: set[str] = set()
    for col, data in ((1, primary_data), (2, ablation_data)):
        by_config = {r["config_key"]: r for r in data}
        rows_in_order: list[dict[str, Any]] = [
            by_config.get(ck, {}) for ck in order
        ]
        for row, (_, true_key, false_key) in enumerate(
            _INCORRECT_FAILURE_MODE_COUNT_PANELS, start=1,
        ):
            max_stack_total = max(max_stack_total, max(
                (r.get(true_key) or 0) + (r.get(false_key) or 0)
                for r in rows_in_order
            ))
            fig.add_trace(
                _failure_mode_tally_bar(
                    "Never expressed",
                    positions,
                    [r.get(false_key) or 0 for r in rows_in_order],
                    config_labels=hover_labels,
                    width=BAR_SLOT_FRACTION,
                    showlegend="Never expressed" not in shown,
                    legendrank=_STACK_LEGEND_RANKS["Never expressed"],
                    meta={"experiment_column": col},
                ),
                row=row,
                col=col,
            )
            shown.add("Never expressed")

            fig.add_trace(
                _failure_mode_tally_bar(
                    "Expressed",
                    positions,
                    [r.get(true_key) or 0 for r in rows_in_order],
                    config_labels=hover_labels,
                    width=BAR_SLOT_FRACTION,
                    showlegend="Expressed" not in shown,
                    legendrank=_STACK_LEGEND_RANKS["Expressed"],
                    meta={"experiment_column": col},
                ),
                row=row,
                col=col,
            )
            shown.add("Expressed")

    fig.update_layout(
        template=FIGURE_TEMPLATE,
        font=dict(family=EXPORT_FONT_FAMILY, size=LANDSCAPE_BASE_FONT_SIZE),
        width=EVERY_TICK_FIGURE_WIDTH,
        height=height,
        barmode="stack",
        title=dict(
            text=(
                "Incorrect-Draft Failure Mode Breakdown by Configuration "
                "and Condition [Primary + Ablation]"
            ),
            font=dict(size=LANDSCAPE_EMPHASIS_FONT_SIZE),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        legend=dict(
            orientation="h", x=0.033, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
            title=dict(text="Operational Interpretation"),
            # Stacked barmode defaults traceorder to "reversed", which would
            # flip the ranked entries; the per-experiment chart's grouped
            # legend draws declaration order, and this pin preserves it.
            traceorder="normal",
        ),
        margin=dict(**_LANDSCAPE_MARGINS_PX),
    )
    # One explicit count scale across the panels (autorange collapses
    # against the two-tier helper's explicit x range), with headroom above
    # the tallest stack.
    fig.update_yaxes(range=[0, math.ceil(max_stack_total * 1.05)])
    fig.add_annotation(
        x=0, xref="paper", xanchor="right", xshift=-_Y_TITLE_SHIFT_PX,
        y=0.5, yref="paper", textangle=-90, showarrow=False,
        text=_WRONG_VERDICT_TRIALS_AXIS_TITLE.replace("<br>", " "),
        font=dict(size=LANDSCAPE_BASE_FONT_SIZE),
    )
    fig.update_annotations(font_size=LANDSCAPE_BASE_FONT_SIZE)
    _finish_structural_axis(
        fig, order,
        band_axis_refs=("x3", "x4"),
        plot_height=plot_height,
        font_px=LANDSCAPE_BASE_FONT_SIZE,
    )
    return fig


def build_paired_failure_mode_correct(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
) -> go.Figure:
    """Correct-draft failure mode counts for both experiments in one composite.

    Two experiment rows on a shared structural configuration axis, stacked
    governing-classification bars from the same factory as the
    per-experiment breakdown chart (each trace carries
    ``meta={"experiment_row": n}`` so tests can assert that equivalence),
    one flat horizontal legend keeping the per-experiment "Verdict Basis"
    title, and the two-tier leaf/band axis on the bottom row. The count
    scale is explicit and shared across the rows; counts are unbounded, so
    the ticks stay automatic.
    """
    order = structural_config_order(primary_data, ablation_data)
    positions, _ = grouped_config_positions(order)
    hover_labels = [short_config_label(ck) for ck in order]
    height = 1000
    vertical_spacing = 0.065
    plot_height = (
        height - _PORTRAIT_MARGINS_PX["t"] - _PORTRAIT_MARGINS_PX["b"]
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_titles=list(_EXPERIMENT_PANEL_TITLES),
    )

    max_stack_total = 0
    for row, data in ((1, primary_data), (2, ablation_data)):
        by_config = {r["config_key"]: r for r in data}
        rows_in_order: list[dict[str, Any]] = [
            by_config.get(ck, {}) for ck in order
        ]
        max_stack_total = max(max_stack_total, max(
            (r.get("correct_field_label") or 0)
            + (r.get("correct_governed") or 0)
            for r in rows_in_order
        ))
        fig.add_trace(
            _failure_mode_tally_bar(
                "Field label matching",
                positions,
                [r.get("correct_field_label") or 0 for r in rows_in_order],
                config_labels=hover_labels,
                width=BAR_SLOT_FRACTION,
                showlegend=row == 1,
                legendrank=_STACK_LEGEND_RANKS["Field label matching"],
                meta={"experiment_row": row},
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            _failure_mode_tally_bar(
                "Operational interpretation",
                positions,
                [r.get("correct_governed") or 0 for r in rows_in_order],
                config_labels=hover_labels,
                width=BAR_SLOT_FRACTION,
                showlegend=row == 1,
                legendrank=_STACK_LEGEND_RANKS["Operational interpretation"],
                meta={"experiment_row": row},
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(
            **_figure_title(
                "Correct-Draft Failure Mode Breakdown by Configuration "
                "[Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        legend=dict(
            orientation="h", x=0, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
            title=dict(text="Verdict Basis"),
            # Stacked barmode defaults traceorder to "reversed", which would
            # flip the ranked entries; the per-experiment chart's grouped
            # legend draws declaration order, and this pin preserves it.
            traceorder="normal",
        ),
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(**_PORTRAIT_MARGINS_PX),
    )
    # One explicit count scale across the rows (autorange collapses against
    # the two-tier helper's explicit x range), with headroom above the
    # tallest stack.
    fig.layout.yaxis2.matches = "y"
    fig.update_yaxes(range=[0, math.ceil(max_stack_total * 1.05)])
    fig.add_annotation(
        x=0, xref="paper", xanchor="right", xshift=-_Y_TITLE_SHIFT_PX,
        y=0.5, yref="paper", textangle=-90, showarrow=False,
        text=_WRONG_VERDICT_TRIALS_AXIS_TITLE.replace("<br>", " "),
        font=dict(size=BASE_FONT_SIZE),
    )
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    _finish_structural_axis(
        fig, order,
        band_axis_refs=("x2",),
        plot_height=plot_height,
        font_px=BASE_FONT_SIZE,
    )
    return fig


_FAILURE_MODE_DOMINANCE_LABELS: dict[str, tuple[str, str]] = {
    "correct": ("Field label matching", "Operational interpretation"),
    "transparent": ("Never expressed", "Expressed"),
    "opaque": ("Never expressed", "Expressed"),
}
"""Condition-specific (negative-side, positive-side) mode labels.

The negative side corresponds to ``dominance == -1`` and the positive side to
``dominance == +1``. On correct-draft the discriminating flag is
``operational_interpretation_governed_judgment``; on transparent and opaque
stimuli it is ``articulated_operational_interpretation`` (see
``failure_mode_dominance_matrix``).
"""


def _dominance_heatmap_traces(
    data: list[dict[str, Any]],
    *,
    value_column: str,
    config_order: list[str],
    showscale: bool = True,
) -> tuple[go.Heatmap, go.Heatmap]:
    """The dominance matrix trace pair: gap-cell background, data foreground.

    Validates alignment and builds both traces — the uniform light-gray
    background revealed through the foreground's ``None`` cells (zero
    classifiable failures), and the diverging blue–gray–orange dominance
    matrix on the fixed [-1, 1] scale with the condition's mode labels on
    the colorbar — so the paired composite and the per-experiment chart
    cannot diverge cell-for-cell. Both traces sit on the gutter column
    geometry and go ``None`` in the gutter columns: a background value
    there would render the gutter in the gap-cell gray and read as a
    column of zero-failure cells. Config labels ride ``customdata`` (the
    foreground's alongside its count payload) because the numeric column
    positions would otherwise surface in hover through ``%{x}``.
    """
    validate_heatmap_alignment_inputs(data, config_order=config_order)

    condition = value_column.replace("_dominance", "")
    neg_label, pos_label = _FAILURE_MODE_DOMINANCE_LABELS.get(
        condition, ("Negative", "Positive"),
    )
    total_column = f"{condition}_total"
    positive_column = f"{condition}_positive"
    negative_column = f"{condition}_negative"

    examples_sorted = sorted_base_examples()
    column_xs, x_to_config = _heatmap_column_layout(list(config_order))
    example_labels = list(examples_sorted)

    # Per-cell lookups keyed by (base_example, config_key)
    dominance_lookup = {
        (r["base_example"], r["config_key"]): r.get(value_column)
        for r in data
    }
    total_lookup = {
        (r["base_example"], r["config_key"]): r.get(total_column, 0)
        for r in data
    }
    positive_lookup = {
        (r["base_example"], r["config_key"]): r.get(positive_column, 0)
        for r in data
    }
    negative_lookup = {
        (r["base_example"], r["config_key"]): r.get(negative_column, 0)
        for r in data
    }

    # Build z-matrix; cells with total=0 become None (transparent → reveals
    # background), and gutter columns are None outright.
    z_matrix = [
        [
            dominance_lookup.get((ex, x_to_config[x]))
            if x in x_to_config
            and total_lookup.get((ex, x_to_config[x]), 0) > 0
            else None
            for x in column_xs
        ]
        for ex in examples_sorted
    ]

    # Custom hover data per real cell:
    # (config label, total, negative_count, positive_count, neg_pct, pos_pct)
    def _cell_payload(ex: str, x: int) -> tuple | None:
        if x not in x_to_config:
            return None
        ck = x_to_config[x]
        total = total_lookup.get((ex, ck), 0)
        negative = negative_lookup.get((ex, ck), 0)
        positive = positive_lookup.get((ex, ck), 0)
        return (
            short_config_label(ck),
            total,
            negative,
            positive,
            negative / total if total > 0 else 0.0,
            positive / total if total > 0 else 0.0,
        )

    custom_data = [
        [_cell_payload(ex, x) for x in column_xs]
        for ex in examples_sorted
    ]
    label_row = [
        short_config_label(x_to_config[x]) if x in x_to_config else None
        for x in column_xs
    ]

    # Diverging blue ← gray → orange (colorblind-accessible)
    _DIVERGING_COLORSCALE = [
        [0.0, "#2563eb"],
        [0.5, "#d1d5db"],
        [1.0, "#ea580c"],
    ]

    # Background trace: uniform light gray for gap cells (total=0 / no RA
    # failures); None in the gutter columns keeps the gutters blank.
    bg_trace = go.Heatmap(
        z=[
            [1 if x in x_to_config else None for x in column_xs]
        ] * len(examples_sorted),
        x=column_xs,
        y=example_labels,
        customdata=[label_row for _ in examples_sorted],
        colorscale=[[0, "#f3f4f6"], [1, "#f3f4f6"]],
        showscale=False,
        hoverongaps=False,
        hovertemplate=(
            "Config: %{customdata}<br>"
            "Example: %{y}<br>"
            "No Rationale Analysis-eligible failures"
            "<extra></extra>"
        ),
    )

    # Foreground data trace: None cells are transparent, revealing the background
    fg_trace = go.Heatmap(
        z=z_matrix,
        x=column_xs,
        y=example_labels,
        customdata=custom_data,
        colorscale=_DIVERGING_COLORSCALE,
        zmid=0,
        zmin=-1,
        zmax=1,
        xgap=0,
        ygap=0,
        hoverongaps=False,
        showscale=showscale,
        hovertemplate=(
            "Config: %{customdata[0]}<br>"
            "Example: %{y}<br>"
            "Dominance: %{z:+.2f}<br>"
            "Total failures: %{customdata[1]}<br>"
            f"{neg_label}: "
            "%{customdata[2]} (%{customdata[4]:.0%})<br>"
            f"{pos_label}: "
            "%{customdata[3]} (%{customdata[5]:.0%})"
            "<extra></extra>"
        ),
        colorbar=dict(
            tickvals=[-1, 0, 1],
            ticktext=[neg_label, "Even split", pos_label],
        ),
    )

    return bg_trace, fg_trace


def build_failure_mode_dominance_heatmap(
    data: list[dict[str, Any]],
    *,
    value_column: str = "transparent_dominance",
    config_order: list[str],
    experiment_label: str,
) -> go.Figure:
    """Heatmap: base_example (y-axis) x config_key (x-axis) with normalized failure mode dominance.

    Shares ``build_example_condition_accuracy_heatmap``'s fixed row order, so a given
    row is the same base example in both figures and readers can
    cross-reference condition accuracy with failure mechanism.

    Each cell plots the within-cell normalized signed dominance produced by
    ``failure_mode_dominance_matrix``: a value in ``[-1, 1]`` where ``-1``
    means the negative-side failure mode accounts for every classifiable
    mismatch, ``+1`` means the positive-side mode accounts for every
    classifiable mismatch, and ``0`` means an even split. The color scale
    is fixed to ``[-1, 1]`` so dominance is directly comparable across
    conditions and across primary vs ablation exports.

    Condition-specific meanings (from ``_FAILURE_MODE_DOMINANCE_LABELS``):

    - transparent / opaque:
        - negative side: "Never expressed"
        - positive side: "Expressed"
    - correct:
        - negative side: "Field label matching"
        - positive side: "Operational interpretation"

    Blue = negative-side mode dominant. Orange = positive-side mode dominant.
    Midpoint gray (#d1d5db) = balanced split among observed failures.

    Cells with zero total classifiable failures render in light gray
    (#f3f4f6) with hover text "No Rationale Analysis-eligible failures",
    visually distinct from a true ``0.0`` dominance split at the midpoint.

    Parameters
    ----------
    config_order:
        The column order and universe, shared across primary and ablation
        renders so paired figures are axis-aligned cell-for-cell — pass
        :func:`structural_config_order` over both experiments' rows. Strict
        validation: every config key present in ``data`` must also be in the
        supplied order, otherwise :func:`validate_heatmap_alignment_inputs`
        raises ``ValueError``. Order entries with no data rows render as the
        gap-cell background.
    """
    if not data:
        return go.Figure()

    condition = value_column.replace("_dominance", "")
    fig = go.Figure(data=list(_dominance_heatmap_traces(
        data, value_column=value_column, config_order=config_order,
    )))

    condition_label = _CONDITION_LEGEND_NAMES[condition]

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Failure Mode Dominance ({condition_label} Condition) [{experiment_label}]"),
        xaxis_title=_CONFIG_GROUPED_BY_MODEL_TITLE,
        yaxis_title=_BASE_EXAMPLE_AXIS_TITLE,
        yaxis_autorange="reversed",
        xaxis_tickangle=CROWDED_CONFIG_TICK_ANGLE,
        height=HEATMAP_FIGURE_HEIGHT,
        width=FIGURE_WIDTH,
        margin=dict(b=CROWDED_CONFIG_BOTTOM_MARGIN),
    )
    _tick_heatmap_columns_with_config_labels(fig, config_order)

    return fig


def build_paired_failure_mode_dominance_heatmap(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
    *,
    value_column: str = "transparent_dominance",
    config_order: list[str],
) -> go.Figure:
    """Failure mode dominance for both experiments in one portrait composite.

    Two experiment matrices stacked on a shared structural configuration
    axis (gutter-separated columns, two-tier labels on the bottom row)
    with one
    diverging colorbar serving both, each matrix's background and
    foreground from the same trace core as the per-experiment chart; each
    trace carries ``meta={"experiment_row": n}`` so tests can assert that
    equivalence. ``config_order`` is the shared column order and universe
    for both experiments.
    """
    height = 1150
    fig, plot_height = _paired_heatmap_scaffold(height=height)

    for row, data in ((1, primary_data), (2, ablation_data)):
        for trace in _dominance_heatmap_traces(
            data, value_column=value_column, config_order=config_order,
            showscale=row == 1,
        ):
            trace.meta = {"experiment_row": row}
            fig.add_trace(trace, row=row, col=1)

    condition_label = _CONDITION_LEGEND_NAMES[value_column.replace("_dominance", "")]
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            **_figure_title(
                f"Failure Mode Dominance ({condition_label} Condition) "
                "[Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(
            t=_PORTRAIT_MARGINS_PX["t"], b=_PORTRAIT_MARGINS_PX["b"],
        ),
    )
    fig.update_yaxes(title_text=_BASE_EXAMPLE_AXIS_TITLE, autorange="reversed")
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    _finish_structural_axis(
        fig, list(config_order),
        band_axis_refs=("x2",),
        plot_height=plot_height,
        font_px=BASE_FONT_SIZE,
        heatmap=True,
    )
    return fig


def build_ablation_comparison_chart(
    data: list[dict[str, Any]],
    significance_data: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """Bar chart showing delta between ablation and primary balanced accuracy.

    Positive delta = ablation performs better. Configs grouped by model and
    ordered by reasoning effort level within each model, labeled by the
    two-tier axis: effort leaf ticks under model band annotations, with the
    structural ordering disclosed by the centered title annotation below.

    When ``significance_data`` is provided (config-level cross-dataset
    comparison rows), bars for significant comparisons are annotated with an
    asterisk (*) riding the bar's grouped numeric position. The delta axis
    range is explicit — data-derived with asterisk ink room past the
    extremes — so the print scale never leans on renderer autorange.
    """
    if not data:
        return go.Figure()

    # Build a lookup from config_key → significant for annotation
    sig_lookup: dict[str, bool] = {}
    if significance_data:
        for r in significance_data:
            sig_lookup[r["config_key"]] = r.get("significant", False)

    config_keys = [r["config_key"] for r in data]
    positions, _ = grouped_config_positions(config_keys)
    hover_labels = [short_config_label(ck) for ck in config_keys]
    colors = [
        "#22c55e" if r["delta"] >= 0 else "#ef4444" for r in data
    ]

    # The two-line title (wrapped ahead of the bracket token, which would
    # overrun the canvas at the emphasis size on one line) deepens the top
    # margin past the portrait default; the height grows to match so the
    # plot area is not the stratum that pays for it.
    height = 660
    top_margin = 200
    plot_height = height - top_margin - _PORTRAIT_MARGINS_PX["b"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=positions,
        y=[r["delta"] for r in data],
        marker_color=colors,
        width=BAR_SLOT_FRACTION,
        showlegend=False,
        hovertemplate=(
            "Config: %{customdata}<br>"
            "Delta: %{y:+.3f}<br>"
            "<extra></extra>"
        ),
        customdata=hover_labels,
    ))

    # Annotate significant bars with an asterisk and add a legend entry
    _has_sig = False
    for i, r in enumerate(data):
        if sig_lookup.get(r["config_key"], False):
            _has_sig = True
            fig.add_annotation(
                x=positions[i],
                y=r["delta"],
                text="*",
                showarrow=False,
                font=dict(size=EMPHASIS_FONT_SIZE, color="#1a202c"),
                yshift=10 if r["delta"] >= 0 else -14,
            )
    if _has_sig:
        # Invisible trace to place an explanatory legend entry
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="asterisk", size=10, color="#1a202c"),
            showlegend=True,
            name="* Significant (e-Bonferroni corrected)",
            hoverinfo="skip",
        ))

    fig.add_hline(y=0, line_color="gray", line_width=1)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            **_figure_title(
                "Ablation Effect (Ablation Experiment − Primary Experiment "
                "Balanced Accuracy)<br>[Primary vs. Ablation]"
            ),
            # Anchored by its middle: a top anchor mis-places multi-line
            # titles (the first line renders above the canvas edge —
            # probed against exported ink extents).
            x=0.033, xanchor="left", y=0.94, yanchor="middle",
        ),
        yaxis_title=_ABLATION_BALANCED_ACCURACY_DELTA_AXIS_TITLE,
        legend=dict(
            orientation="h", x=0, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
        ),
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(**{**_PORTRAIT_MARGINS_PX, "t": top_margin}),
    )
    # Explicit print scale: every delta plus asterisk ink room, priced in
    # this figure's plot pixels.
    deltas = [r["delta"] for r in data]
    delta_min = min(min(deltas), 0.0)
    delta_max = max(max(deltas), 0.0)
    data_per_px = (delta_max - delta_min) / (
        plot_height - 2 * _DELTA_ASTERISK_RESERVE_PX
    )
    fig.update_yaxes(range=[
        delta_min - _DELTA_ASTERISK_RESERVE_PX * data_per_px,
        delta_max + _DELTA_ASTERISK_RESERVE_PX * data_per_px,
    ])
    _finish_structural_axis(
        fig, config_keys,
        band_axis_refs=("x",),
        plot_height=plot_height,
        font_px=BASE_FONT_SIZE,
    )

    return fig


def build_condition_level_ablation_effect_chart(
    data: list[dict[str, Any]],
) -> go.Figure:
    """Stacked 3x1 bar chart of the ablation effect on per-condition accuracy.

    One row per condition (correct-draft, transparent, opaque) on a shared
    structural configuration axis, sized for a landscape page. Within each
    row, one bar per configuration shows the accuracy delta (ablation
    experiment minus primary experiment) for that condition, filled green
    when the ablation experiment improved accuracy and red when it degraded.
    Configurations flagged as significant under the condition-level
    cross-dataset test are marked with an asterisk riding the bar's grouped
    numeric position. Configurations keep the caller's order across all
    three rows; the structural sort lives in the shaper.

    Consumes condition-level ``cross_dataset_significance_data`` rows. Raises
    ``ValueError`` on any row without a per-condition label (config-level rows,
    ``condition`` None, belong to the balanced accuracy ablation effect chart) and
    on any configuration missing from a condition, since real input covers every
    configuration in every condition and a gap would render an incomplete panel.
    """
    _valid_conditions = {condition for _, condition in _CONDITION_PANELS}

    for row in data:
        if row["condition"] not in _valid_conditions:
            raise ValueError(
                "build_condition_level_ablation_effect_chart requires per-condition "
                f"rows; received condition={row['condition']!r} for config "
                f"{row.get('config_key')!r}. Config-level rows (condition None) "
                "belong to the balanced accuracy ablation effect chart."
            )

    if not data:
        return go.Figure()

    # Unique config keys in the caller's structural order.
    seen: set[str] = set()
    config_keys: list[str] = []
    for row in data:
        if row["config_key"] not in seen:
            seen.add(row["config_key"])
            config_keys.append(row["config_key"])
    positions, _ = grouped_config_positions(config_keys)
    hover_labels = [short_config_label(ck) for ck in config_keys]

    # condition -> config_key -> row, for O(1) per-row lookup.
    rows_by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for row in data:
        rows_by_condition.setdefault(row["condition"], {})[row["config_key"]] = row

    # Every configuration must appear in every condition; a gap would silently
    # drop a bar and leave a phantom color-list entry.
    for _, condition in _CONDITION_PANELS:
        panel = rows_by_condition.get(condition, {})
        for ck in config_keys:
            if ck not in panel:
                raise ValueError(
                    "build_condition_level_ablation_effect_chart requires every "
                    f"configuration in every condition; missing config {ck!r} in "
                    f"condition {condition!r}."
                )

    height = 1150
    vertical_spacing = 0.065
    plot_height = (
        height - _LANDSCAPE_MARGINS_PX["t"] - _LANDSCAPE_MARGINS_PX["b"]
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_titles=[
            title.replace(" (", "<br>(") for title, _ in _CONDITION_PANELS
        ],
    )

    _has_significant = False
    for row_idx, (_, condition) in enumerate(_CONDITION_PANELS, start=1):
        panel_rows = rows_by_condition[condition]
        deltas = [panel_rows[ck]["delta"] for ck in config_keys]
        colors = ["#22c55e" if delta >= 0 else "#ef4444" for delta in deltas]

        fig.add_trace(
            go.Bar(
                x=positions,
                y=deltas,
                customdata=hover_labels,
                marker_color=colors,
                width=BAR_SLOT_FRACTION,
                showlegend=False,
                hovertemplate=(
                    "Config: %{customdata}<br>"
                    f"Condition: {condition}<br>"
                    "Delta (Ablation Experiment − Primary Experiment): %{y:+.3f}"
                    "<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )

        for i, ck in enumerate(config_keys):
            row = panel_rows[ck]
            if row.get("significant"):
                _has_significant = True
                delta = row["delta"]
                fig.add_annotation(
                    x=positions[i],
                    y=delta,
                    text="*",
                    showarrow=False,
                    font=dict(
                        size=LANDSCAPE_EMPHASIS_FONT_SIZE, color="#1a202c",
                    ),
                    yshift=10 if delta >= 0 else -14,
                    row=row_idx,
                    col=1,
                )

        fig.add_hline(y=0, line_color="gray", line_width=1, row=row_idx, col=1)

    if _has_significant:
        # Invisible trace to place an explanatory legend entry.
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="asterisk", size=10, color="#1a202c"),
                showlegend=True,
                name="* Significant (e-Bonferroni corrected)",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        template=FIGURE_TEMPLATE,
        font=dict(family=EXPORT_FONT_FAMILY, size=LANDSCAPE_BASE_FONT_SIZE),
        width=EVERY_TICK_FIGURE_WIDTH,
        height=height,
        title=dict(
            text=(
                "Ablation Effect (Ablation Experiment − Primary Experiment "
                "Per-Condition Accuracy) [Primary vs. Ablation]"
            ),
            font=dict(size=LANDSCAPE_EMPHASIS_FONT_SIZE),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        legend=dict(
            orientation="h", x=0.033, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
        ),
        margin=dict(**_LANDSCAPE_MARGINS_PX),
    )
    # The three delta rows read against one scale, so the upper rows match
    # the first row's range the way the old side-by-side panels shared y.
    fig.layout.yaxis2.matches = "y"
    fig.layout.yaxis3.matches = "y"
    # The scale itself is explicit: y autorange collapses to a (-1, 4)
    # fallback when matched y axes meet the two-tier helper's explicit x
    # range, and a print figure should not lean on renderer quirks anyway.
    # The range holds every delta plus ink room for the significance
    # asterisks beyond the bar tips, priced in this figure's row pixels.
    deltas_all = [row["delta"] for row in data]
    delta_min = min(min(deltas_all), 0.0)
    delta_max = max(max(deltas_all), 0.0)
    row_gap_px = vertical_spacing * plot_height
    row_plot_height = (plot_height - 2 * row_gap_px) / 3
    data_per_px = (delta_max - delta_min) / (
        row_plot_height - 2 * _DELTA_ASTERISK_RESERVE_PX
    )
    fig.update_yaxes(range=[
        delta_min - _DELTA_ASTERISK_RESERVE_PX * data_per_px,
        delta_max + _DELTA_ASTERISK_RESERVE_PX * data_per_px,
    ])
    fig.update_yaxes(
        title_text=_ABLATION_CONDITION_ACCURACY_DELTA_AXIS_TITLE,
        row=2, col=1,
    )
    # Resize the subplot chrome annotations, sparing the asterisks their
    # emphasis size.
    for annotation in fig.layout.annotations:
        if annotation.text != "*":
            annotation.font.size = LANDSCAPE_BASE_FONT_SIZE
    _finish_structural_axis(
        fig, config_keys,
        band_axis_refs=("x3",),
        plot_height=plot_height,
        font_px=LANDSCAPE_BASE_FONT_SIZE,
    )

    return fig


def _add_cost_scatter_panel(
    fig: go.Figure,
    data: list[dict[str, Any]],
    *,
    row: int | None = None,
    show_legend: bool = True,
    meta: dict[str, Any] | None = None,
    chart_width: int = 1000,
    chart_height: int = 420,
) -> None:
    """Provider markers, the Pareto frontier, and point labels for one panel.

    ``row`` targets a subplot row on a one-column grid; ``None`` draws onto
    a plain figure. Every mark and label is built here so the paired
    composite and the per-experiment chart cannot diverge point-for-point:
    provider colors, the dashed frontier over cost-efficient configs, and
    the greedily placed per-point labels (bolder ink on frontier points).
    ``chart_width`` and ``chart_height`` describe this panel's plot area in
    pixels for the label placer's collision model; the defaults are the
    per-experiment chart's, and a composite must pass its own row geometry
    or labels sited clear in model space can overprint on the real canvas.
    """
    subplot_kwargs: dict[str, Any] = (
        dict(row=row, col=1) if row is not None else {}
    )

    # Group by provider for color coding, filtering null costs
    providers: dict[str, list[dict[str, Any]]] = {}
    for data_row in data:
        if data_row["cost_per_trial"] is None:
            continue
        providers.setdefault(data_row["provider"], []).append(data_row)

    # Track all plotted points for Pareto frontier computation
    all_points: list[tuple[float, float]] = []
    all_rows: list[dict[str, Any]] = []

    for provider, rows in sorted(providers.items()):
        color = PROVIDER_COLORS.get(provider, "#6b7280")
        provider_label = display_provider(provider)
        fig.add_trace(go.Scatter(
            name=provider_label,
            x=[r["cost_per_trial"] for r in rows],
            y=[r["balanced_accuracy"] for r in rows],
            mode="markers",
            marker=dict(color=color, size=12, opacity=0.8),
            cliponaxis=False,
            showlegend=show_legend,
            meta=meta,
            hovertemplate=(
                "%{customdata[0]}<br>"
                "Provider: %{customdata[1]}<br>"
                "Cost per Trial: $%{x:.4f}<br>"
                "Balanced Accuracy: %{y:.3f}<extra></extra>"
            ),
            customdata=[
                [short_config_label(r["config_key"]), provider_label]
                for r in rows
            ],
        ), **subplot_kwargs)
        for r in rows:
            all_points.append((r["cost_per_trial"], r["balanced_accuracy"]))
            all_rows.append(r)

    # Pareto frontier: connect cost-efficient configs with a dashed line
    frontier_indices = _compute_pareto_frontier(all_points)
    if len(frontier_indices) >= 2:
        frontier_sorted = sorted(
            frontier_indices, key=lambda i: all_points[i][0],
        )
        fig.add_trace(go.Scatter(
            x=[all_points[i][0] for i in frontier_sorted],
            y=[all_points[i][1] for i in frontier_sorted],
            mode="lines",
            line=dict(color="#9ca3af", width=1.5, dash="dash"),
            showlegend=show_legend,
            meta=meta,
            name="Pareto frontier",
            hoverinfo="skip",
        ), **subplot_kwargs)

    # Label every point — frontier labels are bolder, non-frontier subdued.
    # Greedy placement spreads labels to minimize overlap in dense clusters.
    # log₁₀ transform on x is required because Plotly annotations use
    # axis-native coordinates on log-scale axes.
    all_labels = [
        short_config_label(r["config_key"]) for r in all_rows
    ]
    offsets = _place_annotations(
        all_points, all_labels,
        chart_width=chart_width, chart_height=chart_height,
    )
    frontier_set = set(frontier_indices)

    for i, (ax, ay) in enumerate(offsets):
        is_frontier = i in frontier_set
        fig.add_annotation(
            x=math.log10(all_points[i][0]),
            y=all_points[i][1],
            text=all_labels[i],
            showarrow=True,
            arrowhead=0,
            arrowwidth=1.5 if is_frontier else 0.75,
            arrowcolor="#9ca3af" if is_frontier else "#d1d5db",
            ax=ax,
            ay=ay,
            # Frontier labels are set apart by ink, not by size: both tiers sit
            # at the base size because the legibility floor binds them equally.
            font=dict(
                size=BASE_FONT_SIZE,
                color="#374151" if is_frontier else "#9ca3af",
            ),
            xanchor="center",
            **subplot_kwargs,
        )


# The cost axes label a 1-2-5 dollar ladder instead of Plotly's automatic
# log ticks: the automatic mode prints unit-free digits at 75% font beside
# full-size decade labels it nudges off their gridlines, and at the deck's
# calibrated font the two collide outright.
_COST_TICK_MANTISSAS: tuple[int, ...] = (1, 2, 5)
# A cost sitting on (or hugging) the outer rung would straddle the plot
# edge, so the range steps one rung further out whenever the nearest rung
# leaves less air than a marker's half-width (~0.02 decades at the deck
# geometry, doubled for comfort).
_COST_AXIS_MIN_EDGE_AIR_DECADES: float = 0.04
# Drops the tick-label row clear of the y-axis "0", whose glyph dips below
# the panel bottom into the same band; measured at 12 px the two ink runs
# separate by about 10 px where at 0 px they merge.
COST_TICK_LABEL_STANDOFF_PX: int = 12


def cost_log_axis(costs: list[float]) -> dict[str, Any]:
    """Log cost-axis spec snapped to a labeled 1-2-5 dollar ladder.

    The range runs from the ladder rung below the cheapest cost to the rung
    above the costliest, so every marker has a labeled anchor on both
    sides. Rungs render through ``format_cost_per_trial`` — the same $.4f
    as the cost table's Cost per Trial ($) cells — and the digit positions
    between them stay drawn as unlabeled minor ticks and gridlines, keeping
    the log-scale texture the automatic labels carried.
    """
    if not costs:
        raise ValueError(
            "cost_log_axis needs at least one cost to place the ladder"
        )
    lo_value, hi_value = min(costs), max(costs)
    if lo_value <= 0:
        raise ValueError(
            "cost_log_axis needs strictly positive costs for its log "
            f"ladder; got {lo_value}"
        )
    rungs = sorted(
        m * 10.0 ** k
        for k in range(
            math.floor(math.log10(lo_value)) - 1,
            math.floor(math.log10(hi_value)) + 2,
        )
        for m in _COST_TICK_MANTISSAS
    )
    lo_idx = max(i for i, r in enumerate(rungs) if r <= lo_value)
    hi_idx = min(i for i, r in enumerate(rungs) if r >= hi_value)
    if math.log10(lo_value / rungs[lo_idx]) < _COST_AXIS_MIN_EDGE_AIR_DECADES:
        lo_idx -= 1
    if math.log10(rungs[hi_idx] / hi_value) < _COST_AXIS_MIN_EDGE_AIR_DECADES:
        hi_idx += 1
    tickvals = rungs[lo_idx:hi_idx + 1]
    ticktext = [format_cost_per_trial(v) for v in tickvals]
    if "$0.0000" in ticktext or len(set(ticktext)) != len(ticktext):
        raise ValueError(
            "the dollar ladder outgrew the four-decimal Cost per Trial "
            f"($) format and cannot resolve every rung: {ticktext}"
        )
    return dict(
        type="log",
        range=[math.log10(tickvals[0]), math.log10(tickvals[-1])],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        ticklabelstandoff=COST_TICK_LABEL_STANDOFF_PX,
        showgrid=True,
        gridcolor="#e5e7eb",   # gray-200, the deck's major gridline hue
        minor=dict(
            dtick="D1",
            showgrid=True,
            gridcolor="#f1f5f9",   # slate-100, one step fainter than major
            ticks="outside",
            ticklen=4,
            tickcolor="#cbd5e1",   # slate-300
        ),
    )


def build_cost_vs_balanced_accuracy_scatter(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Scatter plot: cost per trial (log scale) vs balanced accuracy.

    Each point is one config, colored by provider. Hover shows model
    and reasoning effort level via short_config_label. A dashed Pareto
    frontier (minimize cost, maximize score) highlights cost-efficient
    configs, with per-point annotation labels.
    """
    fig = go.Figure()
    _add_cost_scatter_panel(fig, data)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Cost per Trial vs. Balanced Accuracy [{experiment_label}]"),
        xaxis_title="Cost per Trial (USD, log scale)",
        yaxis_title="Balanced Accuracy",
        yaxis_range=[0, 1.05],
        legend_title="Provider",
        height=550,
        width=FIGURE_WIDTH,
        margin=dict(t=50),
    )
    costs = [
        r["cost_per_trial"] for r in data if r["cost_per_trial"] is not None
    ]
    # An all-null cost view draws no markers; the empty chart keeps a bare
    # log axis rather than demanding a ladder with nothing to anchor it.
    fig.update_xaxes(**(cost_log_axis(costs) if costs else {"type": "log"}))

    return fig


def build_paired_cost_vs_balanced_accuracy(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
) -> go.Figure:
    """Cost vs balanced accuracy for both experiments, stacked.

    Two experiment rows over one shared log cost axis snapped to the
    dollar-ladder rungs bracketing both experiments' costs (see
    ``cost_log_axis``), so the rows read against one labeled domain. Each row's markers, frontier, and point labels come from the
    same core as the per-experiment chart (``meta={"experiment_row": n}``
    on every trace); one flat horizontal legend keeps the per-experiment
    "Provider" title with the frontier declared once. The height is the
    harness-tuned value: at the budgeted 1140 the greedy label placer ran
    out of clear rows and crossed leader lines through both panels.
    """
    height = 1400
    top_margin = _PORTRAIT_MARGINS_PX["t"]
    bottom_margin = 100
    vertical_spacing = 0.08
    plot_height = height - top_margin - bottom_margin

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_titles=list(_EXPERIMENT_PANEL_TITLES),
    )
    # Each row's true plot geometry feeds the label placer's collision
    # model: the panel width is the canvas less the left margin and the
    # right band the rotated row titles automargin for themselves, and each
    # row takes its share of the plot height less the inter-row gap.
    panel_width = FIGURE_WIDTH - _PORTRAIT_MARGINS_PX["l"] - 125
    row_height = round(plot_height * (1 - vertical_spacing) / 2)
    for row, data in ((1, primary_data), (2, ablation_data)):
        _add_cost_scatter_panel(
            fig, data,
            row=row,
            show_legend=row == 1,
            meta={"experiment_row": row},
            chart_width=panel_width,
            chart_height=row_height,
        )

    costs = [
        r["cost_per_trial"]
        for r in [*primary_data, *ablation_data]
        if r["cost_per_trial"] is not None
    ]
    # Pin the shared log domain to the ladder rungs bracketing the union,
    # so both rows read against one labeled dollar scale.
    fig.update_xaxes(**cost_log_axis(costs))
    fig.update_xaxes(
        title_text="Cost per Trial (USD, log scale)", row=2, col=1,
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            **_figure_title(
                "Cost per Trial vs. Balanced Accuracy [Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        legend=dict(
            orientation="h", x=0, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
            title=dict(text="Provider"),
        ),
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(
            t=top_margin, b=bottom_margin, l=_PORTRAIT_MARGINS_PX["l"],
        ),
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.add_annotation(
        x=0, xref="paper", xanchor="right", xshift=-_Y_TITLE_SHIFT_PX,
        y=0.5, yref="paper", textangle=-90, showarrow=False,
        text="Balanced Accuracy",
        font=dict(size=BASE_FONT_SIZE),
    )
    for annotation in fig.layout.annotations:
        if not annotation.showarrow:
            annotation.font.size = BASE_FONT_SIZE
    return fig


# Stack colors for the token composition charts, bottom segment first.
_TOKEN_STACK_COLORS: dict[str, str] = {
    "Input": "#94a3b8",      # slate-400 (roughly constant across levels)
    "Reasoning": "#f59e0b",  # amber-500 (the variable component)
    "Response": "#3b82f6",   # blue-500 (top of stack for readability)
}


def _token_stack_values(
    data: list[dict[str, Any]],
) -> list[tuple[str, list[float]]]:
    """The three stack segments' values in bottom-to-top order."""
    return [
        ("Input", [r.get("mean_input_tokens") or 0 for r in data]),
        ("Reasoning", [r.get("mean_reasoning_tokens") or 0 for r in data]),
        ("Response", [r.get("mean_response_tokens") or 0 for r in data]),
    ]


def _token_stack_bar(
    name: str,
    x: list[Any],
    values: list[float],
    config_labels: list[str],
    **legend: Any,
) -> go.Bar:
    """A token-composition stack segment: values, color, and hover.

    ``config_labels`` ride ``customdata`` so the hover names the
    configuration even when ``x`` holds numeric grouped positions.
    """
    return go.Bar(
        name=name,
        x=x,
        y=values,
        customdata=config_labels,
        marker_color=_TOKEN_STACK_COLORS[name],
        hovertemplate=(
            "Config: %{customdata}<br>"
            f"{name}: %{{y:,.0f}} tokens/trial<extra></extra>"
        ),
        **legend,
    )


def build_token_composition_chart(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Stacked bar chart: input/response/reasoning tokens per configuration.

    Grouped by model. Within each model, bars ordered by reasoning effort level.
    Dotted vertical lines separate model groups for visual clarity.
    """
    if not data:
        return go.Figure()

    # One bar per configuration, so these are configuration labels like every
    # other per-configuration chart's. The aliasing form belongs to charts whose
    # x-axis is a shared reasoning effort level category; applied here it would
    # print GPT-5.2's "none" as "off", which is Claude Haiku 4.5's own level.
    labels = [short_config_label(r["config_key"]) for r in data]

    fig = go.Figure()
    for name, values in _token_stack_values(data):
        fig.add_trace(_token_stack_bar(
            name, labels, values, config_labels=labels,
        ))

    # Add dotted vertical lines between model groups for visual grouping
    current_model: str | None = None
    for i, r in enumerate(data):
        if r["model_slug"] != current_model:
            if current_model is not None:
                fig.add_vline(
                    x=i - 0.5,
                    line_dash="dot",
                    line_color="#d1d5db",
                    line_width=1,
                )
            current_model = r["model_slug"]

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Token Composition by Configuration [{experiment_label}]"),
        barmode="stack",
        xaxis_title=_CONFIG_GROUPED_BY_MODEL_TITLE,
        yaxis_title="Mean Tokens per Trial",
        yaxis_tickformat=",d",
        legend_title="Token Type",
        xaxis_tickangle=CONFIG_TICK_ANGLE,
        # A y-axis title is bounded by the plot area's height, and at 500 this
        # one overruns it at both ends, colliding with the figure title above
        # and the tick labels below. Height is the lever: the bottom margin
        # already clears the x-axis title with room to spare, and raising it
        # would shorten the plot area and worsen the overrun.
        height=_TOKEN_COMPOSITION_HEIGHT,
        width=FIGURE_WIDTH,
        margin=dict(b=150),
    )

    return fig


def build_paired_token_composition(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
) -> go.Figure:
    """Token composition for both experiments in one portrait composite.

    Two experiment rows on a shared structural configuration axis, stacked
    token bars from the same factory as the per-experiment chart (each
    trace carries ``meta={"experiment_row": n}`` so tests can assert that
    equivalence), one flat horizontal legend keeping the per-experiment
    "Token Type" title, and the two-tier leaf/band axis on the bottom row.
    The group gap carries model membership, so the per-experiment chart's
    dotted separators stay out; the token scale is explicit and shared
    across the rows.
    """
    order = structural_config_order(primary_data, ablation_data)
    positions, _ = grouped_config_positions(order)
    hover_labels = [short_config_label(ck) for ck in order]
    height = 1080
    vertical_spacing = 0.065
    plot_height = (
        height - _PORTRAIT_MARGINS_PX["t"] - _PORTRAIT_MARGINS_PX["b"]
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_titles=list(_EXPERIMENT_PANEL_TITLES),
    )

    max_stack_total = 0.0
    for row, data in ((1, primary_data), (2, ablation_data)):
        by_config = {r["config_key"]: r for r in data}
        rows_in_order: list[dict[str, Any]] = [
            by_config.get(ck, {}) for ck in order
        ]
        stack_segments = _token_stack_values(rows_in_order)
        max_stack_total = max(max_stack_total, max(
            sum(values[i] for _, values in stack_segments)
            for i in range(len(rows_in_order))
        ))
        for name, values in stack_segments:
            fig.add_trace(
                _token_stack_bar(
                    name,
                    positions,
                    values,
                    config_labels=hover_labels,
                    width=BAR_SLOT_FRACTION,
                    showlegend=row == 1,
                    meta={"experiment_row": row},
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(
            **_figure_title(
                "Token Composition by Configuration [Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        # No traceorder pin: plotly's stacked reversed default draws
        # top-of-stack first, exactly as the per-experiment chart's
        # ungrouped stacked legend does.
        legend=dict(
            orientation="h", x=0, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
            title=dict(text="Token Type"),
        ),
        height=height,
        width=FIGURE_WIDTH,
        # The left margin makes room for the shared y title shifted past
        # the six-digit comma-grouped tick ink.
        margin=dict(**{
            **_PORTRAIT_MARGINS_PX,
            "l": _WIDE_TICK_Y_TITLE_SHIFT_PX + 55,
        }),
    )
    # One explicit token scale across the rows (autorange collapses against
    # the two-tier helper's explicit x range), with headroom above the
    # tallest stack.
    fig.layout.yaxis2.matches = "y"
    fig.update_yaxes(
        range=[0, math.ceil(max_stack_total * 1.05)], tickformat=",d",
    )
    fig.add_annotation(
        x=0, xref="paper", xanchor="right",
        xshift=-_WIDE_TICK_Y_TITLE_SHIFT_PX,
        y=0.5, yref="paper", textangle=-90, showarrow=False,
        text="Mean Tokens per Trial",
        font=dict(size=BASE_FONT_SIZE),
    )
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    _finish_structural_axis(
        fig, order,
        band_axis_refs=("x2",),
        plot_height=plot_height,
        font_px=BASE_FONT_SIZE,
    )
    return fig


def build_reasoning_token_scaling_chart(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Line chart: mean reasoning tokens per trial vs reasoning effort level.

    One line per model group, colored via ``MODEL_GROUP_COLORS`` so
    sibling models from the same provider are visually distinguishable.
    Companion to the reasoning effort chart but showing token consumption
    instead of accuracy.  Uses the same canonical reasoning effort level ordering
    for a consistent x-axis.
    """
    fig = go.Figure()

    # Canonical ordering with display aliases applied (e.g., "none" → "off")
    forced_order = _display_levels_union(data)

    for trace in _model_effort_lines(
        data, y_key="mean_reasoning_tokens",
        hover_label="Mean Reasoning Tokens", hover_value_format=",.0f",
    ):
        fig.add_trace(trace)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Reasoning Token Scaling by Effort Level [{experiment_label}]"),
        xaxis_title=_REASONING_EFFORT_LEVEL_TITLE,
        yaxis_title="Mean Reasoning Tokens per Trial",
        yaxis_tickformat=",d",
        legend_title="Model",
        height=500,
        width=FIGURE_WIDTH,
        xaxis=dict(
            categoryorder="array",
            categoryarray=forced_order,
        ),
    )

    return fig


def build_paired_reasoning_token_scaling(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
) -> go.Figure:
    """Reasoning token scaling for both experiments, stacked.

    Two experiment rows over one canonical effort-level axis, model lines
    from the same core as the per-experiment chart, one shared token
    scale (the upper row matches the first row's range), and the flat
    horizontal legend keeping the per-experiment "Model" title.
    """
    height = 980
    fig, plot_height = _paired_effort_figure(
        primary_data, ablation_data,
        y_key="mean_reasoning_tokens",
        hover_label="Mean Reasoning Tokens",
        hover_value_format=",.0f",
        height=height,
        bottom_margin=100,
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            **_figure_title(
                "Reasoning Token Scaling by Effort Level [Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        legend=dict(
            orientation="h", x=0, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
            title=dict(text="Model"),
        ),
        height=height,
        width=FIGURE_WIDTH,
        # The left margin makes room for the shared y title shifted past
        # the six-digit comma-grouped tick ink.
        margin=dict(
            t=_PORTRAIT_MARGINS_PX["t"], b=100,
            l=_WIDE_TICK_Y_TITLE_SHIFT_PX + 55,
        ),
    )
    fig.layout.yaxis2.matches = "y"
    fig.update_yaxes(tickformat=",d")
    fig.add_annotation(
        x=0, xref="paper", xanchor="right",
        xshift=-_WIDE_TICK_Y_TITLE_SHIFT_PX,
        y=0.5, yref="paper", textangle=-90, showarrow=False,
        text="Mean Reasoning Tokens per Trial",
        font=dict(size=BASE_FONT_SIZE),
    )
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    return fig


def _transition_label(r: dict[str, Any]) -> str:
    """The display label for one reasoning effort level transition."""
    return display_reasoning_transition(
        r["model_slug"], r["from_level"], r["to_level"],
    )


def _marginal_cost_traces(
    data: list[dict[str, Any]],
    degradation_data: list[dict[str, Any]] | None,
    positions: dict[tuple[str, str, str], float],
    *,
    show_legend: bool = True,
) -> list[go.Bar | go.Scatter]:
    """The marginal-cost transition traces: bars, flat dashes, triangles.

    Three visual encodings distinguish transition outcomes — improved
    (balanced accuracy increased): a solid bar showing cost per pp gained;
    flat (unchanged): a horizontal dash marker at y=0; regressed
    (decreased): a down triangle at y=0. A category's trace exists only
    when it has items, so the list holds 1 to 3 traces. Marks sit at the
    caller's grouped numeric positions (keyed by ``_transition_key``), so
    every hover leads with the full transition label from
    ``customdata[0]`` — a bare ``%{x}`` would surface slot numbers. When
    ``degradation_data`` is provided, the regressed hover carries the
    adjacent-pair degradation test's adjusted e-value and significance.
    """
    improved = [r for r in data if r["marginal_cost_per_pp"] is not None]
    flat = [
        r for r in data
        if r["marginal_cost_per_pp"] is None and r["delta_balanced_accuracy"] == 0
    ]
    regressed = [
        r for r in data
        if r["marginal_cost_per_pp"] is None and r["delta_balanced_accuracy"] < 0
    ]

    traces: list[go.Bar | go.Scatter] = []

    if improved:
        traces.append(go.Bar(
            x=[positions[_transition_key(r)] for r in improved],
            y=[r["marginal_cost_per_pp"] for r in improved],
            width=BAR_SLOT_FRACTION,
            marker_color=[
                PROVIDER_COLORS.get(r["provider"], "#6b7280")
                for r in improved
            ],
            showlegend=False,
            hovertemplate=(
                "Transition: %{customdata[0]}<br>"
                "Marginal Cost: $%{y:.4f}/pp<br>"
                "Balanced Accuracy: from %{customdata[1]:.1%} to %{customdata[2]:.1%} "
                "(+%{customdata[3]:.1f}pp)<br>"
                "Δ Cost per Trial: %{customdata[4]}"
                "<extra></extra>"
            ),
            customdata=[
                [_transition_label(r), r["from_score"], r["to_score"],
                 r["delta_balanced_accuracy"] * 100, r["delta_cost_display"]]
                for r in improved
            ],
        ))

    if flat:
        traces.append(go.Scatter(
            name="No balanced accuracy gain",
            x=[positions[_transition_key(r)] for r in flat],
            y=[0] * len(flat),
            mode="markers",
            marker=dict(
                symbol="line-ew",
                size=16,
                line=dict(
                    color="#94a3b8",  # slate-400 (matches CELL_STATUS_COLORS["futile"])
                    width=3,
                ),
            ),
            cliponaxis=False,
            showlegend=show_legend,
            hovertemplate=(
                "Transition: %{customdata[0]}<br>"
                "Balanced accuracy unchanged at %{customdata[1]:.1%}<br>"
                "Δ Cost per Trial: %{customdata[2]}"
                "<extra></extra>"
            ),
            customdata=[
                [_transition_label(r), r["from_score"],
                 r["delta_cost_display"]]
                for r in flat
            ],
        ))

    if regressed:
        # Build lookup for degradation test results when available
        deg_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
        if degradation_data:
            for d in degradation_data:
                deg_lookup[(d["model_group"], d["from_level"], d["to_level"])] = d

        # Build customdata with optional degradation test fields
        regressed_customdata = []
        for r in regressed:
            row = [
                _transition_label(r),
                r["from_score"], r["to_score"],
                r["delta_balanced_accuracy"] * 100, r["delta_cost_display"],
            ]
            key = (r["model_group"], r["from_level"], r["to_level"])
            deg = deg_lookup.get(key)
            if deg is not None:
                row.append(deg["adjusted_e_value"])
                row.append("Significant" if deg["significant"] else "Not significant")
            else:
                row.append(None)
                row.append("")
            regressed_customdata.append(row)

        # Use enriched hover when degradation data is available
        if degradation_data:
            hover_template = (
                "Transition: %{customdata[0]}<br>"
                "Balanced accuracy decreased from %{customdata[1]:.1%} "
                "to %{customdata[2]:.1%} (%{customdata[3]:+.1f}pp)<br>"
                "Δ Cost per Trial: %{customdata[4]}<br>"
                "Adjacent-Pair Reasoning Effort Level Degradation Test: %{customdata[6]} "
                "(adjusted E = %{customdata[5]:.3f})"
                "<extra></extra>"
            )
        else:
            hover_template = (
                "Transition: %{customdata[0]}<br>"
                "Balanced accuracy decreased from %{customdata[1]:.1%} "
                "to %{customdata[2]:.1%} (%{customdata[3]:+.1f}pp)<br>"
                "Δ Cost per Trial: %{customdata[4]}"
                "<extra></extra>"
            )

        traces.append(go.Scatter(
            name="Balanced accuracy decreased",
            x=[positions[_transition_key(r)] for r in regressed],
            y=[0] * len(regressed),
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=14,
                color="#ef4444",  # red-500 — regression/warning indicator
            ),
            cliponaxis=False,
            showlegend=show_legend,
            hovertemplate=hover_template,
            customdata=regressed_customdata,
        ))

    return traces


def build_marginal_cost_chart(
    data: list[dict[str, Any]],
    degradation_data: list[dict[str, Any]] | None = None,
    *,
    experiment_label: str,
) -> go.Figure:
    """Bar chart with markers for marginal cost per percentage point of balanced accuracy.

    Each consecutive reasoning effort level transition within a model gets
    a slot on the two-tier transition axis, encoded per
    ``_marginal_cost_traces``. Traces are only added when their category
    has items, so ``len(fig.data)`` ranges from 1 to 3.
    """
    if not data:
        return go.Figure()

    positions, _ = grouped_transition_positions(data)
    position_by_key = dict(
        zip((_transition_key(r) for r in data), positions)
    )

    fig = go.Figure()
    traces = _marginal_cost_traces(data, degradation_data, position_by_key)
    for trace in traces:
        fig.add_trace(trace)

    has_markers = any(trace.name for trace in traces)

    # The two-tier strata are annotation-drawn, which automargin cannot
    # see, so the bottom margin is explicit and the plot height is what
    # the drop constants convert against.
    height = 800
    top_margin = 100
    bottom_margin = 400
    plot_height = height - top_margin - bottom_margin
    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Marginal Cost per Percentage Point of Balanced Accuracy [{experiment_label}]"),
        yaxis_title=_MARGINAL_COST_AXIS_TITLE,
        yaxis_tickprefix="$",
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(t=top_margin, b=bottom_margin),
        **(
            dict(legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ))
            if has_markers else {}
        ),
    )
    _finish_transition_axis(
        fig, data,
        band_axis_refs=("x",),
        plot_height=plot_height,
        font_px=BASE_FONT_SIZE,
    )

    return fig


def build_paired_marginal_cost(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
    primary_degradation: list[dict[str, Any]] | None = None,
    ablation_degradation: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """Marginal cost per percentage point for both experiments, stacked.

    Two experiment rows over one shared two-tier transition axis, traces
    from the same core as the per-experiment chart with each carrying
    ``meta={"experiment_row": n}``, one flat horizontal legend declaring
    each marker kind once, and one explicit dollar scale across the rows.
    Both experiments must cover the identical transition sequence: a
    transition present in only one would take a slot outside its model's
    band on the shared axis, so divergence raises instead of mislabeling.
    """
    primary_keys = [_transition_key(r) for r in primary_data]
    ablation_keys = [_transition_key(r) for r in ablation_data]
    if primary_keys != ablation_keys:
        only_primary = [k for k in primary_keys if k not in ablation_keys]
        only_ablation = [k for k in ablation_keys if k not in primary_keys]
        raise ValueError(
            "build_paired_marginal_cost needs both experiments to cover "
            "the same transition sequence in the same order. Only in "
            f"primary: {only_primary}; only in ablation: {only_ablation}."
        )
    if not primary_data:
        raise ValueError(
            "build_paired_marginal_cost needs at least one transition; "
            "both experiments arrived empty"
        )
    positions, _ = grouped_transition_positions(primary_data)
    position_by_key = dict(zip(primary_keys, positions))

    height = 1250
    vertical_spacing = 0.065
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_titles=list(_EXPERIMENT_PANEL_TITLES),
    )
    # The two-tier transition strata (leaf ticks, rules, band names,
    # disclosure) own the bottom margin; the top holds the title and
    # legend strata.
    top_margin = _PORTRAIT_MARGINS_PX["t"]
    bottom_margin = 380
    plot_height = height - top_margin - bottom_margin

    shown: set[str] = set()
    for row, (data, degradation) in (
        (1, (primary_data, primary_degradation)),
        (2, (ablation_data, ablation_degradation)),
    ):
        for trace in _marginal_cost_traces(data, degradation, position_by_key):
            if trace.name:
                trace.showlegend = trace.name not in shown
                shown.add(trace.name)
            trace.meta = {"experiment_row": row}
            fig.add_trace(trace, row=row, col=1)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            **_figure_title(
                "Marginal Cost per Percentage Point of Balanced Accuracy "
                "[Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        legend=dict(
            orientation="h", x=0, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
        ),
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(
            t=top_margin, b=bottom_margin, l=_PORTRAIT_MARGINS_PX["l"],
        ),
    )
    # One explicit dollar scale across the rows (autorange collapses
    # against the two-tier helper's explicit x range): headroom above the
    # tallest improving bar, room below the deepest cost-saving one (a
    # cheaper-and-better upper level prices a negative marginal cost), and
    # clearance under zero for the y=0 markers. A view with no improving
    # transition still gets a nominal one-dollar scale to place them on.
    improving_costs = [
        r["marginal_cost_per_pp"]
        for r in [*primary_data, *ablation_data]
        if r["marginal_cost_per_pp"] is not None
    ]
    hi_data = max([*improving_costs, 0.0])
    lo_data = min([*improving_costs, 0.0])
    if hi_data > 0:
        top = 1.05 * hi_data
    elif lo_data < 0:
        top = -0.05 * lo_data
    else:
        top = 1.0
    row_height = plot_height * (1 - vertical_spacing) / 2
    fig.layout.yaxis2.matches = "y"
    fig.update_yaxes(
        range=[
            1.05 * lo_data - _ZERO_MARKER_CLEARANCE_PX / row_height * top,
            top,
        ],
        tickprefix="$", title_text=_MARGINAL_COST_AXIS_TITLE,
    )
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    # x and x2 share one domain on this cols=1 stack, so the band
    # stratum anchors to the bottom axis alone; a second ref would
    # overprint every rule and name at identical pixels and set this
    # figure's stratum heavier than the deck's.
    _finish_transition_axis(
        fig, primary_data,
        band_axis_refs=("x2",),
        plot_height=plot_height,
        font_px=BASE_FONT_SIZE,
    )
    return fig


# The stack order of the cell resolution statuses, bottom segment first.
_CELL_STATUSES: tuple[str, ...] = ("rejected", "futile", "active")


def _cell_status_bar(
    status: str,
    x: list[Any],
    counts: list[int],
    config_labels: list[str],
    **legend: Any,
) -> go.Bar:
    """A cell-resolution status bar: counts, color, and hover in one place.

    ``config_labels`` ride ``customdata`` so the hover names the
    configuration even when ``x`` holds numeric grouped positions.
    """
    return go.Bar(
        name=status.capitalize(),
        x=x,
        y=counts,
        customdata=config_labels,
        marker_color=CELL_STATUS_COLORS[status],
        hovertemplate=(
            "Config: %{customdata}<br>"
            + status.capitalize() + ": %{y}<extra></extra>"
        ),
        **legend,
    )


def build_cell_resolution_chart(
    data: list[dict[str, Any]],
    *,
    experiment_label: str,
) -> go.Figure:
    """Stacked bar chart: rejected/futile/active cell counts per config.

    Configs render in the caller's order, which the x-axis title names:
    grouped by model, ordered by reasoning effort level.
    """
    if not data:
        return go.Figure()

    labels = [short_config_label(r["config_key"]) for r in data]

    fig = go.Figure()
    for status in _CELL_STATUSES:
        fig.add_trace(_cell_status_bar(
            status,
            labels,
            [r[status] for r in data],
            config_labels=labels,
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=_figure_title(f"Cell Resolution Status by Configuration [{experiment_label}]"),
        barmode="stack",
        xaxis_title=_CONFIG_GROUPED_BY_MODEL_TITLE,
        yaxis_title="Number of Cells",
        legend_title="Status",
        xaxis_tickangle=CONFIG_TICK_ANGLE,
        height=500,
        width=FIGURE_WIDTH,
        margin=dict(b=CONFIG_TICK_BOTTOM_MARGIN),
    )

    return fig


def build_paired_cell_resolution(
    primary_data: list[dict[str, Any]],
    ablation_data: list[dict[str, Any]],
) -> go.Figure:
    """Cell resolution status for both experiments in one portrait composite.

    Two experiment rows on a shared structural configuration axis, stacked
    status bars from the same factory as the per-experiment chart (each
    trace carries ``meta={"experiment_row": n}`` so tests can assert that
    equivalence), one flat horizontal legend keeping the per-experiment
    "Status" title, and the two-tier leaf/band axis on the bottom row. The
    count axis range is explicit and shared across the rows.
    """
    order = structural_config_order(primary_data, ablation_data)
    positions, _ = grouped_config_positions(order)
    hover_labels = [short_config_label(ck) for ck in order]
    height = 1000
    vertical_spacing = 0.065
    plot_height = (
        height - _PORTRAIT_MARGINS_PX["t"] - _PORTRAIT_MARGINS_PX["b"]
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_titles=list(_EXPERIMENT_PANEL_TITLES),
    )

    max_stack_total = 0
    for row, data in ((1, primary_data), (2, ablation_data)):
        by_config = {r["config_key"]: r for r in data}
        rows_in_order: list[dict[str, Any]] = [
            by_config.get(ck, {}) for ck in order
        ]
        max_stack_total = max(max_stack_total, max(
            sum(r.get(status) or 0 for status in _CELL_STATUSES)
            for r in rows_in_order
        ))
        for status in _CELL_STATUSES:
            fig.add_trace(
                _cell_status_bar(
                    status,
                    positions,
                    [r.get(status) or 0 for r in rows_in_order],
                    config_labels=hover_labels,
                    width=BAR_SLOT_FRACTION,
                    showlegend=row == 1,
                    meta={"experiment_row": row},
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(
            **_figure_title(
                "Cell Resolution Status by Configuration [Primary + Ablation]"
            ),
            x=0.033, xanchor="left", y=0.99, yanchor="top",
        ),
        # No traceorder pin: plotly's stacked reversed default draws
        # top-of-stack first, exactly as the per-experiment chart's
        # ungrouped stacked legend does.
        legend=dict(
            orientation="h", x=0, xanchor="left",
            y=1 + _LEGEND_RISE_PX / plot_height, yanchor="top",
            title=dict(text="Status"),
        ),
        height=height,
        width=FIGURE_WIDTH,
        margin=dict(**_PORTRAIT_MARGINS_PX),
    )
    # One explicit count scale across the rows (autorange collapses against
    # the two-tier helper's explicit x range), with headroom above the
    # tallest stack.
    fig.layout.yaxis2.matches = "y"
    fig.update_yaxes(range=[0, math.ceil(max_stack_total * 1.05)])
    fig.add_annotation(
        x=0, xref="paper", xanchor="right", xshift=-_Y_TITLE_SHIFT_PX,
        y=0.5, yref="paper", textangle=-90, showarrow=False,
        text="Number of Cells",
        font=dict(size=BASE_FONT_SIZE),
    )
    fig.update_annotations(font_size=BASE_FONT_SIZE)
    _finish_structural_axis(
        fig, order,
        band_axis_refs=("x2",),
        plot_height=plot_height,
        font_px=BASE_FONT_SIZE,
    )
    return fig


# ── Static export ──────────────────────────────────────────────────────


def export_figure(
    fig: go.Figure,
    output_dir: Path,
    filename: str,
    *,
    scale: float = 3,
    fmt: str = "png",
) -> Path:
    """Rasterize a chart to ``output_dir/filename.fmt`` via Kaleido.

    Chart builders fix their own canvas in ``update_layout``, so this export
    takes no width or height: passing ``None`` for both leaves each dimension to
    the figure's own layout, and an export-time size would silently contradict
    the builder with no diagnostic. Creates output_dir if it does not exist and
    returns the output path.

    Default scale=3 produces 3x resolution for publication quality.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.{fmt}"
    fig.write_image(str(path), width=None, height=None, scale=scale)
    return path
