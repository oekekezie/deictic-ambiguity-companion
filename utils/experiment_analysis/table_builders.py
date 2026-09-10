"""Plotly table construction for publication export.

Each function accepts pre-shaped data (list[dict] from visualizations_data.py
or other data sources) and returns a plotly.graph_objects.Figure containing a
go.Table trace. This separation keeps Plotly as a presentation concern only —
data transformation lives upstream.

Parallel to figure_builders.py (which handles charts); this module handles
tables rendered as static images for publication export.
"""

import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from utils.experiment_analysis.comparisons import BET_CAP
from utils.experiment_analysis.config_identity import (
    display_config_key,
    display_model_slug,
    display_provider,
    is_authoring_model,
    mark_authoring,
)
from utils.experiment_analysis.figure_builders import EXPORT_FONT_FAMILY
from utils.experiment_analysis.visualizations_data import (
    NOMINAL_VALID_PER_CONDITION,
    NOMINAL_VALID_PER_CONFIG,
    ValidTrialDisclosure,
)
from utils.experiment_analysis.stimulus_metadata import STIMULUS_METADATA_REGISTRY
from utils.experiment_analysis.metrics import ConfigurationSummary
from utils.formatting import (
    format_accuracy,
    format_cost_per_trial,
    format_e_power,
    format_e_value,
    format_optional_int,
    format_percent,
)
from utils.probe_audit.models import ProbeAuditResults
from utils.probe_audit.review_data import (
    pooled_disagreement_category_counts_by_condition,
    pooled_per_condition_probe_correctness,
)
from utils.rationale_analysis.models import (
    MISATTRIBUTION_FLAG_KEYS,
    RATIONALE_ANALYSIS_FLAG_DISPLAY_NAMES,
    RATIONALE_ANALYSIS_FLAG_KEYS,
)

# Reader-facing names for the three stimulus conditions. "Correct" alone would
# collide with the meta-evaluator's verdict value of the same name.
_CONDITION_DISPLAY: dict[str, str] = {
    "correct": "Correct-Draft",
    "transparent": "Transparent",
    "opaque": "Opaque",
}

# ── Shared table styling ──────────────────────────────────────────────

_HEADER_FILL = "#2d3748"       # slate-800
_HEADER_FONT_COLOR = "#ffffff"
_HEADER_FONT_SIZE = 11
_HEADER_HEIGHT = 30
_CELL_FONT_SIZE = 11
_CELL_HEIGHT = 26
_ROW_EVEN_FILL = "#f7fafc"     # slate-50
_ROW_ODD_FILL = "#ffffff"
_CELL_LINE_COLOR = "#e2e8f0"   # slate-200
_CELL_LINE_WIDTH = 1
_TABLE_FONT_FAMILY = EXPORT_FONT_FAMILY
_TITLE_FONT_SIZE = 14
_MARGIN_TOP = 50
_MARGIN_BOTTOM = 10
_MARGIN_LEFT = 20
_MARGIN_RIGHT = 20

# Display mapping for cross-dataset comparison Direction column. Source values
# come from comparisons.py as snake_case literals; this map converts each one
# to its reader-facing form so that the "experiment" head noun appears wherever
# the contrast names the experiment (Sense 2). Indistinguishable carries no
# experiment reference and renders unchanged. Public so notebooks rendering
# their own markdown tables can import the same map and stay in sync.
DIRECTION_DISPLAY: dict[str, str] = {
    "ablation_higher": "Ablation Experiment Higher",
    "ablation_lower": "Ablation Experiment Lower",
    "indistinguishable": "Indistinguishable",
}


def _alternating_row_colors(n_rows: int) -> list[str]:
    """Generate alternating row background colors for visual scanning."""
    return [_ROW_EVEN_FILL if i % 2 == 0 else _ROW_ODD_FILL for i in range(n_rows)]


def _build_table_figure(
    headers: list[str],
    cell_values: list[list[Any]],
    title: str,
    *,
    column_widths: list[float] | None = None,
    cell_align: list[str] | None = None,
) -> go.Figure:
    """Construct a styled Plotly go.Table figure from column-oriented data.

    Parameters
    ----------
    headers:
        Column header labels.
    cell_values:
        One list per column, each containing the column's cell values.
    title:
        Table title displayed as an annotation above the table.
    column_widths:
        Relative column widths (same length as headers). None for equal widths.
    cell_align:
        Per-column text alignment ("left", "center", "right"). Defaults to
        "left" for all columns.
    """
    n_rows = len(cell_values[0]) if cell_values else 0
    row_colors = _alternating_row_colors(n_rows)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color=_HEADER_FILL,
            font=dict(
                color=_HEADER_FONT_COLOR,
                size=_HEADER_FONT_SIZE,
                family=_TABLE_FONT_FAMILY,
            ),
            align="center",
            line=dict(color=_CELL_LINE_COLOR, width=_CELL_LINE_WIDTH),
            height=_HEADER_HEIGHT,
        ),
        cells=dict(
            values=cell_values,
            fill_color=[row_colors] * len(headers),
            font=dict(size=_CELL_FONT_SIZE, family=_TABLE_FONT_FAMILY),
            # go.Table aligns per column, not per cell, so an em-dash null
            # sentinel inherits its column's alignment (right within a
            # numeric column). The paper's LaTeX table path
            # (scripts/build_paper.py) lands in the same place by rule: a
            # column is one visual channel and every cell takes it, with
            # sentinels abstaining from the vote that picks it. The two
            # renderers therefore agree on where a sentinel sits.
            align=cell_align or ["left"] * len(headers),
            line=dict(color=_CELL_LINE_COLOR, width=_CELL_LINE_WIDTH),
            height=_CELL_HEIGHT,
        ),
        columnwidth=column_widths,
    )])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=_TITLE_FONT_SIZE, family=_TABLE_FONT_FAMILY),
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=_MARGIN_LEFT, r=_MARGIN_RIGHT, t=_MARGIN_TOP, b=_MARGIN_BOTTOM),
    )
    return fig


# ── Shared row-formatting primitives ─────────────────────────────────


def _format_yes_no(value: bool | None) -> str:
    """Boolean significance flag rendered as Yes/No (em-dash for None)."""
    if value is None:
        return "\u2014"
    return "Yes" if value else "No"


# How far a row's stated difference may sit from its two levels subtracted and
# still be recognized as their difference. The comparison results subtract the
# very floats they report, so the two agree bit for bit; this only absorbs
# serialization noise, and it stays far below half a display unit, which keeps
# a genuinely different quantity detectable.
_EXACT_DIFFERENCE_TOLERANCE = 1e-9


def format_score_delta_cell(
    minuend: float | None,
    subtrahend: float | None,
    exact_difference: float | None,
) -> str:
    """Render a \u0394 cell as the difference of the two level cells beside it.

    A comparison row prints both levels through ``format_accuracy`` and their
    difference in the next column, and the reader has to be able to subtract
    the first two and land on the third. Differencing the full-precision
    levels does not guarantee that: each level is rounded on its own, so two
    roundings that move toward each other shift their difference by up to one
    display unit, and levels printed as 0.444 and 0.527 can carry a
    full-precision difference of +0.0838. The difference is therefore taken
    between the rendered levels, in exact decimal arithmetic and at whatever
    precision ``format_accuracy`` emitted, which keeps the \u0394 column in step
    with the level formatter rather than with a decimal count restated here.

    Only the display is snapped. *exact_difference* is the full-precision
    difference the comparison result carries and the e-values are computed
    from; it is required so the snap can be checked against it, because a \u0394
    that is not its levels subtracted would be renamed by the snap rather than
    rounded by it.
    """
    difference = reproducible_score_delta(minuend, subtrahend, exact_difference)
    if difference is None:
        return "\u2014"
    decimals = len(format_accuracy(minuend).partition(".")[2])
    return f"{difference:+.{decimals}f}"


def format_percentage_point_delta_cell(
    minuend: float | None,
    subtrahend: float | None,
) -> str:
    """Render a percentage-point cell as the difference of the two rates beside it.

    The score cell's problem, on rates. Two rates each rounded to
    ``format_percent``'s one decimal can carry a full-precision difference
    that is not what the printed rates subtract to, and root ``CLAUDE.md``
    names this case directly: a percentage-point difference carries the
    decimals of the levels printed beside it, and the reader has to be able
    to reproduce it from them. The difference is therefore taken between the
    rendered rates, in exact decimal arithmetic and at whatever precision
    ``format_percent`` emitted.

    There is no *exact_difference* argument to check against, unlike the score
    cell: a shift between two rates is computed here and nowhere else, so
    there is no separately derived value it could contradict.
    """
    rendered_minuend = format_percent(minuend)
    rendered_subtrahend = format_percent(subtrahend)
    if "\u2014" in (rendered_minuend, rendered_subtrahend):
        return "\u2014"
    points = Decimal(rendered_minuend.rstrip("%")) - Decimal(
        rendered_subtrahend.rstrip("%")
    )
    decimals = len(rendered_minuend.rstrip("%").partition(".")[2])
    return f"{points:+.{decimals}f}pp"


def reproducible_score_delta(
    minuend: float | None,
    subtrahend: float | None,
    exact_difference: float | None,
) -> float | None:
    """The snapped difference as a number, for surfaces that cannot take text.

    ``format_score_delta_cell`` suits a markdown export, whose cells are text
    anyway. An interactive table is not: it holds raw values so its columns
    sort numerically, and a preformatted signed string would sort every
    positive difference above every negative one. Both surfaces therefore
    share this one snap and differ only in how they present it.

    Returns ``None`` when either level is missing, matching the em-dash the
    string form renders for that case.
    """
    rendered_minuend = format_accuracy(minuend)
    rendered_subtrahend = format_accuracy(subtrahend)
    if "\u2014" in (rendered_minuend, rendered_subtrahend):
        return None
    if exact_difference is None or not math.isclose(
        exact_difference,
        minuend - subtrahend,
        rel_tol=0.0,
        abs_tol=_EXACT_DIFFERENCE_TOLERANCE,
    ):
        raise ValueError(
            "A \u0394 cell must carry the exact difference of the two levels it "
            f"is printed beside; got {exact_difference!r} beside "
            f"{minuend!r} and {subtrahend!r}."
        )
    return float(Decimal(rendered_minuend) - Decimal(rendered_subtrahend))


# \u2500\u2500 Valid trial (N) inline annotation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#
# When a run falls below the design nominal, the leaderboard annotates each
# affected proportion cell inline with its realized denominator so a reader
# never sees a bare proportion over an unstated, reduced N. The realized-N
# disclosure for the uniform (nominal) case lives in the hand-authored
# captions, not in any generated subtitle.


def annotate_condition_cell(
    value: str,
    config_key: str,
    condition: str,
    disclosure: ValidTrialDisclosure,
) -> str:
    """Append a realized-N annotation to a per-condition proportion cell.

    No-op when the run is uniform or the cell already realized the nominal
    count. When the guard has tripped and this cell's realized N is below
    nominal, the reduced denominator is appended so the proportion is never
    shown bare.
    """
    if disclosure.uniform:
        return value
    n = disclosure.per_config_condition.get((config_key, condition))
    if n is None or n == NOMINAL_VALID_PER_CONDITION:
        return value
    return f"{value} (N = {n})"


def annotate_config_cell(
    value: str,
    config_key: str,
    disclosure: ValidTrialDisclosure,
) -> str:
    """Append a realized-N annotation to a per-configuration proportion cell.

    No-op when the run is uniform or the configuration already realized the
    nominal count. Otherwise the reduced per-configuration denominator is
    appended.
    """
    if disclosure.uniform:
        return value
    n = disclosure.per_config.get(config_key)
    if n is None or n == NOMINAL_VALID_PER_CONFIG:
        return value
    return f"{value} (N = {n})"


DEGRADATION_TABLE_COLUMNS: list[str] = [
    "Experiment",
    "Provider", "Model", "Transition",
    "Lower Level Balanced Accuracy", "Upper Level Balanced Accuracy",
    "\u0394 Balanced Accuracy", "Adjusted E-value", "Significant",
]
"""Canonical column order for adjacent-pair degradation tables.

The accuracy columns name balanced accuracy because that is the quantity the
test bets on: the paired rounds pool all three conditions and are class
reweighted, so their means are balanced accuracy differences.

The "Experiment" column is only present when combined data is used; the
per-experiment variants slice this constant from index 1 onward.
"""


def format_degradation_rows(
    rows: list[dict[str, Any]],
    *,
    include_experiment: bool,
) -> list[dict[str, str]]:
    """Format raw degradation rows into display-ready dicts.

    Single source of truth for the display-column-to-raw-key mapping,
    shared by the Plotly table builder, markdown export, and interactive
    table. When ``include_experiment`` is True, the "Experiment" key is
    populated from ``r["experiment"]``; otherwise it is omitted.
    """
    display_rows: list[dict[str, str]] = []
    for r in rows:
        provider_raw, _, model_raw = r["model_group"].partition("--")
        dr: dict[str, str] = {}
        if include_experiment:
            dr["Experiment"] = str(r["experiment"])
        dr.update({
            "Provider": display_provider(provider_raw),
            "Model": mark_authoring(
                display_model_slug(model_raw),
                is_authoring=is_authoring_model(model_raw),
            ),
            "Transition": f"{r['from_level']} to {r['to_level']}",
            "Lower Level Balanced Accuracy": format_accuracy(
                r.get("lower_balanced_accuracy"),
            ),
            "Upper Level Balanced Accuracy": format_accuracy(
                r.get("upper_balanced_accuracy"),
            ),
            "\u0394 Balanced Accuracy": format_score_delta_cell(
                r.get("upper_balanced_accuracy"),
                r.get("lower_balanced_accuracy"),
                r.get("delta_balanced_accuracy"),
            ),
            "Adjusted E-value": format_e_value(r.get("adjusted_e_value")),
            "Significant": _format_yes_no(r.get("significant")),
        })
        display_rows.append(dr)
    return display_rows


def build_degradation_table(
    rows: list[dict[str, str]], title: str,
) -> go.Figure:
    """Adjacent-pair degradation test results as a Plotly table figure.

    Expects pre-formatted display rows from ``format_degradation_rows``.
    If an "Experiment" key is present (interleaved data), it is shown as
    the first column.
    """
    has_experiment = any("Experiment" in r for r in rows)
    columns = (
        DEGRADATION_TABLE_COLUMNS if has_experiment
        else DEGRADATION_TABLE_COLUMNS[1:]
    )
    cell_values = [[r.get(k, "") for r in rows] for k in columns]
    align = (
        ["left"] if has_experiment else []
    ) + ["left", "left", "left", "right", "right", "right", "right",
         "right"]
    widths = (
        [0.8] if has_experiment else []
    ) + [0.8, 1.5, 1.0, 1.0, 1.0, 0.9, 1.0, 0.6]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


def _model_with_dagger(model_slug: str, *, is_authoring: bool) -> str:
    """Reader-facing model label, with the authoring model dagger appended.

    Annotates the larger *model* in the model size tables (authorship
    decided upstream by exact slug match).
    """
    return mark_authoring(display_model_slug(model_slug), is_authoring=is_authoring)


MODEL_SIZE_SUMMARY_COLUMNS: list[str] = [
    "Experiment",
    "Provider", "Larger Model", "Smaller Model",
    "Larger Model Reasoning Effort Levels", "Smaller Model Configurations",
    "Adjusted E-value", "Significant",
]
"""Canonical column order for the within-provider model size summary sub-tables.

One sub-table renders the every-level test and the other the some-level test;
both share this schema. Neither test selects a configuration, so the two count
columns give the size of each side of the combination the claim quantifies
over rather than naming an endpoint. The "Experiment" column is only present
when combined data is used; the per-experiment variants slice this constant
from index 1 onward.
"""


def format_model_size_summary_rows(
    rows: list[dict[str, Any]],
    *,
    include_experiment: bool,
) -> list[dict[str, str]]:
    """Format raw model size summary rows into display-ready dicts.

    Single source of truth for the display-column-to-raw-key mapping,
    shared by the Plotly table builder, markdown export, and interactive
    table. The larger-model cell carries the authoring model dagger when the
    row's ``is_authoring`` flag is set.
    """
    display_rows: list[dict[str, str]] = []
    for r in rows:
        dr: dict[str, str] = {}
        if include_experiment:
            dr["Experiment"] = str(r["experiment"])
        dr.update({
            "Provider": display_provider(r["provider"]),
            "Larger Model": _model_with_dagger(
                r["larger_model"], is_authoring=r.get("is_authoring", False),
            ),
            "Smaller Model": display_model_slug(r["smaller_model"]),
            "Larger Model Reasoning Effort Levels": format_optional_int(
                r.get("larger_reasoning_effort_levels"),
            ),
            "Smaller Model Configurations": format_optional_int(
                r.get("smaller_configurations"),
            ),
            "Adjusted E-value": format_e_value(r.get("adjusted_e_value")),
            "Significant": _format_yes_no(r.get("significant")),
        })
        display_rows.append(dr)
    return display_rows


def build_model_size_summary_table(
    rows: list[dict[str, str]], title: str,
) -> go.Figure:
    """One within-provider model size summary sub-table as a Plotly table figure.

    Expects pre-formatted display rows from ``format_model_size_summary_rows``.
    If an "Experiment" key is present (interleaved data), it is shown as the
    first column.
    """
    has_experiment = any("Experiment" in r for r in rows)
    columns = (
        MODEL_SIZE_SUMMARY_COLUMNS if has_experiment
        else MODEL_SIZE_SUMMARY_COLUMNS[1:]
    )
    cell_values = [[r.get(k, "") for r in rows] for k in columns]
    # Provider and the two model slugs are text; the two quantifier sizes are
    # integer counts and share the numeric columns' right edge.
    align = (
        ["left"] if has_experiment else []
    ) + ["left", "left", "left", "right", "right", "right", "right"]
    widths = (
        [0.8] if has_experiment else []
    ) + [1.0, 1.5, 1.5, 1.2, 1.2, 1.0, 0.6]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


MODEL_SIZE_DETAIL_COLUMNS: list[str] = [
    "Experiment",
    "Provider", "Larger Model", "Reasoning Effort", "Balanced Accuracy",
    "Valid Trials",
    "Mean E-value Across Smaller Model Configurations",
    "Minimum E-value Across Smaller Model Configurations",
]
"""Canonical column order for the within-provider model size per-level detail table.

The accuracy column names balanced accuracy because that is the quantity the
level's paired processes bet on; "Valid Trials" beside it is the level's sample
size pooled over every condition, not that accuracy's denominator.

The two e-value columns are the inner combinations of the level's paired
processes against every smaller-model configuration: the minimum of the mean
column reproduces the every-level summary e-value, and the arithmetic mean of
the minimum column reproduces the some-level summary e-value, both before the
shared multiplicity correction. The "Experiment" column is only present when
combined data is used; the per-experiment variants slice this constant from
index 1 onward.
"""


def format_model_size_detail_rows(
    rows: list[dict[str, Any]],
    *,
    include_experiment: bool,
) -> list[dict[str, str]]:
    """Format raw model size per-level detail rows into display-ready dicts.

    Single source of truth for the display-column-to-raw-key mapping,
    shared by the Plotly table builder, markdown export, and interactive
    table. The larger-model cell carries the authoring model dagger when the
    row's ``is_authoring`` flag is set. Both e-value columns follow the
    standard display convention via ``format_e_value``.
    """
    display_rows: list[dict[str, str]] = []
    for r in rows:
        dr: dict[str, str] = {}
        if include_experiment:
            dr["Experiment"] = str(r["experiment"])
        dr.update({
            "Provider": display_provider(r["provider"]),
            "Larger Model": _model_with_dagger(
                r["larger_model"], is_authoring=r.get("is_authoring", False),
            ),
            "Reasoning Effort": r["reasoning_effort_level"],
            "Balanced Accuracy": format_accuracy(r.get("balanced_accuracy")),
            "Valid Trials": format_optional_int(r.get("valid_trials")),
            "Mean E-value Across Smaller Model Configurations": format_e_value(
                r.get("mean_e_value_over_smaller_configurations"),
            ),
            "Minimum E-value Across Smaller Model Configurations": format_e_value(
                r.get("min_e_value_over_smaller_configurations"),
            ),
        })
        display_rows.append(dr)
    return display_rows


def build_model_size_detail_table(
    rows: list[dict[str, str]], title: str,
) -> go.Figure:
    """Within-provider model size per-level detail as a Plotly table figure.

    Expects pre-formatted display rows from ``format_model_size_detail_rows``.
    If an "Experiment" key is present (interleaved data), it is shown as the
    first column.
    """
    has_experiment = any("Experiment" in r for r in rows)
    columns = (
        MODEL_SIZE_DETAIL_COLUMNS if has_experiment
        else MODEL_SIZE_DETAIL_COLUMNS[1:]
    )
    cell_values = [[r.get(k, "") for r in rows] for k in columns]
    align = (
        ["left"] if has_experiment else []
    ) + ["left", "left", "left", "right", "right", "right", "right"]
    widths = (
        [0.8] if has_experiment else []
    ) + [1.0, 1.5, 1.1, 1.0, 0.9, 1.8, 1.8]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


COST_TABLE_COLUMNS: list[str] = [
    "Configuration", "Balanced Accuracy", "Cost per Trial ($)", "Total Cost ($)",
    "Input Tokens", "Output Tokens", "Reasoning Tokens",
    "Response Tokens",
]
"""Canonical column order for cost breakdown tables.

Single source of truth shared by the Plotly table builder, the
markdown export formatter, and the interactive mo.ui.table.
"""


_LEGACY_COST_COLUMN_KEYS: frozenset[str] = frozenset({"Cost/Trial", "Total Cost"})


def format_cost_cell(column: str, value: Any) -> str:
    """Format a single cost table cell value for display.

    Single source of truth for cost column formatting, used by the Plotly
    table builder, mo.ui.table format_mapping, and markdown export. Raises
    on legacy column keys to fail loudly if a caller still uses the
    pre-rename spelling.
    """
    if column in _LEGACY_COST_COLUMN_KEYS:
        raise ValueError(
            f"Legacy cost column key {column!r} is no longer supported; "
            "use 'Cost per Trial ($)' or 'Total Cost ($)' instead."
        )
    if value is None:
        return "\u2014"
    if column == "Balanced Accuracy":
        return f"{value:.3f}"
    if column == "Cost per Trial ($)":
        return format_cost_per_trial(value)
    if column == "Total Cost ($)":
        return f"${value:.2f}"
    if column in {"Input Tokens", "Output Tokens", "Reasoning Tokens",
                  "Response Tokens"}:
        return format_optional_int(value)
    return str(value)


def format_cost_breakdown_rows(
    rows: list[dict[str, Any]],
    *,
    include_experiment: bool,
) -> list[dict[str, Any]]:
    """Shape raw cost data into raw-valued rows keyed by COST_TABLE_COLUMNS.

    Returns rows with raw numeric values (not formatted strings) so the
    interactive ``mo.ui.table`` view can apply ``format_mapping`` for
    in-place formatting and the Plotly builder can call ``format_cost_cell``
    per cell. The markdown export path uses ``format_cost_rows_for_markdown``
    on top of the same raw rows.
    """
    display_rows: list[dict[str, Any]] = []
    for r in rows:
        dr: dict[str, Any] = {}
        if include_experiment:
            dr["Experiment"] = r["experiment"]
        dr.update({
            "Configuration": display_config_key(r["config_key"]),
            "Balanced Accuracy": r["balanced_accuracy"],
            "Cost per Trial ($)": r["cost_per_trial"],
            "Total Cost ($)": r["actual_cost"],
            "Input Tokens": r["total_input_tokens"],
            "Output Tokens": r["total_output_tokens"],
            "Reasoning Tokens": r["total_reasoning_tokens"],
            "Response Tokens": r["total_response_tokens"],
        })
        display_rows.append(dr)
    return display_rows


def format_cost_rows_for_markdown(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Convert raw-valued cost rows into string-formatted markdown rows.

    Delegates to ``format_cost_cell`` for consistent formatting across
    Plotly tables, interactive tables, and markdown exports. Surfaces an
    "Experiment" column when present (combined paired data).
    """
    has_experiment = any("Experiment" in r for r in rows)
    formatted: list[dict[str, str]] = []
    for r in rows:
        row: dict[str, str] = {}
        if has_experiment:
            row["Experiment"] = str(r.get("Experiment", ""))
        for col in COST_TABLE_COLUMNS:
            row[col] = format_cost_cell(col, r[col])
        formatted.append(row)
    return formatted


def build_cost_breakdown_table(
    rows: list[dict[str, Any]], title: str,
) -> go.Figure:
    """Cost breakdown by configuration as a Plotly table figure.

    Expects raw numeric rows keyed by COST_TABLE_COLUMNS. Formatting
    is handled by format_cost_cell. If an "Experiment" key is present
    (combined data), it is shown as the first column.
    """
    has_experiment = any("Experiment" in r for r in rows)
    columns = (["Experiment"] if has_experiment else []) + COST_TABLE_COLUMNS
    cell_values = [
        [format_cost_cell(k, r.get(k)) for r in rows]
        for k in columns
    ]
    align = (
        ["left"] if has_experiment else []
    ) + ["left", "right", "right", "right", "right", "right",
         "right", "right"]
    widths = (
        [0.8] if has_experiment else []
    ) + [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


BALANCED_ACCURACY_COLUMNS: list[str] = [
    "Rank", "Configuration", "Balanced Accuracy", "Overall Accuracy",
    "Specificity", "Sensitivity (Transparent)", "Sensitivity (Opaque)",
]
"""Canonical column order for balanced accuracy leaderboard tables."""


def format_leaderboard_rows(
    rows: list[dict[str, Any]],
    *,
    disclosure: ValidTrialDisclosure | None = None,
) -> list[dict[str, str]]:
    """Format raw leaderboard rows into display-ready dicts.

    Single source of truth for the display-column-to-raw-key mapping,
    shared by the Plotly table builder, markdown export, and interactive
    table. If a row carries an ``experiment`` key (combined leaderboard),
    it is surfaced as an ``Experiment`` display column. When a ``disclosure``
    is supplied and the run is non-uniform, each proportion cell whose
    realized denominator fell below nominal is annotated inline with its
    valid trial count so no proportion is rendered bare.
    """
    display_rows: list[dict[str, str]] = []
    for r in rows:
        label = display_config_key(r["config_key"])
        config_key = r["config_key"]

        specificity = format_accuracy(r.get("correct_accuracy"))
        sens_transparent = format_accuracy(r.get("transparent_accuracy"))
        sens_opaque = format_accuracy(r.get("opaque_accuracy"))
        overall = format_accuracy(r.get("overall_accuracy"))
        balanced = format_accuracy(r.get("balanced_accuracy"))
        if disclosure is not None:
            specificity = annotate_condition_cell(
                specificity, config_key, "correct", disclosure,
            )
            sens_transparent = annotate_condition_cell(
                sens_transparent, config_key, "transparent", disclosure,
            )
            sens_opaque = annotate_condition_cell(
                sens_opaque, config_key, "opaque", disclosure,
            )
            overall = annotate_config_cell(overall, config_key, disclosure)
            balanced = annotate_config_cell(balanced, config_key, disclosure)

        dr: dict[str, str] = {}
        if "experiment" in r:
            dr["Experiment"] = str(r["experiment"])
        dr.update({
            "Rank": str(r["rank"]),
            "Configuration": label,
            "Balanced Accuracy": balanced,
            "Overall Accuracy": overall,
            "Specificity": specificity,
            "Sensitivity (Transparent)": sens_transparent,
            "Sensitivity (Opaque)": sens_opaque,
        })
        display_rows.append(dr)
    return display_rows


def build_balanced_accuracy_table(
    rows: list[dict[str, Any]],
    title: str,
    *,
    disclosure: ValidTrialDisclosure | None = None,
) -> go.Figure:
    """Balanced accuracy leaderboard as a Plotly table figure.

    Expects raw rows from ``balanced_accuracy_leaderboard`` (or
    ``combined_balanced_accuracy_leaderboard``). Authoring model configs are
    marked with \u2020. When raw rows carry an ``experiment`` key, an
    ``Experiment`` column is prepended. A ``disclosure`` annotates reduced-N
    proportion cells when the run is non-uniform.
    """
    display_rows = format_leaderboard_rows(rows, disclosure=disclosure)
    has_experiment = any("Experiment" in dr for dr in display_rows)
    columns = (
        ["Experiment"] if has_experiment else []
    ) + BALANCED_ACCURACY_COLUMNS
    cell_values = [[dr.get(k, "") for dr in display_rows] for k in columns]
    align = (
        ["left"] if has_experiment else []
    ) + ["right", "left", "right", "right", "right", "right", "right"]
    widths = (
        [0.8] if has_experiment else []
    ) + [0.5, 2.0, 1.0, 1.0, 1.0, 1.2, 1.2]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


ARTICULATION_GOVERNING_COLUMNS: list[str] = [
    "Condition",
    "Articulation Rate (%) [Primary]",
    "Governing Rate (%) [Primary]",
    "Wrong-Verdict Trial Count [Primary]",
    "Articulation Rate (%) [Ablation]",
    "Governing Rate (%) [Ablation]",
    "Wrong-Verdict Trial Count [Ablation]",
]
"""Column order for the per-condition articulation/governing rate summary.

Seven columns, experiment-major: each experiment's articulation rate,
governing rate, and the wrong-verdict trial count that serves as the rate
denominator form a contiguous group, so the rendered table names the
experiment once in a spanner heading. The misattribution-flag disclosure is
reported separately in its own per-condition table.
"""


def format_articulation_governing_rows(
    primary_rates: list[dict[str, Any]],
    ablation_rates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Format aggregate failure mode rate rows for the summary table.

    Accepts one list per experiment from ``aggregate_failure_mode_rates`` and
    joins by condition, emitting each experiment's columns as a contiguous
    group so the spanner headings can name the experiment once. The
    primary-to-ablation change is read across a row's adjacent groups; no
    delta column restates it.
    """
    ablation_by_cond = {r["condition"]: r for r in ablation_rates}

    def _fmt_rate(rate: float | None) -> str:
        return format_percent(rate) if rate is not None else "\u2014"

    display_rows: list[dict[str, str]] = []
    for pr in primary_rates:
        cond = pr["condition"]
        ar = ablation_by_cond.get(cond, {})
        display_rows.append({
            "Condition": _CONDITION_DISPLAY[cond],
            "Articulation Rate (%) [Primary]": _fmt_rate(pr.get("articulation_rate")),
            "Governing Rate (%) [Primary]": _fmt_rate(pr.get("governing_rate")),
            "Wrong-Verdict Trial Count [Primary]": f"{pr.get('total_mismatches', 0):,d}",
            "Articulation Rate (%) [Ablation]": _fmt_rate(ar.get("articulation_rate")),
            "Governing Rate (%) [Ablation]": _fmt_rate(ar.get("governing_rate")),
            "Wrong-Verdict Trial Count [Ablation]": f"{ar.get('total_mismatches', 0):,d}",
        })
    return display_rows


def build_articulation_governing_table(
    primary_rates: list[dict[str, Any]],
    ablation_rates: list[dict[str, Any]],
    title: str,
) -> go.Figure:
    """Per-condition articulation and governing rate summary table.

    Accepts primary and ablation rate lists from
    ``aggregate_failure_mode_rates``. Columns are experiment-major: each
    experiment's articulation rate, governing rate, and rate denominator
    form a contiguous group, so the change between experiments is read
    across the row rather than from a delta column.
    """
    display_rows = format_articulation_governing_rows(primary_rates, ablation_rates)
    columns = ARTICULATION_GOVERNING_COLUMNS
    cell_values = [[dr[k] for dr in display_rows] for k in columns]
    align = ["left"] + ["right"] * 6
    widths = [1.0, 1.2, 1.2, 1.0, 1.2, 1.2, 1.0]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


MISATTRIBUTION_RATES_COLUMNS: list[str] = [
    "Misattribution Flag",
    "Correct-Draft Rate (%) [Primary]",
    "Transparent Rate (%) [Primary]",
    "Opaque Rate (%) [Primary]",
    "Correct-Draft Rate (%) [Ablation]",
    "Transparent Rate (%) [Ablation]",
    "Opaque Rate (%) [Ablation]",
]
"""Column order for the standalone per-condition misattribution rate table.

One row per misattribution flag plus a closing "Any Misattribution Flag"
row, with six per-condition rate columns (three conditions per experiment).
A confound disclosure rather than an effect measure, so there is no delta
column.
"""


def format_misattribution_rates_rows(
    primary_rates: list[dict[str, Any]],
    ablation_rates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Pivot per-condition misattribution rate rows into per-flag display rows.

    Accepts one list per experiment from ``misattribution_rates_by_condition``
    (one row per condition). Emits one display row per misattribution flag in
    canonical order, plus a closing "Any Misattribution Flag" row, each
    carrying the six per-condition cells that pair the rate with its
    fired-over-resolved count. Returns an empty list when
    either experiment's rates are missing, so the paired table renders only
    when both are present.
    """
    if not primary_rates or not ablation_rates:
        return []

    primary_by_cond = {r["condition"]: r for r in primary_rates}
    ablation_by_cond = {r["condition"]: r for r in ablation_rates}

    def _fmt_cell(cond_row: dict[str, Any], base_key: str) -> str:
        """Render ``rate (fired/resolved)``, pairing each proportion with its
        supporting count; em-dash sentinel when the rate is undefined."""
        rate = cond_row.get(f"{base_key}_rate")
        if rate is None:
            return "—"
        fired = cond_row.get(f"{base_key}_fired")
        resolved = cond_row.get(f"{base_key}_resolved")
        return f"{format_percent(rate)} ({fired}/{resolved})"

    # (display label, base key) per flag row, then the Any row over the
    # determinate any-flag fields. The base key resolves the rate, fired, and
    # resolved fields so each cell pairs the rate with its count.
    flag_specs: list[tuple[str, str]] = [
        (RATIONALE_ANALYSIS_FLAG_DISPLAY_NAMES[flag], flag)
        for flag in MISATTRIBUTION_FLAG_KEYS
    ]
    flag_specs.append(("Any Misattribution Flag", "any_flag"))

    display_rows: list[dict[str, str]] = []
    for label, base_key in flag_specs:
        display_rows.append({
            "Misattribution Flag": label,
            "Correct-Draft Rate (%) [Primary]": _fmt_cell(
                primary_by_cond.get("correct", {}), base_key,
            ),
            "Transparent Rate (%) [Primary]": _fmt_cell(
                primary_by_cond.get("transparent", {}), base_key,
            ),
            "Opaque Rate (%) [Primary]": _fmt_cell(
                primary_by_cond.get("opaque", {}), base_key,
            ),
            "Correct-Draft Rate (%) [Ablation]": _fmt_cell(
                ablation_by_cond.get("correct", {}), base_key,
            ),
            "Transparent Rate (%) [Ablation]": _fmt_cell(
                ablation_by_cond.get("transparent", {}), base_key,
            ),
            "Opaque Rate (%) [Ablation]": _fmt_cell(
                ablation_by_cond.get("opaque", {}), base_key,
            ),
        })
    return display_rows


def build_misattribution_rates_table(
    primary_rates: list[dict[str, Any]],
    ablation_rates: list[dict[str, Any]],
    title: str,
) -> go.Figure:
    """Standalone per-condition misattribution flag rate table as a Plotly figure.

    Accepts primary and ablation rate lists from
    ``misattribution_rates_by_condition``. One row per flag plus an
    "Any Misattribution Flag" row, with per-condition rates side by side.
    """
    display_rows = format_misattribution_rates_rows(primary_rates, ablation_rates)
    columns = MISATTRIBUTION_RATES_COLUMNS
    cell_values = [[dr.get(k, "") for dr in display_rows] for k in columns]
    align = ["left"] + ["right"] * 6
    widths = [2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


CAP_SENSITIVITY_COLUMNS: list[str] = [
    "Experiment", "Betting Cap", "Significant Comparisons",
    "Total Comparisons", "Comparisons Changing Status",
]
"""Canonical column order for the pairwise comparison cap sensitivity table."""


def format_cap_sensitivity_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Format raw cap sensitivity rows into display-ready dicts.

    The reported cap's own row is the one every other row is compared
    against, so its change count is the null sentinel rather than zero: no
    change is definable there, which is a different statement from none
    having occurred.

    Counts are formatted directly rather than through ``format_optional_int``,
    which maps zero to the null sentinel. That reading suits token counts,
    where zero means a capability is absent, but here a zero is a measured
    result: no comparison surviving at a cap is the finding such a row exists
    to report, and rendering it as unavailable would hide it.
    """
    display_rows: list[dict[str, str]] = []
    for r in rows:
        is_reference = r["bet_cap"] == BET_CAP
        display_rows.append({
            "Experiment": str(r["experiment"]),
            "Betting Cap": f"{r['bet_cap']:.2f}",
            "Significant Comparisons": f"{r['significant_comparisons']:,d}",
            "Total Comparisons": f"{r['total_comparisons']:,d}",
            "Comparisons Changing Status": (
                "\u2014" if is_reference else f"{len(r['status_changes']):,d}"
            ),
        })
    return display_rows


def build_cap_sensitivity_table(
    rows: list[dict[str, str]], title: str,
) -> go.Figure:
    """Pairwise comparison cap sensitivity as a Plotly table figure.

    Expects pre-formatted display rows from ``format_cap_sensitivity_rows``.
    If an "Experiment" key is present (interleaved data), it is shown as the
    first column; the exported markdown drops it, because the cut-in transform
    turns it into one heading row per experiment block.
    """
    has_experiment = any("Experiment" in r for r in rows)
    columns = (
        CAP_SENSITIVITY_COLUMNS if has_experiment
        else CAP_SENSITIVITY_COLUMNS[1:]
    )
    cell_values = [[r.get(k, "") for r in rows] for k in columns]
    align = (
        ["left"] if has_experiment else []
    ) + ["right", "right", "right", "right"]
    widths = (
        [0.8] if has_experiment else []
    ) + [0.7, 1.2, 1.0, 1.3]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


CROSS_DATASET_SIGNIFICANCE_COLUMNS: list[str] = [
    "Configuration", "Condition", "Per-Condition Accuracy [Primary]",
    "Per-Condition Accuracy [Ablation]",
    "\u0394 Per-Condition Accuracy", "Adjusted E-value", "Significant", "Direction",
]
"""Canonical column order for the condition-level cross-dataset table.

Every row is one condition of one configuration, so its accuracy is that
condition's hits over its valid trials. The header says so rather than leaving
a bare "Accuracy" for the reader to resolve against the configuration-level
table's balanced accuracy.
"""


def format_cross_dataset_significance_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Format raw cross-dataset significance rows into display-ready dicts.

    Single source of truth for the display-column-to-raw-key mapping,
    shared by the Plotly table builder, markdown export, and interactive
    table.
    """
    display_rows: list[dict[str, str]] = []
    for r in rows:
        display_rows.append({
            "Configuration": display_config_key(r["config_key"]),
            "Condition": _CONDITION_DISPLAY[r["condition"]],
            "Per-Condition Accuracy [Primary]": format_accuracy(
                r.get("primary_accuracy"),
            ),
            "Per-Condition Accuracy [Ablation]": format_accuracy(
                r.get("ablation_accuracy"),
            ),
            "\u0394 Per-Condition Accuracy": format_score_delta_cell(
                r.get("ablation_accuracy"),
                r.get("primary_accuracy"),
                r.get("delta"),
            ),
            "Adjusted E-value": format_e_value(r.get("adjusted_e_value")),
            "Significant": _format_yes_no(r.get("significant")),
            "Direction": DIRECTION_DISPLAY.get(r.get("direction"), "\u2014"),
        })
    return display_rows


def build_cross_dataset_significance_table(
    rows: list[dict[str, str]], title: str,
) -> go.Figure:
    """Cross-dataset ablation significance test results as a Plotly table figure.

    Expects pre-formatted display rows from
    ``format_cross_dataset_significance_rows`` keyed by
    ``CROSS_DATASET_SIGNIFICANCE_COLUMNS``.
    """
    columns = CROSS_DATASET_SIGNIFICANCE_COLUMNS
    cell_values = [[r.get(k, "") for r in rows] for k in columns]
    align = ["left", "left", "right", "right", "right", "right", "right", "left"]
    widths = [2.0, 1.0, 1.0, 1.0, 0.9, 1.2, 0.6, 1.2]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


CROSS_DATASET_CONFIG_SIGNIFICANCE_COLUMNS: list[str] = [
    "Configuration", "Balanced Accuracy [Primary]",
    "Balanced Accuracy [Ablation]",
    "Δ Balanced Accuracy", "Adjusted E-value", "Significant", "Direction",
]
"""Canonical column order for the configuration-level cross-dataset table.

Each row covers all three conditions of one configuration, and its paired
rounds are class reweighted, so the tested and reported quantity is balanced
accuracy.
"""


def format_cross_dataset_config_significance_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Format raw cross-dataset configuration-level rows for display.

    Sibling of ``format_cross_dataset_significance_rows`` for the K=21
    family that covers all three conditions of a configuration in one test,
    reporting the balanced accuracy its class-reweighted rounds bet on. The
    output schema has no ``Condition`` column; the title announces the
    configuration-level scope.
    """
    display_rows: list[dict[str, str]] = []
    for r in rows:
        display_rows.append({
            "Configuration": display_config_key(r["config_key"]),
            "Balanced Accuracy [Primary]": format_accuracy(r.get("primary_accuracy")),
            "Balanced Accuracy [Ablation]": format_accuracy(
                r.get("ablation_accuracy"),
            ),
            "Δ Balanced Accuracy": format_score_delta_cell(
                r.get("ablation_accuracy"),
                r.get("primary_accuracy"),
                r.get("delta"),
            ),
            "Adjusted E-value": format_e_value(r.get("adjusted_e_value")),
            "Significant": _format_yes_no(r.get("significant")),
            "Direction": DIRECTION_DISPLAY.get(r.get("direction"), "—"),
        })
    return display_rows


def build_cross_dataset_config_significance_table(
    rows: list[dict[str, str]], title: str,
) -> go.Figure:
    """Cross-dataset configuration-level ablation significance table figure.

    Expects pre-formatted display rows from
    ``format_cross_dataset_config_significance_rows`` keyed by
    ``CROSS_DATASET_CONFIG_SIGNIFICANCE_COLUMNS``.
    """
    columns = CROSS_DATASET_CONFIG_SIGNIFICANCE_COLUMNS
    cell_values = [[r.get(k, "") for r in rows] for k in columns]
    align = ["left", "right", "right", "right", "right", "right", "left"]
    widths = [2.0, 1.0, 1.0, 0.9, 1.2, 0.6, 1.2]

    return _build_table_figure(
        columns, cell_values, title,
        column_widths=widths, cell_align=align,
    )


# ── Base example inventory ────────────────────────────────────────────

BASE_EXAMPLE_INVENTORY_COLUMNS: list[str] = [
    "Example", "Risk", "Value Under Test",
    "Current Value", "Historical Value", "Proposed Value",
]
"""Column order for the base example inventory table."""


def format_base_example_inventory_rows() -> list[dict[str, str]]:
    """Build display rows from the stimulus metadata registry.

    Returns one row per base example, sorted by example key
    (example_01 through example_10), with values taken directly
    from ``STIMULUS_METADATA_REGISTRY``.
    """
    return [
        {
            "Example": key,
            "Risk": meta.risk,
            "Value Under Test": meta.domain_noun,
            "Current Value": meta.current_value,
            "Historical Value": meta.historical_value,
            "Proposed Value": meta.proposed_value,
        }
        for key, meta in sorted(STIMULUS_METADATA_REGISTRY.items())
    ]


# ── Paired supplementary tables: leaderboard and cross-dataset tests ──

def _format_decimal_3(value: float | None) -> str:
    """Three-decimal value or the em-dash null sentinel for missing data.

    Used for log space quantities (log e-values) and unitless widths
    (confidence sequence width), which render at three decimals rather than
    through the e-value fixed/scientific magnitude threshold.
    """
    return f"{value:.3f}" if value is not None else "—"


# --- Pairwise condition comparison e-values ---

PAIRWISE_COMPARISON_E_VALUE_COLUMNS: list[str] = [
    "Configuration",
    "Transparent vs Opaque E-value [Primary]",
    "Transparent vs Correct-Draft E-value [Primary]",
    "Opaque vs Correct-Draft E-value [Primary]",
    "Transparent vs Opaque E-value [Ablation]",
    "Transparent vs Correct-Draft E-value [Ablation]",
    "Opaque vs Correct-Draft E-value [Ablation]",
]


def format_pairwise_comparison_e_value_rows(
    primary_by_config: list[dict[str, Any]],
    ablation_by_config: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Format paired pairwise-comparison e-value rows for the summary table.

    Accepts one list per experiment from
    ``pairwise_comparison_e_values_by_config`` (already ordered), joins by
    config key, and renders every e-value through the shared display contract
    (fixed notation below 10,000, scientific at or above). Keys are emitted
    experiment-major, matching ``PAIRWISE_COMPARISON_E_VALUE_COLUMNS``,
    because the notebook Interactive Table renders the dict key order.
    """
    ablation_lookup = {r["config_key"]: r for r in ablation_by_config}
    display_rows: list[dict[str, str]] = []
    for pr in primary_by_config:
        ar = ablation_lookup.get(pr["config_key"], {})
        display_rows.append({
            "Configuration": display_config_key(pr["config_key"]),
            "Transparent vs Opaque E-value [Primary]": format_e_value(
                pr.get("transparent_vs_opaque_e_value"),
            ),
            "Transparent vs Correct-Draft E-value [Primary]": format_e_value(
                pr.get("transparent_vs_correct_e_value"),
            ),
            "Opaque vs Correct-Draft E-value [Primary]": format_e_value(
                pr.get("opaque_vs_correct_e_value"),
            ),
            "Transparent vs Opaque E-value [Ablation]": format_e_value(
                ar.get("transparent_vs_opaque_e_value"),
            ),
            "Transparent vs Correct-Draft E-value [Ablation]": format_e_value(
                ar.get("transparent_vs_correct_e_value"),
            ),
            "Opaque vs Correct-Draft E-value [Ablation]": format_e_value(
                ar.get("opaque_vs_correct_e_value"),
            ),
        })
    return display_rows


# --- E-value distribution and power diagnostics ---

E_VALUE_DISTRIBUTION_COLUMNS: list[str] = [
    "Condition",
    "Experiment",
    "Log E-value Minimum",
    "Log E-value First Quartile",
    "Log E-value Median",
    "Log E-value Third Quartile",
    "Log E-value Maximum",
    "Mean Empirical E-power",
    "Mean Confidence Sequence Width",
    "Cells Rejected (%)",
]


def format_e_value_distribution_rows(
    primary_distribution: list[dict[str, Any]],
    ablation_distribution: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Format stacked per-(experiment, condition) e-value distribution rows.

    Accepts one list per experiment from ``e_value_distribution_by_condition``
    and stacks them into one row per (experiment, condition) with an
    ``Experiment`` column, mirroring the other paired stacked tables
    (leaderboard, cost, degradation). Rows are experiment-major: every primary
    condition, then every ablation condition. Log e-value quantiles, empirical
    e-power, and confidence sequence width render at three decimals; the
    rejection fraction renders as a percentage; a missing value renders as the
    em-dash null sentinel.
    """
    display_rows: list[dict[str, str]] = []
    for experiment_label, distribution in (
        ("Primary", primary_distribution),
        ("Ablation", ablation_distribution),
    ):
        for row in distribution:
            display_rows.append({
                "Condition": _CONDITION_DISPLAY[row["condition"]],
                "Experiment": experiment_label,
                "Log E-value Minimum": _format_decimal_3(row.get("log_e_value_min")),
                "Log E-value First Quartile": _format_decimal_3(row.get("log_e_value_q1")),
                "Log E-value Median": _format_decimal_3(row.get("log_e_value_median")),
                "Log E-value Third Quartile": _format_decimal_3(row.get("log_e_value_q3")),
                "Log E-value Maximum": _format_decimal_3(row.get("log_e_value_max")),
                "Mean Empirical E-power": format_e_power(row.get("mean_empirical_e_power")),
                "Mean Confidence Sequence Width": _format_decimal_3(row.get("mean_cs_width")),
                "Cells Rejected (%)": format_percent(row.get("cells_rejected_rate")),
            })
    return display_rows


# --- Experiment cut-in rows and the probe stub layout ---

_EXPERIMENT_ORDER: tuple[str, str] = ("Primary", "Ablation")


def insert_experiment_cut_in_rows(
    rows: list[dict[str, str]], *, cut_in_column: str,
) -> list[dict[str, str]]:
    """Replace the Experiment column with a cut-in heading row per block.

    A union table's rows arrive grouped by experiment; each block gains a
    leading ``**<Experiment> Experiment**`` heading carried in
    ``cut_in_column``, and the data rows drop their "Experiment" key, since
    the heading now says what the column repeated. Interleaved blocks or an
    experiment label outside the primary/ablation pair raise rather than
    exporting a table whose headings misgroup its rows.
    """
    out: list[dict[str, str]] = []
    seen: list[str] = []
    for row in rows:
        experiment = row["Experiment"]
        if experiment not in _EXPERIMENT_ORDER:
            raise ValueError(f"unknown experiment label: {experiment!r}")
        if not seen or seen[-1] != experiment:
            if experiment in seen:
                raise ValueError(
                    f"experiment blocks must be contiguous; {experiment!r} "
                    "reappears after another block"
                )
            seen.append(experiment)
            out.append({cut_in_column: f"**{experiment} Experiment**"})
        out.append({k: v for k, v in row.items() if k != "Experiment"})
    return out


def format_probe_stub_layout_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Regroup flag-major probe rows into the experiment-major stub layout.

    The row builder emits flag-major rows with an Experiment key. The stub
    layout instead cuts the table into one block per experiment, opens each
    flag's group with a ``*<flag>*`` stub heading in the Condition column, and
    drops the Flag and Experiment keys from the condition rows beneath it, so
    the widest column of the old layout disappears entirely.
    """
    unknown = {r["Experiment"] for r in rows} - set(_EXPERIMENT_ORDER)
    if unknown:
        raise ValueError(f"unknown experiment label: {sorted(unknown)!r}")
    out: list[dict[str, str]] = []
    for experiment in _EXPERIMENT_ORDER:
        block = [r for r in rows if r["Experiment"] == experiment]
        if not block:
            continue
        out.append({"Condition": f"**{experiment} Experiment**"})
        current_flag: str | None = None
        for row in block:
            if row["Flag"] != current_flag:
                current_flag = row["Flag"]
                out.append({"Condition": f"*{current_flag}*"})
            out.append({
                k: v for k, v in row.items() if k not in ("Flag", "Experiment")
            })
    return out


# --- Supplementary Table: probe-auditor alignment (paired) ---

# The audit answers two questions over one row set, so the row builder emits
# every field and each table selects what it needs. These three constants are
# the row-field manifests of that flag-major output, not rendered column
# orders: the exported tables pass the rows through
# format_probe_stub_layout_rows, which lifts the Flag and Experiment keys
# into stub and cut-in heading rows, so each export projects only the
# Condition column and its own metric columns.
PROBE_AUDITOR_KEY_COLUMNS: list[str] = ["Flag", "Condition", "Experiment"]

PROBE_AUDITOR_ALIGNMENT_COLUMNS: list[str] = [
    *PROBE_AUDITOR_KEY_COLUMNS,
    "Audited Trial Count",
    "Aligned Count",
    "Alignment Rate (%)",
    "Confidence Sequence Lower Bound",
]

PROBE_AUDITOR_DISAGREEMENT_TRANSITION_COLUMNS: list[str] = [
    *PROBE_AUDITOR_KEY_COLUMNS,
    "Stable Agreement",
    "Shifted Toward Probe",
    "Stable Disagreement",
    "Shifted Against Probe",
]


def _format_rate_or_dash(rate: float) -> str:
    """Percentage rate, or the em-dash null sentinel when undefined (NaN)."""
    return format_percent(None) if math.isnan(rate) else format_percent(rate)


def _format_bound_or_dash(value: float) -> str:
    """One-sided confidence sequence lower bound as a three-decimal score.

    Renders the em-dash null sentinel when the bound is undefined (NaN) or
    suppressed for a coverage-convergent floored flag — never a fabricated
    numeric bound.
    """
    return format_accuracy(None) if math.isnan(value) else format_accuracy(value)


def resolve_floored_flags(
    pool_results: list[ProbeAuditResults],
) -> frozenset[str]:
    """Flags whose pooled confidence sequence bound must be suppressed.

    Only a coverage-convergent draw floors rare flags (recorded in
    ``provenance.classified_rare_flags``); a stratified pool floors nothing.
    Mirrors the probe-audit summary's suppression decision so the table's
    em-dash bounds match the rendered reports.
    """
    materialized = list(pool_results)
    if not materialized:
        return frozenset()
    provenance = materialized[0].provenance
    if provenance.batch_allocation != "coverage_convergent":
        return frozenset()
    return frozenset(provenance.classified_rare_flags or ())


def format_probe_auditor_alignment_rows(
    experiment_pools: list[tuple[str, list[ProbeAuditResults], frozenset[str]]],
) -> list[dict[str, str]]:
    """Format per-flag per-condition probe-auditor alignment rows for both experiments.

    ``experiment_pools`` is a sequence of ``(experiment_label, pool_results,
    floored_flags)`` — one per experiment. Rows are emitted flag-major, then in
    the given experiment order, then condition (correct-draft, transparent,
    opaque), matching the per-flag alignment tables of the probe-audit summary.

    Each row's counts, alignment rate, and four disagreement-transition
    categories come straight from the canonical pooled aggregation
    (``pooled_per_condition_probe_correctness`` and
    ``pooled_disagreement_category_counts_by_condition``); the aligned count is
    the probe-correct count and the four transition counts sum to the audited
    trial count. For a flag in that experiment's ``floored_flags`` the pooled
    lower bound is suppressed to the em-dash sentinel (the pooled rate mixes the
    probe's classification classes at an enriched ratio and is not a pool rate);
    the rate and counts are still reported, exactly as the summary does.
    """
    display_rows: list[dict[str, str]] = []
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        flag_label = RATIONALE_ANALYSIS_FLAG_DISPLAY_NAMES[flag]
        for experiment_label, pool_results, floored_flags in experiment_pools:
            results = list(pool_results)
            pcc_by_cond = {
                r["condition"]: r
                for r in pooled_per_condition_probe_correctness(results, flag=flag)
            }
            cat_by_cond = {
                r["condition"]: r
                for r in pooled_disagreement_category_counts_by_condition(
                    results, flag=flag,
                )
            }
            suppress = flag in floored_flags
            for cond in ("correct", "transparent", "opaque"):
                pcc = pcc_by_cond[cond]
                cat = cat_by_cond[cond]
                bound_value = math.nan if suppress else pcc["cs_lower"]
                display_rows.append({
                    "Flag": flag_label,
                    "Condition": _CONDITION_DISPLAY[cond],
                    "Experiment": experiment_label,
                    "Audited Trial Count": str(pcc["n"]),
                    "Aligned Count": str(pcc["probe_correct"]),
                    "Alignment Rate (%)": _format_rate_or_dash(
                        pcc["probe_correctness_rate"],
                    ),
                    "Confidence Sequence Lower Bound": _format_bound_or_dash(bound_value),
                    "Stable Agreement": str(cat["stable_agreement"]),
                    "Shifted Toward Probe": str(cat["shifted_to_probe"]),
                    "Stable Disagreement": str(cat["stable_disagreement"]),
                    "Shifted Against Probe": str(cat["shifted_against_probe"]),
                })
    return display_rows


# --- Per-configuration failure mode breakdown ---

FAILURE_MODE_BY_CONFIG_COLUMNS: list[str] = [
    "Configuration",
    "Condition",
    "Balanced Accuracy [Primary]",
    "Wrong-Verdict Trial Count [Primary]",
    "Articulation Rate (%) [Primary]",
    "Governing Rate (%) [Primary]",
    "Balanced Accuracy [Ablation]",
    "Wrong-Verdict Trial Count [Ablation]",
    "Articulation Rate (%) [Ablation]",
    "Governing Rate (%) [Ablation]",
]


def format_failure_mode_by_config_rows(
    primary_rates_by_config: list[dict[str, Any]],
    ablation_rates_by_config: list[dict[str, Any]],
    primary_summaries: tuple[ConfigurationSummary, ...],
    ablation_summaries: tuple[ConfigurationSummary, ...],
) -> list[dict[str, str]]:
    """Format paired per-configuration failure mode rows for the summary table.

    Joins the per (config, condition) articulation and governing rates from
    ``aggregate_failure_mode_rates_by_config`` (one list per experiment) with
    each experiment's per-configuration balanced accuracy. The wrong-verdict
    trial count is the per (config, condition) denominator; balanced accuracy
    is per configuration and repeats across a configuration's three condition
    rows. A rate over a zero denominator renders as the em-dash null sentinel.
    Each experiment's columns form a contiguous group so the spanner headings
    can name the experiment once; the primary-to-ablation change is read
    across a row's adjacent groups rather than restated in a delta column.
    """
    ablation_lookup = {
        (r["config_key"], r["condition"]): r for r in ablation_rates_by_config
    }
    primary_balanced = {s.config_key: s.balanced_accuracy for s in primary_summaries}
    ablation_balanced = {s.config_key: s.balanced_accuracy for s in ablation_summaries}

    def _fmt_rate(rate: float | None) -> str:
        return format_percent(rate) if rate is not None else "—"

    display_rows: list[dict[str, str]] = []
    for pr in primary_rates_by_config:
        config_key = pr["config_key"]
        ar = ablation_lookup.get((config_key, pr["condition"]), {})
        display_rows.append({
            "Configuration": display_config_key(config_key),
            "Condition": _CONDITION_DISPLAY[pr["condition"]],
            "Balanced Accuracy [Primary]": format_accuracy(primary_balanced.get(config_key)),
            "Wrong-Verdict Trial Count [Primary]": f"{pr.get('total_mismatches', 0):,d}",
            "Articulation Rate (%) [Primary]": _fmt_rate(pr.get("articulation_rate")),
            "Governing Rate (%) [Primary]": _fmt_rate(pr.get("governing_rate")),
            "Balanced Accuracy [Ablation]": format_accuracy(ablation_balanced.get(config_key)),
            "Wrong-Verdict Trial Count [Ablation]": f"{ar.get('total_mismatches', 0):,d}",
            "Articulation Rate (%) [Ablation]": _fmt_rate(ar.get("articulation_rate")),
            "Governing Rate (%) [Ablation]": _fmt_rate(ar.get("governing_rate")),
        })
    return display_rows


def export_table_markdown(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    *,
    columns: list[str] | None = None,
) -> Path:
    """Write table data as a markdown file with a pipe-delimited table.

    The first content line is always the pipe-delimited header row. The
    realized valid trial disclosure for each inserted table lives in the
    hand-authored caption, not in a generated preamble, so the build's table
    parser never has to skip a non-table line.

    Parameters
    ----------
    rows:
        Display-ready row dicts (keys become column headers).
    output_dir:
        Target directory (created if absent).
    filename:
        Base filename without extension (.md appended automatically).
    columns:
        Column keys to include and their order. Defaults to all keys
        from the first row.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.md"

    if not rows:
        path.write_text("*(empty table)*\n")
        return path

    cols = columns or list(rows[0].keys())
    def _esc(value: Any) -> str:
        """Cell text with the null sentinel applied and pipe characters escaped."""
        return str("\u2014" if value is None else value).replace("|", "\\|")

    header_line = "| " + " | ".join(_esc(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    data_lines = [
        "| " + " | ".join(_esc(r.get(c, "")) for c in cols) + " |"
        for r in rows
    ]
    content = "\n".join([header_line, separator, *data_lines, ""])
    path.write_text(content)
    return path
