"""Data transformation functions for results communication visualizations.

Each function transforms ExperimentAnalysisResults (or RationaleAnalysisResults)
into a flat list[dict] suitable for Plotly figures and tables. No Plotly
dependency — pure data reshaping following the functional pipeline pattern.
"""

import math

from typing import Any

from pydantic import BaseModel, ConfigDict

from utils.experiment_analysis.config_identity import (
    REASONING_SCALES,
    config_tie_break_key,
    parse_config_key,
    require_compute_position,
)
from utils.experiment_analysis.metrics import (
    usage_cost_summary,
)
from utils.experiment_analysis.ground_truth import parse_example_id
from utils.experiment_analysis.comparisons import (
    BET_CAP,
    ComparisonSlice,
    CrossDatasetComparison,
    ModelSizeComparison,
    compute_all_comparisons,
    compute_global_null_comparison,
)
from utils.experiment_analysis.models import TrialOutcome
from utils.experiment_analysis.pipeline import ExperimentAnalysisResults
from utils.experiment_analysis.stimulus_metadata import STIMULUS_METADATA_REGISTRY
from utils.rationale_analysis.models import (
    MISATTRIBUTION_FLAG_KEYS,
    RationaleAnalysisResults,
)
from utils.token_costs import compute_response_tokens


_EXPERIMENT_SORT_ORDER: dict[str, int] = {"Primary": 0, "Ablation": 1}
"""Deliberate display order for combined [Primary + Ablation] tables.

Used as the first element of sort keys in every ``combined_*`` helper so the
Primary block renders above the Ablation block. Lexicographic sorting on the
raw ``"experiment"`` string would put Ablation first, which reads backward to
researchers — Primary is the canonical experiment and should lead.
"""


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Compute rate, returning None when the denominator is zero."""
    return numerator / denominator if denominator > 0 else None


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (type 7) of an already-sorted list.

    ``q`` is a fraction in [0, 1]. Matches numpy's default ``percentile``
    interpolation so the reported quantiles line up with any numpy-based
    cross-check. Raises on empty input, since a quantile of no data is
    undefined.
    """
    if not sorted_values:
        raise ValueError("percentile of an empty sequence is undefined")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


# Design nominal valid trial counts. These are the comparison baseline the
# disclosure guard checks against, NOT a denominator to display: the disclosed
# N is always the realized count read from the data. A run with no parse
# failures has 200 valid trials per (configuration, condition) and 600 per
# configuration; any shortfall (a parse failure excludes its trial) drops the
# realized count below these values and trips the guard.
NOMINAL_VALID_PER_CONDITION: int = 200
NOMINAL_VALID_PER_CONFIG: int = 600


class ValidTrialDisclosure(BaseModel):
    """Realized valid trial counts behind a table's proportion denominators.

    Sourced entirely from ``cell_metrics_list`` so the disclosed N is the
    actual denominator each proportion was computed over, never a hard-coded
    nominal. ``uniform`` is True only when every (configuration, condition)
    realized exactly ``NOMINAL_VALID_PER_CONDITION`` valid trials (which also
    makes every configuration total ``NOMINAL_VALID_PER_CONFIG`` and equalizes
    the per-condition counts within each configuration). When ``uniform`` is
    False the disclosure guard has tripped: at least one cell fell below
    nominal or the per-condition counts within a configuration are unequal,
    and a table must enumerate or annotate the affected denominators rather
    than render bare proportions.
    """

    model_config = ConfigDict(frozen=True)

    per_config_condition: dict[tuple[str, str], int]
    per_config: dict[str, int]
    uniform: bool


def valid_trial_disclosure(
    results: ExperimentAnalysisResults,
) -> ValidTrialDisclosure:
    """Read realized valid trial counts per (configuration, condition) and per configuration.

    Aggregates ``cell_metrics_list`` (one entry per stimulus × configuration ×
    condition) into per-(configuration, condition) and per-configuration valid
    totals, then decides uniformity against the design nominal. Uniformity
    requires every cell to realize exactly ``NOMINAL_VALID_PER_CONDITION``;
    any shortfall or any within-configuration per-condition inequality marks
    the disclosure non-uniform so the table surfaces the reduced denominator.
    """
    per_cc: dict[tuple[str, str], int] = {}
    per_config: dict[str, int] = {}
    for cm in results.cell_metrics_list:
        config_key = cm.cell_key.config_key
        cc_key = (config_key, cm.condition)
        per_cc[cc_key] = per_cc.get(cc_key, 0) + cm.valid_trials
        per_config[config_key] = per_config.get(config_key, 0) + cm.valid_trials

    cells_at_nominal = all(
        v == NOMINAL_VALID_PER_CONDITION for v in per_cc.values()
    )
    # Per-condition equality within each configuration is implied when every
    # cell equals the nominal; checked explicitly so the invariant is guarded
    # rather than inferred.
    by_config_conditions: dict[str, list[int]] = {}
    for (config_key, _condition), valid in per_cc.items():
        by_config_conditions.setdefault(config_key, []).append(valid)
    per_condition_equal = all(
        len(set(counts)) == 1 for counts in by_config_conditions.values()
    )
    uniform = bool(per_cc) and cells_at_nominal and per_condition_equal

    return ValidTrialDisclosure(
        per_config_condition=per_cc,
        per_config=per_config,
        uniform=uniform,
    )


def cross_dataset_valid_trial_disclosure(
    rows: list[dict[str, Any]],
    *,
    level: str,
) -> ValidTrialDisclosure:
    """Build a valid trial disclosure from cross-dataset comparison rows.

    Sources N from each comparison's ``primary_valid`` and ``ablation_valid``
    (added by ``cross_dataset_significance_data``) rather than from a single
    experiment's ``cell_metrics_list``: each cross-dataset cell has a
    realized denominator in each experiment, and the disclosed N is the
    smaller of the two so a reduced count on either side surfaces.
    ``level`` is ``"condition"`` (nominal ``NOMINAL_VALID_PER_CONDITION`` per
    experiment) or ``"config"`` (nominal ``NOMINAL_VALID_PER_CONFIG`` per
    experiment, pooled across conditions).
    """
    if level not in ("condition", "config"):
        raise ValueError(f"level must be 'condition' or 'config', got {level!r}")
    nominal = (
        NOMINAL_VALID_PER_CONDITION if level == "condition"
        else NOMINAL_VALID_PER_CONFIG
    )

    per_cc: dict[tuple[str, str], int] = {}
    per_config: dict[str, int] = {}
    uniform = bool(rows)
    for r in rows:
        primary_valid = r["primary_valid"]
        ablation_valid = r["ablation_valid"]
        realized = min(primary_valid, ablation_valid)
        if primary_valid != nominal or ablation_valid != nominal:
            uniform = False
        if level == "condition":
            per_cc[(r["config_key"], r["condition"])] = realized
        else:
            per_config[r["config_key"]] = realized

    return ValidTrialDisclosure(
        per_config_condition=per_cc,
        per_config=per_config,
        uniform=uniform,
    )


def _empirical_e_power(slices: tuple[ComparisonSlice, ...]) -> float | None:
    """Mean log e-value gained per betting round of a paired fold.

    The evidence growth rate ("speed" in Experiment Design Section 7) of
    the sequential test, including the calibration round, which places no
    bet and contributes zero. None when no round slices exist — a comparison
    with nothing to pair, whose fold never ran.
    """
    if not slices:
        return None
    return sum(s.log_e_value for s in slices) / len(slices)


def condition_accuracy_by_config(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """Flat table: one row per (config_key, condition) with accuracy.

    Powers the per-condition accuracy chart. Rows follow the structural order
    (``config_tie_break_key``: provider, model, ascending reasoning effort),
    then condition order (transparent, opaque, correct) within each
    configuration, so every configuration axis in the deck reads the same way.
    """
    condition_order = {"transparent": 0, "opaque": 1, "correct": 2}
    rows: list[dict[str, Any]] = []

    for summary in results.config_summaries:
        provider, model_slug, reasoning_effort_level = parse_config_key(
            summary.config_key,
        )
        for condition, accuracy in summary.condition_accuracies.items():
            rows.append({
                "config_key": summary.config_key,
                "provider": provider,
                "model_slug": model_slug,
                "reasoning_effort_level": reasoning_effort_level,
                "condition": condition,
                "accuracy": accuracy,
                "overall_accuracy": summary.overall_accuracy,
                "balanced_accuracy": summary.balanced_accuracy,
            })

    # The structural key precedes condition_order so a configuration's three
    # rows stay contiguous rather than interleaving across configurations.
    return sorted(
        rows,
        key=lambda r: (
            *config_tie_break_key(r["config_key"]),
            condition_order.get(r["condition"], 99),
        ),
    )


def example_accuracy_matrix(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """Flat table: one row per (base_example, config_key) with per-condition accuracy.

    Powers the per-example condition accuracy heatmaps. Includes the per-config
    balanced accuracy the column ordering is derived from.
    """
    # Group trial outcomes by (base_example, config_key, condition)
    groups: dict[tuple[str, str, str], list[bool]] = {}
    for outcome in results.trial_outcomes:
        if outcome.matches_ground_truth is None:
            continue
        key = (outcome.base_example, outcome.config_key, outcome.condition)
        groups.setdefault(key, []).append(outcome.matches_ground_truth)

    # Build balanced accuracy lookup from config summaries
    score_lookup = {
        s.config_key: s.balanced_accuracy for s in results.config_summaries
    }

    # Pivot to one row per (base_example, config_key) with per-condition accuracy
    cell_data: dict[tuple[str, str], dict[str, Any]] = {}
    for (base_example, config_key, condition), matches in groups.items():
        cell_key = (base_example, config_key)
        if cell_key not in cell_data:
            cell_data[cell_key] = {
                "base_example": base_example,
                "config_key": config_key,
                "balanced_accuracy": score_lookup.get(config_key, 0.0),
                "correct_acc": 0.0,
                "transparent_acc": 0.0,
                "opaque_acc": 0.0,
            }
        acc = sum(matches) / len(matches) if matches else 0.0
        cell_data[cell_key][f"{condition}_acc"] = acc

    return list(cell_data.values())


def stimuli_summary_rows(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """One row per base example merging design metadata with pooled performance.

    Accuracies and valid trial counts are computed directly from
    ``results.trial_outcomes`` — exact hit counts pooled across all
    configurations. A condition with no valid trials yields accuracy None
    and 0 valid trials; the per-stimulus overall accuracy is the unweighted
    mean of the condition accuracies present for that base example (None when
    no condition has valid trials), which is a per-stimulus quantity rather
    than the pooled configuration-level metric of the same name. Rows are
    sorted ascending by that value (hardest first) with None last.

    Powers the Stimuli Summary table in the experiment analysis notebook.
    """
    # Exact (hits, valid) per (base_example, condition), pooled across configs.
    # Bases are collected from ALL outcomes so an example whose every trial
    # failed to parse still gets a row (all-None accuracies, 0 valid).
    bases: set[str] = set()
    tallies: dict[tuple[str, str], list[int]] = {}
    for o in results.trial_outcomes:
        bases.add(o.base_example)
        if o.matches_ground_truth is None:
            continue
        slot = tallies.setdefault((o.base_example, o.condition), [0, 0])
        slot[1] += 1
        if o.matches_ground_truth:
            slot[0] += 1

    rows: list[dict[str, Any]] = []
    for base in sorted(bases):
        meta = STIMULUS_METADATA_REGISTRY.get(base)

        per_condition: dict[str, tuple[float | None, int]] = {}
        for condition in ("correct", "transparent", "opaque"):
            hits, valid = tallies.get((base, condition), (0, 0))
            accuracy = hits / valid if valid > 0 else None
            per_condition[condition] = (accuracy, valid)

        present = [acc for acc, _ in per_condition.values() if acc is not None]
        overall = sum(present) / len(present) if present else None

        rows.append({
            "example_number": base.replace("example_", ""),
            "domain": meta.domain if meta else "",
            "risk": meta.risk if meta else "",
            "correct_fallback": meta.current_value if meta else "",
            "historical_previous": meta.historical_value if meta else "",
            "proposed": meta.proposed_value if meta else "",
            "correct_draft_accuracy": per_condition["correct"][0],
            "correct_draft_valid_trials": per_condition["correct"][1],
            "transparent_accuracy": per_condition["transparent"][0],
            "transparent_valid_trials": per_condition["transparent"][1],
            "opaque_accuracy": per_condition["opaque"][0],
            "opaque_valid_trials": per_condition["opaque"][1],
            "overall_accuracy": overall,
        })

    # Ascending by overall accuracy (hardest first); None sorts last
    return sorted(
        rows,
        key=lambda r: (
            r["overall_accuracy"] is None,
            r["overall_accuracy"] if r["overall_accuracy"] is not None else 0.0,
        ),
    )


def reasoning_effort_curves(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """One row per (model_group, reasoning_effort_level) with balanced_accuracy.

    Powers the reasoning effort line charts. Only includes
    model groups with 2+ reasoning effort levels in REASONING_SCALES.
    """
    rows: list[dict[str, Any]] = []
    for summary in results.config_summaries:
        provider, model_slug, reasoning_effort_level = parse_config_key(
            summary.config_key,
        )
        model_group = f"{provider}--{model_slug}"

        # Only include models with graduated reasoning scales
        scale_key = (provider, model_slug)
        if scale_key not in REASONING_SCALES:
            continue
        if len(REASONING_SCALES[scale_key]) < 2:
            continue

        pos = require_compute_position(summary.config_key)
        rows.append({
            "model_group": model_group,
            "provider": provider,
            "model_slug": model_slug,
            "reasoning_effort_level": reasoning_effort_level,
            "ordinal": pos[0],
            "scale_size": pos[1],
            "balanced_accuracy": summary.balanced_accuracy,
        })

    return sorted(
        rows,
        key=lambda r: (r["provider"], r["model_slug"], r["ordinal"]),
    )


def failure_mode_by_config(
    ra_results: RationaleAnalysisResults,
    *,
    config_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    """One row per config_key with per-condition failure mode counts.

    The discriminating flag differs by condition per the Methods: Stimulus Design
    specification:

    - **transparent / opaque:** ``articulated_operational_interpretation``
      (the articulation classification). ``True`` → Expressed;
      ``False`` → Never expressed.
    - **correct:** ``operational_interpretation_governed_judgment``
      (the governing classification). ``True`` → Operational interpretation;
      ``False`` → Field label matching.

    Records where the discriminating flag is None (insufficient parses to
    resolve) are excluded.

    ``config_order`` is both the config universe and the output order: one row
    per entry, in that sequence, so configs with zero classifiable failures
    still get a row (all counts zero). The charts disclose that order on their
    x-axis, so the caller owns it — pass
    ``figure_builders.structural_config_order`` for the grouped-by-model order
    the failure mode figures claim. ``total_failures`` remains a magnitude
    field and no longer orders anything. Raises ``ValueError`` when a trial
    record names a config absent from ``config_order``.

    Powers the per-condition failure mode faceted chart and the workbook
    failure mode table.
    """
    _ZERO_CONDITION = {"flag_true": 0, "flag_false": 0, "total": 0}

    # Seed the whole universe up front: every config in config_order gets a
    # bucket (so zero-failure configs still produce a row), and any record
    # naming a config outside it is a data integrity error rather than an
    # extra bar the caller never asked to plot.
    config_cond_counts: dict[str, dict[str, dict[str, int]]] = {
        config_key: {
            c: {**_ZERO_CONDITION} for c in ("correct", "transparent", "opaque")
        }
        for config_key in config_order
    }
    for record in ra_results.trial_records:
        _, condition = parse_example_id(record.example_id)

        if record.config_key not in config_cond_counts:
            raise ValueError(
                f"Trial record config {record.config_key!r} is absent from "
                "config_order"
            )

        # Select the condition-specific discriminating flag
        flag = (
            record.operational_interpretation_governed_judgment
            if condition == "correct"
            else record.articulated_operational_interpretation
        )
        if flag is None:
            continue

        bucket = config_cond_counts[record.config_key][condition]
        bucket["total"] += 1
        if flag:
            bucket["flag_true"] += 1
        else:
            bucket["flag_false"] += 1

    rows: list[dict[str, Any]] = []
    for config_key in config_order:
        cond_map = config_cond_counts[config_key]
        cor = cond_map["correct"]
        tra = cond_map["transparent"]
        opa = cond_map["opaque"]
        total = cor["total"] + tra["total"] + opa["total"]
        rows.append({
            "config_key": config_key,
            # Correct-draft: governing classification discriminates governed vs field label
            "correct_governed": cor["flag_true"],
            "correct_field_label": cor["flag_false"],
            "correct_total": cor["total"],
            # Transparent: articulation classification discriminates expressed-but-dismissed vs never-expressed
            "transparent_selection_failed": tra["flag_true"],
            "transparent_capability_absent": tra["flag_false"],
            "transparent_total": tra["total"],
            # Opaque: articulation classification discriminates expressed-but-dismissed vs never-expressed
            "opaque_selection_failed": opa["flag_true"],
            "opaque_capability_absent": opa["flag_false"],
            "opaque_total": opa["total"],
            "total_failures": total,
        })

    return rows


def failure_mode_rates_by_config(
    ra_results: RationaleAnalysisResults,
    *,
    config_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    """One row per config_key with per-condition failure mode RATES.

    Normalizes the integer counts from ``failure_mode_by_config`` to rates
    (count / total). Rates are directly comparable across configurations
    with different mismatch totals.

    Rows follow ``config_order`` exactly, inherited from
    ``failure_mode_by_config`` — see there for the ordering contract and the
    unknown-config error.

    Powers the two failure mode rate charts, which are notebook analysis
    views; the paper's placed failure mode figures are the count
    breakdown composites.
    """
    count_rows = failure_mode_by_config(ra_results, config_order=config_order)

    rows: list[dict[str, Any]] = []
    for r in count_rows:
        rows.append({
            "config_key": r["config_key"],
            # Correct-draft: governed vs field label rates
            "correct_governed_rate": _safe_rate(r["correct_governed"], r["correct_total"]),
            "correct_field_label_rate": _safe_rate(r["correct_field_label"], r["correct_total"]),
            "correct_total": r["correct_total"],
            # Transparent: expressed-but-dismissed vs never-expressed rates
            "transparent_selection_failed_rate": _safe_rate(
                r["transparent_selection_failed"], r["transparent_total"],
            ),
            "transparent_capability_absent_rate": _safe_rate(
                r["transparent_capability_absent"], r["transparent_total"],
            ),
            "transparent_total": r["transparent_total"],
            # Opaque: expressed-but-dismissed vs never-expressed rates
            "opaque_selection_failed_rate": _safe_rate(
                r["opaque_selection_failed"], r["opaque_total"],
            ),
            "opaque_capability_absent_rate": _safe_rate(
                r["opaque_capability_absent"], r["opaque_total"],
            ),
            "opaque_total": r["opaque_total"],
            "total_failures": r["total_failures"],
        })

    return rows


def aggregate_failure_mode_rates(
    ra_results: RationaleAnalysisResults,
) -> list[dict[str, Any]]:
    """Aggregate articulation and governing rates per condition across all configs.

    Returns one row per condition with:
    - articulation_rate: proportion of mismatches where the operational
      interpretation was articulated
    - governing_rate: proportion of mismatches where the operational
      interpretation governed the judgment
    - misattribution_flagged_rate: proportion of mismatches that fired at
      least one misattribution flag — the confound-disclosure annotation of
      docs/rationale_analysis.md Section 2.4
    - misattribution_flagged: numerator of that rate (mismatches with >=1
      misattribution flag True)
    - total_mismatches: the shared denominator (records with both deictic
      flags resolved)

    All three rates share the ``total_mismatches`` denominator, so the
    misattribution annotation is a coherent fraction of the same rows the
    articulation/governing rates describe. This differs from
    ``misattribution_rates_by_condition``, whose any-flag denominator is the
    determinate-record count; the two coincide on real snapshots (zero
    unresolved flags) but can diverge on records with unresolved flags.

    Call once per experiment (primary, ablation) and join downstream for
    side-by-side comparison. Consistent with the single-results-in pattern
    used throughout this module.
    """
    # Accumulate resolved flag counts per condition
    condition_counts: dict[str, dict[str, int]] = {
        cond: {"articulated": 0, "governed": 0, "misattribution_flagged": 0, "total": 0}
        for cond in ("correct", "transparent", "opaque")
    }

    for record in ra_results.trial_records:
        _, condition = parse_example_id(record.example_id)
        bucket = condition_counts[condition]

        # Only count records where both flags resolved (not None)
        art = record.articulated_operational_interpretation
        gov = record.operational_interpretation_governed_judgment
        if art is None or gov is None:
            continue

        bucket["total"] += 1
        if art:
            bucket["articulated"] += 1
        if gov:
            bucket["governed"] += 1
        # Confound annotation: did this mismatch misreport at least one fact?
        if any(getattr(record, k) is True for k in MISATTRIBUTION_FLAG_KEYS):
            bucket["misattribution_flagged"] += 1

    rows: list[dict[str, Any]] = []
    for condition in ("correct", "transparent", "opaque"):
        c = condition_counts[condition]
        rows.append({
            "condition": condition,
            "articulation_rate": _safe_rate(c["articulated"], c["total"]),
            "governing_rate": _safe_rate(c["governed"], c["total"]),
            "misattribution_flagged_rate": _safe_rate(
                c["misattribution_flagged"], c["total"],
            ),
            "misattribution_flagged": c["misattribution_flagged"],
            "total_mismatches": c["total"],
        })

    return rows


_PAIRWISE_CONDITION_KEYS: dict[frozenset[str], str] = {
    frozenset(("transparent", "opaque")): "transparent_vs_opaque_e_value",
    frozenset(("transparent", "correct")): "transparent_vs_correct_e_value",
    frozenset(("opaque", "correct")): "opaque_vs_correct_e_value",
}
"""Maps each unordered condition pair to its pairwise-comparison output column."""

_DISTRIBUTION_CONDITIONS: tuple[str, ...] = ("correct", "transparent", "opaque")


def pairwise_comparison_e_values_by_config(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """One row per config_key with the three pairwise condition comparison e-values.

    Reads the materialized testing-by-betting e-values from a single
    experiment snapshot's ``comparisons`` and pivots the three condition pairs
    (transparent vs opaque, transparent vs correct-draft, opaque vs
    correct-draft) into columns. The e-values are pinned in the snapshot;
    nothing is recomputed here. Rows are ordered by provider, model, and
    ascending reasoning effort.
    """
    by_config: dict[str, dict[str, float]] = {}
    for comp in results.comparisons:
        column = _PAIRWISE_CONDITION_KEYS.get(
            frozenset((comp.condition_a, comp.condition_b)),
        )
        if column is None:
            raise ValueError(
                f"Unexpected condition pair ({comp.condition_a}, "
                f"{comp.condition_b}) for config {comp.config_key!r}"
            )
        by_config.setdefault(comp.config_key, {})[column] = comp.running_e_value

    rows: list[dict[str, Any]] = []
    for config_key in sorted(by_config, key=config_tie_break_key):
        pairs = by_config[config_key]
        rows.append({
            "config_key": config_key,
            "transparent_vs_opaque_e_value": pairs.get("transparent_vs_opaque_e_value"),
            "transparent_vs_correct_e_value": pairs.get("transparent_vs_correct_e_value"),
            "opaque_vs_correct_e_value": pairs.get("opaque_vs_correct_e_value"),
        })
    return rows


def e_value_distribution_by_condition(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """One row per condition summarizing the per-cell log e-value distribution.

    Aggregates the snapshot's ``cell_metrics_list`` within each condition into
    log e-value quantiles (minimum, first quartile, median, third quartile,
    maximum), the mean empirical e-power (mean log e-value gained per batch),
    the mean confidence sequence width (``cs_upper - cs_lower``), and the
    fraction of cells resolved by rejection. Every quantity is read from pinned
    per-cell metrics; nothing is recomputed.
    """
    rows: list[dict[str, Any]] = []
    for condition in _DISTRIBUTION_CONDITIONS:
        cells = [cm for cm in results.cell_metrics_list if cm.condition == condition]
        if not cells:
            raise ValueError(f"No cells found for condition {condition!r}")
        log_e_values = sorted(cm.log_e_value for cm in cells)
        rejected = sum(1 for cm in cells if cm.status == "rejected")
        rows.append({
            "condition": condition,
            "log_e_value_min": log_e_values[0],
            "log_e_value_q1": _percentile(log_e_values, 0.25),
            "log_e_value_median": _percentile(log_e_values, 0.5),
            "log_e_value_q3": _percentile(log_e_values, 0.75),
            "log_e_value_max": log_e_values[-1],
            "mean_empirical_e_power": sum(cm.empirical_e_power for cm in cells) / len(cells),
            "mean_cs_width": sum(cm.cs_upper - cm.cs_lower for cm in cells) / len(cells),
            "cells_rejected_rate": _safe_rate(rejected, len(cells)),
            "cell_count": len(cells),
        })
    return rows


def aggregate_failure_mode_rates_by_config(
    ra_results: RationaleAnalysisResults,
    all_config_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Per (config_key, condition) articulation and governing rates.

    The per-configuration analogue of ``aggregate_failure_mode_rates``: it
    partitions ``trial_records`` by both config and condition and, using only
    records where both deictic flags resolved, reports the articulation rate,
    the governing rate, and their shared denominator (``total_mismatches``).
    Every config in ``all_config_keys`` yields three rows (correct,
    transparent, opaque); a (config, condition) pair with no eligible trials
    reports a None rate and a zero denominator so downstream renders the
    em-dash null sentinel rather than a fabricated 0%. Rows are ordered by
    provider, model, ascending reasoning effort, then condition.
    """
    counts: dict[tuple[str, str], dict[str, int]] = {
        (config_key, condition): {"articulated": 0, "governed": 0, "total": 0}
        for config_key in all_config_keys
        for condition in _DISTRIBUTION_CONDITIONS
    }
    for record in ra_results.trial_records:
        _, condition = parse_example_id(record.example_id)
        bucket = counts.get((record.config_key, condition))
        if bucket is None:
            raise ValueError(
                f"Trial record config {record.config_key!r} is absent from "
                "all_config_keys"
            )
        art = record.articulated_operational_interpretation
        gov = record.operational_interpretation_governed_judgment
        if art is None or gov is None:
            continue
        bucket["total"] += 1
        if art:
            bucket["articulated"] += 1
        if gov:
            bucket["governed"] += 1

    rows: list[dict[str, Any]] = []
    for config_key in sorted(all_config_keys, key=config_tie_break_key):
        for condition in _DISTRIBUTION_CONDITIONS:
            c = counts[(config_key, condition)]
            rows.append({
                "config_key": config_key,
                    "condition": condition,
                "articulation_rate": _safe_rate(c["articulated"], c["total"]),
                "governing_rate": _safe_rate(c["governed"], c["total"]),
                "total_mismatches": c["total"],
            })
    return rows


def deconfounded_failure_mode_rates(
    ra_results: RationaleAnalysisResults,
) -> list[dict[str, Any]]:
    """Articulation and governing rates per condition over the clean subset.

    The de-confounded view: identical in shape to
    ``aggregate_failure_mode_rates`` (one row per condition with
    ``articulation_rate``, ``governing_rate``, ``total_mismatches``) but
    computed over the *clean subset* — records where both deictic flags
    resolved, all five misattribution flags resolved, and none of the five
    fired. Removing the confound candidates (docs/rationale_analysis.md
    Section 2.4) isolates the deictic capability from the factual-misread
    confound; the gap between this rate and ``aggregate_failure_mode_rates``
    measures how much the confound inflated the unfiltered number — the
    "computed over the subset of trials with no misattribution flag set"
    option of docs/rationale_analysis.md Section 7.2.

    A record with any misattribution flag None is indeterminate and
    excluded: a None is never read as False, mirroring the determinate
    policy of ``misattribution_rates_by_condition``. The
    both-deictic-flags-resolved gate matches ``aggregate_failure_mode_rates``.
    The misattribution annotation is omitted from the row because it is
    vacuously zero over a subset defined by the absence of fired flags.

    Call once per experiment (primary, ablation) and join downstream for the
    side-by-side full-vs-clean comparison. Consistent with the
    single-results-in pattern used throughout this module.
    """
    condition_counts: dict[str, dict[str, int]] = {
        cond: {"articulated": 0, "governed": 0, "total": 0}
        for cond in ("correct", "transparent", "opaque")
    }

    for record in ra_results.trial_records:
        _, condition = parse_example_id(record.example_id)

        # Both deictic flags must resolve (mirrors aggregate_failure_mode_rates).
        art = record.articulated_operational_interpretation
        gov = record.operational_interpretation_governed_judgment
        if art is None or gov is None:
            continue

        # Clean subset: every misattribution flag resolved and none fired.
        # An unresolved (None) flag is indeterminate, never coerced to False,
        # so the record is excluded rather than silently counted as clean.
        misattribution = [getattr(record, k) for k in MISATTRIBUTION_FLAG_KEYS]
        if any(value is None for value in misattribution):
            continue
        if any(value is True for value in misattribution):
            continue

        bucket = condition_counts[condition]
        bucket["total"] += 1
        if art:
            bucket["articulated"] += 1
        if gov:
            bucket["governed"] += 1

    rows: list[dict[str, Any]] = []
    for condition in ("correct", "transparent", "opaque"):
        c = condition_counts[condition]
        rows.append({
            "condition": condition,
            "articulation_rate": _safe_rate(c["articulated"], c["total"]),
            "governing_rate": _safe_rate(c["governed"], c["total"]),
            "total_mismatches": c["total"],
        })

    return rows


def misattribution_rates_by_condition(
    ra_results: RationaleAnalysisResults,
) -> list[dict[str, Any]]:
    """Per-condition misattribution-flag rates across all configs.

    Returns one row per condition (``correct``, ``transparent``,
    ``opaque``). For each of the five misattribution flags the row carries
    ``{flag}_fired`` (records classified True), ``{flag}_resolved`` (records
    where the flag is not None), and ``{flag}_rate`` (fired over resolved,
    None when nothing resolved). The per-flag denominator excludes records
    where that flag is None, mirroring how ``failure_mode_by_config`` skips
    an unresolved discriminating flag — a None is never read as False.

    The any-flag fields (``any_flag_fired``, ``any_flag_resolved``,
    ``any_flag_rate``) use a *determinate* denominator: a record contributes
    when it is determinately flagged (at least one flag True) or fully
    resolved (no flag None); a record with no True and at least one None is
    indeterminate and excluded rather than silently counted as unflagged.
    ``record_count`` is the per-condition record total for context.

    Realizes the misattribution rates of docs/rationale_analysis.md Section
    7.2 — the confound disclosure of Section 2.4. Distinct from
    ``aggregate_failure_mode_rates``, whose misattribution-flagged annotation
    is taken over ``total_mismatches`` (records with both deictic flags
    resolved); the two denominators are deliberately different populations
    and can diverge on records with unresolved flags.

    Call once per experiment (primary, ablation) and join downstream for the
    paired summary table. Consistent with the single-results-in pattern used
    throughout this module.
    """
    conditions = ("correct", "transparent", "opaque")
    fired = {c: {k: 0 for k in MISATTRIBUTION_FLAG_KEYS} for c in conditions}
    resolved = {c: {k: 0 for k in MISATTRIBUTION_FLAG_KEYS} for c in conditions}
    record_count = {c: 0 for c in conditions}
    any_fired = {c: 0 for c in conditions}
    any_resolved = {c: 0 for c in conditions}

    for record in ra_results.trial_records:
        _, condition = parse_example_id(record.example_id)
        record_count[condition] += 1

        values = [getattr(record, k) for k in MISATTRIBUTION_FLAG_KEYS]
        for k, value in zip(MISATTRIBUTION_FLAG_KEYS, values):
            if value is not None:
                resolved[condition][k] += 1
                if value:
                    fired[condition][k] += 1

        # Any-flag is determinate when at least one flag is True or every
        # flag resolved; otherwise the record's any-flag status is unknown.
        has_true = any(value is True for value in values)
        fully_resolved = all(value is not None for value in values)
        if has_true or fully_resolved:
            any_resolved[condition] += 1
            if has_true:
                any_fired[condition] += 1

    rows: list[dict[str, Any]] = []
    for c in conditions:
        row: dict[str, Any] = {"condition": c, "record_count": record_count[c]}
        for k in MISATTRIBUTION_FLAG_KEYS:
            row[f"{k}_fired"] = fired[c][k]
            row[f"{k}_resolved"] = resolved[c][k]
            row[f"{k}_rate"] = _safe_rate(fired[c][k], resolved[c][k])
        row["any_flag_fired"] = any_fired[c]
        row["any_flag_resolved"] = any_resolved[c]
        row["any_flag_rate"] = _safe_rate(any_fired[c], any_resolved[c])
        rows.append(row)

    return rows


def failure_mode_dominance_matrix(
    ra_results: RationaleAnalysisResults,
    all_config_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Flat table: one row per (base_example, config_key) with per-condition failure mode dominance.

    Powers the failure mode dominance heatmaps. Carries the same 10
    base_example rows as ``example_accuracy_matrix``, which the shared
    row-ordering rule then renders in the same sequence, so the two figures
    align visually.

    Per-condition flag selection follows the rationale analysis spec:
    - transparent / opaque: ``articulated_operational_interpretation``
      classifies each mismatch as "expressed but dismissed" (flag=True,
      the positive side) or "never expressed" (flag=False, the negative side).
    - correct: ``operational_interpretation_governed_judgment`` classifies
      each mismatch as "operational interpretation governed" (flag=True,
      the positive side) or "field label matching governed" (flag=False,
      the negative side).

    Per condition, the output encodes both the raw component counts
    (``*_positive``, ``*_negative``, ``*_total``) and a normalized signed
    dominance score

        dominance = (positive - negative) / (positive + negative)

    bounded to ``[-1, 1]`` and set to ``None`` when ``total == 0``. A value
    of ``+1`` means the positive-side failure mode fully dominates the cell,
    ``-1`` means the negative-side mode fully dominates, and ``0`` means an
    even split. This normalization lets the heatmap use a fixed color scale
    that is comparable across conditions and across primary vs ablation.

    Cells for all ``all_config_keys`` × all base examples in
    ``STIMULUS_METADATA_REGISTRY`` are guaranteed to exist (zero-filled
    when no classifiable failures are present) so downstream heatmaps
    always show the full config × example grid.
    """
    # Condition-neutral bucket names: "positive" and "negative" refer to the
    # two sides of the diverging scale, with condition-specific meanings
    # documented in the function docstring.
    cell_data: dict[tuple[str, str], dict[str, Any]] = {}

    def _empty_cell(base_example: str, config_key: str) -> dict[str, Any]:
        return {
            "base_example": base_example,
            "config_key": config_key,
            "correct_positive": 0, "correct_negative": 0,
            "transparent_positive": 0, "transparent_negative": 0,
            "opaque_positive": 0, "opaque_negative": 0,
        }

    for record in ra_results.trial_records:
        base_example, condition = parse_example_id(record.example_id)

        # Select the discriminating flag per the spec (Section 2.3).
        # For transparent/opaque, flag=True means the meta-evaluator
        # expressed the operational interpretation but did not let it
        # govern (the positive side); flag=False means it never expressed
        # the interpretation (the negative side). For correct-draft,
        # flag=True means the operational interpretation governed the
        # verdict (the positive side); flag=False means field label
        # matching governed (the negative side).
        flag = (
            record.operational_interpretation_governed_judgment
            if condition == "correct"
            else record.articulated_operational_interpretation
        )
        if flag is None:
            continue

        cell_key = (base_example, record.config_key)
        if cell_key not in cell_data:
            cell_data[cell_key] = _empty_cell(base_example, record.config_key)

        suffix = "positive" if flag else "negative"
        cell_data[cell_key][f"{condition}_{suffix}"] += 1

    # Ensure every (base_example, config_key) cell exists
    for base_example in STIMULUS_METADATA_REGISTRY:
        for config_key in all_config_keys:
            cell_key = (base_example, config_key)
            if cell_key not in cell_data:
                cell_data[cell_key] = _empty_cell(base_example, config_key)

    # Derive dominance and total columns from the raw component counts.
    rows: list[dict[str, Any]] = []
    for cell in cell_data.values():
        row: dict[str, Any] = {
            "base_example": cell["base_example"],
            "config_key": cell["config_key"],
        }
        for cond in ("correct", "transparent", "opaque"):
            pos = cell[f"{cond}_positive"]
            neg = cell[f"{cond}_negative"]
            total = pos + neg
            row[f"{cond}_positive"] = pos
            row[f"{cond}_negative"] = neg
            row[f"{cond}_total"] = total
            row[f"{cond}_dominance"] = (
                (pos - neg) / total if total > 0 else None
            )
        rows.append(row)

    return sorted(rows, key=lambda r: r["base_example"])


def primary_vs_ablation_deltas(
    primary_results: ExperimentAnalysisResults,
    ablation_results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """One row per config_key with primary vs ablation balanced accuracy and delta.

    Matches configs by config_key across both result sets. Unmatched
    configs are silently dropped.

    Powers the balanced accuracy ablation effect chart.
    """
    primary_scores = {
        s.config_key: s.balanced_accuracy for s in primary_results.config_summaries
    }
    ablation_scores = {
        s.config_key: s.balanced_accuracy for s in ablation_results.config_summaries
    }

    # Only include configs present in both datasets
    common_keys = set(primary_scores) & set(ablation_scores)

    rows: list[dict[str, Any]] = []
    for config_key in common_keys:
        provider, model_slug, reasoning_effort_level = parse_config_key(config_key)
        pos = require_compute_position(config_key)
        primary = primary_scores[config_key]
        ablation = ablation_scores[config_key]
        rows.append({
            "config_key": config_key,
            "provider": provider,
            "model_slug": model_slug,
            "reasoning_effort_level": reasoning_effort_level,
            "ordinal": pos[0],
            "primary_score": primary,
            "ablation_score": ablation,
            "delta": ablation - primary,
        })

    return sorted(rows, key=lambda r: config_tie_break_key(r["config_key"]))


def cost_vs_balanced_accuracy(
    results: ExperimentAnalysisResults,
    *,
    is_batch: bool = True,
) -> list[dict[str, Any]]:
    """One row per config_key with balanced_accuracy, cost, and token breakdown.

    Powers the cost vs. balanced accuracy scatter plot and cost breakdown table.
    """
    # Pre-index outcomes by config_key for O(N) total instead of O(C×N)
    outcomes_by_config: dict[str, list[TrialOutcome]] = {}
    for o in results.trial_outcomes:
        outcomes_by_config.setdefault(o.config_key, []).append(o)

    rows: list[dict[str, Any]] = []

    for summary in results.config_summaries:
        provider, model_slug, reasoning_effort_level = parse_config_key(
            summary.config_key,
        )

        config_outcomes = outcomes_by_config.get(summary.config_key, [])
        usage = usage_cost_summary(config_outcomes, is_batch=is_batch)

        # Configs that report no reasoning tokens sit at the 0 baseline of
        # their scale (e.g., haiku-off), the same rule reasoning_token_scaling
        # applies: reasoning 0 by construction makes response equal output.
        # The row's total_reasoning_tokens stays None so the table's em-dash
        # sentinel holds for the reasoning column.
        effective_reasoning = (
            usage.total_reasoning_tokens
            if usage.total_reasoning_tokens is not None
            else 0
        )
        response_tokens = compute_response_tokens(
            usage.total_output_tokens,
            effective_reasoning,
            provider,
        )

        cost_per_trial = (
            usage.actual_cost.total_cost / usage.num_trials
            if usage.cost_available and usage.num_trials > 0
            else None
        )

        rows.append({
            "config_key": summary.config_key,
            "provider": provider,
            "model_slug": model_slug,
            "reasoning_effort_level": reasoning_effort_level,
            "balanced_accuracy": summary.balanced_accuracy,
            "actual_cost": (
                usage.actual_cost.total_cost if usage.cost_available else None
            ),
            "cost_per_trial": cost_per_trial,
            "total_input_tokens": usage.total_input_tokens,
            "total_output_tokens": usage.total_output_tokens,
            "total_reasoning_tokens": usage.total_reasoning_tokens,
            "total_response_tokens": response_tokens,
            "num_trials": usage.num_trials,
            "cost_available": usage.cost_available,
        })

    return sorted(
        rows,
        key=lambda r: (-r["balanced_accuracy"], *config_tie_break_key(r["config_key"])),
    )


def reasoning_token_scaling(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """One row per (model_group, reasoning_effort_level) with mean tokens per trial.

    Powers the reasoning token scaling chart and token composition bars.
    Only includes model groups with graduated reasoning scales.
    """
    # Pre-index outcomes by config_key for O(N) total instead of O(C×N)
    outcomes_by_config: dict[str, list[TrialOutcome]] = {}
    for o in results.trial_outcomes:
        outcomes_by_config.setdefault(o.config_key, []).append(o)

    rows: list[dict[str, Any]] = []

    for summary in results.config_summaries:
        provider, model_slug, reasoning_effort_level = parse_config_key(
            summary.config_key,
        )

        # Only include models with graduated reasoning scales
        scale_key = (provider, model_slug)
        if scale_key not in REASONING_SCALES:
            continue
        if len(REASONING_SCALES[scale_key]) < 2:
            continue

        pos = require_compute_position(summary.config_key)
        model_group = f"{provider}--{model_slug}"

        config_outcomes = outcomes_by_config.get(summary.config_key, [])
        n = len(config_outcomes)
        if n == 0:
            continue

        total_input = sum(o.usage.input_tokens for o in config_outcomes)
        total_output = sum(o.usage.output_tokens for o in config_outcomes)
        total_reasoning = sum(
            o.usage.reasoning_tokens
            for o in config_outcomes
            if o.usage.reasoning_tokens is not None
        )
        has_reasoning = any(
            o.usage.reasoning_tokens is not None for o in config_outcomes
        )

        # Configs in REASONING_SCALES that report no reasoning tokens are
        # at the 0 baseline of their scale (e.g., haiku-off, gpt-5.2-none).
        # Use 0 instead of None so they appear on scaling/composition charts.
        effective_reasoning = total_reasoning if has_reasoning else 0
        response = compute_response_tokens(
            total_output, effective_reasoning, provider,
        )

        rows.append({
            "config_key": summary.config_key,
            "model_group": model_group,
            "provider": provider,
            "model_slug": model_slug,
            "reasoning_effort_level": reasoning_effort_level,
            "ordinal": pos[0],
            "mean_input_tokens": total_input / n,
            "mean_output_tokens": total_output / n,
            "mean_reasoning_tokens": effective_reasoning / n,
            "mean_response_tokens": response / n if response is not None else None,
            "total_trials": n,
            "overall_accuracy": summary.overall_accuracy,
        })

    return sorted(rows, key=lambda r: config_tie_break_key(r["config_key"]))


def _signed_cost_display(value: float) -> str:
    """Signed dollar string for hover text, e.g. "+$0.0020" or "-$0.0200".

    Hover templates print this verbatim, so the sign always comes from the
    value rather than from a hardcoded "+$" prefix in the template.
    """
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.4f}"


def marginal_cost_per_balanced_accuracy_point(
    results: ExperimentAnalysisResults,
    *,
    is_batch: bool = True,
) -> list[dict[str, Any]]:
    """One row per consecutive reasoning effort level pair within each model.

    Computes delta_cost and delta_balanced_accuracy between adjacent reasoning
    levels, then marginal_cost = delta_cost / (delta_balanced_accuracy * 100)
    to express cost per percentage point, where one percentage point denotes
    0.01 of balanced accuracy by this metric's explicit definition.

    Powers the marginal cost analysis chart.
    """
    cost_data = cost_vs_balanced_accuracy(
        results, is_batch=is_batch,
    )

    # Group by model
    model_configs: dict[str, list[dict[str, Any]]] = {}
    for row in cost_data:
        model_group = f"{row['provider']}--{row['model_slug']}"
        model_configs.setdefault(model_group, []).append(row)

    rows: list[dict[str, Any]] = []
    for model_group, configs in model_configs.items():
        # Only models with graduated reasoning
        provider, model_slug = model_group.split("--", 1)
        scale_key = (provider, model_slug)
        if scale_key not in REASONING_SCALES:
            continue
        if len(REASONING_SCALES[scale_key]) < 2:
            continue

        # Sort by reasoning effort level ordinal; .index() raises ValueError
        # if a level is not in REASONING_SCALES (stale scale definition)
        sorted_configs = sorted(
            configs,
            key=lambda r: REASONING_SCALES[scale_key].index(r["reasoning_effort_level"]),
        )

        # Compute marginal cost between consecutive levels
        for i in range(len(sorted_configs) - 1):
            lower = sorted_configs[i]
            upper = sorted_configs[i + 1]

            # Skip if cost unavailable for either level
            if lower["cost_per_trial"] is None or upper["cost_per_trial"] is None:
                continue

            delta_balanced_accuracy = (
                upper["balanced_accuracy"] - lower["balanced_accuracy"]
            )
            delta_cost = upper["cost_per_trial"] - lower["cost_per_trial"]

            # Only meaningful when balanced accuracy increases; None encodes
            # "not applicable"
            marginal = (
                delta_cost / (delta_balanced_accuracy * 100)
                if delta_balanced_accuracy > 0
                else None
            )

            rows.append({
                "model_group": model_group,
                "provider": provider,
                "model_slug": model_slug,
                "from_level": lower["reasoning_effort_level"],
                "to_level": upper["reasoning_effort_level"],
                "from_ordinal": REASONING_SCALES[scale_key].index(
                    lower["reasoning_effort_level"],
                ),
                "to_ordinal": REASONING_SCALES[scale_key].index(
                    upper["reasoning_effort_level"],
                ),
                "from_score": lower["balanced_accuracy"],
                "to_score": upper["balanced_accuracy"],
                "delta_balanced_accuracy": delta_balanced_accuracy,
                "delta_cost": delta_cost,
                "delta_cost_display": _signed_cost_display(delta_cost),
                "marginal_cost_per_pp": marginal,
            })

    return sorted(
        rows,
        key=lambda r: (r["provider"], r["model_slug"], r["from_ordinal"]),
    )


CAP_SENSITIVITY_GRID: tuple[float, ...] = (0.25, 0.40, 0.50, 0.60, 0.75)
"""Betting caps the pairwise robustness check reports, fixed in advance.

``BET_CAP`` is not selected from this grid. It is fixed a priori, and the grid
exists only to report how the significance verdicts would have moved had it
been set elsewhere, which is a robustness statement about the reported result
rather than a tuning procedure.

**Scope: condition pairwise comparisons only.** Those rounds carry raw binary
differences, so every cap inside (0, 1) folds them. The configuration-level
families pool all three conditions and class reweight, which puts a
split-dependent ceiling on the cap: a completely paired round weighs 1.5 and
admits nothing at or above two thirds, so 0.60 is the largest entry here they
could take. Sweeping one of those families over this grid raises rather than
emitting a process the positivity argument does not cover; extend the grid or
the family with that in mind.
"""


def _significance_by_cell(
    results: ExperimentAnalysisResults,
    bet_cap: float,
) -> dict[tuple[str, str, str], bool]:
    """Significance of every pairwise condition comparison at one cap.

    Recomputes each configuration's three comparisons from raw trial outcomes
    at ``bet_cap`` and feeds them to the shipped closed test, so the double
    threshold that decides significance is applied by
    ``compute_global_null_comparison`` rather than restated here. Keys are
    ``(config_key, condition_a, condition_b)`` in the orientation the
    comparison reports.
    """
    outcomes = list(results.trial_outcomes)
    verdicts: dict[tuple[str, str, str], bool] = {}
    for summary in results.config_summaries:
        comparisons = compute_all_comparisons(
            outcomes, summary.config_key, bet_cap=bet_cap,
        )
        for pair in compute_global_null_comparison(comparisons).pair_results:
            key = (summary.config_key, pair.condition_a, pair.condition_b)
            verdicts[key] = pair.significant
    return verdicts


def pairwise_comparison_cap_sensitivity(
    primary_results: ExperimentAnalysisResults,
    ablation_results: ExperimentAnalysisResults,
    *,
    caps: tuple[float, ...] = CAP_SENSITIVITY_GRID,
) -> list[dict[str, Any]]:
    """One row per (experiment, betting cap) with significance counts.

    Recomputes every pairwise condition comparison in both experiments at each
    cap and counts how many clear the double threshold. Each row also carries
    ``status_changes``, the cells whose verdict differs from the one the
    reported cap produces, so a reader can see which comparisons the count
    moved rather than only that it moved. The reported cap's own row carries
    no changes, since it is the row the others are compared against.

    Caps are validated by the comparison code itself: a cap outside the open
    interval (0, 1) raises before any row is built.

    Powers the pairwise comparison cap sensitivity table.
    """
    rows: list[dict[str, Any]] = []
    for experiment, results in (
        ("Primary", primary_results), ("Ablation", ablation_results),
    ):
        default = _significance_by_cell(results, BET_CAP)
        for cap in caps:
            verdicts = (
                default if cap == BET_CAP else _significance_by_cell(results, cap)
            )
            changes = tuple(
                {
                    "config_key": config_key,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "significant_at_default_cap": default[key],
                    "significant_at_cap": significant,
                }
                for key, significant in sorted(verdicts.items())
                for config_key, condition_a, condition_b in (key,)
                if default[key] != significant
            )
            rows.append({
                "experiment": experiment,
                "bet_cap": cap,
                "significant_comparisons": sum(verdicts.values()),
                "total_comparisons": len(verdicts),
                "status_changes": changes,
            })
    return rows


def degradation_test_summary(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """One row per adjacent pair with adjacent-pair reasoning effort level degradation test results.

    Powers the adjacent-pair reasoning effort level degradation test table in the
    results workbook and results figures notebooks. Each row contains the
    model group, transition levels, the balanced accuracies the test bets on,
    their delta, the pooled valid trial counts behind them, the e-value
    components, and the significance verdict.
    """
    rows: list[dict[str, Any]] = []
    for test in results.degradation_tests:
        rows.append({
            "model_group": test.model_group,
            "from_level": test.lower_level,
            "to_level": test.upper_level,
            "lower_balanced_accuracy": test.lower_balanced_accuracy,
            "upper_balanced_accuracy": test.upper_balanced_accuracy,
            "delta_balanced_accuracy": test.delta_balanced_accuracy,
            "lower_valid": test.lower_valid,
            "upper_valid": test.upper_valid,
            "raw_e_value": test.raw_e_value,
            "adjusted_e_value": test.adjusted_e_value,
            "empirical_e_power": _empirical_e_power(test.e_slices),
            "bonferroni_k": test.bonferroni_k,
            "significant": test.significant,
        })
    return rows


def combined_degradation_test_summary(
    primary_results: ExperimentAnalysisResults,
    ablation_results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """Merged degradation test rows with an 'experiment' column.

    Concatenates primary and ablation rows, each tagged with experiment
    identity. Sorted by (experiment, model_group, from_level ordinal) so
    all rows for one experiment appear together — Primary block first,
    then Ablation block — making it easy to read the table by experiment
    rather than per-transition pairs.
    """
    primary_rows = [
        {**r, "experiment": "Primary"}
        for r in degradation_test_summary(primary_results)
    ]
    ablation_rows = [
        {**r, "experiment": "Ablation"}
        for r in degradation_test_summary(ablation_results)
    ]
    all_rows = primary_rows + ablation_rows

    def _sort_key(r: dict[str, Any]) -> tuple:
        """Group by experiment first, then by model group and reasoning level ordinal."""
        mg = r["model_group"]
        provider, model_slug = mg.split("--", 1)
        scale = REASONING_SCALES.get((provider, model_slug), ())
        from_ord = scale.index(r["from_level"]) if r["from_level"] in scale else 0
        return (_EXPERIMENT_SORT_ORDER[r["experiment"]], mg, from_ord)

    return sorted(all_rows, key=_sort_key)


_PROVIDER_DISPLAY_ORDER: dict[str, int] = {"openai": 0, "anthropic": 1, "gemini": 2}
"""Reader-facing provider order for model size tables: OpenAI, Anthropic, Google.

Sorting on the raw provider slug would order them alphabetically (anthropic,
gemini, openai), which does not match the canonical reading order used
throughout the paper's model size surfaces.
"""


def model_size_summary_rows(
    results: ExperimentAnalysisResults,
    *,
    comparison_type: str,
    authoring_model_slug: str,
) -> list[dict[str, Any]]:
    """One row per provider for a single within-provider model size claim.

    ``comparison_type`` selects which of the two claims to render:
    ``"worst"`` for the every-level test or ``"best"`` for the some-level
    test. Each row carries the two model slugs, the size of each side of the
    double combination — how many larger-model reasoning effort levels and
    how many smaller-model configurations the claim quantifies over — the
    adjusted e-value, and the significance verdict. ``is_authoring`` flags
    the rows whose larger model is the authoring model, so the dagger
    annotation is data-sourced rather than hard-coded.
    """
    if comparison_type not in ("worst", "best"):
        raise ValueError(
            f"comparison_type must be 'worst' or 'best', got {comparison_type!r}"
        )

    rows: list[dict[str, Any]] = []
    for test in results.model_size_comparisons:
        if test.comparison_type != comparison_type:
            continue
        rows.append({
            "provider": test.provider,
            "smaller_model": test.smaller_model_slug,
            "larger_model": test.larger_model_slug,
            "comparison_type": test.comparison_type,
            "larger_reasoning_effort_levels": len(test.larger_config_keys),
            "smaller_configurations": len(test.smaller_config_keys),
            "adjusted_e_value": test.adjusted_e_value,
            "significant": test.significant,
            "bonferroni_k": test.bonferroni_k,
            "is_authoring": test.larger_model_slug == authoring_model_slug,
        })
    return sorted(rows, key=lambda r: _PROVIDER_DISPLAY_ORDER.get(r["provider"], 99))


def combined_model_size_summary_rows(
    primary_results: ExperimentAnalysisResults,
    ablation_results: ExperimentAnalysisResults,
    *,
    comparison_type: str,
    authoring_model_slug: str,
) -> list[dict[str, Any]]:
    """Merged model size summary rows for one claim with an 'experiment' column.

    Concatenates the per-provider rows for the requested ``comparison_type``
    from both experiments, each tagged with experiment identity, with the
    Primary block above the Ablation block.
    """
    primary_rows = [
        {**r, "experiment": "Primary"}
        for r in model_size_summary_rows(
            primary_results,
            comparison_type=comparison_type,
            authoring_model_slug=authoring_model_slug,
        )
    ]
    ablation_rows = [
        {**r, "experiment": "Ablation"}
        for r in model_size_summary_rows(
            ablation_results,
            comparison_type=comparison_type,
            authoring_model_slug=authoring_model_slug,
        )
    ]
    return sorted(
        primary_rows + ablation_rows,
        key=lambda r: (
            _EXPERIMENT_SORT_ORDER[r["experiment"]],
            _PROVIDER_DISPLAY_ORDER.get(r["provider"], 99),
        ),
    )


def model_size_level_detail_rows(
    results: ExperimentAnalysisResults,
    *,
    authoring_model_slug: str,
) -> list[dict[str, Any]]:
    """One row per (provider, larger-model reasoning effort level) audit fold.

    Each row reports the two inner combinations of that level's paired
    e-processes against every smaller-model configuration: the mean e-value
    across configurations, which the every-level claim minimizes over levels,
    and the minimum e-value across configurations, which the some-level claim
    averages over levels. The per-level balanced accuracy and valid trial count
    come from the ``LevelFold``. Levels are emitted in REASONING_SCALES order
    within each provider, and providers follow the reader-facing order
    (OpenAI, Anthropic, Google).

    The minimum of a provider's mean column reproduces that provider's
    every-level summary e-value, and the arithmetic mean of its minimum
    column reproduces the some-level summary e-value, so the detail rows are
    the traceable inputs behind both summaries.
    """
    by_provider: dict[str, dict[str, ModelSizeComparison]] = {}
    for test in results.model_size_comparisons:
        by_provider.setdefault(test.provider, {})[test.comparison_type] = test

    rows: list[dict[str, Any]] = []
    for provider in sorted(by_provider, key=lambda p: _PROVIDER_DISPLAY_ORDER.get(p, 99)):
        pair = by_provider[provider]
        if "worst" not in pair or "best" not in pair:
            continue
        # Both claims read the same per-pair processes, so either record
        # carries the audit. Divergence would mean the two claims were
        # computed from different evidence, which is a data error.
        every_level, some_level = pair["worst"], pair["best"]
        if every_level.level_folds != some_level.level_folds:
            raise ValueError(
                f"The two model size claims for provider {provider!r} carry "
                "different level folds; both must be combinations of one set "
                "of per-pair processes."
            )
        for level_fold in every_level.level_folds:
            _, _, level = parse_config_key(level_fold.config_key)
            rows.append({
                "provider": provider,
                "larger_model": every_level.larger_model_slug,
                "reasoning_effort_level": level,
                "balanced_accuracy": level_fold.balanced_accuracy,
                "valid_trials": level_fold.valid,
                "mean_e_value_over_smaller_configurations": math.exp(
                    level_fold.log_mean_e_value,
                ),
                "min_e_value_over_smaller_configurations": math.exp(
                    level_fold.log_min_e_value,
                ),
                "is_authoring": (
                    every_level.larger_model_slug == authoring_model_slug
                ),
            })
    return rows


def combined_model_size_level_detail_rows(
    primary_results: ExperimentAnalysisResults,
    ablation_results: ExperimentAnalysisResults,
    *,
    authoring_model_slug: str,
) -> list[dict[str, Any]]:
    """Merged per-level model size detail rows with an 'experiment' column.

    Concatenates the per-level audit rows from both experiments, each tagged
    with experiment identity, with the Primary block above the Ablation
    block.
    """
    primary_rows = [
        {**r, "experiment": "Primary"}
        for r in model_size_level_detail_rows(
            primary_results,
            authoring_model_slug=authoring_model_slug,
        )
    ]
    ablation_rows = [
        {**r, "experiment": "Ablation"}
        for r in model_size_level_detail_rows(
            ablation_results,
            authoring_model_slug=authoring_model_slug,
        )
    ]
    return primary_rows + ablation_rows


def combined_balanced_accuracy_leaderboard(
    primary_results: ExperimentAnalysisResults,
    ablation_results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """Merged leaderboard rows with an 'experiment' column.

    Concatenates per-experiment leaderboards (each ranked independently
    within its own experiment) and tags every row with experiment identity.
    Sorted by (experiment, rank) so the Primary block appears first,
    followed by the Ablation block.
    """
    primary_rows = [
        {**r, "experiment": "Primary"}
        for r in balanced_accuracy_leaderboard(
            primary_results,
        )
    ]
    ablation_rows = [
        {**r, "experiment": "Ablation"}
        for r in balanced_accuracy_leaderboard(
            ablation_results,
        )
    ]
    return sorted(
        primary_rows + ablation_rows,
        key=lambda r: (_EXPERIMENT_SORT_ORDER[r["experiment"]], r["rank"]),
    )


def combined_cost_vs_balanced_accuracy(
    primary_results: ExperimentAnalysisResults,
    ablation_results: ExperimentAnalysisResults,
    *,
    is_batch: bool = True,
) -> list[dict[str, Any]]:
    """Merged cost-and-score rows with an 'experiment' column.

    Concatenates per-experiment ``cost_vs_balanced_accuracy`` rows and tags
    every row with experiment identity. Sorted by (experiment, -balanced_accuracy)
    so the Primary block appears first, followed by the Ablation block,
    with each block ordered from best to worst configuration. The
    """
    primary_rows = [
        {**r, "experiment": "Primary"}
        for r in cost_vs_balanced_accuracy(
            primary_results,
                is_batch=is_batch,
        )
    ]
    ablation_rows = [
        {**r, "experiment": "Ablation"}
        for r in cost_vs_balanced_accuracy(
            ablation_results,
                is_batch=is_batch,
        )
    ]
    return sorted(
        primary_rows + ablation_rows,
        key=lambda r: (
            _EXPERIMENT_SORT_ORDER[r["experiment"]],
            -r["balanced_accuracy"],
            *config_tie_break_key(r["config_key"]),
        ),
    )


def cell_resolution_by_config(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """One row per config_key with rejected/futile/active cell counts.

    Powers the cell resolution status charts. Rows follow the structural
    configuration order (config_tie_break_key).
    """
    rows: list[dict[str, Any]] = []

    for summary in results.config_summaries:
        rows.append({
            "config_key": summary.config_key,
            "balanced_accuracy": summary.balanced_accuracy,
            "rejected": summary.cells_rejected,
            "futile": summary.cells_futile,
            "active": summary.cells_active,
            "fully_resolved": summary.fully_resolved,
        })

    return sorted(rows, key=lambda r: config_tie_break_key(r["config_key"]))


def balanced_accuracy_leaderboard(
    results: ExperimentAnalysisResults,
) -> list[dict[str, Any]]:
    """Flat table: one row per configuration with accuracy metrics and rank.

    Powers the balanced accuracy leaderboard table (Table 1). Ordered by
    balanced accuracy descending, sharing the score-ranked figures' tie-break
    so a rank never disagrees with a bar position.
    """
    sorted_configs = sorted(
        results.config_summaries,
        key=lambda cs: (-cs.balanced_accuracy, *config_tie_break_key(cs.config_key)),
    )

    rows: list[dict[str, Any]] = []
    for rank, summary in enumerate(sorted_configs, 1):
        rows.append({
            "rank": rank,
            "config_key": summary.config_key,
            "balanced_accuracy": summary.balanced_accuracy,
            "overall_accuracy": summary.overall_accuracy,
            "correct_accuracy": summary.condition_accuracies.get("correct", 0.0),
            "transparent_accuracy": summary.condition_accuracies.get("transparent", 0.0),
            "opaque_accuracy": summary.condition_accuracies.get("opaque", 0.0),
        })

    return rows


def cross_dataset_significance_data(
    comparisons: list[CrossDatasetComparison],
) -> list[dict[str, Any]]:
    """One row per cross-dataset comparison with significance metadata.

    Transforms CrossDatasetComparison records into flat dicts for the
    significance table and figure annotations. Sorted by the structural order
    (``config_tie_break_key``), then condition, matching every configuration
    axis in the deck.
    """
    rows: list[dict[str, Any]] = []
    for c in comparisons:
        provider, model_slug, reasoning_effort_level = parse_config_key(c.config_key)
        pos = require_compute_position(c.config_key)
        rows.append({
            "config_key": c.config_key,
            "provider": provider,
            "model_slug": model_slug,
            "reasoning_effort_level": reasoning_effort_level,
            "ordinal": pos[0],
            "condition": c.condition,
            "primary_accuracy": c.primary_accuracy,
            "ablation_accuracy": c.ablation_accuracy,
            "primary_valid": c.primary_valid,
            "ablation_valid": c.ablation_valid,
            "delta": c.delta,
            "raw_e_value": c.raw_e_value,
            "adjusted_e_value": c.adjusted_e_value,
            "significant": c.significant,
            "direction": c.direction,
        })

    # Condition sort key: correct=0, transparent=1, opaque=2, None (config-level)=3
    _condition_order = {"correct": 0, "transparent": 1, "opaque": 2}

    return sorted(rows, key=lambda r: (
        *config_tie_break_key(r["config_key"]),
        _condition_order.get(r["condition"], 3),
    ))
