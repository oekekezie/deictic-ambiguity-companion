"""Metric computation from accumulators and trial outcomes.

All functions are pure — they compute metrics from CellAccumulator state
and TrialOutcome collections without mutating any input.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from utils.experiment_analysis.confidence_sequences import bernoulli_cs_bounds
from utils.experiment_analysis.models import (
    CellAccumulator,
    CellKey,
    CellMetrics,
    Condition,
    TrialOutcome,
)
from utils.token_costs.calculation import calculate_actual_cost, calculate_worst_case_cost
from utils.token_costs.models import TokenCost


def _sum_optional(values: Iterable[int | None]) -> int | None:
    """Sum values that may be None, returning None if all are None."""
    total = 0
    any_present = False
    for v in values:
        if v is not None:
            total += v
            any_present = True
    return total if any_present else None


class ConfigurationSummary(BaseModel):
    """Aggregate metrics for one configuration across all 30 cells."""

    model_config = ConfigDict(frozen=True)

    config_key: str
    condition_accuracies: dict[str, float]  # condition → accuracy
    balanced_accuracy: float
    overall_accuracy: float
    cells_rejected: int
    cells_futile: int
    cells_active: int
    fully_resolved: bool
    total_trials: int  # sum of total_trials across all cells for this config
    valid_trials: int  # sum of valid_trials across all cells for this config
    max_batches: int   # max batches accumulated across cells for this config


class UsageCostSummary(BaseModel):
    """Aggregate token usage and cost for a group of trials.

    Composes TokenCost objects for worst-case (no caching) and actual
    (with cache discounts) costs, following the same pattern as JobCostSummary.
    The delta between them quantifies prompt cache savings.
    """

    model_config = ConfigDict(frozen=True)

    num_trials: int
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int | None
    total_cache_read_tokens: int | None
    total_cache_write_tokens: int | None
    worst_case_cost: TokenCost
    actual_cost: TokenCost
    cache_savings: float          # worst_case.total_cost - actual.total_cost
    cost_available: bool          # False when pricing lookup failed


_ZERO_COST = TokenCost(
    input_cost=0.0, output_cost=0.0,
    cache_read_cost=0.0, cache_write_cost=0.0, total_cost=0.0,
)


def usage_cost_summary(
    outcomes: list[TrialOutcome],
    *,
    is_batch: bool = True,
) -> UsageCostSummary:
    """Aggregate token usage and compute cost breakdown for trial outcomes.

    Sums token fields across all outcomes, then computes per-trial costs
    using the token_costs infrastructure. If any trial's provider/model
    is not in the pricing registry, cost fields are zeroed and
    cost_available is set to False.
    """
    if not outcomes:
        return UsageCostSummary(
            num_trials=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_reasoning_tokens=None,
            total_cache_read_tokens=None,
            total_cache_write_tokens=None,
            worst_case_cost=_ZERO_COST,
            actual_cost=_ZERO_COST,
            cache_savings=0.0,
            cost_available=False,
        )

    # Aggregate token counts
    total_input = sum(o.usage.input_tokens for o in outcomes)
    total_output = sum(o.usage.output_tokens for o in outcomes)
    total_reasoning = _sum_optional(o.usage.reasoning_tokens for o in outcomes)
    total_cache_read = _sum_optional(o.usage.cache_read_tokens for o in outcomes)
    total_cache_write = _sum_optional(o.usage.cache_write_tokens for o in outcomes)

    # Per-trial cost calculation; bail to zeroed costs on pricing lookup failure
    worst_costs: list[TokenCost] = []
    actual_costs: list[TokenCost] = []
    cost_available = True

    for o in outcomes:
        try:
            worst_costs.append(
                calculate_worst_case_cost(
                    o.provider, o.model_slug, o.usage, is_batch=is_batch,
                )
            )
            actual_costs.append(
                calculate_actual_cost(
                    o.provider, o.model_slug, o.usage, is_batch=is_batch,
                )
            )
        except ValueError:
            cost_available = False
            break

    if cost_available:
        total_worst = TokenCost(
            input_cost=sum(c.input_cost for c in worst_costs),
            output_cost=sum(c.output_cost for c in worst_costs),
            cache_read_cost=sum(c.cache_read_cost for c in worst_costs),
            cache_write_cost=sum(c.cache_write_cost for c in worst_costs),
            total_cost=sum(c.total_cost for c in worst_costs),
        )
        total_actual = TokenCost(
            input_cost=sum(c.input_cost for c in actual_costs),
            output_cost=sum(c.output_cost for c in actual_costs),
            cache_read_cost=sum(c.cache_read_cost for c in actual_costs),
            cache_write_cost=sum(c.cache_write_cost for c in actual_costs),
            total_cost=sum(c.total_cost for c in actual_costs),
        )
        savings = total_worst.total_cost - total_actual.total_cost
    else:
        total_worst = _ZERO_COST
        total_actual = _ZERO_COST
        savings = 0.0

    return UsageCostSummary(
        num_trials=len(outcomes),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_reasoning_tokens=total_reasoning,
        total_cache_read_tokens=total_cache_read,
        total_cache_write_tokens=total_cache_write,
        worst_case_cost=total_worst,
        actual_cost=total_actual,
        cache_savings=savings,
        cost_available=cost_available,
    )


def cell_metrics(
    acc: CellAccumulator,
    alpha: float = 0.05,
) -> CellMetrics:
    """Compute reportable metrics from a cell's accumulator state."""
    accuracy = acc.hit_count / acc.valid_trials if acc.valid_trials > 0 else 0.0
    parse_failure_rate = (
        acc.parse_failures / acc.total_trials if acc.total_trials > 0 else 0.0
    )

    cs_lower, cs_upper = bernoulli_cs_bounds(
        hits=acc.hit_count,
        valid=acc.valid_trials,
        alpha=alpha,
        side="two-sided",
    )

    return CellMetrics(
        cell_key=acc.cell_key,
        condition=acc.condition,
        accuracy=accuracy,
        cs_lower=cs_lower,
        cs_upper=cs_upper,
        parse_failure_rate=parse_failure_rate,
        e_value=acc.running_e_value,
        log_e_value=acc.running_log_e_value,
        empirical_e_power=acc.empirical_e_power,
        status=acc.status,
        batches_used=len(acc.slices),
        total_trials=acc.total_trials,
        valid_trials=acc.valid_trials,
        resolved_at_batch=acc.resolved_at_batch,
    )


def condition_accuracy(
    accumulators: dict[CellKey, CellAccumulator],
    condition: Condition,
    config_key: str,
) -> float:
    """Pool accuracy across all stimuli within a condition for one configuration.

    Returns Σ hit_count / Σ valid_trials for matching cells.
    Returns 0.0 if no valid trials exist.
    """
    total_hits = 0
    total_valid = 0
    for key, acc in accumulators.items():
        if key.config_key == config_key and acc.condition == condition:
            total_hits += acc.hit_count
            total_valid += acc.valid_trials
    return total_hits / total_valid if total_valid > 0 else 0.0


def composite_metrics(
    accumulators: dict[CellKey, CellAccumulator],
    config_key: str,
) -> dict[str, float]:
    """Compute balanced accuracy and overall accuracy for one configuration.

    Balanced accuracy = (sensitivity + specificity) / 2, where specificity is
    the correct-draft accuracy and sensitivity pools both incorrect-draft
    conditions. Weighting the two classes equally is what makes it the
    headline metric: every stimulus-blind strategy scores exactly 0.500.

    Overall accuracy = pooled hits / pooled valid trials across the three
    conditions. It equals the unweighted mean of the three condition
    accuracies whenever those conditions carry equal valid counts, which the
    balanced design intends, and it is reported as a descriptive column only.
    It performs no class imbalance correction: a stimulus-blind strategy with
    acceptance bias q scores (1 + q) / 3, so the whole band [1/3, 2/3] is
    attainable without reading a single stimulus.
    """
    correct_acc = condition_accuracy(accumulators, "correct", config_key)
    transparent_acc = condition_accuracy(accumulators, "transparent", config_key)
    opaque_acc = condition_accuracy(accumulators, "opaque", config_key)

    specificity = correct_acc
    # Sensitivity pools both incorrect-draft conditions and overall accuracy
    # pools all three, so one pass over the configuration's cells feeds both.
    sensitivity_hits = 0
    sensitivity_valid = 0
    total_hits = 0
    total_valid = 0
    for key, acc in accumulators.items():
        if key.config_key != config_key:
            continue
        if acc.condition not in ("correct", "transparent", "opaque"):
            continue
        total_hits += acc.hit_count
        total_valid += acc.valid_trials
        if acc.condition in ("transparent", "opaque"):
            sensitivity_hits += acc.hit_count
            sensitivity_valid += acc.valid_trials
    sensitivity = sensitivity_hits / sensitivity_valid if sensitivity_valid > 0 else 0.0

    balanced_accuracy = (sensitivity + specificity) / 2.0
    overall_accuracy = total_hits / total_valid if total_valid > 0 else 0.0

    return {
        "correct_accuracy": correct_acc,
        "transparent_accuracy": transparent_acc,
        "opaque_accuracy": opaque_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "overall_accuracy": overall_accuracy,
    }


def configuration_summary(
    accumulators: dict[CellKey, CellAccumulator],
    config_key: str,
) -> ConfigurationSummary:
    """Build an aggregate summary for one configuration."""
    composites = composite_metrics(accumulators, config_key)

    condition_accs = {
        "correct": composites["correct_accuracy"],
        "transparent": composites["transparent_accuracy"],
        "opaque": composites["opaque_accuracy"],
    }

    # Aggregate cell statuses, trial counts, and batch depth
    status_counts: dict[str, int] = {
        "rejected": 0, "futile": 0, "active": 0,
    }
    agg_total_trials = 0
    agg_valid_trials = 0
    agg_max_batches = 0
    for key, acc in accumulators.items():
        if key.config_key == config_key:
            status_counts[acc.status] = status_counts.get(acc.status, 0) + 1
            agg_total_trials += acc.total_trials
            agg_valid_trials += acc.valid_trials
            agg_max_batches = max(agg_max_batches, len(acc.slices))

    total_cells = sum(status_counts.values())
    fully_resolved = total_cells > 0 and status_counts["active"] == 0

    return ConfigurationSummary(
        config_key=config_key,
        condition_accuracies=condition_accs,
        balanced_accuracy=composites["balanced_accuracy"],
        overall_accuracy=composites["overall_accuracy"],
        cells_rejected=status_counts["rejected"],
        cells_futile=status_counts["futile"],
        cells_active=status_counts["active"],
        fully_resolved=fully_resolved,
        total_trials=agg_total_trials,
        valid_trials=agg_valid_trials,
        max_batches=agg_max_batches,
    )


def thinking_rate(outcomes: list[TrialOutcome]) -> float:
    """Proportion of valid trials with a non-empty reasoning trace.

    Only meaningful for configurations where reasoning is enabled.
    """
    valid = [o for o in outcomes if o.predicted_score is not None]
    if not valid:
        return 0.0
    return sum(1 for o in valid if o.has_reasoning_trace) / len(valid)


