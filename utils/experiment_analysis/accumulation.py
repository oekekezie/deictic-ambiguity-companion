"""E-value accumulation via a fold over batches.

The fold is the core abstraction: each batch produces a new CellAccumulator
from the previous one plus the new data. The fold enforces sequential validity
by deriving each batch's alternative entirely from prior data (the alternative
comes from ``prior``, the data comes from ``outcomes``).
"""

import math

from utils.experiment_analysis.e_values import bernoulli_lr_log_e_value, clamp_alternative
from utils.experiment_analysis.ground_truth import ground_truth_for_condition, parse_example_id
from utils.experiment_analysis.models import (
    AccumulationParams,
    CellAccumulator,
    CellKey,
    IterationSlice,
    TrialOutcome,
)


def make_initial_accumulator(
    cell_key: CellKey,
    seed_alternative: float,
) -> CellAccumulator:
    """Create a zero-state accumulator for a cell before any data.

    The condition and ground truth are derived from the cell_key's
    example_id, ensuring consistency with the ground truth mapping.
    """
    _, condition = parse_example_id(cell_key.example_id)
    gt_score = ground_truth_for_condition(condition)

    return CellAccumulator(
        cell_key=cell_key,
        condition=condition,
        ground_truth_score=gt_score,
        total_trials=0,
        valid_trials=0,
        hit_count=0,
        parse_failures=0,
        slices=(),
        running_log_e_value=0.0,
        running_e_value=1.0,
        next_alternative=seed_alternative,
        status="active",
        resolved_at_batch=None,
        empirical_e_power=0.0,
    )


def accumulate_batch(
    prior: CellAccumulator,
    outcomes: tuple[TrialOutcome, ...],
    p_null: float,
    alpha: float,
    futility_bound: float,
    clamp_epsilon: float = 0.01,
) -> CellAccumulator:
    """Fold one batch of outcomes into a cell's accumulator.

    This is the most delicate function in the analysis layer. The ordering
    constraint — prior.next_alternative is determined before seeing the
    current batch's data — is enforced structurally by the signature.

    Data is always accumulated regardless of the cell's status, and the
    status is recomputed from the running log e-value on every fold: it
    reflects the CURRENT evidence and may move in any direction as
    subsequent batches fold in (optional continuation). resolved_at_batch
    is a set-once audit record of the first threshold crossing — retained
    even if the status later reverts, None only if no crossing has
    occurred. Inspecting the running e-value at any batch boundary is
    valid by Ville's inequality, which holds for any stopping rule.
    """
    if not outcomes:
        return prior

    # Tally this batch's results
    trials = len(outcomes)
    valid = sum(1 for o in outcomes if o.predicted_score is not None)
    hits = sum(1 for o in outcomes if o.matches_ground_truth is True)
    parse_failures = trials - valid

    # Compute this batch's log e-value using the pre-determined alternative
    batch_log_e = bernoulli_lr_log_e_value(
        hits=hits,
        valid=valid,
        p_null=p_null,
        p_alt=prior.next_alternative,
    )

    # Accumulate into the running log sum (log space product)
    new_running_log = prior.running_log_e_value + batch_log_e

    # Update cumulative counts
    new_total_trials = prior.total_trials + trials
    new_valid = prior.valid_trials + valid
    new_hits = prior.hit_count + hits
    new_parse_failures = prior.parse_failures + parse_failures

    # Derive next_alternative from cumulative accuracy (for the NEXT batch)
    if new_valid > 0:
        p_hat = new_hits / new_valid
        new_next_alt = clamp_alternative(p_hat, p_null, clamp_epsilon)
    else:
        new_next_alt = prior.next_alternative

    # Status: pure function of the running log e-value, recomputed on
    # every fold — it may move in any direction as evidence accumulates
    log_rejection = math.log(1.0 / alpha)
    log_futility = math.log(futility_bound)

    if new_running_log >= log_rejection:
        new_status = "rejected"
    elif new_running_log <= log_futility:
        new_status = "futile"
    else:
        new_status = "active"

    # resolved_at_batch: set-once audit record of the first crossing
    if prior.resolved_at_batch is not None:
        new_resolved_at = prior.resolved_at_batch
    elif new_status != "active":
        new_resolved_at = outcomes[0].batch_index
    else:
        new_resolved_at = None

    # Build the audit record for this batch
    new_slice = IterationSlice(
        batch_index=outcomes[0].batch_index,
        trials=trials,
        valid=valid,
        hits=hits,
        alternative_used=prior.next_alternative,
        e_value=math.exp(batch_log_e),
        log_e_value=batch_log_e,
    )

    new_slices = prior.slices + (new_slice,)

    # Empirical e-power: mean log e-value across all batches
    total_log_e = sum(s.log_e_value for s in new_slices)
    new_e_power = total_log_e / len(new_slices)

    return CellAccumulator(
        cell_key=prior.cell_key,
        condition=prior.condition,
        ground_truth_score=prior.ground_truth_score,
        total_trials=new_total_trials,
        valid_trials=new_valid,
        hit_count=new_hits,
        parse_failures=new_parse_failures,
        slices=new_slices,
        running_log_e_value=new_running_log,
        running_e_value=math.exp(new_running_log),
        next_alternative=new_next_alt,
        status=new_status,
        resolved_at_batch=new_resolved_at,
        empirical_e_power=new_e_power,
    )


def fold_outcomes(
    outcomes: list[TrialOutcome],
    params: AccumulationParams,
) -> dict[CellKey, CellAccumulator]:
    """Group outcomes by cell and fold across batches.

    This is the shared accumulation pipeline behind ``analyze_experiment``,
    which operates on pre-enriched TrialOutcomes.

    Outcomes are grouped by CellKey, then within each cell the batches
    are processed in ascending batch_index order. Cells that do not
    cross the rejection or futility threshold remain "active".
    """
    # Group by (CellKey, batch_index)
    by_cell: dict[CellKey, dict[int, list[TrialOutcome]]] = {}
    for o in outcomes:
        key = CellKey(example_id=o.example_id, config_key=o.config_key)
        by_cell.setdefault(key, {}).setdefault(o.batch_index, []).append(o)

    # Fold each cell across batches in order
    accumulators: dict[CellKey, CellAccumulator] = {}
    for cell_key, batches in by_cell.items():
        acc = make_initial_accumulator(cell_key, params.seed_alternative)
        for batch_idx in sorted(batches.keys()):
            acc = accumulate_batch(
                prior=acc,
                outcomes=tuple(batches[batch_idx]),
                p_null=params.p_null,
                alpha=params.alpha,
                futility_bound=params.futility_bound,
                clamp_epsilon=params.clamp_epsilon,
            )
        accumulators[cell_key] = acc

    return accumulators

