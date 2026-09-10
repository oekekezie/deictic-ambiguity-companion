"""Comparison e-values built on paired differences.

Every comparison in this module tests a difference between two sets of
outcomes drawn from the same stimulus grid, using the testing-by-betting /
plug-in (aGRAPA) framework for bounded means (Waudby-Smith and Ramdas, 2024).
A betting round is one ``(batch_index, trial)`` position and carries one
paired difference per cell; the skeptic bets a predictable, capped fraction
of its wealth on those differences and Ville's inequality makes the running
wealth an anytime-valid e-process.

What a round mean estimates depends on how its cells are weighted. A
comparison restricted to one condition carries binary differences, so its
round mean is that condition's accuracy difference. A comparison pooling all
three conditions scales each cell through ``_class_reweighted_differences``,
which makes the round mean the balanced accuracy difference — the headline
metric, and what the configuration-level families report.

Four families share that engine, differing only in what a cell is and which
half-line the bet may occupy:

- **Condition comparisons** pair two conditions of one configuration; a cell
  is a base example and the bet is unconstrained, because the null is an
  equality.
- **Adjacent-pair degradation** pairs two reasoning effort levels of one
  model; a cell is a stimulus and the bet is non-positive.
- **Within-provider model size** pairs each larger-model level with each
  smaller-model configuration; the bet is non-negative and the per-pair
  processes are combined twice, once per claim.
- **Cross-dataset** pairs one configuration across the two experiments and
  runs both one-sided processes, reporting the larger.

``_accumulate_difference_rounds`` carries the validity argument for all of
them; the sign constraint is what makes it hold against a one-sided null.
"""

import math

from pydantic import BaseModel, ConfigDict, model_validator

from typing import Literal

from utils.experiment_analysis.config_identity import (
    MODEL_SIZE_PAIRS,
    REASONING_SCALES,
    parse_config_key,
)
from utils.experiment_analysis.models import Condition, TrialOutcome


class ComparisonSlice(BaseModel):
    """Audit record for one betting round's contribution to a paired comparison.

    Shared by every family built on the paired-difference primitive: the
    condition comparisons, which pair two conditions within a configuration,
    and the cross-sample families, which pair two samples cell by cell.
    """

    model_config = ConfigDict(frozen=True)

    round_index: int  # sequential position in the betting sequence (0 = calibration)
    batch_index: int  # source batch for traceability
    trial: int  # source trial number (1-indexed) for traceability
    # Per-cell paired differences: {-1.0, 0.0, 1.0} for a condition-restricted
    # round, and those values class-reweighted for a pooled one.
    differences: tuple[float, ...]
    mean_difference: float
    bet_size: float  # λ used for this round
    log_e_value: float


class PairwiseComparison(BaseModel):
    """Accumulated e-value for one condition pair within one configuration."""

    model_config = ConfigDict(frozen=True)

    config_key: str
    condition_a: Condition
    condition_b: Condition
    slices: tuple[ComparisonSlice, ...]
    running_log_e_value: float
    running_e_value: float
    observed_mean_difference: float  # cumulative mean of all differences
    total_pairs: int  # total per-example paired differences across all rounds


def _build_trial_rounds(
    outcomes: list[TrialOutcome],
    config_key: str,
    condition_a: Condition,
    condition_b: Condition,
) -> list[tuple[int, int, list[float]]]:
    """Pair trial-level outcomes across two conditions and compute binary differences.

    For each (batch_index, trial) pair, matches outcome_A and outcome_B by
    base example, then computes d = float(outcome_A.matches_ground_truth)
    - float(outcome_B.matches_ground_truth) ∈ {-1, 0, 1}. Rounds with zero
    paired differences are omitted.

    Returns (batch_index, trial, differences) tuples sorted by (batch_index, trial).
    Raises ValueError on duplicate (base_example, batch_index, trial, condition) keys.
    """
    # Build lookup: (base_example, batch_index, trial, condition) → TrialOutcome
    lookup: dict[tuple[str, int, int, str], TrialOutcome] = {}
    for o in outcomes:
        if o.config_key != config_key:
            continue
        if o.condition not in (condition_a, condition_b):
            continue
        if o.matches_ground_truth is None:
            continue

        key = (o.base_example, o.batch_index, o.trial, o.condition)
        if key in lookup:
            raise ValueError(
                f"Duplicate trial outcome for {key!r}. "
                f"Each (base_example, batch_index, trial, condition) "
                f"must be unique within a config."
            )
        lookup[key] = o

    # Collect unique (batch_index, trial) pairs, sorted deterministically
    round_keys: set[tuple[int, int]] = set()
    for _be, bi, t, _cond in lookup:
        round_keys.add((bi, t))

    # Collect unique base examples that appear in either condition
    base_examples: set[str] = set()
    for _be, _bi, _t, _cond in lookup:
        base_examples.add(_be)
    sorted_bases = sorted(base_examples)

    rounds: list[tuple[int, int, list[float]]] = []
    for batch_idx, trial_num in sorted(round_keys):
        diffs: list[float] = []
        for base in sorted_bases:
            key_a = (base, batch_idx, trial_num, condition_a)
            key_b = (base, batch_idx, trial_num, condition_b)
            outcome_a = lookup.get(key_a)
            outcome_b = lookup.get(key_b)

            # Both conditions must have valid data for this base example in this round
            if outcome_a is None or outcome_b is None:
                continue

            d = float(outcome_a.matches_ground_truth) - float(outcome_b.matches_ground_truth)
            diffs.append(d)

        if diffs:
            rounds.append((batch_idx, trial_num, diffs))

    return rounds


# Which accuracy class each condition belongs to. Balanced accuracy is the
# unweighted mean of specificity (correct-draft) and sensitivity (the two
# incorrect-draft conditions pooled over their trials), so a pooled round must
# know which side each cell falls on before it can weight the two equally.
_ACCURACY_CLASSES: dict[Condition, str] = {
    "correct": "specificity",
    "transparent": "sensitivity",
    "opaque": "sensitivity",
}


def _class_reweighted_differences(
    cell_differences: list[tuple[Condition, float]],
    *,
    bet_cap: float,
) -> list[float]:
    """Scale one pooled round's differences so its mean is the balanced one.

    Balanced accuracy weights the two accuracy classes equally, so each class
    must carry half of the round's total weight. With ``n`` paired cells split
    ``n_spec`` to ``n_sens``, giving every specificity cell ``n / (2 * n_spec)``
    and every sensitivity cell ``n / (2 * n_sens)`` puts ``n / 2`` on each side,
    and the equal-weight mean of the scaled values is then the unweighted mean
    of the two class means. Equal class sizes make both weights exactly 1,
    which is why a balanced grid folds identically scaled or unscaled.

    That mean is the balanced accuracy difference exactly when every cell
    carries the same number of valid trials, the complete balanced pairing the
    design intends: only then is an unweighted mean over a class's cells the
    same number as that class's accuracy pooled over trials.

    **Why the cap is checked here.** A cell's weight ``w_c`` bounds both the
    difference the round realizes, since ``|d_c| <= w_c``, and the true mean
    difference ``δ_c`` the validity argument's arithmetic-geometric mean step
    quantifies over, since ``δ_c`` is a mean of values in ``[-w_c, +w_c]``.
    Every wealth factor ``1 + λ·δ_c`` is therefore strictly positive exactly
    while ``bet_cap * max weight < 1``, and this is where the weights are
    known. Bounding the realized differences instead would leave the step
    unproven: a round whose specificity cells all happen to tie carries no
    large value at all while still carrying a large weight.

    Cell order is preserved, so the fold's arithmetic stays reproducible.
    Returns an empty list when either class is absent from the round, since a
    balanced difference is undefined without both; the caller drops such a
    round rather than betting on half of the metric.
    """
    by_class: dict[str, list[float]] = {"specificity": [], "sensitivity": []}
    for condition, difference in cell_differences:
        accuracy_class = _ACCURACY_CLASSES.get(condition)
        if accuracy_class is None:
            raise ValueError(
                f"Condition {condition!r} belongs to no accuracy class, so it "
                "cannot be weighted into a balanced difference. Balanced "
                f"accuracy is defined over {sorted(_ACCURACY_CLASSES)}."
            )
        by_class[accuracy_class].append(difference)

    if not by_class["specificity"] or not by_class["sensitivity"]:
        return []

    total = len(cell_differences)
    weights = {
        accuracy_class: total / (2 * len(members))
        for accuracy_class, members in by_class.items()
    }

    heaviest = max(weights, key=lambda accuracy_class: weights[accuracy_class])
    max_weight = weights[heaviest]
    if bet_cap * max_weight >= 1.0:
        raise ValueError(
            f"A pooled round splitting {len(by_class['specificity'])} "
            f"correct-draft cells to {len(by_class['sensitivity'])} "
            f"incorrect-draft cells weights its {heaviest} class at "
            f"{max_weight}, which a bet capped at {bet_cap} cannot carry: "
            "positivity requires bet_cap x class weight < 1, and this round "
            f"gives {bet_cap * max_weight}."
        )

    return [
        weights[_ACCURACY_CLASSES[condition]] * difference
        for condition, difference in cell_differences
    ]


def _build_cross_sample_rounds(
    test_outcomes: list[TrialOutcome],
    reference_outcomes: list[TrialOutcome],
    test_config_key: str,
    reference_config_key: str,
    *,
    bet_cap: float,
    condition: Condition | None = None,
) -> list[tuple[int, int, list[float]]]:
    """Pair two samples cell by cell and compute their paired differences.

    A cell is one stimulus — a ``(base_example, condition)`` pair — and a
    round is one ``(batch_index, trial)`` position, taken in ascending order.
    Within a round, every cell present on both sides contributes

        d = float(test.matches_ground_truth) - float(reference.matches_ground_truth)

    ordered by cell so the fold's arithmetic is reproducible. Cells missing on
    either side are omitted from that round, and a round left with no
    differences is omitted entirely.

    ``condition`` restricts the pairing to one condition, where the raw binary
    differences lie in {-1, 0, +1} and the round mean is that condition's
    accuracy difference. ``None`` pools all three, which is what the
    configuration-level comparisons test; those differences pass through
    ``_class_reweighted_differences`` so the round mean is the balanced
    accuracy difference. A pooled round carrying only one accuracy class has no
    balanced difference and is dropped.

    ``bet_cap`` is the cap the caller will fold these rounds at. A pooled
    round's class weights are the bound the fold's positivity argument rests
    on, and they are known here and nowhere later, so the builder refuses a
    round whose weights that cap cannot carry rather than handing back one the
    fold would have to trust. A condition-restricted round is one accuracy
    class, whose raw differences are already bounded by 1, so only the cap's
    own admissibility applies to it.

    The two samples are identified by ``(outcomes, config_key)`` pairs rather
    than by role, so the same builder serves a within-experiment comparison of
    two configurations and a cross-dataset comparison of one configuration in
    two experiments.

    Returns ``(batch_index, trial, differences)`` tuples sorted by
    ``(batch_index, trial)``. Raises ValueError when either sample carries a
    duplicate ``(base_example, condition, batch_index, trial)`` key, which
    would leave the pairing undefined, and when two samples that both carry
    valid data produce no round at all — whether because they share no cell or
    because no round carried both accuracy classes. Either way there is no
    paired test to report, and neutral evidence would read as "no effect" for a
    comparison that never ran. A sample with no valid data pairs to nothing
    legitimately and yields no rounds.
    """
    _validate_bet_cap(bet_cap)

    def _index(
        outcomes: list[TrialOutcome], config_key: str, label: str,
    ) -> dict[tuple[str, str, int, int], TrialOutcome]:
        lookup: dict[tuple[str, str, int, int], TrialOutcome] = {}
        for o in outcomes:
            if o.config_key != config_key:
                continue
            if condition is not None and o.condition != condition:
                continue
            if o.matches_ground_truth is None:
                continue
            key = (o.base_example, o.condition, o.batch_index, o.trial)
            if key in lookup:
                raise ValueError(
                    f"Duplicate trial outcome for {key!r} in the {label} "
                    f"sample of {config_key!r}. Each "
                    f"(base_example, condition, batch_index, trial) must be "
                    f"unique within a sample."
                )
            lookup[key] = o
        return lookup

    test_lookup = _index(test_outcomes, test_config_key, "test")
    reference_lookup = _index(reference_outcomes, reference_config_key, "reference")

    round_keys = {
        (batch_index, trial)
        for _base, _cond, batch_index, trial in (*test_lookup, *reference_lookup)
    }
    cells = sorted(
        {(base, cond) for base, cond, _bi, _t in (*test_lookup, *reference_lookup)}
    )

    rounds: list[tuple[int, int, list[float]]] = []
    dropped_for_one_class = False
    for batch_index, trial in sorted(round_keys):
        paired: list[tuple[Condition, float]] = []
        for base, cond in cells:
            key = (base, cond, batch_index, trial)
            test_outcome = test_lookup.get(key)
            reference_outcome = reference_lookup.get(key)
            if test_outcome is None or reference_outcome is None:
                continue
            paired.append((
                cond,
                float(test_outcome.matches_ground_truth)
                - float(reference_outcome.matches_ground_truth),
            ))
        if not paired:
            continue
        if condition is not None:
            differences = [difference for _cond, difference in paired]
        else:
            differences = _class_reweighted_differences(paired, bet_cap=bet_cap)
            if not differences:
                dropped_for_one_class = True
                continue
        rounds.append((batch_index, trial, differences))

    if test_lookup and reference_lookup and not rounds:
        cause = (
            "every round that paired at all carried a single accuracy class, "
            "and a balanced difference needs both, so there is no paired "
            "balanced difference to bet on"
            if dropped_for_one_class
            else "they share no (base_example, condition, batch_index, trial) "
            "cell, so there is no paired difference to bet on"
        )
        raise ValueError(
            f"No paired rounds between {test_config_key!r} and "
            f"{reference_config_key!r}"
            + (f" for condition {condition!r}" if condition is not None else "")
            + f". Both samples carry valid trials but {cause}."
        )

    return rounds


# Fixed a priori, never selected against results. See _compute_bet_size.
BET_CAP: float = 0.5

# Which half-line a family's bet may occupy. ``non_positive`` and
# ``non_negative`` carry the one-sided nulls: a bet on the side the null
# forbids would let the wealth grow under the null itself.
BetSign = Literal["any", "non_positive", "non_negative"]

_BET_SIGNS: frozenset[str] = frozenset({"any", "non_positive", "non_negative"})


def _validate_bet_sign(bet_sign: str) -> None:
    """Reject a sign constraint the validity argument does not cover.

    Each one-sided family bounds its wealth by holding the bet on the side
    where ``bet x mean difference`` is at most zero under its null. An
    unrecognized constraint read as "unconstrained" would silently drop that
    bound, so it raises instead.
    """
    if bet_sign not in _BET_SIGNS:
        raise ValueError(
            f"bet_sign must be one of {sorted(_BET_SIGNS)}; got {bet_sign!r}."
        )


def _validate_bet_cap(bet_cap: float) -> None:
    """Reject a cap the wealth update cannot carry.

    A condition-restricted paired difference is -1, 0, or +1, so a factor
    1 + λ·d stays strictly positive only while |λ| < 1. A cap of 1 or more
    admits a bet that drives some factor to zero or below, which has no
    logarithm; a cap of zero silently disables betting; a negative cap inverts
    the clip and returns a bet of the wrong sign.

    The interval is the cap's admissibility on its own, not the whole
    positivity condition, so passing it does not make every cap usable
    everywhere. A pooled round carries class weights above 1 whenever the two
    accuracy classes contribute unequal numbers of cells, and those weights
    bound its differences, so such a round admits only the part of this
    interval below ``1 / max weight``: the complete grid's ten-to-twenty split
    weighs 1.5 and stops at two thirds, and a round that loses correct-draft
    pairing stops lower still. ``_build_cross_sample_rounds`` applies that
    split-dependent bound where the weights are known.
    """
    if not 0.0 < bet_cap < 1.0:
        raise ValueError(
            f"bet_cap must lie in the open interval (0, 1); got {bet_cap!r}. "
            "A condition-restricted paired difference is -1, 0, or +1, so the "
            "wealth factor 1 + bet x difference stays positive only for a bet "
            "strictly inside the ruin boundary at magnitude 1."
        )


def _compute_bet_size(
    running_mean: float,
    running_var: float,
    *,
    bet_cap: float = BET_CAP,
    bet_sign: BetSign = "any",
) -> float:
    """Compute the capped plug-in bet for one round of paired differences.

    The bet is the running mean-to-variance ratio clipped to
    ``[-bet_cap, +bet_cap]``; zero or negative variance places no bet.

    **The sign constraint.** ``bet_sign`` narrows that interval to one half
    of it for the one-sided families: ``non_positive`` gives
    ``[-bet_cap, 0]`` and ``non_negative`` gives ``[0, +bet_cap]``. Under a
    one-sided null the permitted half is the one on which
    ``bet x mean difference`` cannot be positive, which is what holds each
    round's expected wealth factor at or below 1. A ratio pointing the other
    way is clamped to zero — the process declines the bet rather than taking
    one the null would reward. The constraint only ever narrows the interval,
    so the cap still bounds the bet in both directions.

    **Why one half.** A condition-restricted difference lies in {-1, 0, +1},
    so a bet of magnitude one sits on the ruin boundary: an adverse pair
    meeting it sends the wealth factor to exactly zero and the accumulated
    evidence to negative infinity. A pooled round's class weights move that
    boundary inward, which ``_build_cross_sample_rounds`` enforces when it
    builds the round. The plug-in ratio reaches the boundary whenever the mean
    of the observed differences dominates their variance. A one-sided pool
    guarantees it — a pool of {0, +1} at fraction f has ratio
    f / (f(1 − f)) = 1/(1 − f) ≥ 1 — and a mixed pool reaches it too once
    the lean is strong enough. Those are the pools that most favor the
    alternative, so an uncapped ratio destroys the most evidence in the
    cases that carry the most.

    Ramdas and Wang, *Hypothesis Testing with E-values*, Definition 7.21
    bounds the bets of the empirically adaptive e-process by a parameter γ
    and takes γ = 1/2 as its uninformative default, with a first bet of zero
    that this module's round 0 mirrors. Under the correspondence
    E_s = 1 + d_s the book's wealth factor (1 − λ) + λE_s is identically the
    1 + λ·d_s used here, so γ = 1/2 maps exactly onto ``BET_CAP`` for λ ≥ 0,
    and the symmetric cap extends the book's one-sided [0, γ] class by
    mirroring it for signed differences. The book's bounded mean analysis
    (§6.7.4) places the log-optimal bet strictly inside the ruin boundary
    whenever the adverse outcome has positive probability, since expected
    log growth has derivative −∞ there.

    ``BET_CAP`` is fixed a priori and not tuned to the results; ``bet_cap``
    exists to report sensitivity to that commitment, never to select it.
    """
    _validate_bet_cap(bet_cap)
    _validate_bet_sign(bet_sign)

    if running_var <= 0.0:
        return 0.0

    lower = 0.0 if bet_sign == "non_negative" else -bet_cap
    upper = 0.0 if bet_sign == "non_positive" else bet_cap
    raw_lambda = running_mean / running_var
    return max(lower, min(upper, raw_lambda))


def _accumulate_difference_rounds(
    rounds: list[tuple[int, int, list[float]]],
    *,
    bet_sign: BetSign = "any",
    bet_cap: float = BET_CAP,
) -> tuple[tuple[ComparisonSlice, ...], float, list[float]]:
    """Fold a sequence of paired-difference rounds into a wealth process.

    The engine shared by every comparison family. Round 0 is calibration
    (λ = 0); each later round's bet is the plug-in mean-to-variance ratio of
    all differences observed before it, capped and sign-constrained by
    ``_compute_bet_size``, and the round's contribution is the sum of
    ``log(1 + λ·d)`` over its differences.

    **Why the process is anytime-valid.** (1) Round t's bet is a function of
    rounds 0..t−1 alone, taken in a fixed, data-independent round order, so it
    is predictable with respect to the round filtration. (2) Trials are
    independent across rounds, so conditioning on the past leaves round t's
    cells with their marginal law, and cells within a round are independent of
    one another, so the conditional expected factor is
    ``Π_c (1 + λ_t·δ_c)`` with ``δ_c`` the true per-cell mean difference.
    Cross-round independence licenses the conditional-to-marginal step;
    within-round independence licenses the factorization. (3) Every factor
    ``1 + λ_t·δ_c`` is strictly positive, so the arithmetic-geometric mean
    inequality gives ``Π_c (1 + λ_t·δ_c) ≤ (1 + λ_t·δ̄_t)^{m_t}`` for ``δ̄_t``
    the equal-weight mean over the round's ``m_t`` present cells. Positivity
    holds because each cell's weight ``w_c`` bounds ``|δ_c|`` — a true mean of
    values in ``[-w_c, +w_c]`` — and the round builders admit only weights with
    ``bet_cap * w_c < 1``: a condition-restricted round has ``w_c = 1``, which
    the cap's own interval covers, and a pooled round's class weights are
    checked against the cap in ``_class_reweighted_differences``. Bounding the
    *realized* ``|d_c|`` would not establish this, since a cell can realize a
    tie in every round it appears in and still carry a large ``δ_c``. (4) Under
    a one-sided null with its matching ``bet_sign``, ``λ_t·δ̄_t ≤ 0`` every
    round, so the wealth is a nonnegative supermartingale and Ville's
    inequality applies.

    **What the null is about.** Each round's condition is on its own
    equal-weight mean, and what that mean estimates is set by how the round's
    cells were weighted before they arrived. A condition-restricted round
    carries binary differences, so its mean is that condition's accuracy
    difference. A pooled round carries class-reweighted differences, so its
    mean is the balanced accuracy difference.

    Under complete balanced pairing every round carries the same cells, so the
    per-round condition is equivalent to the pooled one-sided null over all
    rounds. That pooled quantity is the reported accuracy difference exactly
    when every cell also carries the same number of valid trials, since only
    then does an unweighted mean over cells equal an accuracy pooled over
    trials. When pairs are missing, round membership varies and no single
    pooled equal-weight mean is licensed: validity then rests on the per-round
    nulls — each round's equal-weight mean over its own present cells — or on
    the stronger per-cell null that every ``δ_c`` lies on the null side, which
    implies the per-round condition whatever the membership.

    Returns ``(slices, running_log_e_value, all_differences)``.
    """
    _validate_bet_cap(bet_cap)
    _validate_bet_sign(bet_sign)

    slices: list[ComparisonSlice] = []
    running_log_e = 0.0
    all_differences: list[float] = []

    for round_idx, (batch_idx, trial_num, diffs) in enumerate(rounds):
        # Bet size from prior data (non-anticipatory)
        if all_differences:
            running_mean = sum(all_differences) / len(all_differences)
            running_var = (
                sum((d - running_mean) ** 2 for d in all_differences)
                / len(all_differences)
            )
        else:
            # Round 0: conservative default — no bet (calibration)
            running_mean = 0.0
            running_var = 0.0

        bet = _compute_bet_size(
            running_mean, running_var, bet_cap=bet_cap, bet_sign=bet_sign,
        )
        mean_diff = sum(diffs) / len(diffs)

        # E_round = Π (1 + λ × d_i)
        round_log_e = 0.0
        for d in diffs:
            round_log_e += math.log(1.0 + bet * d)

        running_log_e += round_log_e
        all_differences.extend(diffs)

        slices.append(ComparisonSlice(
            round_index=round_idx,
            batch_index=batch_idx,
            trial=trial_num,
            differences=tuple(diffs),
            mean_difference=mean_diff,
            bet_size=bet,
            log_e_value=round_log_e,
        ))

    return tuple(slices), running_log_e, all_differences


def _is_significant(adjusted_log_e_value: float, alpha: float) -> bool:
    """Whether a corrected process has reached its rejection bar.

    The comparison runs in log space against ``log(1/alpha)`` and is
    inclusive at the bar. That is monotone-equivalent to comparing e-values
    but free of the exp/log round-trip: ``exp(log(20.0))`` is
    19.999999999999996, so an e-scale comparison would miss a process sitting
    exactly on the bar.
    """
    return adjusted_log_e_value >= math.log(1.0 / alpha)


def compute_pairwise_comparison(
    outcomes: list[TrialOutcome],
    config_key: str,
    condition_a: Condition,
    condition_b: Condition,
    *,
    bet_cap: float = BET_CAP,
) -> PairwiseComparison:
    """Compute the pairwise comparison e-value for two conditions.

    Processes all trial-level rounds sequentially, using the
    predictable plug-in betting strategy: at each round, the bet is
    determined from prior observations (non-anticipatory), then applied
    to the new per-example differences. Round 0 is calibration (λ = 0).

    Differences lie in {-1, 0, +1} and the bet in ``[-bet_cap, +bet_cap]``,
    so every wealth factor 1 + λ·d lies in ``[1 - bet_cap, 1 + bet_cap]`` —
    [0.5, 1.5] at the default cap. Because ``bet_cap`` is validated into
    (0, 1), that interval stays strictly positive and the fold needs no
    guard on the logarithm.
    """
    _validate_bet_cap(bet_cap)

    rounds = _build_trial_rounds(outcomes, config_key, condition_a, condition_b)
    slices, running_log_e, all_differences = _accumulate_difference_rounds(
        rounds, bet_cap=bet_cap,
    )

    observed_mean = (
        sum(all_differences) / len(all_differences) if all_differences else 0.0
    )

    return PairwiseComparison(
        config_key=config_key,
        condition_a=condition_a,
        condition_b=condition_b,
        slices=slices,
        running_log_e_value=running_log_e,
        running_e_value=math.exp(running_log_e),
        observed_mean_difference=observed_mean,
        total_pairs=sum(len(s.differences) for s in slices),
    )


# The three pairwise comparisons available from the condition structure
CONDITION_PAIRS: list[tuple[Condition, Condition]] = [
    ("correct", "transparent"),
    ("correct", "opaque"),
    ("transparent", "opaque"),
]


def compute_all_comparisons(
    outcomes: list[TrialOutcome],
    config_key: str,
    *,
    bet_cap: float = BET_CAP,
) -> list[PairwiseComparison]:
    """Compute all three pairwise condition comparison e-values."""
    return [
        compute_pairwise_comparison(
            outcomes, config_key, cond_a, cond_b, bet_cap=bet_cap,
        )
        for cond_a, cond_b in CONDITION_PAIRS
    ]


# ---------------------------------------------------------------------------
# Global null condition comparison and the collapsed closed test
# ---------------------------------------------------------------------------


class ConditionPairResult(BaseModel):
    """One condition pair's e-value and its collapsed-closed-test outcome."""

    model_config = ConfigDict(frozen=True)

    condition_a: Condition
    condition_b: Condition
    log_e_value: float
    e_value: float
    # significant ⟺ this pair AND its configuration's global null both reach 1/α
    significant: bool


class GlobalNullComparison(BaseModel):
    """Closed test over the three condition comparisons of one configuration.

    **Global null statistic.** The equal-weight arithmetic mean of the
    three pairwise e-values, computed in log space via ``_log_mean_exp``.
    It tests H_G: p_correct = p_transparent = p_opaque within this
    configuration — known in the analysis-of-variance tradition as an
    omnibus test. Under H_G each pairwise e-process has expectation at
    most 1, so by linearity their mean does too, valid under arbitrary
    dependence between the three, which matters because they share trial
    rounds. Weights are fixed at equal by design and are deliberately not
    a parameter: a tunable weight would invite selection after seeing the
    data and break the bound.

    **Collapsed closed test.** The three nulls are equalities, so any two
    of them imply the third. Every 2-way intersection of the pairwise
    nulls is therefore H_G itself, and the closure over the three
    hypotheses collapses from seven intersections to four distinct ones:
    the three singletons and H_G. Closed testing rejects a pair exactly
    when every intersection containing it is rejected, which reduces to
    two gates — the pair's own e-value reaching 1/α and the global null
    statistic reaching 1/α. Each is an anytime-valid e-process test, so
    Ville's inequality gives family-wise error at most α at every stopping
    time, with no union bound needed.

    **What the second gate can do.** Because an arithmetic mean never
    exceeds its maximum, the global null gate can only ever *block* a pair
    that clears 1/α on its own; it can never *rescue* one that does not.
    Equivalently, ``significant`` on this model holds exactly when at
    least one ``pair_results`` entry is significant.

    **Scope.** The transitivity collapse is a three-condition theorem. It
    is NOT valid for K > 3 without redesign: with four or more conditions
    the 2-way intersections are strictly weaker than the global null and
    must be tested separately. ``compute_global_null_comparison``
    enforces the premise rather than generalizing silently.

    The family the error rate covers is the three condition pairs **within
    one configuration**. Nothing here corrects across configurations.
    """

    model_config = ConfigDict(frozen=True)

    config_key: str

    # One entry per CONDITION_PAIRS pair, in CONDITION_PAIRS order.
    pair_results: tuple[ConditionPairResult, ...]

    # Test of H_G, in both representations (log is primary).
    global_null_log_e_value: float
    global_null_e_value: float
    significant: bool  # global_null_log_e_value >= log(1/alpha)

    alpha: float

    @model_validator(mode="after")
    def _validate_pair_structure(self) -> "GlobalNullComparison":
        """The collapse presumes the three CONDITION_PAIRS, once each, in order."""
        pairs = [frozenset((r.condition_a, r.condition_b)) for r in self.pair_results]
        if pairs != [frozenset(pair) for pair in CONDITION_PAIRS]:
            raise ValueError(
                "GlobalNullComparison must carry the three CONDITION_PAIRS "
                f"exactly once each in CONDITION_PAIRS order; got "
                f"{[sorted(pair) for pair in pairs]}"
            )
        return self


def compute_global_null_comparison(
    comparisons: list[PairwiseComparison],
    *,
    alpha: float = 0.05,
) -> GlobalNullComparison:
    """Collapse one configuration's three pairwise comparisons into a closed test.

    ``comparisons`` must be exactly the three ``CONDITION_PAIRS``
    comparisons for a single ``config_key``; pairs are matched as
    unordered sets, so the orientation of ``condition_a``/``condition_b``
    is irrelevant. Anything else raises ``ValueError`` rather than
    averaging a structure the closure argument does not cover.

    All merge arithmetic runs in log space from ``running_log_e_value``.
    That is the module's primary representation and the only one that
    stays finite across the full range: ``running_e_value`` is
    ``exp(running_log_e_value)`` and underflows to exactly 0.0 once the
    log passes about -745, at which point the mean would silently lose
    the term.
    """
    if len(comparisons) != len(CONDITION_PAIRS):
        raise ValueError(
            "The collapsed closed test is a three-condition theorem and "
            "requires exactly three pairwise comparisons (one per "
            f"CONDITION_PAIRS entry); got {len(comparisons)}"
        )

    config_keys = {comparison.config_key for comparison in comparisons}
    if len(config_keys) != 1:
        raise ValueError(
            "All pairwise comparisons must describe a single configuration; "
            f"got {sorted(config_keys)}"
        )

    by_pair = {
        frozenset((comparison.condition_a, comparison.condition_b)): comparison
        for comparison in comparisons
    }
    if by_pair.keys() != {frozenset(pair) for pair in CONDITION_PAIRS}:
        raise ValueError(
            "Pairwise comparisons must cover the three CONDITION_PAIRS exactly "
            f"once each; got {sorted(sorted(pair) for pair in by_pair)}"
        )

    # Fix a canonical CONDITION_PAIRS order once, so the merge and the
    # emitted results read the same K inputs in the same sequence.
    ordered = [by_pair[frozenset(pair)] for pair in CONDITION_PAIRS]

    # K is read from the validated input rather than assumed, and the
    # equal-weight merge is the module's existing mean-of-e-values helper.
    global_null_log_e = _log_mean_exp([c.running_log_e_value for c in ordered])
    if math.isnan(global_null_log_e):
        # Reachable only from synthetic input where every log is -inf; the
        # max-shift then evaluates -inf - -inf. Raise rather than emit a
        # NaN e-value that would silently read as not significant.
        raise ValueError(
            "The global null merge produced NaN, which means every pairwise "
            f"log e-value was -inf for {ordered[0].config_key!r}. A real fold "
            "holds every factor strictly above zero and cannot reach this "
            "state."
        )
    global_null_e = math.exp(global_null_log_e)

    # Both gates compare in log space against log(1/α). That is monotone-
    # equivalent to comparing e-values but free of the exp/log round-trip:
    # ``exp(log(20.0))`` is 19.999999999999996, so an e-scale comparison
    # would miss a process sitting exactly on the bar.
    log_threshold = math.log(1.0 / alpha)
    global_null_significant = global_null_log_e >= log_threshold

    pair_results = tuple(
        ConditionPairResult(
            condition_a=comparison.condition_a,
            condition_b=comparison.condition_b,
            log_e_value=comparison.running_log_e_value,
            e_value=comparison.running_e_value,
            significant=(
                comparison.running_log_e_value >= log_threshold
                and global_null_significant
            ),
        )
        for comparison in ordered
    )

    return GlobalNullComparison(
        config_key=next(iter(config_keys)),
        pair_results=pair_results,
        global_null_log_e_value=global_null_log_e,
        global_null_e_value=global_null_e,
        significant=global_null_significant,
        alpha=alpha,
    )


def compute_all_global_null_comparisons(
    comparisons: list[PairwiseComparison],
    *,
    alpha: float = 0.05,
) -> list[GlobalNullComparison]:
    """One global null closed test per configuration, ordered by ``config_key``.

    Groups a flat comparison sequence — such as
    ``ExperimentAnalysisResults.comparisons``, which holds all three pairs
    for every configuration — by ``config_key``. A configuration whose
    group is not the three ``CONDITION_PAIRS`` raises rather than being
    skipped: a missing pair means the closure has nothing to gate on.
    """
    grouped: dict[str, list[PairwiseComparison]] = {}
    for comparison in comparisons:
        grouped.setdefault(comparison.config_key, []).append(comparison)

    return [
        compute_global_null_comparison(grouped[config_key], alpha=alpha)
        for config_key in sorted(grouped)
    ]


# ---------------------------------------------------------------------------
# Adjacent-pair reasoning effort level degradation tests
# ---------------------------------------------------------------------------


class AdjacentPairDegradation(BaseModel):
    """Adjacent-pair reasoning effort level degradation test result for one transition.

    Tests whether accuracy decreased when moving from a lower to a higher
    reasoning effort level within a model family. The test is one-sided:
    H₀: p_upper ≥ p_lower (no degradation) vs H₁: p_upper < p_lower.

    The two configurations share a stimulus grid, so the test bets on the
    paired differences ``upper − lower`` cell by cell, with the bet held at
    or below zero. Under the null the equal-weight mean difference is
    non-negative, so a non-positive bet keeps every round's expected wealth
    factor at or below 1.

    **What the reported accuracy is.** The rounds pool all three conditions and
    are class-reweighted, so the tested quantity is balanced accuracy and the
    three ``*_balanced_accuracy`` fields report it. The hit and valid counts
    beside them are sample sizes rather than a second accuracy: they pool every
    condition's trials, so ``lower_hits / lower_valid`` is the descriptive
    overall accuracy and deliberately does not equal
    ``lower_balanced_accuracy``.
    """

    model_config = ConfigDict(frozen=True)

    model_group: str  # e.g. "openai--gpt-5.2"
    lower_config_key: str  # e.g. "openai--gpt-5.2--medium"
    upper_config_key: str  # e.g. "openai--gpt-5.2--high"
    lower_level: str  # e.g. "medium"
    upper_level: str  # e.g. "high"

    # Sample sizes behind the test, pooled over every condition's trials. Not
    # the numerator and denominator of the balanced accuracies below.
    lower_hits: int
    lower_valid: int
    upper_hits: int
    upper_valid: int

    # Observed balanced accuracies, the quantity the paired fold bets on
    lower_balanced_accuracy: float
    upper_balanced_accuracy: float
    delta_balanced_accuracy: float  # upper - lower; negative means degradation observed

    # E-value components
    raw_log_e_value: float
    raw_e_value: float

    # Multiplicity correction
    bonferroni_k: int  # total number of tests in the family
    adjusted_log_e_value: float  # raw_log - log(K)
    adjusted_e_value: float  # raw / K
    significant: bool  # adjusted_log_e_value >= log(1/alpha)

    # Per-round audit trail of the paired fold.
    e_slices: tuple[ComparisonSlice, ...]

    @model_validator(mode="after")
    def _validate_audit_trail(self) -> "AdjacentPairDegradation":
        """A test with valid data on both sides must carry its audit trail."""
        if self.lower_valid > 0 and self.upper_valid > 0 and not self.e_slices:
            raise ValueError(
                "AdjacentPairDegradation with valid trials on both sides "
                "must carry a non-empty e_slices audit trail"
            )
        return self


def _tally_config_outcomes(
    outcomes: list[TrialOutcome],
    config_key: str,
) -> tuple[int, int]:
    """Count hits and valid trials for a config, pooled across all conditions.

    A trial is valid when matches_ground_truth is not None. A hit is a
    valid trial where matches_ground_truth is True.
    """
    hits = 0
    valid = 0
    for o in outcomes:
        if o.config_key != config_key:
            continue
        if o.matches_ground_truth is None:
            continue
        valid += 1
        if o.matches_ground_truth:
            hits += 1
    return hits, valid


def _tally_config_condition_outcomes(
    outcomes: list[TrialOutcome],
    config_key: str,
    condition: Condition,
) -> tuple[int, int]:
    """Count hits and valid trials for one (config, condition) pair.

    Like ``_tally_config_outcomes`` but restricted to a single condition
    instead of pooling across all three.
    """
    hits = 0
    valid = 0
    for o in outcomes:
        if o.config_key != config_key:
            continue
        if o.condition != condition:
            continue
        if o.matches_ground_truth is None:
            continue
        valid += 1
        if o.matches_ground_truth:
            hits += 1
    return hits, valid


def compute_adjacent_pair_degradation(
    outcomes: list[TrialOutcome],
    lower_config_key: str,
    upper_config_key: str,
    bonferroni_k: int,
    *,
    bet_cap: float = BET_CAP,
    alpha: float = 0.05,
) -> AdjacentPairDegradation:
    """Compute the degradation e-value for one adjacent reasoning effort level pair.

    The two configurations ran the same stimulus grid, so each
    ``(base_example, condition, batch_index, trial)`` cell yields one paired
    difference ``upper − lower`` and each ``(batch_index, trial)`` position is
    one betting round. The bet is held at or below zero, which is the side on
    which ``bet x mean difference`` cannot be positive under
    H₀: p_upper ≥ p_lower.

    The displayed accuracies and the e-value both describe balanced accuracy
    but aggregate it differently:

    - **Displayed values** come from ``_balanced_accuracy_from_outcomes``,
      which pools hits over valid trials within each accuracy class and takes
      the unweighted mean of the two.
    - **E-value** bets on the class-reweighted per-cell differences, whose
      equal-weight round means carry that same balanced quantity under
      complete pairing at equal valid counts.
    """
    _validate_bet_cap(bet_cap)

    lower_provider, lower_model, lower_level = parse_config_key(lower_config_key)
    _, _, upper_level = parse_config_key(upper_config_key)
    model_group = f"{lower_provider}--{lower_model}"

    lower_hits, lower_valid = _tally_config_outcomes(outcomes, lower_config_key)
    upper_hits, upper_valid = _tally_config_outcomes(outcomes, upper_config_key)

    lower_balanced = _balanced_accuracy_from_outcomes(outcomes, lower_config_key)
    upper_balanced = _balanced_accuracy_from_outcomes(outcomes, upper_config_key)

    rounds = _build_cross_sample_rounds(
        outcomes, outcomes, upper_config_key, lower_config_key,
        bet_cap=bet_cap,
    )
    slices, log_e, _differences = _accumulate_difference_rounds(
        rounds, bet_sign="non_positive", bet_cap=bet_cap,
    )
    raw_e = math.exp(log_e)

    # e-Bonferroni: divide raw e-value by K
    adjusted_log_e = log_e - math.log(bonferroni_k) if bonferroni_k > 0 else log_e
    adjusted_e = raw_e / bonferroni_k if bonferroni_k > 0 else raw_e

    return AdjacentPairDegradation(
        model_group=model_group,
        lower_config_key=lower_config_key,
        upper_config_key=upper_config_key,
        lower_level=lower_level,
        upper_level=upper_level,
        lower_hits=lower_hits,
        lower_valid=lower_valid,
        upper_hits=upper_hits,
        upper_valid=upper_valid,
        lower_balanced_accuracy=lower_balanced,
        upper_balanced_accuracy=upper_balanced,
        delta_balanced_accuracy=upper_balanced - lower_balanced,
        raw_log_e_value=log_e,
        raw_e_value=raw_e,
        bonferroni_k=bonferroni_k,
        adjusted_log_e_value=adjusted_log_e,
        adjusted_e_value=adjusted_e,
        significant=_is_significant(adjusted_log_e, alpha),
        e_slices=slices,
    )


def compute_all_degradation_tests(
    outcomes: list[TrialOutcome],
    *,
    bet_cap: float = BET_CAP,
    alpha: float = 0.05,
) -> list[AdjacentPairDegradation]:
    """Compute adjacent-pair reasoning effort level degradation tests for all adjacent pairs.

    Enumerates adjacent pairs from REASONING_SCALES for every model
    whose config_keys are actually present in the outcomes. K (the
    Bonferroni correction factor) is computed dynamically from the
    pairs that have data, not from REASONING_SCALES alone.
    """
    _validate_bet_cap(bet_cap)

    # Collect config_keys present in the outcomes
    present_keys: set[str] = {o.config_key for o in outcomes}

    # Enumerate all adjacent pairs that have data for both levels
    pairs: list[tuple[str, str]] = []
    for (provider, model_slug), scale in REASONING_SCALES.items():
        if len(scale) < 2:
            continue
        for i in range(len(scale) - 1):
            lower_key = f"{provider}--{model_slug}--{scale[i]}"
            upper_key = f"{provider}--{model_slug}--{scale[i + 1]}"
            if lower_key in present_keys and upper_key in present_keys:
                pairs.append((lower_key, upper_key))

    bonferroni_k = len(pairs)

    return [
        compute_adjacent_pair_degradation(
            outcomes, lower_key, upper_key, bonferroni_k,
            bet_cap=bet_cap, alpha=alpha,
        )
        for lower_key, upper_key in pairs
    ]


# ---------------------------------------------------------------------------
# Within-provider model size comparison tests
# ---------------------------------------------------------------------------

# The three conditions the composite accuracy metrics are computed over,
# derived from the class map so the two cannot drift apart.
_CONDITIONS: tuple[Condition, ...] = tuple(_ACCURACY_CLASSES)


def _log_mean_exp(logs: list[float]) -> float:
    """log of the arithmetic mean of exp(values), max-shifted for stability.

    The generic e-value merge for an intersection null: when each input is
    a valid e-value under its own component null, their arithmetic mean
    has expectation ≤ 1 under the conjunction of those nulls, and that
    holds under arbitrary dependence between the inputs.

    Three call sites instantiate it, over different component sets.
    ``compute_all_model_size_tests`` uses it twice: once per larger-model
    level, where the components are that level's processes against each
    smaller-model configuration and the conjunction is "this level beats no
    configuration", and once across levels, where the components are the
    per-level minima and the conjunction is "every level is beaten by some
    configuration". ``compute_global_null_comparison`` uses it for the three
    condition comparisons, where the components are the three pairwise
    equalities and the conjunction is the global null that all three
    condition accuracies agree.
    """
    if not logs:
        raise ValueError("logs must be non-empty")
    peak = max(logs)
    return peak + math.log(
        sum(math.exp(x - peak) for x in logs) / len(logs)
    )


class SmallerConfigurationFold(BaseModel):
    """One larger-model level's paired e-process against one smaller configuration.

    The atom of the model size audit. Every pair of a larger-model reasoning
    effort level with a smaller-model configuration gets one of these, so no
    configuration on either side is ever selected out.
    """

    model_config = ConfigDict(frozen=True)

    smaller_config_key: str
    smaller_valid: int
    log_e_value: float
    slices: tuple[ComparisonSlice, ...]


class LevelFold(BaseModel):
    """One larger-model level's family of processes, with both inner combinations.

    Holds this level's process against every smaller-model configuration plus
    the two quantities the outer combinations consume: the log of the mean
    e-value across configurations, which the every-level claim minimizes over
    levels, and the minimum log e-value across configurations, which the
    some-level claim averages over levels. A level with zero valid trials
    carries neutral folds (log e-value 0.0, empty slices).

    **What the reported accuracy is.** The rounds pool all three conditions and
    are class-reweighted, so the tested quantity is balanced accuracy and
    ``balanced_accuracy`` reports it. ``valid`` beside it is a sample size
    rather than that accuracy's denominator: it counts every condition's valid
    trials, which is the denominator of the descriptive overall accuracy
    instead.
    """

    model_config = ConfigDict(frozen=True)

    config_key: str  # the larger-model level
    valid: int  # sample size, pooled over every condition's trials
    balanced_accuracy: float
    configuration_folds: tuple[SmallerConfigurationFold, ...]
    log_mean_e_value: float  # log of the mean e-value over configurations
    log_min_e_value: float  # minimum log e-value over configurations

    @model_validator(mode="after")
    def _validate_audit_trails(self) -> "LevelFold":
        """A level has no statistic without its per-configuration folds."""
        if not self.configuration_folds:
            raise ValueError(
                "LevelFold must carry one entry in configuration_folds per "
                "smaller-model configuration"
            )
        for fold in self.configuration_folds:
            if self.valid > 0 and fold.smaller_valid > 0 and not fold.slices:
                raise ValueError(
                    f"The pair of {self.config_key!r} with "
                    f"{fold.smaller_config_key!r} has valid trials on both "
                    "sides and must carry a non-empty slices audit trail"
                )
        return self


class ModelSizeComparison(BaseModel):
    """E-value test result for one within-provider model size claim.

    No configuration is selected on either side. For every pair of a
    larger-model reasoning effort level i with a smaller-model configuration
    j there is one sign-constrained paired e-process E_ij, held to a
    non-negative bet because the null is H₀: p_larger ≤ p_smaller. The two
    claims differ only in how those processes are combined.

    - ``comparison_type="worst"`` (the every-level claim): the statistic is
      the minimum over levels of the mean over configurations. Its null is
      the union over levels of the intersection over configurations — some
      level beats no configuration — so rejecting it makes the claim true of
      the smaller model's *true worst* configuration, since beating any
      configuration implies beating the one with the lowest rate.
    - ``comparison_type="best"`` (the some-level claim): the statistic is the
      mean over levels of the minimum over configurations. Its null is the
      intersection over levels of the union over configurations — every level
      is beaten by some configuration — so rejecting it speaks to the *true
      best* configuration.

    Both are valid under arbitrary dependence between the per-pair processes,
    by the same two facts the module uses elsewhere: a mean of e-values is an
    e-value under the conjunction of their nulls, and a minimum is bounded by
    any single member, so its expectation is at most 1 under whichever
    component of the union holds.

    **What the combinations cost.** Each statistic composes one averaging
    step and one minimum step, and only the averaging step is bounded. A mean
    of n processes loses at most log n nats relative to the strongest of
    them: log J for the every-level statistic's inner mean over the J
    smaller-model configurations, log m for the some-level statistic's outer
    mean over the m levels. A minimum is bounded by no such quantity — it can
    sit arbitrarily below every other input, and one weak member drives it
    alone. That unbounded conservatism is the price of the universal
    quantifier each claim carries: over levels for the every-level claim,
    over configurations for the some-level claim. Neither statistic has a
    bounded total cost.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    smaller_model_slug: str
    larger_model_slug: str
    comparison_type: Literal["worst", "best"]

    # Every configuration on each side, in REASONING_SCALES order. Both
    # claims quantify over all of them; neither names an endpoint.
    smaller_config_keys: tuple[str, ...]
    larger_config_keys: tuple[str, ...]

    # Valid trials pooled over every configuration on each side.
    smaller_valid: int
    larger_valid: int

    # E-value components. raw_* is the combined statistic for this claim.
    raw_log_e_value: float
    raw_e_value: float

    # Multiplicity correction
    bonferroni_k: int  # total number of model size tests in the family
    adjusted_log_e_value: float  # raw_log - log(K)
    adjusted_e_value: float  # raw / K
    significant: bool  # adjusted_log_e_value >= log(1/alpha)

    # One fold per larger-model level, in REASONING_SCALES order — the
    # selection-free audit behind the combined statistic. Both claims read
    # the same folds.
    level_folds: tuple[LevelFold, ...]

    @model_validator(mode="after")
    def _validate_audit_trails(self) -> "ModelSizeComparison":
        """A test with valid data on both sides must carry its audit trails."""
        if self.smaller_valid > 0 and self.larger_valid > 0 and not self.level_folds:
            raise ValueError(
                "ModelSizeComparison with valid trials on both sides "
                "must carry non-empty level_folds"
            )
        return self


def _balanced_accuracy_from_outcomes(
    outcomes: list[TrialOutcome],
    config_key: str,
) -> float:
    """Compute balanced accuracy for a config directly from trial outcomes.

    Mirrors ``composite_metrics()`` in metrics.py: specificity is correct-draft
    hits over correct-draft valid trials, sensitivity pools the two
    incorrect-draft conditions as hits over their valid trials, and balanced
    accuracy is the unweighted mean of the two. A class with no valid trials
    contributes 0.0, as it does there. This implementation is intentionally
    separate because it operates on raw ``TrialOutcome`` lists rather than the
    ``dict[CellKey, CellAccumulator]`` that function takes.
    """
    hits: dict[str, int] = {"specificity": 0, "sensitivity": 0}
    valid: dict[str, int] = {"specificity": 0, "sensitivity": 0}
    for o in outcomes:
        if o.config_key != config_key or o.condition not in _CONDITIONS:
            continue
        if o.matches_ground_truth is None:
            continue
        accuracy_class = _ACCURACY_CLASSES[o.condition]
        valid[accuracy_class] += 1
        if o.matches_ground_truth:
            hits[accuracy_class] += 1

    rates = [
        hits[accuracy_class] / valid[accuracy_class]
        if valid[accuracy_class] > 0 else 0.0
        for accuracy_class in ("specificity", "sensitivity")
    ]
    return sum(rates) / len(rates)


def _ordered_by_reasoning_scale(
    config_keys: list[str],
    provider: str,
    model_slug: str,
) -> list[str]:
    """Sort a model's config_keys by REASONING_SCALES position.

    Levels missing from the scale sort last, alphabetically — a
    deterministic order for the audit regardless of input order.
    """
    scale = REASONING_SCALES.get((provider, model_slug), [])

    def sort_key(ck: str) -> tuple[int, str]:
        _, _, level = parse_config_key(ck)
        pos = scale.index(level) if level in scale else len(scale)
        return (pos, level)

    return sorted(config_keys, key=sort_key)


def compute_all_model_size_tests(
    outcomes: list[TrialOutcome],
    *,
    bet_cap: float = BET_CAP,
    alpha: float = 0.05,
) -> list[ModelSizeComparison]:
    """Compute both model size claims for every within-provider pair.

    Two tests per provider — the every-level and some-level claims — with K
    (the Bonferroni correction factor) computed dynamically from the pairs
    that have data for both models.

    Neither model's configurations are selected. Each larger-model level is
    paired against each smaller-model configuration through the shared
    paired-difference fold with a non-negative bet, and the resulting
    processes are combined twice: minimum over levels of the mean over
    configurations for the every-level claim, mean over levels of the minimum
    over configurations for the some-level claim. Both claims read the same
    per-pair processes, so the per-level audit is shared between them. A level
    with zero valid trials contributes neutral evidence, which blocks the
    every-level claim and dilutes the some-level mean.
    """
    _validate_bet_cap(bet_cap)

    present_keys: set[str] = {o.config_key for o in outcomes}

    # Enumerate provider pairs and collect config_keys per model
    provider_tests: list[tuple[str, str, str, list[str], list[str]]] = []
    for provider, (smaller_slug, larger_slug) in MODEL_SIZE_PAIRS.items():
        smaller_configs = _ordered_by_reasoning_scale(
            [ck for ck in present_keys if ck.startswith(f"{provider}--{smaller_slug}--")],
            provider, smaller_slug,
        )
        larger_configs = _ordered_by_reasoning_scale(
            [ck for ck in present_keys if ck.startswith(f"{provider}--{larger_slug}--")],
            provider, larger_slug,
        )
        if not smaller_configs or not larger_configs:
            continue
        provider_tests.append(
            (provider, smaller_slug, larger_slug, smaller_configs, larger_configs)
        )

    # 2 tests per provider that has data
    bonferroni_k = 2 * len(provider_tests)

    results: list[ModelSizeComparison] = []
    for provider, smaller_slug, larger_slug, smaller_configs, larger_configs in provider_tests:
        smaller_valid_by_key = {
            ck: _tally_config_outcomes(outcomes, ck)[1] for ck in smaller_configs
        }

        level_folds: list[LevelFold] = []
        for larger_ck in larger_configs:
            _level_hits, level_valid = _tally_config_outcomes(outcomes, larger_ck)
            configuration_folds: list[SmallerConfigurationFold] = []
            for smaller_ck in smaller_configs:
                rounds = _build_cross_sample_rounds(
                    outcomes, outcomes, larger_ck, smaller_ck,
                    bet_cap=bet_cap,
                )
                slices, log_e, _differences = _accumulate_difference_rounds(
                    rounds, bet_sign="non_negative", bet_cap=bet_cap,
                )
                configuration_folds.append(SmallerConfigurationFold(
                    smaller_config_key=smaller_ck,
                    smaller_valid=smaller_valid_by_key[smaller_ck],
                    log_e_value=log_e,
                    slices=slices,
                ))
            pair_logs = [fold.log_e_value for fold in configuration_folds]
            level_folds.append(LevelFold(
                config_key=larger_ck,
                valid=level_valid,
                balanced_accuracy=_balanced_accuracy_from_outcomes(
                    outcomes, larger_ck,
                ),
                configuration_folds=tuple(configuration_folds),
                log_mean_e_value=_log_mean_exp(pair_logs),
                log_min_e_value=min(pair_logs),
            ))

        combined_logs = {
            # Every level beats the smaller model's worst configuration.
            "worst": min(lf.log_mean_e_value for lf in level_folds),
            # Some level beats the smaller model's best configuration.
            "best": _log_mean_exp([lf.log_min_e_value for lf in level_folds]),
        }
        shared_folds = tuple(level_folds)

        for comparison_type in ("worst", "best"):
            log_e = combined_logs[comparison_type]
            raw_e = math.exp(log_e)
            adjusted_log_e = (
                log_e - math.log(bonferroni_k) if bonferroni_k > 0 else log_e
            )
            adjusted_e = raw_e / bonferroni_k if bonferroni_k > 0 else raw_e

            results.append(ModelSizeComparison(
                provider=provider,
                smaller_model_slug=smaller_slug,
                larger_model_slug=larger_slug,
                comparison_type=comparison_type,
                smaller_config_keys=tuple(smaller_configs),
                larger_config_keys=tuple(larger_configs),
                smaller_valid=sum(smaller_valid_by_key.values()),
                larger_valid=sum(lf.valid for lf in shared_folds),
                raw_log_e_value=log_e,
                raw_e_value=raw_e,
                bonferroni_k=bonferroni_k,
                adjusted_log_e_value=adjusted_log_e,
                adjusted_e_value=adjusted_e,
                significant=_is_significant(adjusted_log_e, alpha),
                level_folds=shared_folds,
            ))

    return results


# ---------------------------------------------------------------------------
# Cross-dataset ablation comparison
# ---------------------------------------------------------------------------


class CrossDatasetComparison(BaseModel):
    """E-value test result for an accuracy difference between primary and ablation datasets.

    Tests whether the ablation changed accuracy for a specific
    (config, condition) pair or for a configuration across all three
    conditions. The two datasets ran the same stimulus grid, so the test bets
    on the paired differences ``ablation − primary`` cell by cell.

    Two sign-constrained processes run over the same rounds: an
    ablation-higher process with a non-negative bet, whose null is
    H₀: p_ablation ≤ p_primary, and an ablation-lower process with a
    non-positive bet, whose null is H₀: p_ablation ≥ p_primary. The reported
    statistic is their maximum and the correction is ``2 * bonferroni_k``,
    which buys family-wise control over all 2K one-sided nulls and therefore
    licenses reporting a direction rather than only a difference.

    ``condition`` is None for config-level tests, whose rounds pool all three
    conditions and are class reweighted, and a specific Condition value for
    condition-level tests.

    **What the reported accuracy is.** A condition-level row reports that
    condition's accuracy, which is its hits over its valid trials. A
    config-level row reports balanced accuracy, the quantity its class-
    reweighted fold bets on, so ``primary_accuracy`` there is deliberately not
    ``primary_hits / primary_valid``: the hit and valid counts remain the
    sample sizes behind the test, and the accuracy is the balanced quantity.
    """

    model_config = ConfigDict(frozen=True)

    config_key: str
    condition: Condition | None  # None = config-level, all three conditions

    # Primary dataset sample. The accuracy is per-condition for a
    # condition-level row and balanced for a config-level one.
    primary_hits: int
    primary_valid: int
    primary_accuracy: float

    # Ablation dataset sample, on the same terms.
    ablation_hits: int
    ablation_valid: int
    ablation_accuracy: float

    # Observed difference (ablation - primary)
    delta: float

    # E-value components
    raw_log_e_value: float
    raw_e_value: float

    # Multiplicity correction (includes the factor of 2 for the two directions)
    bonferroni_k: int  # total tests in this family (condition-level or config-level)
    adjusted_log_e_value: float  # raw_log - log(2) - log(K)
    adjusted_e_value: float  # raw / (2K)
    significant: bool  # adjusted_log_e_value >= log(1/alpha)

    # Which one-sided process crossed: ablation_higher when the non-negative
    # process accumulated positive log evidence, ablation_lower when the
    # non-positive one did, indistinguishable when neither did — regardless
    # of the observed sign of delta.
    direction: Literal["ablation_higher", "ablation_lower", "indistinguishable"]

    # Per-round audit trails for both directional processes. Both are
    # populated regardless of which one wins; consumers can identify the
    # winning process from ``direction``.
    e_slices_ablation_higher: tuple[ComparisonSlice, ...]
    e_slices_ablation_lower: tuple[ComparisonSlice, ...]

    @model_validator(mode="after")
    def _validate_audit_trails(self) -> "CrossDatasetComparison":
        """A comparison with valid data on both sides must carry both trails."""
        if self.primary_valid > 0 and self.ablation_valid > 0:
            if not self.e_slices_ablation_higher:
                raise ValueError(
                    "CrossDatasetComparison with valid trials on both sides "
                    "must carry a non-empty e_slices_ablation_higher audit trail"
                )
            if not self.e_slices_ablation_lower:
                raise ValueError(
                    "CrossDatasetComparison with valid trials on both sides "
                    "must carry a non-empty e_slices_ablation_lower audit trail"
                )
        return self


def compute_cross_dataset_comparison(
    primary_outcomes: list[TrialOutcome],
    ablation_outcomes: list[TrialOutcome],
    config_key: str,
    condition: Condition | None,
    bonferroni_k: int,
    *,
    bet_cap: float = BET_CAP,
    alpha: float = 0.05,
) -> CrossDatasetComparison:
    """Compute the cross-dataset e-value for one (config, condition) pair.

    Both experiments ran the same configuration over the same stimulus grid,
    so each ``(base_example, condition, batch_index, trial)`` cell yields one
    paired difference ``ablation − primary``. Two sign-constrained processes
    run over those rounds — one for each direction — and the larger is
    reported.

    Under either one-sided null the corresponding process has expectation at
    most 1, so ``max(E_higher, E_lower) / 2`` is a valid two-sided
    e-variable by the union bound. The factor of 2 is applied alongside
    Bonferroni so the family-wise control extends over all 2K one-sided
    nulls, which is what makes ``direction`` reportable.

    ``direction`` names the process that accumulated positive evidence. When
    neither did — including every case where the sequence is too short to
    have placed a bet — it is ``indistinguishable``, regardless of the
    observed sign of delta.
    """
    _validate_bet_cap(bet_cap)

    if condition is None:
        primary_hits, primary_valid = _tally_config_outcomes(
            primary_outcomes, config_key,
        )
        ablation_hits, ablation_valid = _tally_config_outcomes(
            ablation_outcomes, config_key,
        )
        # The config-level fold bets on the balanced difference, so the
        # reported accuracies are balanced rather than pooled over trials.
        p_primary = _balanced_accuracy_from_outcomes(primary_outcomes, config_key)
        p_ablation = _balanced_accuracy_from_outcomes(ablation_outcomes, config_key)
    else:
        primary_hits, primary_valid = _tally_config_condition_outcomes(
            primary_outcomes, config_key, condition,
        )
        ablation_hits, ablation_valid = _tally_config_condition_outcomes(
            ablation_outcomes, config_key, condition,
        )
        p_primary = primary_hits / primary_valid if primary_valid > 0 else 0.0
        p_ablation = ablation_hits / ablation_valid if ablation_valid > 0 else 0.0

    rounds = _build_cross_sample_rounds(
        ablation_outcomes, primary_outcomes, config_key, config_key,
        bet_cap=bet_cap, condition=condition,
    )
    higher_slices, higher_log, _higher_differences = _accumulate_difference_rounds(
        rounds, bet_sign="non_negative", bet_cap=bet_cap,
    )
    lower_slices, lower_log, _lower_differences = _accumulate_difference_rounds(
        rounds, bet_sign="non_positive", bet_cap=bet_cap,
    )

    direction: Literal["ablation_higher", "ablation_lower", "indistinguishable"]
    if higher_log >= lower_log and higher_log > 0:
        log_e = higher_log
        direction = "ablation_higher"
    elif lower_log > higher_log and lower_log > 0:
        log_e = lower_log
        direction = "ablation_lower"
    else:
        # Neither process accumulated positive evidence — report no signal,
        # regardless of which side the observed delta tilts.
        log_e = max(higher_log, lower_log)
        direction = "indistinguishable"

    raw_e = math.exp(log_e)
    # Two directions (factor of 2) + Bonferroni (factor of K)
    correction = 2 * bonferroni_k if bonferroni_k > 0 else 2
    adjusted_log_e = log_e - math.log(correction)

    return CrossDatasetComparison(
        config_key=config_key,
        condition=condition,
        primary_hits=primary_hits,
        primary_valid=primary_valid,
        primary_accuracy=p_primary,
        ablation_hits=ablation_hits,
        ablation_valid=ablation_valid,
        ablation_accuracy=p_ablation,
        delta=p_ablation - p_primary,
        raw_log_e_value=log_e,
        raw_e_value=raw_e,
        bonferroni_k=bonferroni_k,
        adjusted_log_e_value=adjusted_log_e,
        adjusted_e_value=raw_e / correction,
        significant=_is_significant(adjusted_log_e, alpha),
        direction=direction,
        e_slices_ablation_higher=higher_slices,
        e_slices_ablation_lower=lower_slices,
    )


def compute_all_cross_dataset_comparisons(
    primary_outcomes: list[TrialOutcome],
    ablation_outcomes: list[TrialOutcome],
    *,
    bet_cap: float = BET_CAP,
    alpha: float = 0.05,
) -> tuple[list[CrossDatasetComparison], list[CrossDatasetComparison]]:
    """Compute cross-dataset significance tests at condition and config levels.

    Returns two Bonferroni-corrected families:
    - **Condition-level** (K = number of (config, condition) pairs present in both
      datasets): tests per-condition accuracy differences (expected ~200 trials/side).
    - **Config-level** (K = number of configs present in both datasets): tests
      balanced accuracy differences (expected ~600 trials/side).

    Each family has its own independent Bonferroni correction, computed
    dynamically from pairs that actually have data in both datasets.
    """
    _validate_bet_cap(bet_cap)

    primary_keys: set[str] = {o.config_key for o in primary_outcomes}
    ablation_keys: set[str] = {o.config_key for o in ablation_outcomes}
    common_keys = sorted(primary_keys & ablation_keys)

    # -- Condition-level family ------------------------------------------------
    condition_targets: list[tuple[str, Condition]] = [
        (config_key, condition)
        for config_key in common_keys
        for condition in _CONDITIONS
        if _tally_config_condition_outcomes(
            primary_outcomes, config_key, condition,
        )[1] > 0
        and _tally_config_condition_outcomes(
            ablation_outcomes, config_key, condition,
        )[1] > 0
    ]
    condition_k = len(condition_targets)

    condition_results = [
        compute_cross_dataset_comparison(
            primary_outcomes, ablation_outcomes,
            config_key, condition, condition_k,
            bet_cap=bet_cap, alpha=alpha,
        )
        for config_key, condition in condition_targets
    ]

    # -- Config-level family ---------------------------------------------------
    config_targets = [
        config_key for config_key in common_keys
        if _tally_config_outcomes(primary_outcomes, config_key)[1] > 0
        and _tally_config_outcomes(ablation_outcomes, config_key)[1] > 0
    ]
    config_k = len(config_targets)

    config_results = [
        compute_cross_dataset_comparison(
            primary_outcomes, ablation_outcomes,
            config_key, None, config_k,
            bet_cap=bet_cap, alpha=alpha,
        )
        for config_key in config_targets
    ]

    return condition_results, config_results
