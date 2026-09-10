"""Per-flag aggregation for rationale analysis results.

Groups parsed responses by trial identity and computes per-flag
aggregated values, vote tallies, and audit trails (spec Section 3.12,
Layer 3). The five misattribution flags and the articulation flag use
any-affirmative aggregation; the governing flag
(``operational_interpretation_governed_judgment``) uses majority vote.
Every flag requires at least ceil(K/2) parsed reps to resolve.
"""

import math
from collections.abc import Callable
from typing import Any

from utils.batch_inference.processed import ProcessedResult, parse_custom_id
from utils.rationale_analysis.custom_id import parse_rationale_analysis_custom_id
from utils.rationale_analysis.models import (
    RationaleAnalysisParsedResponse,
    RationaleAnalysisProvenance,
    RationaleAnalysisResults,
    RationaleAnalysisTrialRecord,
)
from utils.rationale_analysis.response_parser import parse_rationale_analysis_response


def _any_affirmative(
    classifications: list[bool],
    k: int,
) -> tuple[bool | None, int, int]:
    """Any-affirmative aggregation across K repetitions.

    If *any* parsed rep classifies True, the result is True; it is only
    False when *all* parsed reps classify False. Returns None when fewer
    than ceil(K/2) reps parsed successfully — the same threshold as
    majority vote — so all flags resolve or fail to resolve together.
    """
    votes_for = sum(1 for c in classifications if c is True)
    votes_against = sum(1 for c in classifications if c is False)

    min_required = math.ceil(k / 2)
    if len(classifications) < min_required:
        return None, votes_for, votes_against

    # Any True among parsed reps → True; all False → False
    flag_value = votes_for > 0
    return flag_value, votes_for, votes_against


def _majority_vote(
    classifications: list[bool],
    k: int,
) -> tuple[bool | None, int, int]:
    """Majority-vote aggregation for the governing flag across K repetitions.

    Returns ``(flag_value, votes_for, votes_against)`` where flag_value is
    None when fewer than ceil(K/2) repetitions parsed successfully
    (insufficient data for a majority).
    """
    votes_for = sum(1 for c in classifications if c is True)
    votes_against = sum(1 for c in classifications if c is False)

    min_required = math.ceil(k / 2)
    if len(classifications) < min_required:
        return None, votes_for, votes_against

    # With sufficient parses, majority wins; K is odd per spec so ties
    # cannot occur when all K parse. If some fail and votes are tied,
    # votes_for > votes_against is false → flag_value is False.
    flag_value = votes_for > votes_against
    return flag_value, votes_for, votes_against


# Per-flag K-repetition aggregation policy. The five misattribution flags
# and articulation use any-affirmative (a flag fires if any rep sees it);
# governing uses majority vote.
_AggregationPolicy = Callable[[list[bool], int], tuple[bool | None, int, int]]
_FLAG_AGGREGATION_POLICIES: dict[str, _AggregationPolicy] = {
    "misattributed_scenario_current_value": _any_affirmative,
    "misattributed_scenario_historical_value": _any_affirmative,
    "misattributed_scenario_proposed_value": _any_affirmative,
    "misattributed_assistant_fallback_value": _any_affirmative,
    "misattributed_grader_claim": _any_affirmative,
    "articulated_operational_interpretation": _any_affirmative,
    "operational_interpretation_governed_judgment": _majority_vote,
}


def flag_aggregation_method_names() -> tuple[tuple[str, str], ...]:
    """Return each flag's aggregation policy as ``(flag_key, policy_name)`` pairs.

    Exposes the per-flag policy table as a stable, serializable descriptor
    (e.g. for snapshot provenance) without leaking the policy callables.
    Policy names drop the leading underscore — ``any_affirmative`` /
    ``majority_vote`` — and preserve the canonical flag order.
    """
    return tuple(
        (flag, policy.__name__.lstrip("_"))
        for flag, policy in _FLAG_AGGREGATION_POLICIES.items()
    )


def aggregate_repetitions(
    trial_identity: tuple[str, str, int, int],
    parsed_responses: tuple[RationaleAnalysisParsedResponse | None, ...],
) -> RationaleAnalysisTrialRecord:
    """Aggregate K parsed responses into a single trial record.

    ``trial_identity`` is ``(example_id, config_key, batch_index, trial)``.
    ``parsed_responses`` contains K entries — one per repetition — where
    None indicates a parse failure for that repetition.

    Each of the seven flags is aggregated by its policy in
    ``_FLAG_AGGREGATION_POLICIES``. Every flag requires ceil(K/2) parsed
    reps to resolve, else its aggregated value is None.
    """
    example_id, config_key, batch_index, trial = trial_identity
    k = len(parsed_responses)

    record_fields: dict[str, Any] = {
        "config_key": config_key,
        "example_id": example_id,
        "batch_index": batch_index,
        "trial": trial,
        "repetitions": parsed_responses,
    }

    for flag_key, policy in _FLAG_AGGREGATION_POLICIES.items():
        classifications = [
            getattr(resp, flag_key).classification
            for resp in parsed_responses
            if resp is not None
        ]
        value, votes_for, votes_against = policy(classifications, k)
        record_fields[flag_key] = value
        record_fields[f"{flag_key}_votes_for"] = votes_for
        record_fields[f"{flag_key}_votes_against"] = votes_against

    return RationaleAnalysisTrialRecord(**record_fields)


def build_rationale_analysis_results(
    processed_results: list[ProcessedResult],
    k: int,
    provenance: RationaleAnalysisProvenance,
) -> RationaleAnalysisResults:
    """Full Layer 2+3 pipeline: parse responses, group by trial, aggregate.

    The provided ``provenance`` is attached to the returned results so the
    persisted snapshot is self-describing.

    1. Strip ``_trial_{KKK}`` suffix via ``parse_custom_id()`` to recover
       the assembly-time custom_id and the repetition number.
    2. Parse the assembly-time custom_id via
       ``parse_rationale_analysis_custom_id()`` to recover
       ``(example_id, config_key, batch_index, trial)``.
    3. Parse ``raw_response_text`` via
       ``parse_rationale_analysis_response()``.
    4. Attach ``reasoning_text`` from the ``ProcessedResult`` to the
       parsed response (extended thinking, when the model provides it).
    5. Group by trial identity (the four dimensions).
    6. Aggregate each group via ``aggregate_repetitions()``.

    Raises ValueError if any custom_id does not conform to the ``ra__``
    format (non-rationale-analysis results must not be passed in).
    """
    # Group parsed responses by trial identity
    groups: dict[
        tuple[str, str, int, int],
        dict[int, RationaleAnalysisParsedResponse | None],
    ] = {}

    for result in processed_results:
        # Layer 1 already ran — result has raw_response_text
        assembly_custom_id, repetition = parse_custom_id(result.custom_id)
        example_id, config_key, batch_index, trial = (
            parse_rationale_analysis_custom_id(assembly_custom_id)
        )

        trial_key = (example_id, config_key, batch_index, trial)
        if trial_key not in groups:
            groups[trial_key] = {}

        # Parse the rationale analyst's seven-flag response
        parsed, _parse_error = parse_rationale_analysis_response(
            result.raw_response_text
        )
        # Attach extended thinking from the ProcessedResult (if available)
        if parsed is not None and result.reasoning_text is not None:
            parsed = parsed.model_copy(
                update={"reasoning_text": result.reasoning_text}
            )
        groups[trial_key][repetition] = parsed

    # Aggregate each trial group
    records: list[RationaleAnalysisTrialRecord] = []
    for trial_key in sorted(groups.keys()):
        repetition_map = groups[trial_key]

        # Build ordered tuple of K responses (1-indexed repetitions)
        ordered: list[RationaleAnalysisParsedResponse | None] = [
            repetition_map.get(i) for i in range(1, k + 1)
        ]

        records.append(
            aggregate_repetitions(trial_key, tuple(ordered))
        )

    return RationaleAnalysisResults(
        trial_records=tuple(records), provenance=provenance
    )
