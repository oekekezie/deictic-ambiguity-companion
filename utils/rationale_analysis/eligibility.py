"""Trial eligibility filtering for rationale analysis assembly.

Partitions experiment TrialOutcome records into three mutually exclusive
categories via two filters applied in order:

1. Parse failures (envelope, structural, or domain) → excluded
2. Correct trials (matches_ground_truth is True) → excluded
3. All remaining (incorrect trials with successful parse) → eligible

Only incorrect trials are eligible because the rationale analysis spec
(v3.0) restricts the pipeline to trials where the meta-evaluator got
the answer wrong, making its stated reasoning diagnostic of failure mode.
"""

from utils.experiment_analysis.models import TrialOutcome
from utils.rationale_analysis.models import (
    RationaleAnalysisEligibilityResult,
    RationaleAnalysisExcludedTrial,
)


def filter_eligible_trials(
    outcomes: list[TrialOutcome],
) -> RationaleAnalysisEligibilityResult:
    """Partition trials into eligible and excluded sets.

    Everything needed for filtering lives on ``TrialOutcome`` directly:
    ``parse_error`` gates parse failures, ``matches_ground_truth`` gates
    correct trials, and ``rationale`` is carried for downstream assembly.

    A trial is eligible only when it has no parse error AND its
    prediction is incorrect (``matches_ground_truth is False``).
    """
    eligible: list[TrialOutcome] = []
    excluded_parse_failure: list[RationaleAnalysisExcludedTrial] = []
    excluded_correct_trial: list[RationaleAnalysisExcludedTrial] = []

    for outcome in outcomes:
        # Filter 1: parse failure (envelope, structural, or domain)
        if outcome.parse_error is not None:
            excluded_parse_failure.append(
                RationaleAnalysisExcludedTrial(
                    example_id=outcome.example_id,
                    config_key=outcome.config_key,
                    batch_index=outcome.batch_index,
                    trial=outcome.trial,
                )
            )
            continue

        # Filter 2: correct trial (matches_ground_truth is guaranteed
        # bool after gate 1 since parse_error is None ⇒ predicted_score
        # is non-None ⇒ matches_ground_truth is bool)
        if outcome.matches_ground_truth is True:
            excluded_correct_trial.append(
                RationaleAnalysisExcludedTrial(
                    example_id=outcome.example_id,
                    config_key=outcome.config_key,
                    batch_index=outcome.batch_index,
                    trial=outcome.trial,
                )
            )
            continue

        # Gate 3: incorrect trial with successful parse → eligible
        eligible.append(outcome)

    return RationaleAnalysisEligibilityResult(
        eligible=tuple(eligible),
        excluded_parse_failure=tuple(excluded_parse_failure),
        excluded_correct_trial=tuple(excluded_correct_trial),
    )
