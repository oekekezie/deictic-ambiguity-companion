"""Enrichment — bridging from ProcessedJob to TrialOutcome.

Transforms raw provider-specific ProcessedResult records into the flat,
analysis-ready TrialOutcome records by attaching experimental design
coordinates (condition, base example), ground truth, and configuration
identity metadata.
"""

from utils.batch_inference.processed import ProcessedJob, ProcessedResult
from utils.experiment_analysis.config_identity import parse_config_key
from utils.experiment_analysis.ground_truth import ground_truth_for_condition, parse_example_id
from utils.experiment_analysis.models import TrialOutcome
from utils.experiment_analysis.response_parser import parse_experiment_response


def _enrich_result(
    result: ProcessedResult,
    config_key: str,
    provider: str,
    model_slug: str,
    reasoning_effort_level: str,
    batch_index: int,
) -> TrialOutcome:
    """Enrich a single ProcessedResult into a TrialOutcome.

    Performs Layer 2 (domain-specific) parsing via
    ``parse_experiment_response()`` to extract the predicted score from
    the raw response text. The combined parse_error reflects both
    envelope-level failures (from ProcessedResult) and domain-level
    failures (from the experiment response parser).
    """
    base_example, condition = parse_example_id(result.example_id)
    ground_truth_score = ground_truth_for_condition(condition)

    # Layer 2: parse domain-specific score from raw response
    parsed, domain_parse_error = (
        parse_experiment_response(result.raw_response_text)
        if result.parse_error is None
        else (None, None)
    )
    predicted_score = parsed.score if parsed is not None else None
    rationale = parsed.rationale if parsed is not None else None

    matches_ground_truth: bool | None = None
    if predicted_score is not None:
        matches_ground_truth = predicted_score == ground_truth_score

    # Combined parse_error: envelope-level takes precedence, else domain-level
    parse_error = result.parse_error if result.parse_error is not None else domain_parse_error

    has_reasoning_trace = bool(
        result.reasoning_text is not None and len(result.reasoning_text) > 0
    )

    return TrialOutcome(
        custom_id=result.custom_id,
        example_id=result.example_id,
        trial=result.trial,
        base_example=base_example,
        condition=condition,
        predicted_score=predicted_score,
        ground_truth_score=ground_truth_score,
        matches_ground_truth=matches_ground_truth,
        rationale=rationale,
        config_key=config_key,
        provider=provider,
        model_slug=model_slug,
        reasoning_effort_level=reasoning_effort_level,
        batch_index=batch_index,
        usage=result.usage,
        has_reasoning_trace=has_reasoning_trace,
        parse_error=parse_error,
    )


def enrich_job(
    job: ProcessedJob,
    config_key: str,
    batch_index: int,
) -> list[TrialOutcome]:
    """Transform a ProcessedJob's results into enriched TrialOutcome records.

    Attaches experimental design coordinates, ground truth mapping, and
    configuration identity to each raw ProcessedResult. The config_key
    and batch_index are provided by the caller (derived from the manifest
    and job ordering at load time, not from the job itself).
    """
    provider, model_slug, reasoning_effort_level = parse_config_key(config_key)

    return [
        _enrich_result(
            result,
            config_key=config_key,
            provider=provider,
            model_slug=model_slug,
            reasoning_effort_level=reasoning_effort_level,
            batch_index=batch_index,
        )
        for result in job.results
    ]
