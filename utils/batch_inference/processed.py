"""Pydantic models and utilities for standardized benchmark results.

Defines the output schema for processed batch inference job results,
normalizing the wildly different response shapes from Anthropic, OpenAI,
Gemini, and Fireworks into a single structure suitable for aggregation
and downstream analysis.
"""

import re

from pydantic import BaseModel, ConfigDict

from utils.batch_inference.llm_configs import LLMConfig

# Matches the _trial_{NNN} suffix appended by expand_trials() in
# serializers/expansion.py — anchored to end, greedy base capture
# handles example IDs that themselves contain underscores.
_TRIAL_SUFFIX_PATTERN: re.Pattern[str] = re.compile(r"^(.+)_trial_(\d+)$")


def parse_custom_id(custom_id: str) -> tuple[str, int]:
    """Decompose a trial-expanded custom_id into (example_id, trial_number).

    Reverses the encoding applied by expand_trials(), which appends
    '_trial_{NNN}' to the original example custom_id.

    >>> parse_custom_id("example_01_trial_003")
    ('example_01', 3)
    """
    match = _TRIAL_SUFFIX_PATTERN.match(custom_id)
    if match is None:
        raise ValueError(
            f"custom_id {custom_id!r} does not match expected "
            f"pattern '{{example_id}}_trial_{{NNN}}'"
        )
    return match.group(1), int(match.group(2))


class TokenUsage(BaseModel):
    """Normalized token counts across all providers."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    # Separately reported by OpenAI (output_tokens_details.reasoning_tokens)
    # and Gemini (thoughtsTokenCount). For Anthropic and Fireworks, derived via
    # post-processing: enrich_anthropic_reasoning_tokens / enrich_fireworks_reasoning_tokens
    # count raw_response_text tokens and subtract from output_tokens. None when
    # not yet enriched or when the provider doesn't report reasoning tokens.
    reasoning_tokens: int | None = None
    # Tokens read from provider cache; reported by all four providers in
    # API responses. None when the provider didn't report cache data.
    cache_read_tokens: int | None = None
    # Tokens written to provider cache (Anthropic only via
    # cache_creation_input_tokens; None for other providers).
    cache_write_tokens: int | None = None


class ProcessedResult(BaseModel):
    """A single standardized response extracted from provider-specific output.

    Each instance corresponds to one JSONL line from a job's output.jsonl,
    with provider-specific nesting flattened into uniform fields.

    ``parse_error is not None`` is the canonical failure indicator: the
    response either failed envelope extraction (provider-level error) or
    structural validation (malformed JSON, empty freetext). Domain-specific
    parsing (e.g. experiment ``{score, rationale}``) is performed at
    consumption time, not here.
    """

    model_config = ConfigDict(frozen=True)

    # Identity — decomposed from the trial-expanded custom_id
    custom_id: str
    example_id: str
    trial: int

    # Raw model output for audit / debugging / domain-specific parsing
    raw_response_text: str
    reasoning_text: str | None

    # Provider metadata (normalized)
    model_version: str
    stop_reason: str | None
    usage: TokenUsage

    # Non-None when envelope extraction or structural validation failed
    parse_error: str | None = None


class BatchError(BaseModel):
    """An API-level request failure that never produced a usable result.

    Written to errors.jsonl by the retrieval SDK; persisted in ProcessedJob
    so downstream analysis knows exactly which trials are missing.
    """

    model_config = ConfigDict(frozen=True)

    custom_id: str
    example_id: str
    trial: int
    # "errored" | "expired" | "canceled" (Anthropic); varies by provider
    error_type: str
    # Human-readable error message when the provider supplies one
    detail: str | None = None


class ProcessedJob(BaseModel):
    """Processed output from a single completed batch inference job.

    Bundles job-level metadata (mirrored from JobRecord) with the
    standardized per-response results, creating a self-contained
    artifact for downstream aggregation and analysis.
    """

    model_config = ConfigDict(frozen=True)

    # Job identity (from JobRecord)
    job_id: str
    provider: str
    model: str
    llm_config: LLMConfig
    num_trials: int
    num_requests: int
    created_at: str
    completed_at: str | None
    submission_group: str | None

    # Lineage — non-None when this job retries a prior job's failed requests
    retry_of: str | None = None  # job_id of the original job this retries

    # Processing metadata
    processed_at: str
    total_results: int
    successful_parses: int
    failed_parses: int

    results: list[ProcessedResult]
    # API-level failures from errors.jsonl (requests the batch API couldn't process)
    batch_errors: list[BatchError] = []
