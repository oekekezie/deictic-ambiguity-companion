"""Pydantic models for token cost calculation.

TokenCost represents the dollar cost breakdown for a single result or
aggregation. JobCostSummary provides job-level rollups with worst-case
and actual costs for quantifying cache savings. TokenUsageCostSummary is
the multi-job counterpart, aggregating token usage and cost across a set
of jobs with a cost_available flag for graceful degradation.
InputTokenCountsManifest stores precomputed input token counts per
(provider, model) x example as a companion file to assembled_examples.jsonl.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class TokenCost(BaseModel):
    """Dollar cost breakdown for token usage.

    All values are in US dollars. For pre-flight estimates where cache
    behavior is unknown, cache_read_cost and cache_write_cost are 0.0
    since pre-flight assumes no caching.
    """

    model_config = ConfigDict(frozen=True)

    input_cost: float
    output_cost: float
    cache_read_cost: float
    cache_write_cost: float
    total_cost: float


class JobCostSummary(BaseModel):
    """Aggregated cost summary for a completed batch inference job.

    Pairs worst-case estimates (all input tokens at uncached rates) with
    actual costs (using real cache data from API responses). The delta
    quantifies cache savings across the batch.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str
    provider: str
    model: str
    num_results: int
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int | None
    total_cache_read_tokens: int | None
    total_cache_write_tokens: int | None
    worst_case_cost: TokenCost
    actual_cost: TokenCost
    cache_savings: float


class TokenUsageCostSummary(BaseModel):
    """Aggregate token usage and cost across a set of batch jobs.

    The multi-job counterpart to JobCostSummary. ``cost_available`` is False
    when any job's provider/model lacks a pricing policy, in which case the
    cost fields are zeroed; token counts remain valid.
    """

    model_config = ConfigDict(frozen=True)

    num_results: int
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int | None
    worst_case_cost: TokenCost
    actual_cost: TokenCost
    cache_savings: float
    cost_available: bool


class InputTokenCountsManifest(BaseModel):
    """Precomputed input token counts per (provider, model) x example.

    Companion file to assembled_examples.jsonl. Input token counts are
    provider-specific (different tokenizers produce different counts for
    the same text) but config-agnostic (reasoning effort level, temperature,
    max_tokens do not affect input tokenization).

    Outer key of input_token_counts is "{provider}/{model}" (e.g.
    "openai/gpt-5.2"). Inner key is the example's custom_id. Value
    is the integer count of input tokens for that example under that
    provider/model's tokenizer.
    """

    model_config = ConfigDict(frozen=True)

    dataset_file: str
    dataset_sha256: str
    generated_at: str
    input_token_counts: dict[str, dict[str, int]]

    @model_validator(mode="after")
    def validate_non_empty(self) -> "InputTokenCountsManifest":
        """At least one provider/model entry with at least one example."""
        if not self.input_token_counts:
            raise ValueError("input_token_counts must contain at least one provider/model entry")
        for key, counts in self.input_token_counts.items():
            if not counts:
                raise ValueError(f"input_token_counts[{key!r}] must contain at least one example")
        return self


def derive_input_token_counts_path(dataset_path: Path) -> Path:
    """Derive the companion input token counts filepath from a dataset path.

    Applies the naming convention established by the precompute script:
    replace '.jsonl' with '_input_token_counts.json', same directory.
    """
    name = dataset_path.name
    if not name.endswith(".jsonl"):
        raise ValueError(f"Dataset filename must end with '.jsonl': {name!r}")
    output_name = name.removesuffix(".jsonl") + "_input_token_counts.json"
    return dataset_path.parent / output_name
