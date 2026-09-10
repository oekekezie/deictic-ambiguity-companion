"""Token-usage and cost aggregation for batch inference results.

``summarize_job_costs`` rolls up one job into a JobCostSummary;
``summarize_token_usage_cost`` rolls up a set of jobs into a single
TokenUsageCostSummary. Per-result costs are calculated individually to
respect per-request tier thresholds (e.g., Gemini Pro ≤200k vs >200k
context pricing); when a job's provider/model has no pricing policy,
``summarize_token_usage_cost`` degrades gracefully (cost fields zeroed,
``cost_available`` False) while ``summarize_job_costs`` raises.
"""

from collections.abc import Iterable

from utils.batch_inference.processed import ProcessedJob
from utils.token_costs.calculation import calculate_actual_cost, calculate_worst_case_cost
from utils.token_costs.models import JobCostSummary, TokenCost, TokenUsageCostSummary


def _sum_optional(values: Iterable[int | None]) -> int | None:
    """Sum values that may be None, returning None if all are None."""
    total = 0
    any_present = False
    for v in values:
        if v is not None:
            total += v
            any_present = True
    return total if any_present else None


def _sum_costs(costs: list[TokenCost]) -> TokenCost:
    """Sum a list of TokenCost instances into a single aggregate."""
    if not costs:
        return TokenCost(
            input_cost=0.0, output_cost=0.0,
            cache_read_cost=0.0, cache_write_cost=0.0, total_cost=0.0,
        )
    return TokenCost(
        input_cost=sum(c.input_cost for c in costs),
        output_cost=sum(c.output_cost for c in costs),
        cache_read_cost=sum(c.cache_read_cost for c in costs),
        cache_write_cost=sum(c.cache_write_cost for c in costs),
        total_cost=sum(c.total_cost for c in costs),
    )


def summarize_job_costs(job: ProcessedJob) -> JobCostSummary:
    """Compute aggregated cost summary for a completed batch inference job.

    Per-result costs are calculated individually to respect per-request
    tier thresholds, then summed to the job level. The cache_savings
    field captures the difference between worst-case (all uncached)
    and actual (with cache discounts applied) total costs.
    """
    worst_costs: list[TokenCost] = []
    actual_costs: list[TokenCost] = []

    for result in job.results:
        worst_costs.append(
            calculate_worst_case_cost(job.provider, job.model, result.usage)
        )
        actual_costs.append(
            calculate_actual_cost(job.provider, job.model, result.usage)
        )

    total_worst = _sum_costs(worst_costs)
    total_actual = _sum_costs(actual_costs)

    return JobCostSummary(
        job_id=job.job_id,
        provider=job.provider,
        model=job.model,
        num_results=len(job.results),
        total_input_tokens=sum(r.usage.input_tokens for r in job.results),
        total_output_tokens=sum(r.usage.output_tokens for r in job.results),
        total_reasoning_tokens=_sum_optional(
            r.usage.reasoning_tokens for r in job.results
        ),
        total_cache_read_tokens=_sum_optional(
            r.usage.cache_read_tokens for r in job.results
        ),
        total_cache_write_tokens=_sum_optional(
            r.usage.cache_write_tokens for r in job.results
        ),
        worst_case_cost=total_worst,
        actual_cost=total_actual,
        cache_savings=total_worst.total_cost - total_actual.total_cost,
    )


def summarize_token_usage_cost(jobs: list[ProcessedJob]) -> TokenUsageCostSummary:
    """Aggregate token usage and cost across a set of batch jobs.

    Sums token counts over every result in every job, then computes
    per-result worst-case and actual costs (per-result calculation
    respects per-request tier thresholds). If any job's provider/model
    lacks a pricing policy, the cost fields are zeroed and
    ``cost_available`` is False — token counts remain valid regardless.
    Mirrors the graceful-degradation contract of ``usage_cost_summary``
    in experiment_analysis, but takes ProcessedJob input.
    """
    pairs = [(job, result) for job in jobs for result in job.results]
    if not pairs:
        return TokenUsageCostSummary(
            num_results=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_reasoning_tokens=None,
            worst_case_cost=_sum_costs([]),
            actual_cost=_sum_costs([]),
            cache_savings=0.0,
            cost_available=False,
        )

    total_input = sum(r.usage.input_tokens for _, r in pairs)
    total_output = sum(r.usage.output_tokens for _, r in pairs)
    total_reasoning = _sum_optional(r.usage.reasoning_tokens for _, r in pairs)

    # Per-result cost calculation; bail to zeroed costs on pricing lookup failure
    worst_costs: list[TokenCost] = []
    actual_costs: list[TokenCost] = []
    cost_available = True
    for job, result in pairs:
        try:
            worst_costs.append(
                calculate_worst_case_cost(job.provider, job.model, result.usage)
            )
            actual_costs.append(
                calculate_actual_cost(job.provider, job.model, result.usage)
            )
        except ValueError:
            cost_available = False
            break

    if cost_available:
        total_worst = _sum_costs(worst_costs)
        total_actual = _sum_costs(actual_costs)
        cache_savings = total_worst.total_cost - total_actual.total_cost
    else:
        total_worst = _sum_costs([])
        total_actual = _sum_costs([])
        cache_savings = 0.0

    return TokenUsageCostSummary(
        num_results=len(pairs),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_reasoning_tokens=total_reasoning,
        worst_case_cost=total_worst,
        actual_cost=total_actual,
        cache_savings=cache_savings,
        cost_available=cost_available,
    )
