"""Token cost calculation for batch inference results.

Provides three cost computation modes:

  - Pre-flight: input-only estimate before job submission (no output, no cache)
  - Worst-case: post-hoc using actual token counts but all input at uncached rates
  - Actual: post-hoc with cache discounts from real API response data

The delta between worst-case and actual quantifies cache savings.
"""

from utils.batch_inference.processed import TokenUsage
from utils.token_costs.accounting import (
    INPUT_EXCLUDES_CACHE_PROVIDERS,
    OUTPUT_EXCLUDES_REASONING_PROVIDERS,
)
from utils.token_costs.models import TokenCost
from utils.token_costs.pricing import PROVIDERS, get_pricing_policy


def estimate_preflight_cost(
    provider: str,
    model: str,
    input_tokens: int,
    *,
    is_batch: bool = True,
) -> TokenCost:
    """Pre-flight input-only cost estimate before job submission.

    Output tokens are unknown pre-flight, so output_cost is 0.0.
    All input tokens are assumed uncached (worst-case).
    """
    policy = get_pricing_policy(provider, model)
    return policy(input_tokens, 0, None, None, None, is_batch)


def calculate_worst_case_cost(
    provider: str,
    model: str,
    usage: TokenUsage,
    *,
    is_batch: bool = True,
) -> TokenCost:
    """Post-hoc cost with all input at uncached rates (no cache savings).

    Uses actual input and output token counts from the API response,
    but bills the entire prompt context at the uncached rate. Serves
    as the baseline for quantifying cache savings when compared to
    actual cost.

    For Anthropic, whose input_tokens excludes the cache fields, the
    full context is reconstructed as input_tokens + cache_read_tokens +
    cache_write_tokens before billing at the uncached rate. For every
    other provider input_tokens already contains that total, so it is
    passed through directly.
    """
    policy = get_pricing_policy(provider, model)

    if provider in INPUT_EXCLUDES_CACHE_PROVIDERS:
        total_input = (
            usage.input_tokens
            + (usage.cache_read_tokens or 0)
            + (usage.cache_write_tokens or 0)
        )
        return policy(
            total_input, usage.output_tokens,
            usage.reasoning_tokens, None, None, is_batch,
        )

    return policy(
        usage.input_tokens, usage.output_tokens,
        usage.reasoning_tokens, None, None, is_batch,
    )


def calculate_actual_cost(
    provider: str,
    model: str,
    usage: TokenUsage,
    *,
    is_batch: bool = True,
) -> TokenCost:
    """Post-hoc cost using real cache data from the API response.

    Applies cache discounts to the cached portion of input tokens.
    The difference between worst-case and actual quantifies cache savings.
    """
    policy = get_pricing_policy(provider, model)
    return policy(
        usage.input_tokens, usage.output_tokens,
        usage.reasoning_tokens, usage.cache_read_tokens,
        usage.cache_write_tokens, is_batch,
    )


def compute_response_tokens(
    total_output: int,
    total_reasoning: int | None,
    provider: str,
) -> int | None:
    """Derive the response-only output tokens from reported usage fields.

    Providers whose ``output_tokens`` excludes reasoning (Gemini) report
    the response count directly, so it stands whether or not reasoning
    was reported at all. Everywhere else ``output_tokens`` includes
    reasoning, so the response is the difference, and it is unrecoverable
    when reasoning is unavailable: un-enriched Anthropic and Fireworks
    jobs derive reasoning by post-processing, and a trial run with
    reasoning disabled reports none.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}"
        )
    if provider in OUTPUT_EXCLUDES_REASONING_PROVIDERS:
        return total_output
    if total_reasoning is None:
        return None
    return total_output - total_reasoning
