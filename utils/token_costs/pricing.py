"""Pricing policy registry for batch inference token costs.

Maps (provider, model) pairs to pricing policy callables. Each policy
encodes the billing logic for its model: context-length tiers, cache
discounts, and batch discounts.

Policy callables have a uniform signature:
    (input_tokens, output_tokens, reasoning_tokens, cache_read_tokens,
     cache_write_tokens, is_batch) → TokenCost

Pre-flight calls pass cache fields as None (worst-case, no caching).
Post-hoc calls pass actual cache data from API responses.

Two accounting models govern how "uncached input" is derived:

  Subset (OpenAI, Gemini, Fireworks):
    input_tokens is the *total* including cached tokens.
    uncached = input_tokens - cache_read_tokens.

  Additive (Anthropic):
    input_tokens *excludes* cache fields.
    uncached = input_tokens (as-is).
    Total context = input_tokens + cache_read + cache_write.

Output-side reasoning accounting is an independent axis, and the two
must not be conflated: Anthropic is exceptional on the input axis,
Gemini on the output axis. Every policy derives what to bill as output
from ``OUTPUT_EXCLUDES_REASONING_PROVIDERS`` in ``accounting.py``, which
is the single home for both relationships. The registry below binds
rates to (provider, family); it states no field relationships of its own.
"""

from collections.abc import Callable

from utils.batch_inference.schema_utils import find_model_family
from utils.token_costs.accounting import OUTPUT_EXCLUDES_REASONING_PROVIDERS
from utils.token_costs.models import TokenCost

# Type alias for pricing policy functions
PricingPolicy = Callable[
    [int, int, int | None, int | None, int | None, bool],
    TokenCost,
]

# Batch API discount factor — 50% for all four providers
_BATCH_DISCOUNT: float = 0.5

# Pricing unit: dollars per million tokens
_MTOK: int = 1_000_000


# -- Shared cost arithmetic ----------------------------------------------------


def _billed_output(
    output_tokens: int,
    reasoning_tokens: int | None,
    reasoning_in_output: bool,
) -> int:
    """Resolve the output token count to bill at the output rate.

    When the provider already folds reasoning into output_tokens
    (OpenAI, Anthropic, Fireworks), output_tokens is billed as-is.
    When reasoning is reported separately (Gemini thinking tokens),
    it is added, since Google bills thinking at the output rate.
    """
    if reasoning_in_output:
        return output_tokens
    return output_tokens + (reasoning_tokens or 0)


def _apply_rates(
    uncached_input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    input_rate: float,
    cached_input_rate: float,
    output_rate: float,
    cache_write_rate: float,
    is_batch: bool,
) -> TokenCost:
    """Apply per-MTok rates to token counts and return a cost breakdown.

    All rates are in $/MTok. The batch discount (50%) is applied
    uniformly to all cost components when is_batch is True.
    """
    discount = _BATCH_DISCOUNT if is_batch else 1.0

    input_cost = uncached_input_tokens * input_rate / _MTOK * discount
    cache_read_cost = cache_read_tokens * cached_input_rate / _MTOK * discount
    cache_write_cost = cache_write_tokens * cache_write_rate / _MTOK * discount
    output_cost = output_tokens * output_rate / _MTOK * discount

    return TokenCost(
        input_cost=input_cost,
        output_cost=output_cost,
        cache_read_cost=cache_read_cost,
        cache_write_cost=cache_write_cost,
        total_cost=input_cost + output_cost + cache_read_cost + cache_write_cost,
    )


# -- Policy factories ---------------------------------------------------------
#
# Four factory functions covering two accounting models × two rate structures.
# Each returns a PricingPolicy closure that captures the rate constants.


def _flat_subset_policy(
    *,
    provider: str,
    input_rate: float,
    cached_input_rate: float,
    output_rate: float,
) -> PricingPolicy:
    """Flat-rate policy for subset-accounting providers (no cache writes).

    Used by OpenAI, Gemini Flash, and all Fireworks models. What counts
    as billable output follows from ``provider`` alone, so a rate row
    never restates it.
    """
    reasoning_in_output = provider not in OUTPUT_EXCLUDES_REASONING_PROVIDERS

    def policy(
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None,
        cache_read_tokens: int | None,
        cache_write_tokens: int | None,
        is_batch: bool,
    ) -> TokenCost:
        cr = cache_read_tokens or 0
        uncached = input_tokens - cr
        billed_output = _billed_output(output_tokens, reasoning_tokens, reasoning_in_output)
        return _apply_rates(
            uncached, billed_output, cr, 0,
            input_rate, cached_input_rate, output_rate, 0.0, is_batch,
        )

    return policy


def _flat_additive_policy(
    *,
    provider: str,
    input_rate: float,
    cached_input_rate: float,
    output_rate: float,
    cache_write_rate: float,
) -> PricingPolicy:
    """Flat-rate policy for additive-accounting providers (Anthropic).

    Anthropic's input_tokens field excludes cached tokens, so
    uncached_input = input_tokens as-is. The output axis is independent
    of that: what counts as billable output follows from ``provider``,
    the same way it does for the subset policies.
    """
    reasoning_in_output = provider not in OUTPUT_EXCLUDES_REASONING_PROVIDERS

    def policy(
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None,
        cache_read_tokens: int | None,
        cache_write_tokens: int | None,
        is_batch: bool,
    ) -> TokenCost:
        cr = cache_read_tokens or 0
        cw = cache_write_tokens or 0
        billed_output = _billed_output(
            output_tokens, reasoning_tokens, reasoning_in_output,
        )
        return _apply_rates(
            input_tokens, billed_output, cr, cw,
            input_rate, cached_input_rate, output_rate, cache_write_rate,
            is_batch,
        )

    return policy


def _tiered_subset_policy(
    *,
    provider: str,
    threshold: int,
    low_input_rate: float,
    low_cached_input_rate: float,
    low_output_rate: float,
    high_input_rate: float,
    high_cached_input_rate: float,
    high_output_rate: float,
) -> PricingPolicy:
    """Tiered-rate policy for subset-accounting providers (no cache writes).

    Tier is determined by input_tokens (the total including cached
    tokens in the subset accounting model). Used by Gemini Pro.
    """
    reasoning_in_output = provider not in OUTPUT_EXCLUDES_REASONING_PROVIDERS

    def policy(
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None,
        cache_read_tokens: int | None,
        cache_write_tokens: int | None,
        is_batch: bool,
    ) -> TokenCost:
        cr = cache_read_tokens or 0
        uncached = input_tokens - cr
        billed_output = _billed_output(output_tokens, reasoning_tokens, reasoning_in_output)

        if input_tokens <= threshold:
            return _apply_rates(
                uncached, billed_output, cr, 0,
                low_input_rate, low_cached_input_rate, low_output_rate,
                0.0, is_batch,
            )
        return _apply_rates(
            uncached, billed_output, cr, 0,
            high_input_rate, high_cached_input_rate, high_output_rate,
            0.0, is_batch,
        )

    return policy


def _tiered_additive_policy(
    *,
    provider: str,
    threshold: int,
    low_input_rate: float,
    low_cached_input_rate: float,
    low_output_rate: float,
    low_cache_write_rate: float,
    high_input_rate: float,
    high_cached_input_rate: float,
    high_output_rate: float,
    high_cache_write_rate: float,
) -> PricingPolicy:
    """Tiered-rate policy for additive-accounting providers (Anthropic).

    Tier is determined by total context length: input_tokens +
    cache_read + cache_write (since input_tokens excludes cache in
    the additive model). Used by Anthropic Opus 4.6 (1M context beta).
    Billable output follows from ``provider`` on the independent output
    axis.
    """
    reasoning_in_output = provider not in OUTPUT_EXCLUDES_REASONING_PROVIDERS

    def policy(
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None,
        cache_read_tokens: int | None,
        cache_write_tokens: int | None,
        is_batch: bool,
    ) -> TokenCost:
        cr = cache_read_tokens or 0
        cw = cache_write_tokens or 0
        total_context = input_tokens + cr + cw
        billed_output = _billed_output(
            output_tokens, reasoning_tokens, reasoning_in_output,
        )

        if total_context <= threshold:
            return _apply_rates(
                input_tokens, billed_output, cr, cw,
                low_input_rate, low_cached_input_rate, low_output_rate,
                low_cache_write_rate, is_batch,
            )
        return _apply_rates(
            input_tokens, billed_output, cr, cw,
            high_input_rate, high_cached_input_rate, high_output_rate,
            high_cache_write_rate, is_batch,
        )

    return policy


# -- Model family constants for registry lookup --------------------------------

_OPENAI_FAMILIES: frozenset[str] = frozenset({"gpt-5.2", "gpt-5-mini"})

_ANTHROPIC_FAMILIES: frozenset[str] = frozenset({
    "claude-opus-4-6", "claude-opus-4-5",
    "claude-sonnet-4-6", "claude-sonnet-4-5",
    "claude-haiku-4-5",
})

_GEMINI_FAMILIES: frozenset[str] = frozenset({
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    # DEPRECATED: discontinued by Google effective March 9, 2026.
    # Retained for cost analysis of existing experiment results.
    "gemini-3-pro-preview",
})

_FIREWORKS_FAMILIES: frozenset[str] = frozenset({
    "kimi-k2p5", "kimi-k2p6", "glm-4p7", "glm-5", "gpt-oss-120b",
})

_PROVIDER_FAMILIES: dict[str, frozenset[str]] = {
    "openai": _OPENAI_FAMILIES,
    "anthropic": _ANTHROPIC_FAMILIES,
    "gemini": _GEMINI_FAMILIES,
    "fireworks": _FIREWORKS_FAMILIES,
}

PROVIDERS: frozenset[str] = frozenset(_PROVIDER_FAMILIES)
"""Every provider the registry prices, for validating caller input."""

# Context-length tier threshold (tokens)
_TIER_THRESHOLD: int = 200_000


# -- Policy registry -----------------------------------------------------------
#
# All rates are standard non-batch $/MTok (February 2026).
# The batch discount is applied inside _apply_rates when is_batch=True.
#
# Cache write rates for Anthropic use the 5-minute ephemeral rate
# (1.25× base input), which is the default cache behavior for batch.

_REGISTRY: dict[tuple[str, str], PricingPolicy] = {
    # ── OpenAI (subset input accounting, no cache writes) ──
    ("openai", "gpt-5.2"): _flat_subset_policy(
        provider="openai",
        input_rate=1.75, cached_input_rate=0.175, output_rate=14.00,
    ),
    ("openai", "gpt-5-mini"): _flat_subset_policy(
        provider="openai",
        input_rate=0.25, cached_input_rate=0.025, output_rate=2.00,
    ),

    # ── Anthropic (additive input accounting, cache writes at 5m ephemeral rate) ──
    ("anthropic", "claude-opus-4-6"): _tiered_additive_policy(
        provider="anthropic",
        threshold=_TIER_THRESHOLD,
        low_input_rate=5.00, low_cached_input_rate=0.50,
        low_output_rate=25.00, low_cache_write_rate=6.25,
        high_input_rate=10.00, high_cached_input_rate=1.00,
        high_output_rate=37.50, high_cache_write_rate=12.50,
    ),
    ("anthropic", "claude-opus-4-5"): _flat_additive_policy(
        provider="anthropic",
        input_rate=5.00, cached_input_rate=0.50,
        output_rate=25.00, cache_write_rate=6.25,
    ),
    ("anthropic", "claude-sonnet-4-6"): _flat_additive_policy(
        provider="anthropic",
        input_rate=3.00, cached_input_rate=0.30,
        output_rate=15.00, cache_write_rate=3.75,
    ),
    ("anthropic", "claude-sonnet-4-5"): _flat_additive_policy(
        provider="anthropic",
        input_rate=3.00, cached_input_rate=0.30,
        output_rate=15.00, cache_write_rate=3.75,
    ),
    ("anthropic", "claude-haiku-4-5"): _flat_additive_policy(
        provider="anthropic",
        input_rate=1.00, cached_input_rate=0.10,
        output_rate=5.00, cache_write_rate=1.25,
    ),

    # ── Gemini (subset input accounting, no cache writes) ──
    ("gemini", "gemini-3-flash-preview"): _flat_subset_policy(
        provider="gemini",
        input_rate=0.50, cached_input_rate=0.05, output_rate=3.00,
    ),
    ("gemini", "gemini-3.1-pro-preview"): _tiered_subset_policy(
        provider="gemini",
        threshold=_TIER_THRESHOLD,
        low_input_rate=2.00, low_cached_input_rate=0.20, low_output_rate=12.00,
        high_input_rate=4.00, high_cached_input_rate=0.40, high_output_rate=18.00,
    ),
    # DEPRECATED: discontinued by Google effective March 9, 2026.
    # Retained for cost analysis of existing experiment results.
    ("gemini", "gemini-3-pro-preview"): _tiered_subset_policy(
        provider="gemini",
        threshold=_TIER_THRESHOLD,
        low_input_rate=2.00, low_cached_input_rate=0.20, low_output_rate=12.00,
        high_input_rate=4.00, high_cached_input_rate=0.40, high_output_rate=18.00,
    ),

    # ── Fireworks (subset input accounting, no cache writes) ──
    ("fireworks", "gpt-oss-120b"): _flat_subset_policy(
        provider="fireworks",
        input_rate=0.15, cached_input_rate=0.075, output_rate=0.60,
    ),
    ("fireworks", "kimi-k2p5"): _flat_subset_policy(
        provider="fireworks",
        input_rate=0.60, cached_input_rate=0.10, output_rate=3.00,
    ),
    ("fireworks", "kimi-k2p6"): _flat_subset_policy(
        provider="fireworks",
        input_rate=0.95, cached_input_rate=0.16, output_rate=4.00,
    ),
    ("fireworks", "glm-4p7"): _flat_subset_policy(
        provider="fireworks",
        input_rate=0.60, cached_input_rate=0.30, output_rate=2.20,
    ),
    ("fireworks", "glm-5"): _flat_subset_policy(
        provider="fireworks",
        input_rate=1.00, cached_input_rate=0.20, output_rate=3.20,
    ),
}


def get_pricing_policy(provider: str, model: str) -> PricingPolicy:
    """Look up the pricing policy for a (provider, model) pair.

    Uses substring matching via find_model_family() to handle versioned
    model IDs (e.g., "claude-haiku-4-5-20251001" matches "claude-haiku-4-5")
    and Fireworks path-prefixed IDs ("accounts/fireworks/models/glm-5").

    Raises ValueError if no policy matches the provider/model combination.
    """
    families = _PROVIDER_FAMILIES.get(provider)
    if families is None:
        raise ValueError(f"Unknown provider: {provider!r}")

    family = find_model_family(model, families)
    if family is None:
        raise ValueError(
            f"No pricing policy for model {model!r} under provider {provider!r}"
        )

    return _REGISTRY[(provider, family)]
