"""Provider registry for batch inference.

Centralizes provider-level knowledge: supported models per provider,
API key environment variables, batch request limits, LLM config
construction, model capability queries, batch size validation, and
provider-string-based SDK dispatch.
"""

import os
from pathlib import Path
from typing import Any, Literal

from utils.batch_inference.llm_configs import LLMConfig
from utils.batch_inference.llm_configs.anthropic import (
    AnthropicLLMConfig,
    _ANTHROPIC_EFFORT_LEVELS,
    _ANTHROPIC_KNOWN_FAMILIES,
)
from utils.batch_inference.llm_configs.fireworks import (
    FireworksLLMConfig,
    _FIREWORKS_KNOWN_FAMILIES,
)
from utils.batch_inference.llm_configs.gemini import (
    GeminiLLMConfig,
    _GEMINI_KNOWN_FAMILIES,
    _GEMINI_THINKING_LEVELS,
)
from utils.batch_inference.llm_configs.openai import (
    OpenAILLMConfig,
    _OPENAI_KNOWN_FAMILIES,
    _OPENAI_REASONING_EFFORT,
)
from utils.batch_inference.schema_utils import find_model_family
from utils.batch_inference.sdk.anthropic import (
    cancel_anthropic,
    check_status_anthropic,
    retrieve_anthropic,
)
from utils.batch_inference.sdk.fireworks import (
    cancel_fireworks,
    check_status_fireworks,
    retrieve_fireworks,
)
from utils.batch_inference.sdk.gemini import (
    cancel_gemini,
    check_status_gemini,
    retrieve_gemini,
)
from utils.batch_inference.sdk.openai import (
    cancel_openai,
    check_status_openai,
    retrieve_openai,
)

Provider = Literal["openai", "anthropic", "gemini", "fireworks"]

# -- Provider catalog ---------------------------------------------------------

API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
}

# Additional required environment variables beyond the API key
ADDITIONAL_ENV_VARS: dict[str, list[str]] = {
    "fireworks": ["FIREWORKS_ACCOUNT_ID"],
}

# Display label → full model identifier, per provider
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "openai": {
        "gpt-5.2": "gpt-5.2",
        "gpt-5-mini": "gpt-5-mini",
    },
    "anthropic": {
        "claude-opus-4-6": "claude-opus-4-6",
        "claude-opus-4-5": "claude-opus-4-5",
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-sonnet-4-5": "claude-sonnet-4-5",
        "claude-haiku-4-5": "claude-haiku-4-5",
    },
    "gemini": {
        "gemini-3-flash-preview": "gemini-3-flash-preview",
        "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
        # gemini-3-pro-preview: DEPRECATED — discontinued by Google effective
        # March 9, 2026. Removed from batch submit UI; retained in config
        # validators and pricing for existing experiment result compatibility.
    },
    "fireworks": {
        "kimi-k2p6": "accounts/fireworks/models/kimi-k2p6",
        "kimi-k2p5": "accounts/fireworks/models/kimi-k2p5",
        "glm-4p7": "accounts/fireworks/models/glm-4p7",
        "glm-5": "accounts/fireworks/models/glm-5",
        "gpt-oss-120b": "accounts/fireworks/models/gpt-oss-120b",
    },
}

# Maximum batch requests per provider
PROVIDER_LIMITS: dict[str, int] = {
    "openai": 50_000,
    "anthropic": 100_000,
    "gemini": 100_000,
    "fireworks": 100_000,
}


# -- API key check ------------------------------------------------------------


def check_api_key(provider: str) -> bool:
    """Return True if all required environment variables for the provider are set.

    Checks the primary API key and any additional required variables
    (e.g., FIREWORKS_ACCOUNT_ID for Fireworks).
    """
    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown provider: {provider!r}")

    if not os.environ.get(env_var):
        return False

    # Check additional required env vars (e.g., FIREWORKS_ACCOUNT_ID)
    for additional_var in ADDITIONAL_ENV_VARS.get(provider, []):
        if not os.environ.get(additional_var):
            return False

    return True


# -- Model capability queries -------------------------------------------------
#
# Intentional scaffold surface: nothing calls these yet, because the notebooks
# load pre-authored JSON configs rather than building effort-level pickers.


def get_reasoning_options_openai(model: str) -> list[str]:
    """Return allowed reasoning_effort values for an OpenAI model family, ordered low-to-high."""
    family = find_model_family(model, _OPENAI_KNOWN_FAMILIES)
    if family is None:
        return []
    return list(_OPENAI_REASONING_EFFORT.get(family, []))


def get_thinking_options_gemini(model: str) -> list[str]:
    """Return allowed thinking_level values for a Gemini model family, ordered low-to-high."""
    family = find_model_family(model, _GEMINI_KNOWN_FAMILIES)
    if family is None:
        return []
    return list(_GEMINI_THINKING_LEVELS.get(family, []))


def get_reasoning_options_fireworks(model: str) -> dict[str, str | bool]:
    """Return reasoning_effort UI options for a Fireworks model.

    Returns {display_label: actual_value} since GLM uses booleans
    while other models use string effort levels.
    """
    family = find_model_family(model, _FIREWORKS_KNOWN_FAMILIES)
    if family in {"glm-4p7", "glm-5"}:
        return {"Enabled": True, "Disabled": False}
    return {"low": "low", "medium": "medium", "high": "high"}


def get_effort_options_anthropic(model: str) -> list[str]:
    """Return allowed effort values for an Anthropic model family, ordered low-to-high."""
    family = find_model_family(model, _ANTHROPIC_KNOWN_FAMILIES)
    if family is None:
        return []
    return list(_ANTHROPIC_EFFORT_LEVELS.get(family, []))


# -- Reasoning effort level extraction ----------------------------------------------
# One place knows "for this LLMConfig variant, which field is the reasoning
# parameter and what is its value." Both summarize_reasoning() (display) and
# the analysis layer's resolve_config_key() (identity) build on this.


def extract_reasoning_effort_level(config: LLMConfig) -> str | None:
    """Extract the raw reasoning effort level label from any LLMConfig variant.

    Returns the reasoning parameter value as a string suitable for config
    keys. Returns None when no reasoning parameter is configured.

    For Anthropic, prefers ``effort`` (graduated depth control) over
    ``thinking_type`` (binary enable/disable). When both are set,
    ``effort`` is the more granular signal for config identity.
    """
    if isinstance(config, OpenAILLMConfig):
        return config.reasoning_effort
    if isinstance(config, AnthropicLLMConfig):
        if config.effort is not None:
            return config.effort
        if config.thinking_type is not None:
            return config.thinking_type
        return None
    if isinstance(config, GeminiLLMConfig):
        return config.thinking_level
    if isinstance(config, FireworksLLMConfig):
        return _extract_fireworks_level(config)
    raise ValueError(f"Unrecognized config type: {type(config).__name__}")


def _extract_fireworks_level(config: FireworksLLMConfig) -> str | None:
    """Extract reasoning effort level for Fireworks configs.

    Handles the three mutually exclusive mechanisms: thinking object,
    boolean reasoning_effort (GLM models), and string reasoning_effort.
    """
    if config.thinking is not None:
        return config.thinking.type
    if config.reasoning_effort is not None:
        # bool check before int — isinstance(True, int) is True in Python
        if isinstance(config.reasoning_effort, bool):
            return "on" if config.reasoning_effort else "off"
        return str(config.reasoning_effort)
    return None


# -- Reasoning summary -------------------------------------------------------


def summarize_reasoning(config: LLMConfig) -> str:
    """Produce a human-readable summary of a config's reasoning/thinking settings.

    Covers each provider's distinct mechanism for controlling reasoning intensity.
    Output formatting fields (e.g. reasoning_summary for OpenAI, include_thoughts
    for Gemini) are intentionally excluded — they affect output format, not depth.
    """
    level = extract_reasoning_effort_level(config)
    if level is None:
        return "—"

    # Anthropic and Fireworks need richer display formatting
    # that includes context beyond the raw level label
    if isinstance(config, AnthropicLLMConfig):
        return _format_anthropic_summary(config)
    if isinstance(config, FireworksLLMConfig):
        return _format_fireworks_summary(config)

    # OpenAI and Gemini: the raw level IS the display string
    return level


def _format_anthropic_summary(config: AnthropicLLMConfig) -> str:
    """Format Anthropic reasoning summary when reasoning is active."""
    if config.thinking_type == "enabled":
        budget = f"{config.thinking_budget_tokens:,}" if config.thinking_budget_tokens else "?"
        base = f"enabled (budget: {budget})"
        if config.effort:
            return f"{base[:-1]}, effort: {config.effort})"
        return base

    if config.thinking_type == "adaptive":
        if config.effort:
            return f"adaptive (effort: {config.effort})"
        return "adaptive"

    # effort only (no thinking_type)
    return f"effort: {config.effort}"


def _format_fireworks_summary(config: FireworksLLMConfig) -> str:
    """Format Fireworks reasoning summary when reasoning is active.

    The thinking and reasoning_effort mechanisms are mutually exclusive
    (enforced by the config validator).
    """
    if config.thinking is not None:
        label = f"thinking: {config.thinking.type}"
        if config.thinking.budget_tokens is not None:
            return f"{label} (budget: {config.thinking.budget_tokens:,})"
        return label

    if isinstance(config.reasoning_effort, bool):
        return "on" if config.reasoning_effort else "off"
    return str(config.reasoning_effort)


# -- LLM config factory ------------------------------------------------------


def build_llm_config(provider: str, model: str, params: dict[str, Any]) -> LLMConfig:
    """Construct the appropriate LLMConfig from provider, model, and params."""
    if provider == "openai":
        return OpenAILLMConfig(model=model, **params)
    if provider == "anthropic":
        return AnthropicLLMConfig(model=model, **params)
    if provider == "gemini":
        return GeminiLLMConfig(model=model, **params)
    if provider == "fireworks":
        return FireworksLLMConfig(model=model, **params)
    raise ValueError(f"Unknown provider: {provider!r}")


# -- Batch size validation ----------------------------------------------------


def validate_batch_size(
    num_examples: int, num_trials: int, provider: str
) -> dict[str, Any]:
    """Validate total request count against provider limits.

    Returns a dict with keys:
        total: int — total requests after expansion
        warning: str | None — warning message if total > 10,000
        error: str | None — error message if exceeding provider limit
    """
    total = num_examples * num_trials
    limit = PROVIDER_LIMITS.get(provider, 100_000)

    warning = None
    error = None

    if total > limit:
        error = f"Exceeds {provider} limit of {limit:,} requests (total: {total:,})"
    elif total > 10_000:
        warning = f"Large batch: {total:,} total requests"

    return {"total": total, "warning": warning, "error": error}


# -- Provider-string SDK dispatch ---------------------------------------------
# The sdk/__init__.py dispatch functions require a BatchConfig (type-based).
# These alternatives dispatch by provider string, for use cases like the
# monitor notebook where only a JobRecord (with a provider field) is available.

_CHECK_STATUS_DISPATCH: dict[str, Any] = {
    "openai": check_status_openai,
    "anthropic": check_status_anthropic,
    "gemini": check_status_gemini,
    "fireworks": check_status_fireworks,
}


async def check_status_by_provider(provider: str, job_id: str) -> dict[str, Any]:
    """Dispatch check_status by provider string rather than BatchConfig."""
    handler = _CHECK_STATUS_DISPATCH.get(provider)
    if handler is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    return await handler(job_id)


_CANCEL_DISPATCH: dict[str, Any] = {
    "openai": cancel_openai,
    "anthropic": cancel_anthropic,
    "gemini": cancel_gemini,
    "fireworks": cancel_fireworks,
}


async def cancel_by_provider(provider: str, job_id: str) -> None:
    """Dispatch cancel by provider string rather than BatchConfig.

    Cancellation is asynchronous — callers should follow up with
    check_status_by_provider to capture the transitional state.
    """
    handler = _CANCEL_DISPATCH.get(provider)
    if handler is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    await handler(job_id)


async def retrieve_by_provider(
    provider: str,
    job_id: str,
    output_path: Path,
    errors_path: Path | None = None,
) -> None:
    """Dispatch retrieve by provider string rather than BatchConfig.

    OpenAI and Anthropic require errors_path; Fireworks uses it when
    provided; Gemini ignores it (errors are inline in results).
    """
    if provider == "openai":
        if errors_path is None:
            raise ValueError("errors_path is required for OpenAI retrieval")
        return await retrieve_openai(job_id, output_path, errors_path)
    if provider == "anthropic":
        if errors_path is None:
            raise ValueError("errors_path is required for Anthropic retrieval")
        return await retrieve_anthropic(job_id, output_path, errors_path)
    if provider == "gemini":
        return await retrieve_gemini(job_id, output_path)
    if provider == "fireworks":
        return await retrieve_fireworks(job_id, output_path, errors_path)
    raise ValueError(f"Unknown provider: {provider!r}")
