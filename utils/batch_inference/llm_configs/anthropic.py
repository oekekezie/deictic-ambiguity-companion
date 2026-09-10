"""Anthropic LLM configuration for batch inference."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from utils.batch_inference.llm_configs.validation import validate_temperature
from utils.batch_inference.schema_utils import find_model_family

_ANTHROPIC_KNOWN_FAMILIES: set[str] = {
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
}

# Families that require the beta namespace when effort is set
_ANTHROPIC_EFFORT_BETA_FAMILIES: set[str] = {"claude-opus-4-5"}

# Allowed effort values per model family, ordered low-to-high
_ANTHROPIC_EFFORT_LEVELS: dict[str, list[str]] = {
    "claude-opus-4-6": ["low", "medium", "high", "max"],
    "claude-sonnet-4-6": ["low", "medium", "high"],
    "claude-opus-4-5": ["low", "medium", "high"],
}


class AnthropicLLMConfig(BaseModel):
    """Immutable configuration for Anthropic batch inference requests.

    This model is frozen to prevent accidental mutation of shared instances.
    To create a modified configuration, use model_copy(update={...}):

        custom = config.model_copy(update={"max_tokens": 8192})

    max_tokens is required per the Anthropic API.

    Thinking modes:
    - "enabled" (4.5 era): requires temperature=1.0, thinking_budget_tokens >= 1024
    - "adaptive" (4.6 era): no budget_tokens needed, depth controlled via effort
    """

    model_config = ConfigDict(frozen=True)

    provider: Literal["anthropic"] = "anthropic"
    model: str
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    thinking_type: Literal["enabled", "adaptive"] | None = None
    thinking_budget_tokens: int | None = None
    effort: Literal["low", "medium", "high", "max"] | None = None

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float | None) -> float | None:
        # Anthropic temperature range is 0.0-1.0
        return validate_temperature(v, max_temp=1.0)

    @model_validator(mode="after")
    def validate_thinking_constraints(self) -> "AnthropicLLMConfig":
        """Enforce thinking constraints based on thinking_type.

        "enabled" (4.5 era):
        - temperature, if set, must be exactly 1.0
        - thinking_budget_tokens is required (>= 1024, < max_tokens)

        "adaptive" (4.6 era):
        - thinking_budget_tokens is not used (depth via effort parameter)
        """
        if self.thinking_type is None:
            return self

        if self.thinking_type == "enabled":
            if self.temperature is not None and self.temperature != 1.0:
                raise ValueError(
                    "temperature must be 1.0 when thinking_type is 'enabled', "
                    f"got {self.temperature}"
                )
            if self.thinking_budget_tokens is None:
                raise ValueError(
                    "thinking_budget_tokens is required when thinking_type is 'enabled'"
                )
            if self.thinking_budget_tokens < 1024:
                raise ValueError(
                    f"thinking_budget_tokens must be >= 1024, "
                    f"got {self.thinking_budget_tokens}"
                )
            if self.thinking_budget_tokens >= self.max_tokens:
                raise ValueError(
                    f"thinking_budget_tokens ({self.thinking_budget_tokens}) must be less "
                    f"than max_tokens ({self.max_tokens})"
                )

        return self

    @model_validator(mode="after")
    def validate_effort_for_model(self) -> "AnthropicLLMConfig":
        """Validate effort parameter against model family support."""
        if self.effort is None:
            return self

        family = find_model_family(self.model, _ANTHROPIC_KNOWN_FAMILIES)
        if family is None:
            raise ValueError(
                f"Unrecognized model family for {self.model!r}. "
                f"Known families: {sorted(_ANTHROPIC_KNOWN_FAMILIES)}"
            )

        allowed = _ANTHROPIC_EFFORT_LEVELS.get(family)
        if allowed is None:
            raise ValueError(
                f"{family} does not support the effort parameter. "
                f"Models with effort support: {sorted(_ANTHROPIC_EFFORT_LEVELS.keys())}"
            )

        if self.effort not in allowed:
            raise ValueError(
                f"effort {self.effort!r} is not valid for {family}. "
                f"Allowed: {allowed}"
            )

        return self
