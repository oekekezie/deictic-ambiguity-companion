"""Fireworks AI LLM configuration for batch inference."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from utils.batch_inference.llm_configs.validation import validate_temperature
from utils.batch_inference.schema_utils import find_model_family

_FIREWORKS_KNOWN_FAMILIES: set[str] = {
    "gpt-oss-120b",
    "glm-4p7",
    "glm-5",
    "kimi-k2p5",
    "kimi-k2p6",
}

_FIREWORKS_THINKING_FAMILIES: frozenset[str] = frozenset({
    "kimi-k2p5",
    "kimi-k2p6",
})


class FireworksThinking(BaseModel):
    """Anthropic-compatible thinking configuration for Kimi-family models on Fireworks.

    Controls thinking mode on/off. budget_tokens is accepted by the API
    but is not meaningful for Kimi K2.5 / K2.6 (thinking is binary on/off).
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["enabled", "disabled"]
    budget_tokens: int | None = None

    @field_validator("budget_tokens")
    @classmethod
    def check_budget_tokens(cls, v: int | None) -> int | None:
        """Enforce minimum of 1024 per the Fireworks API spec."""
        if v is not None and v < 1024:
            raise ValueError(f"budget_tokens must be >= 1024, got {v}")
        return v


class FireworksLLMConfig(BaseModel):
    """Immutable configuration for Fireworks AI batch inference requests.

    Thinking/reasoning can be configured via two mutually exclusive
    mechanisms — the Fireworks API rejects requests with both:

    - thinking: Anthropic-compatible thinking object (Kimi K2.5 only).
      The documented mechanism for controlling Kimi K2.5 reasoning.
    - reasoning_effort: General-purpose reasoning control. See the
      Fireworks API docs for the model-specific behavior list.

    reasoning_effort semantics vary by model (per Fireworks API docs):
    - GLM 4.7 / GLM 5: binary on/off — effort levels all behave the same;
      use False/"none" to disable, any other value enables reasoning
    - GPT-OSS 120b: only "low"/"medium"/"high" (cannot disable)

    Recommended defaults per model (from model creator documentation):

        Kimi K2.5:
            temperature=1.0, top_p=0.95,
            max_tokens=262144, thinking={"type": "enabled"}

        Kimi K2.6:
            temperature=1.0, top_p=0.95,
            max_tokens=262144, thinking={"type": "enabled"}

        GLM 4.7:
            temperature=1.0, top_p=0.95,
            max_tokens=202752, reasoning_effort=True

        GLM 5:
            temperature=1.0, top_p=0.95,
            max_tokens=202752, reasoning_effort=True

        GPT-OSS 120b:
            temperature=1.0, max_tokens=131072, reasoning_effort="medium"
    """

    model_config = ConfigDict(frozen=True)

    provider: Literal["fireworks"] = "fireworks"
    model: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    reasoning_effort: Literal["low", "medium", "high"] | bool | int | None = None
    reasoning_history: Literal["disabled", "interleaved", "preserved"] | None = None
    thinking: FireworksThinking | None = None

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float | None) -> float | None:
        return validate_temperature(v, max_temp=2.0)

    @model_validator(mode="after")
    def validate_reasoning_for_model(self) -> "FireworksLLMConfig":
        """Enforce thinking/reasoning_effort constraints.

        The Fireworks API rejects requests with both thinking and
        reasoning_effort. The thinking parameter is only supported for
        Kimi-family models. GPT-OSS 120b requires string reasoning_effort.
        """
        # Mutual exclusivity — Fireworks rejects requests with both
        if self.thinking is not None and self.reasoning_effort is not None:
            raise ValueError(
                "Cannot specify both 'thinking' and 'reasoning_effort' — "
                "the Fireworks API rejects requests with both parameters."
            )

        # thinking is only supported for Kimi-family models (K2.5, K2.6)
        if self.thinking is not None:
            family = find_model_family(self.model, _FIREWORKS_KNOWN_FAMILIES)
            if family not in _FIREWORKS_THINKING_FAMILIES:
                raise ValueError(
                    "The 'thinking' parameter is only supported for Kimi-family "
                    "models (Kimi K2.5, Kimi K2.6). Use 'reasoning_effort' for "
                    "other models."
                )
            return self

        if self.reasoning_effort is None:
            return self

        family = find_model_family(self.model, _FIREWORKS_KNOWN_FAMILIES)
        if family is None:
            raise ValueError(
                f"Unrecognized model family for {self.model!r}. "
                f"Known families: {sorted(_FIREWORKS_KNOWN_FAMILIES)}"
            )

        if family == "gpt-oss-120b":
            if not isinstance(self.reasoning_effort, str):
                raise ValueError(
                    f"{family} requires reasoning_effort to be 'low', 'medium', or 'high' "
                    f"(cannot be disabled or set to an integer token limit)"
                )

        # No current Fireworks model supports integer token limits
        if isinstance(self.reasoning_effort, int) and not isinstance(
            self.reasoning_effort, bool
        ):
            raise ValueError(
                "Integer reasoning_effort (token limit) is not supported. "
                "Use string values ('low', 'medium', 'high') or booleans."
            )

        return self


def is_reasoning_active(config: FireworksLLMConfig) -> bool:
    """Determine whether reasoning output is expected for this config.

    Checks both mutually exclusive reasoning mechanisms:
    - thinking: active when type is "enabled" (disabled means off)
    - reasoning_effort: active when truthy (True, "low"/"medium"/"high")
    """
    if config.thinking is not None:
        return config.thinking.type == "enabled"
    if config.reasoning_effort is not None:
        return bool(config.reasoning_effort)
    return False
