"""OpenAI LLM configuration for batch inference.

Targets the Responses API (/v1/responses), which does NOT support stop
sequences, presence_penalty, or frequency_penalty — those parameters
are Chat Completions–only.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from utils.batch_inference.llm_configs.validation import validate_temperature
from utils.batch_inference.schema_utils import find_model_family

_OPENAI_KNOWN_FAMILIES: set[str] = {
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5-mini",
}

# Model family -> allowed reasoning effort values, ordered low-to-high
_OPENAI_REASONING_EFFORT: dict[str, list[str]] = {
    "gpt-5": ["minimal", "low", "medium", "high"],
    "gpt-5.1": ["none", "low", "medium", "high"],
    "gpt-5.2": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.2-codex": ["low", "medium", "high", "xhigh"],
    "gpt-5-mini": ["minimal", "low", "medium", "high"],
}


class OpenAILLMConfig(BaseModel):
    """Immutable configuration for OpenAI Responses API batch inference.

    This model is frozen to prevent accidental mutation of shared instances.
    To create a modified configuration, use model_copy(update={...}):

        custom = config.model_copy(update={"temperature": 0.7})

    Model is specified per-request. Reasoning effort is validated
    against model-specific allowed values when both are provided.
    """

    model_config = ConfigDict(frozen=True)

    provider: Literal["openai"] = "openai"
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = "auto"
    verbosity: Literal["low", "medium", "high"] | None = None

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float | None) -> float | None:
        return validate_temperature(v, max_temp=2.0)

    @model_validator(mode="after")
    def validate_reasoning_effort_for_model(self) -> "OpenAILLMConfig":
        """Ensure reasoning_effort is valid for the specified model family."""
        if self.reasoning_effort is None:
            return self
        family = find_model_family(self.model, _OPENAI_KNOWN_FAMILIES)
        if family is None:
            raise ValueError(
                f"Unrecognized model family for {self.model!r}. "
                f"Known families: {sorted(_OPENAI_KNOWN_FAMILIES)}"
            )
        allowed = _OPENAI_REASONING_EFFORT[family]
        if self.reasoning_effort not in allowed:
            raise ValueError(
                f"reasoning_effort '{self.reasoning_effort}' is not valid for {family}. "
                f"Allowed: {allowed}"
            )
        return self
