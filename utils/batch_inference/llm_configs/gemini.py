"""Google Gemini LLM configuration for batch inference."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from utils.batch_inference.llm_configs.validation import validate_temperature
from utils.batch_inference.schema_utils import find_model_family

_GEMINI_KNOWN_FAMILIES: set[str] = {
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    # DEPRECATED: discontinued by Google effective March 9, 2026.
    # Retained for existing experiment config compatibility.
    "gemini-3-pro-preview",
}

# Allowed thinking levels per model family, ordered low-to-high
_GEMINI_THINKING_LEVELS: dict[str, list[str]] = {
    "gemini-3-flash-preview": ["minimal", "low", "medium", "high"],
    "gemini-3.1-pro-preview": ["low", "medium", "high"],
    # DEPRECATED: discontinued by Google effective March 9, 2026.
    # Retained for existing experiment config compatibility.
    "gemini-3-pro-preview": ["low", "high"],
}


class GeminiLLMConfig(BaseModel):
    """Immutable configuration for Gemini batch inference requests.

    This model is frozen to prevent accidental mutation of shared instances.
    To create a modified configuration, use model_copy(update={...}):

        custom = config.model_copy(update={"temperature": 0.7})

    Model is specified at the job level. Thinking level is validated
    against model-specific constraints (e.g., 3 Pro supports low/high;
    3.1 Pro supports low/medium/high; Flash supports all levels).
    """

    model_config = ConfigDict(frozen=True)

    provider: Literal["gemini"] = "gemini"
    model: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    include_thoughts: bool | None = True

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float | None) -> float | None:
        return validate_temperature(v, max_temp=2.0)

    @model_validator(mode="after")
    def validate_thinking_level_for_model(self) -> "GeminiLLMConfig":
        """Ensure thinking_level is valid for the specified model family."""
        if self.thinking_level is None:
            return self
        family = find_model_family(self.model, _GEMINI_KNOWN_FAMILIES)
        if family is None:
            raise ValueError(
                f"Unrecognized model family for {self.model!r}. "
                f"Known families: {sorted(_GEMINI_KNOWN_FAMILIES)}"
            )
        allowed = _GEMINI_THINKING_LEVELS[family]
        if self.thinking_level not in allowed:
            raise ValueError(
                f"thinking_level '{self.thinking_level}' is not valid for {family}. "
                f"Allowed: {allowed}"
            )
        return self
