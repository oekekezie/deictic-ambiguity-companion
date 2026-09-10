"""Serializers for converting batch configs to provider-specific JSONL formats.

Dispatches to the appropriate provider serializer based on the LLMConfig type.
"""

from typing import Any

from utils.batch_inference.batch_config import BatchConfig
from utils.batch_inference.dataset import Dataset
from utils.batch_inference.llm_configs.anthropic import AnthropicLLMConfig
from utils.batch_inference.llm_configs.fireworks import FireworksLLMConfig
from utils.batch_inference.llm_configs.gemini import GeminiLLMConfig
from utils.batch_inference.llm_configs.openai import OpenAILLMConfig
from utils.batch_inference.serializers.anthropic import serialize_anthropic
from utils.batch_inference.serializers.expansion import expand_trials
from utils.batch_inference.serializers.fireworks import (
    embed_schema_in_system_prompt,
    serialize_fireworks,
)
from utils.batch_inference.serializers.gemini import serialize_gemini
from utils.batch_inference.serializers.openai import serialize_openai

# Provider serializer dispatch table
_SERIALIZERS: dict[type, Any] = {
    OpenAILLMConfig: serialize_openai,
    AnthropicLLMConfig: serialize_anthropic,
    GeminiLLMConfig: serialize_gemini,
    FireworksLLMConfig: serialize_fireworks,
}


def serialize(config: BatchConfig, dataset: Dataset) -> list[dict[str, Any]]:
    """Dispatch to the appropriate provider serializer.

    Selects the serializer based on the runtime type of `config.llm_config`.
    Raises TypeError for unrecognized LLMConfig types.
    """
    serializer = _SERIALIZERS.get(type(config.llm_config))
    if serializer is None:
        raise TypeError(
            f"No serializer for LLMConfig type: {type(config.llm_config).__name__}"
        )
    return serializer(config, dataset)


__all__ = [
    "embed_schema_in_system_prompt",
    "expand_trials",
    "serialize",
    "serialize_anthropic",
    "serialize_fireworks",
    "serialize_gemini",
    "serialize_openai",
]
