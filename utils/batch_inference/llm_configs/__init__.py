"""Provider-specific LLM configuration models.

The LLMConfig union type uses the `provider` field as a discriminator,
enabling Pydantic to automatically select the correct variant during
JSON deserialization.
"""

from typing import Annotated, Union

from pydantic import Discriminator, Tag

from utils.batch_inference.llm_configs.anthropic import AnthropicLLMConfig
from utils.batch_inference.llm_configs.fireworks import FireworksLLMConfig
from utils.batch_inference.llm_configs.gemini import GeminiLLMConfig
from utils.batch_inference.llm_configs.openai import OpenAILLMConfig

LLMConfig = Annotated[
    Union[
        Annotated[OpenAILLMConfig, Tag("openai")],
        Annotated[AnthropicLLMConfig, Tag("anthropic")],
        Annotated[GeminiLLMConfig, Tag("gemini")],
        Annotated[FireworksLLMConfig, Tag("fireworks")],
    ],
    Discriminator("provider"),
]

__all__ = [
    "LLMConfig",
    "OpenAILLMConfig",
    "GeminiLLMConfig",
    "AnthropicLLMConfig",
    "FireworksLLMConfig",
]
