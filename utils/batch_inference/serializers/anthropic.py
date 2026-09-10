"""Anthropic batch inference serializer.

Converts a BatchConfig + Dataset into Anthropic Messages Batch format.
"""

from typing import Any

from utils.batch_inference.batch_config import BatchConfig
from utils.batch_inference.dataset import Dataset
from utils.batch_inference.example import Example
from utils.batch_inference.llm_configs.anthropic import AnthropicLLMConfig
from utils.batch_inference.schema_utils import resolve_schema


def serialize_anthropic(config: BatchConfig, dataset: Dataset) -> list[dict[str, Any]]:
    """Serialize each example to an Anthropic batch request dict.

    Maps messages to `params.messages`, system prompt to `params.system` (string).
    The `output_config` block is included when either a response schema is set
    on the example (placed in `output_config.format`) or when `effort` is set
    on the LLM config (placed in `output_config.effort`). The `max_tokens` field
    is always present since Anthropic requires it.

    Thinking serialization depends on thinking_type:
    - "enabled": emits {"type": "enabled", "budget_tokens": N}
    - "adaptive": emits {"type": "adaptive"}
    - None: thinking block omitted entirely
    """
    llm: AnthropicLLMConfig = config.llm_config  # type: ignore[assignment]
    return [_serialize_example(llm, example) for example in dataset.examples]


def _serialize_example(llm: AnthropicLLMConfig, example: Example) -> dict[str, Any]:
    """Build a single Anthropic batch request dict from an example."""
    params: dict[str, Any] = {
        "model": llm.model,
        "max_tokens": llm.max_tokens,
        "messages": [{"role": m.role, "content": m.content} for m in example.messages],
    }

    output_config: dict[str, Any] = {}

    if example.response_schema is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": resolve_schema(example.response_schema),
        }

    if llm.effort is not None:
        output_config["effort"] = llm.effort

    if output_config:
        params["output_config"] = output_config

    if example.system_prompt is not None:
        params["system"] = example.system_prompt

    # Optional params — omit if None
    if llm.temperature is not None:
        params["temperature"] = llm.temperature
    if llm.top_p is not None:
        params["top_p"] = llm.top_p
    if llm.top_k is not None:
        params["top_k"] = llm.top_k
    if llm.stop_sequences is not None:
        params["stop_sequences"] = llm.stop_sequences

    # Thinking block — depends on thinking_type
    if llm.thinking_type == "enabled":
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": llm.thinking_budget_tokens,
        }
    elif llm.thinking_type == "adaptive":
        params["thinking"] = {"type": "adaptive"}

    return {
        "custom_id": example.custom_id,
        "params": params,
    }
