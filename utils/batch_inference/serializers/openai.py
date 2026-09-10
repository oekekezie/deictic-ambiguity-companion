"""OpenAI batch inference serializer.

Converts a BatchConfig + Dataset into OpenAI Responses API batch JSONL format.
"""

from typing import Any

from utils.batch_inference.batch_config import BatchConfig
from utils.batch_inference.dataset import Dataset
from utils.batch_inference.example import Example
from utils.batch_inference.llm_configs.openai import OpenAILLMConfig
from utils.batch_inference.schema_utils import resolve_schema


def serialize_openai(config: BatchConfig, dataset: Dataset) -> list[dict[str, Any]]:
    """Serialize each example to an OpenAI Responses API batch request dict.

    System prompt maps to a developer-role message prepended to `body.input`,
    matching OpenAI's authority model where developer messages provide
    system-level guidance. When a response schema is set on the example,
    it is placed in `body.text.format` with `strict: true` and passed
    through as-is — the caller must ensure it meets OpenAI's strict mode
    requirements (e.g., additionalProperties: false on all object types).
    """
    llm: OpenAILLMConfig = config.llm_config  # type: ignore[assignment]
    return [_serialize_example(llm, example) for example in dataset.examples]


def _serialize_example(llm: OpenAILLMConfig, example: Example) -> dict[str, Any]:
    """Build a single OpenAI batch request dict from an example."""
    # Developer message goes first when a system prompt is provided
    input_messages: list[dict[str, str]] = []
    if example.system_prompt is not None:
        input_messages.append({"role": "developer", "content": example.system_prompt})
    input_messages.extend(
        {"role": m.role, "content": m.content} for m in example.messages
    )

    body: dict[str, Any] = {
        "model": llm.model,
        "input": input_messages,
    }

    if example.response_schema is not None:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "response_schema",
                "strict": True,
                "schema": resolve_schema(example.response_schema),
            }
        }

    # Reasoning block — included by default (reasoning_summary defaults to "auto")
    reasoning: dict[str, Any] = {}
    if llm.reasoning_effort is not None:
        reasoning["effort"] = llm.reasoning_effort
    if llm.reasoning_summary is not None:
        reasoning["summary"] = llm.reasoning_summary
    if reasoning:
        body["reasoning"] = reasoning

    # Optional scalar params — omit if None
    if llm.temperature is not None:
        body["temperature"] = llm.temperature
    if llm.top_p is not None:
        body["top_p"] = llm.top_p
    if llm.max_output_tokens is not None:
        body["max_output_tokens"] = llm.max_output_tokens
    # Responses API nests verbosity under text (not top-level)
    if llm.verbosity is not None:
        if "text" not in body:
            body["text"] = {}
        body["text"]["verbosity"] = llm.verbosity

    return {
        "custom_id": example.custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }
