"""Fireworks AI batch inference serializer.

Converts a Dataset into Fireworks batch JSONL format. Sampling parameters
(temperature, top_p, etc.) are set at the batch job level via
``inferenceParameters``. Reasoning parameters (``reasoning_effort``,
``thinking``, ``reasoning_history``) are included per-request in the JSONL
body to match the Fireworks chat completions API schema.

When reasoning is active, the Fireworks API disables reasoning output if
``response_format`` is set (both ``json_schema`` and ``json_object``).
To preserve reasoning traces, this serializer omits ``response_format``
and instead injects the JSON schema into the system prompt, following
the Fireworks documentation's recommended workaround. JSON conformance
is validated during result processing by ``validate_response()``, with
failures eligible for retry via the existing batch retry infrastructure.
GPT-OSS-120b is excluded from this workaround due to a separate
control-token leak issue that makes its batch output non-functional.
"""

import json
from typing import Any

from utils.batch_inference.batch_config import BatchConfig
from utils.batch_inference.dataset import Dataset
from utils.batch_inference.example import Example
from utils.batch_inference.llm_configs.fireworks import (
    FireworksLLMConfig,
    _FIREWORKS_KNOWN_FAMILIES,
    is_reasoning_active,
)
from utils.batch_inference.schema_utils import find_model_family, resolve_schema


def embed_schema_in_system_prompt(
    system_prompt: str | None, resolved_schema: dict[str, Any]
) -> str:
    """Fold a JSON schema into a system prompt for Fireworks reasoning models.

    Fireworks disables a model's reasoning output when ``response_format`` is
    set, so for reasoning models the schema is delivered inside the system
    prompt instead (the Fireworks-documented workaround). Returns the system
    prompt with the schema instruction appended; when ``system_prompt`` is
    None, the instruction is returned on its own.
    """
    # ensure_ascii=False keeps non-ASCII schema content (degree signs, curly
    # quotes) literal — the model reads this embedded schema as prompt text,
    # so \uXXXX escapes would degrade it.
    instruction = (
        "\n\nIMPORTANT: Your response must be ONLY a JSON object "
        "matching this schema (no markdown code fences, no extra "
        "text):\n"
        + json.dumps(resolved_schema, indent=2, ensure_ascii=False)
    )
    if system_prompt is None:
        return instruction.strip()
    return system_prompt + instruction


def _should_omit_response_format(llm: FireworksLLMConfig) -> bool:
    """Determine whether to omit response_format in favor of prompt-based schema.

    Fireworks disables reasoning output when response_format is set.
    For models where reasoning can produce usable output, we omit
    response_format and inject the schema into the prompt instead.

    GPT-OSS-120b is excluded: its control-token leak issue makes batch
    output non-functional regardless of the response_format setting.
    """
    if not is_reasoning_active(llm):
        return False
    family = find_model_family(llm.model, _FIREWORKS_KNOWN_FAMILIES)
    return family != "gpt-oss-120b"


def serialize_fireworks(config: BatchConfig, dataset: Dataset) -> list[dict[str, Any]]:
    """Serialize each example to a Fireworks batch request dict.

    System prompt is prepended as a message with role "system" (Fireworks
    uses OpenAI-style chat format). Reasoning parameters are included
    per-request to match the chat completions API schema. Sampling
    parameters are set at the batch job level via ``inferenceParameters``.
    """
    llm: FireworksLLMConfig = config.llm_config  # type: ignore[assignment]
    return [_serialize_example(llm, example) for example in dataset.examples]


def _serialize_example(llm: FireworksLLMConfig, example: Example) -> dict[str, Any]:
    """Build a single Fireworks batch request dict from an example."""
    # System prompt goes first as a system-role message
    messages: list[dict[str, str]] = []
    if example.system_prompt is not None:
        messages.append({"role": "system", "content": example.system_prompt})
    messages.extend(
        {"role": m.role, "content": m.content} for m in example.messages
    )

    body: dict[str, Any] = {
        "messages": messages,
    }

    if example.response_schema is not None:
        resolved = resolve_schema(example.response_schema)
        if _should_omit_response_format(llm):
            # Fireworks disables reasoning when response_format is set.
            # Omit response_format and embed schema in the prompt instead
            # (per Fireworks docs: "include the schema in your prompt").
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": embed_schema_in_system_prompt(
                        messages[0]["content"], resolved
                    ),
                }
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": embed_schema_in_system_prompt(None, resolved),
                })
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": resolved,
                },
            }

    # Reasoning parameters — per-request to match the Fireworks chat
    # completions API schema (same pattern as OpenAI/Anthropic/Gemini
    # serializers). Mutual exclusivity between thinking and
    # reasoning_effort is enforced by FireworksLLMConfig's model validator.
    if llm.reasoning_effort is not None:
        body["reasoning_effort"] = llm.reasoning_effort
    if llm.thinking is not None:
        thinking_obj: dict[str, Any] = {"type": llm.thinking.type}
        if llm.thinking.budget_tokens is not None:
            thinking_obj["budget_tokens"] = llm.thinking.budget_tokens
        body["thinking"] = thinking_obj
    if llm.reasoning_history is not None:
        body["reasoning_history"] = llm.reasoning_history

    return {
        "custom_id": example.custom_id,
        "body": body,
    }
