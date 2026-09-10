"""Google Gemini batch inference serializer.

Converts a BatchConfig + Dataset into Gemini BatchGenerateContent JSONL format.

Schema normalization: The Gemini Batch API processes uploaded JSONL files as-is,
without the transformations the SDK applies to inline requests. This serializer
delegates schema normalization to ``Schema.from_json_schema()``, which handles
type uppercasing, ``$ref`` inlining, nullable conversion, and stripping of
unsupported fields like ``additionalProperties``.

Field naming: Uses camelCase (e.g., ``generationConfig``, ``systemInstruction``)
matching the Gemini REST API format.
"""

from typing import Any

from google.genai.types import JSONSchema, Schema

from utils.batch_inference.batch_config import BatchConfig
from utils.batch_inference.dataset import Dataset
from utils.batch_inference.example import Example
from utils.batch_inference.llm_configs.gemini import GeminiLLMConfig
from utils.batch_inference.schema_utils import resolve_schema

# Gemini uses "model" instead of "assistant"
_ROLE_MAP: dict[str, str] = {"assistant": "model", "user": "user"}

def _normalize_schema_for_gemini(raw_schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON Schema dict for Gemini's structured output API.

    Delegates to the SDK's ``Schema.from_json_schema()`` which handles:
    - Type uppercasing (``"object"`` → ``"OBJECT"``)
    - ``$defs``/``$ref`` inlining
    - Nullable conversion (``anyOf: [{type: X}, {type: null}]`` → ``nullable``)
    - Stripping unsupported fields (e.g., ``additionalProperties``)
    """
    json_schema = JSONSchema.model_validate(raw_schema)
    gemini_schema = Schema.from_json_schema(
        json_schema=json_schema,
        api_option="GEMINI_API",
    )
    return gemini_schema.model_dump(by_alias=True, exclude_none=True, mode="json")


def serialize_gemini(config: BatchConfig, dataset: Dataset) -> list[dict[str, Any]]:
    """Serialize each example to a Gemini batch request dict.

    Maps ``custom_id`` → ``key``, messages → ``request.contents`` with Gemini's
    ``{role, parts: [{text}]}`` format and ``"assistant"`` → ``"model"`` role
    mapping. Response schemas are normalized for Gemini compatibility via the
    SDK's ``Schema.from_json_schema()``.
    """
    llm: GeminiLLMConfig = config.llm_config  # type: ignore[assignment]
    return [_serialize_example(llm, example) for example in dataset.examples]


def _serialize_example(llm: GeminiLLMConfig, example: Example) -> dict[str, Any]:
    """Build a single Gemini batch request dict from an example.

    Uses camelCase field names matching the Gemini REST API format.
    """
    # Gemini uses {role, parts: [{text}]} content format
    contents = [
        {"role": _ROLE_MAP[m.role], "parts": [{"text": m.content}]}
        for m in example.messages
    ]

    generation_config: dict[str, Any] = {}

    if example.response_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = _normalize_schema_for_gemini(
            resolve_schema(example.response_schema)
        )

    # Optional generation params — omit if None (using camelCase field names)
    if llm.temperature is not None:
        generation_config["temperature"] = llm.temperature
    if llm.top_p is not None:
        generation_config["topP"] = llm.top_p
    if llm.top_k is not None:
        generation_config["topK"] = llm.top_k
    if llm.max_output_tokens is not None:
        generation_config["maxOutputTokens"] = llm.max_output_tokens

    # Thinking config — included by default (include_thoughts defaults to True).
    # IMPORTANT: thinkingLevel MUST be uppercased. The Batch API processes JSONL
    # as-is without the SDK's automatic enum normalization, so enum values like
    # thinkingLevel require SCREAMING_CASE (e.g., "HIGH" not "high"). Same
    # reason schema type values are uppercased. See test:
    # test_thinking_level_uppercased_for_batch_jsonl
    thinking_config: dict[str, Any] = {}
    if llm.thinking_level is not None:
        thinking_config["thinkingLevel"] = llm.thinking_level.upper()
    if llm.include_thoughts is not None:
        thinking_config["includeThoughts"] = llm.include_thoughts
    if thinking_config:
        generation_config["thinkingConfig"] = thinking_config

    request: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    if example.system_prompt is not None:
        request["systemInstruction"] = {"parts": [{"text": example.system_prompt}]}

    return {
        "key": example.custom_id,
        "request": request,
    }
