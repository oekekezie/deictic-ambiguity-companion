"""Fireworks-specific result extraction from batch output.

Navigates the Fireworks chat-completion-style batch envelope, handling
two reasoning output variants:

1. **Structured reasoning** (glm-4p7, glm-5, kimi-k2p5, kimi-k2p6, gpt-oss-120b):
   A separate ``reasoning_content`` field on the message object;
   ``content`` contains clean JSON.

2. **Inline reasoning**: No ``reasoning_content`` field; ``content``
   contains ``<think>...</think>`` tags followed by JSON.

When ``reasoning_content`` is absent and no think tags are present,
``content`` is treated as clean JSON with no reasoning trace. This is
the expected output when reasoning is disabled. When reasoning is
enabled, pilot testing revealed that the Fireworks batch API
systematically omits the ``reasoning_content`` field when structured
outputs (``response_format: json_schema``) are also enabled — this is
an infrastructure-level incompatibility in the batch serving layer, not
a property of the underlying models. The ``reasoning_text`` field on
``ProcessedResult`` therefore doubles as a post-hoc indicator of
whether reasoning actually occurred on a given trial.

When ``response_format`` is omitted to preserve reasoning traces (the
workaround for the above incompatibility), models may wrap their JSON
output in markdown code fences (````json ... ````). Code fences are
stripped before JSON parsing as a defensive measure.
"""

from typing import Any

from utils.batch_inference.processed import BatchError, ProcessedResult, TokenUsage, parse_custom_id


def extract_fireworks_batch_error(raw_line: dict[str, Any]) -> BatchError:
    """Extract a BatchError from a Fireworks error file line.

    Fireworks batch error lines carry the error at the top level:
    {"custom_id": "...", "error": {"code": "...", "message": "..."}}.
    """
    custom_id: str = raw_line["custom_id"]
    example_id, trial = parse_custom_id(custom_id)

    error_obj = raw_line.get("error") or {}
    return BatchError(
        custom_id=custom_id,
        example_id=example_id,
        trial=trial,
        error_type=error_obj.get("code", "unknown"),
        detail=error_obj.get("message"),
    )


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences wrapping a JSON payload.

    Models without response_format constraints sometimes wrap JSON in
    ````` ```json ... ``` ````` or ````` ``` ... ``` ````` fences. Returns the inner
    content if fences are detected, otherwise returns the input unchanged.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (with optional language tag like ```json)
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        # Remove closing fence
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    return stripped


def _split_think_tags(content: str) -> tuple[str, str]:
    """Split content on </think> delimiter into (reasoning, response).

    Strips the leading <think> tag from reasoning if present.
    Returns (reasoning_text, response_text).
    """
    parts = content.split("</think>", 1)
    reasoning = parts[0].strip()
    # Strip leading <think> tag if present
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :].strip()
    response = parts[1].strip() if len(parts) > 1 else ""
    return reasoning, response


def extract_fireworks(raw_line: dict[str, Any]) -> ProcessedResult:
    """Extract a ProcessedResult from a Fireworks batch output line."""
    custom_id: str = raw_line["custom_id"]
    example_id, trial = parse_custom_id(custom_id)

    response = raw_line["response"]
    model_version: str = response.get("model", "")

    choices = response.get("choices")
    if not choices:
        return ProcessedResult(
            custom_id=custom_id,
            example_id=example_id,
            trial=trial,
            raw_response_text="",
            reasoning_text=None,
            model_version=model_version,
            stop_reason=None,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            parse_error="No choices in response",
        )

    choice = choices[0]
    message = choice["message"]
    stop_reason = choice.get("finish_reason")
    content: str = message.get("content") or ""
    reasoning_content: str | None = message.get("reasoning_content")

    # Determine reasoning and response text based on the variant
    reasoning_text: str | None = None
    raw_response_text: str = content

    if reasoning_content:
        # Variant 1: structured reasoning_content field (glm-4p7, glm-5, etc.)
        reasoning_text = reasoning_content
    elif "</think>" in content:
        # Variant 2: inline <think>...</think> tags
        reasoning_text, raw_response_text = _split_think_tags(content)

    usage_data = response.get("usage") or {}
    cache_read_tokens = (usage_data.get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    usage = TokenUsage(
        input_tokens=usage_data.get("prompt_tokens", 0),
        output_tokens=usage_data.get("completion_tokens", 0),
        cache_read_tokens=cache_read_tokens,
    )

    # Strip markdown code fences that models produce when response_format
    # is omitted to preserve reasoning traces
    raw_response_text = _strip_code_fences(raw_response_text)

    return ProcessedResult(
        custom_id=custom_id,
        example_id=example_id,
        trial=trial,
        raw_response_text=raw_response_text,
        reasoning_text=reasoning_text,
        model_version=model_version,
        stop_reason=stop_reason,
        usage=usage,
    )
