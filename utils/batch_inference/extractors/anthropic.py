"""Anthropic-specific result extraction from batch output.

Navigates the Anthropic MessageBatch result envelope:
  - custom_id at top level
  - result.type indicates success vs. error/expired/canceled
  - result.message.content is an array of typed blocks (thinking, text)
  - Usage in result.message.usage
"""

from typing import Any

from utils.batch_inference.processed import BatchError, ProcessedResult, TokenUsage, parse_custom_id


def extract_anthropic_batch_error(raw_line: dict[str, Any]) -> BatchError:
    """Extract a BatchError from an Anthropic errors.jsonl line.

    Anthropic error envelope nests the human-readable message two levels
    deep: result.error.error.message (the outer "error" is the error
    wrapper, the inner "error" holds the actual API error details).
    """
    custom_id: str = raw_line["custom_id"]
    example_id, trial = parse_custom_id(custom_id)

    result = raw_line["result"]
    error_type: str = result.get("type", "unknown")
    # Two levels of nesting: result.error.error.message
    detail: str | None = (
        result.get("error", {}).get("error", {}).get("message")
    )

    return BatchError(
        custom_id=custom_id,
        example_id=example_id,
        trial=trial,
        error_type=error_type,
        detail=detail,
    )


def extract_anthropic(raw_line: dict[str, Any]) -> ProcessedResult:
    """Extract a ProcessedResult from an Anthropic batch output line."""
    custom_id: str = raw_line["custom_id"]
    example_id, trial = parse_custom_id(custom_id)

    result = raw_line["result"]

    # Non-succeeded results (errored, expired, canceled) have no message
    if result["type"] != "succeeded":
        return ProcessedResult(
            custom_id=custom_id,
            example_id=example_id,
            trial=trial,
            raw_response_text="",
            reasoning_text=None,
            model_version="",
            stop_reason=None,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            parse_error=f"Non-succeeded result type: {result['type']}",
        )

    message = result["message"]

    # Walk content blocks to separate thinking from text response
    raw_response_text = ""
    reasoning_text = None
    for block in message["content"]:
        if block["type"] == "text":
            raw_response_text = block["text"]
        elif block["type"] == "thinking":
            reasoning_text = block["thinking"]

    usage_data = message["usage"]
    usage = TokenUsage(
        input_tokens=usage_data["input_tokens"],
        output_tokens=usage_data["output_tokens"],
        cache_read_tokens=usage_data.get("cache_read_input_tokens"),
        cache_write_tokens=usage_data.get("cache_creation_input_tokens"),
    )

    return ProcessedResult(
        custom_id=custom_id,
        example_id=example_id,
        trial=trial,
        raw_response_text=raw_response_text,
        reasoning_text=reasoning_text,
        model_version=message["model"],
        stop_reason=message.get("stop_reason"),
        usage=usage,
    )
