"""Gemini-specific result extraction from batch output.

Navigates the Gemini BatchGenerateContent response envelope:
  - "key" field (not "custom_id") holds the example identifier
  - response.candidates[0].content.parts contains thought and text parts
  - Thought parts have {"thought": true}; the response text part may carry
    a "thoughtSignature" but thought is absent or false
  - Errors appear in two forms: (1) top-level with no response envelope
    (cancelled, deadline exceeded), or (2) inline within response (missing
    candidates or a status field)
  - Usage in response.usageMetadata with separate thoughtsTokenCount
"""

from typing import Any

from utils.batch_inference.processed import ProcessedResult, TokenUsage, parse_custom_id


def extract_gemini(raw_line: dict[str, Any]) -> ProcessedResult:
    """Extract a ProcessedResult from a Gemini batch output line."""
    # Gemini uses "key" instead of "custom_id"
    custom_id: str = raw_line["key"]
    example_id, trial = parse_custom_id(custom_id)

    # Top-level error (cancelled, deadline exceeded, etc.) — no response envelope
    if "response" not in raw_line:
        error = raw_line.get("error", {})
        error_msg = (
            error.get("message", "unknown error")
            if isinstance(error, dict)
            else str(error)
        )
        return ProcessedResult(
            custom_id=custom_id,
            example_id=example_id,
            trial=trial,
            raw_response_text="",
            reasoning_text=None,
            model_version="",
            stop_reason=None,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            parse_error=f"Top-level error (no response): {error_msg}",
        )

    response = raw_line["response"]
    model_version = response.get("modelVersion", "")

    # Missing candidates signals an inline error
    candidates = response.get("candidates")
    if not candidates:
        error = response.get("error", {})
        error_msg = (
            error.get("message", "unknown error")
            if isinstance(error, dict)
            else str(error)
        )
        return ProcessedResult(
            custom_id=custom_id,
            example_id=example_id,
            trial=trial,
            raw_response_text="",
            reasoning_text=None,
            model_version=model_version,
            stop_reason=None,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            parse_error=f"No candidates in response: {error_msg}",
        )

    candidate = candidates[0]
    stop_reason = candidate.get("finishReason")

    # Walk parts: thought parts (thought==true) are reasoning,
    # everything else is the response text
    raw_response_text = ""
    reasoning_text = None
    for part in (candidate.get("content") or {}).get("parts") or []:
        if part.get("thought") is True:
            reasoning_text = part.get("text", "")
        else:
            raw_response_text = part.get("text", "")

    usage_meta = response.get("usageMetadata") or {}
    usage = TokenUsage(
        input_tokens=usage_meta.get("promptTokenCount", 0),
        output_tokens=usage_meta.get("candidatesTokenCount", 0),
        reasoning_tokens=usage_meta.get("thoughtsTokenCount"),
        cache_read_tokens=usage_meta.get("cachedContentTokenCount"),
    )

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
