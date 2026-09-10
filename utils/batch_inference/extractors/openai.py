"""OpenAI-specific result extraction from batch output.

Navigates the OpenAI Responses API batch envelope:
  - custom_id at top level
  - error field for API-level failures
  - response.body.output is an array of typed blocks (reasoning, message)
  - Message content is nested: output[type=="message"].content[type=="output_text"].text
  - Reasoning summaries in output[type=="reasoning"].summary[*].text
  - Usage in response.body.usage with output_tokens_details.reasoning_tokens
"""

from typing import Any

from utils.batch_inference.processed import BatchError, ProcessedResult, TokenUsage, parse_custom_id


def extract_openai_batch_error(raw_line: dict[str, Any]) -> BatchError:
    """Extract a BatchError from an OpenAI error file line.

    OpenAI batch error lines carry the error at the top level:
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


def extract_openai(raw_line: dict[str, Any]) -> ProcessedResult:
    """Extract a ProcessedResult from an OpenAI batch output line."""
    custom_id: str = raw_line["custom_id"]
    example_id, trial = parse_custom_id(custom_id)

    # API-level error (distinct from HTTP status on the response)
    if raw_line.get("error") is not None:
        return ProcessedResult(
            custom_id=custom_id,
            example_id=example_id,
            trial=trial,
            raw_response_text="",
            reasoning_text=None,
            model_version="",
            stop_reason=None,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            parse_error=f"API error: {raw_line['error']}",
        )

    body = raw_line["response"]["body"]

    # Walk output blocks to separate reasoning from the message
    raw_response_text = ""
    reasoning_text = None
    stop_reason = None

    for block in body.get("output") or []:
        if block["type"] == "message":
            stop_reason = block.get("status")
            for content_block in block.get("content") or []:
                if content_block["type"] == "output_text":
                    raw_response_text = content_block["text"]
        elif block["type"] == "reasoning":
            summaries = block.get("summary") or []
            texts = [s["text"] for s in summaries if s.get("text")]
            if texts:
                reasoning_text = "\n\n".join(texts)

    usage_data = body.get("usage") or {}
    reasoning_tokens = (usage_data.get("output_tokens_details") or {}).get(
        "reasoning_tokens"
    )
    cache_read_tokens = (usage_data.get("input_tokens_details") or {}).get(
        "cached_tokens"
    )
    usage = TokenUsage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
    )

    return ProcessedResult(
        custom_id=custom_id,
        example_id=example_id,
        trial=trial,
        raw_response_text=raw_response_text,
        reasoning_text=reasoning_text,
        model_version=body.get("model", ""),
        stop_reason=stop_reason,
        usage=usage,
    )
