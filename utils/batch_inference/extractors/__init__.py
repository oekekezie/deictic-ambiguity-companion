"""Dispatch for provider-specific batch output extraction.

Routes each raw output JSONL line to the appropriate provider-specific
extractor, mirroring the dispatch pattern in serializers/__init__.py
and sdk/__init__.py.
"""

from typing import Any

from utils.batch_inference.extractors.anthropic import extract_anthropic, extract_anthropic_batch_error
from utils.batch_inference.extractors.fireworks import extract_fireworks, extract_fireworks_batch_error
from utils.batch_inference.extractors.gemini import extract_gemini
from utils.batch_inference.extractors.openai import extract_openai, extract_openai_batch_error
from utils.batch_inference.processed import BatchError, ProcessedResult

_EXTRACTORS: dict[str, Any] = {
    "anthropic": extract_anthropic,
    "openai": extract_openai,
    "gemini": extract_gemini,
    "fireworks": extract_fireworks,
}

# Batch error extractors — only providers that produce a separate errors.jsonl.
# Gemini reports errors inline in output.jsonl (handled by extract_gemini).
_BATCH_ERROR_EXTRACTORS: dict[str, Any] = {
    "anthropic": extract_anthropic_batch_error,
    "openai": extract_openai_batch_error,
    "fireworks": extract_fireworks_batch_error,
}


def extract_result(provider: str, raw_line: dict[str, Any]) -> ProcessedResult:
    """Extract a standardized ProcessedResult from a raw provider output line.

    Dispatches to the provider-specific extractor based on the provider name.
    """
    handler = _EXTRACTORS.get(provider)
    if handler is None:
        raise ValueError(f"No extractor for provider: {provider!r}")
    return handler(raw_line)


def extract_batch_error(provider: str, raw_line: dict[str, Any]) -> BatchError:
    """Extract a BatchError from a raw provider errors.jsonl line.

    Dispatches to the provider-specific batch error extractor.
    Raises ValueError for providers without a separate errors file
    (e.g., Gemini, whose errors appear inline in output.jsonl).
    """
    handler = _BATCH_ERROR_EXTRACTORS.get(provider)
    if handler is None:
        raise ValueError(f"No batch error extractor for provider: {provider!r}")
    return handler(raw_line)


__all__ = [
    "BatchError",
    "extract_batch_error",
    "extract_result",
]
