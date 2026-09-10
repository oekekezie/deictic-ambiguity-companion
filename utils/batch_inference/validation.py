"""Schema-aware validation for batch inference responses.

Validates a ``ProcessedResult``'s ``raw_response_text`` against the
expected response format (JSON or freetext) derived from the original
request's ``response_schema``. When a JSON schema is provided,
validation goes beyond structural checks (parseable JSON, top-level
dict) to verify required fields, types, and enum constraints using
the ``jsonschema`` library.

Called by ``process_job_output.py`` after provider-specific envelope
extraction, using the ``input_original.jsonl`` to retrieve each
request's ``response_schema`` (or ``None`` for freetext requests).
"""

import json
from typing import Any

import jsonschema

from utils.batch_inference.processed import ProcessedResult


def validate_response(
    result: ProcessedResult,
    response_schema: dict[str, Any] | None,
) -> ProcessedResult:
    """Validate a result's raw_response_text against the expected format.

    If the extractor already set ``parse_error`` (envelope-level failure),
    the result is returned unchanged -- validation is moot when the
    provider envelope itself was broken.

    Otherwise, checks conformance based on the schema parameter:
      - ``response_schema`` is a dict: response must be parseable JSON
        that decodes to a dict, then must conform to the JSON schema
      - ``response_schema`` is None: response must be non-empty text

    Returns the result unchanged when valid, or a copy with ``parse_error``
    set to a diagnostic message when validation fails.
    """
    # Envelope already failed -- skip validation
    if result.parse_error is not None:
        return result

    if response_schema is not None:
        return _validate_json_response(result, response_schema)
    return _validate_freetext_response(result)


def _validate_json_response(
    result: ProcessedResult,
    schema: dict[str, Any],
) -> ProcessedResult:
    """Validate that raw_response_text is parseable JSON conforming to schema.

    Two-phase validation: first checks JSON parseability and top-level
    dict structure, then validates against the JSON schema for required
    fields, types, enum constraints, etc.
    """
    try:
        payload = json.loads(result.raw_response_text)
    except json.JSONDecodeError as e:
        return result.model_copy(
            update={"parse_error": f"JSON decode failed: {e}"}
        )

    if not isinstance(payload, dict):
        return result.model_copy(
            update={
                "parse_error": (
                    f"Expected JSON object, got {type(payload).__name__}"
                )
            }
        )

    return _validate_against_schema(result, payload, schema)


def _validate_against_schema(
    result: ProcessedResult,
    payload: dict[str, Any],
    schema: dict[str, Any],
) -> ProcessedResult:
    """Validate a parsed JSON dict against a JSON Schema.

    Uses ``jsonschema.validate()`` which raises ``ValidationError`` on the
    first schema violation. The error message is captured into
    ``parse_error`` for downstream retry eligibility.

    ``jsonschema.SchemaError`` (malformed schema) is intentionally NOT
    caught -- that indicates a programming error in the schema definition,
    not a runtime data issue.
    """
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.ValidationError as e:
        return result.model_copy(
            update={"parse_error": f"Schema validation failed: {e.message}"}
        )
    return result


def _validate_freetext_response(result: ProcessedResult) -> ProcessedResult:
    """Validate that raw_response_text contains non-empty text."""
    if not result.raw_response_text.strip():
        return result.model_copy(
            update={"parse_error": "Empty response text"}
        )
    return result
