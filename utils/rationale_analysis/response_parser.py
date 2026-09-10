"""Parse rationale analyst JSON responses into structured results.

Validates raw response text against the seven-flag response schema and
extracts the seven flag results. This is the rationale analysis analogue
of ``parse_experiment_response()`` in the experiment analysis layer: a
domain-specific parser for a domain-specific response structure.
"""

import json
from typing import Any

from utils.rationale_analysis.models import (
    RATIONALE_ANALYSIS_FLAG_KEYS,
    RationaleAnalysisFlagResult,
    RationaleAnalysisParsedResponse,
)

# Required fields within each flag object.
_FLAG_FIELDS: tuple[str, ...] = (
    "analysis",
    "classification",
)


def _validate_flag(flag_key: str, raw: Any) -> RationaleAnalysisFlagResult:
    """Validate and construct a single flag result from raw JSON.

    Raises ValueError with a descriptive message on any schema violation.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"'{flag_key}' must be an object, got {type(raw).__name__}"
        )

    missing = [f for f in _FLAG_FIELDS if f not in raw]
    if missing:
        raise ValueError(
            f"'{flag_key}' missing required fields: {missing}"
        )

    analysis = raw["analysis"]
    if not isinstance(analysis, str):
        raise ValueError(
            f"'{flag_key}.analysis' must be a string, "
            f"got {type(analysis).__name__}"
        )

    classification = raw["classification"]
    if not isinstance(classification, bool):
        raise ValueError(
            f"'{flag_key}.classification' must be a boolean, "
            f"got {type(classification).__name__}"
        )

    return RationaleAnalysisFlagResult(
        analysis=analysis,
        classification=classification,
    )


def parse_rationale_analysis_response(
    raw_response_text: str,
) -> tuple[RationaleAnalysisParsedResponse | None, str | None]:
    """Parse a rationale analyst's JSON response into structured fields.

    Every response carries the seven binary flags named by
    ``RATIONALE_ANALYSIS_FLAG_KEYS``, each an object with an ``analysis``
    string and a ``classification`` boolean.

    Returns ``(parsed_response, parse_error)`` where:
      - On success: ``parsed_response`` is populated, ``parse_error`` is None
      - On failure: ``parsed_response`` is None, ``parse_error`` describes
        what went wrong
    """
    try:
        data = json.loads(raw_response_text)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"

    if not isinstance(data, dict):
        return None, f"Expected JSON object, got {type(data).__name__}"

    missing_flags = [k for k in RATIONALE_ANALYSIS_FLAG_KEYS if k not in data]
    if missing_flags:
        return None, f"Missing required flag keys: {missing_flags}"

    try:
        flags = {
            flag_key: _validate_flag(flag_key, data[flag_key])
            for flag_key in RATIONALE_ANALYSIS_FLAG_KEYS
        }
    except ValueError as e:
        return None, str(e)

    return RationaleAnalysisParsedResponse(**flags), None
