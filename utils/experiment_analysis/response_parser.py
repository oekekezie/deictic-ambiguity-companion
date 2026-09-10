"""Parse experiment meta-evaluator JSON responses into score and rationale.

Validates raw response text against the experiment response schema
(``synthetic_dataset/response_schema.json``) and extracts the binary
verdict and its justification. This is the experiment domain analogue
of ``parse_rationale_analysis_response()`` in the rationale analysis
layer -- a domain-specific Layer 2 parser applied at consumption time,
not during generic envelope extraction.
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Valid experiment score values, matching the response schema's enum
_VALID_SCORES: frozenset[str] = frozenset({"correct", "incorrect"})


class ExperimentParsedResponse(BaseModel):
    """Parsed output from one experiment meta-evaluator trial response.

    Contains the binary verdict (did the grader get it right?) and
    the model's stated justification. Both fields are required and
    must be the correct types for a parse to succeed.
    """

    model_config = ConfigDict(frozen=True)

    score: Literal["correct", "incorrect"]
    rationale: str


def parse_experiment_response(
    raw_response_text: str,
) -> tuple[ExperimentParsedResponse | None, str | None]:
    """Parse an experiment response's JSON into score and rationale.

    Returns ``(parsed_response, parse_error)`` where:
      - On success: ``parsed_response`` is populated, ``parse_error`` is None
      - On failure: ``parsed_response`` is None, ``parse_error`` describes
        what went wrong

    Validates that ``score`` is one of ``{"correct", "incorrect"}`` and
    ``rationale`` is a string. Any structural or type violation is a
    total failure -- there is no partial-success case.
    """
    try:
        data = json.loads(raw_response_text)
    except json.JSONDecodeError as e:
        return None, f"JSON decode failed: {e}"

    if not isinstance(data, dict):
        return None, f"Expected JSON object, got {type(data).__name__}"

    score = data.get("score")
    if score not in _VALID_SCORES:
        return None, f"Invalid score value: {score!r}"

    rationale = data.get("rationale")
    if not isinstance(rationale, str):
        return None, (
            f"Invalid rationale type: {type(rationale).__name__}"
            if rationale is not None
            else "Missing required field: rationale"
        )

    return ExperimentParsedResponse(score=score, rationale=rationale), None
