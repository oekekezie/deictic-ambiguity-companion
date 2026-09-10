"""Utilities for JSON schema resolution and model family identification."""

import copy
import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel


def resolve_schema(schema: type[BaseModel] | dict[str, Any] | str) -> dict[str, Any]:
    """Convert a schema specification to a plain JSON Schema dict.

    Always returns a fresh copy so callers can safely mutate the result
    (e.g., injecting additionalProperties) without affecting the original.

    Accepts:
        - A Pydantic BaseModel subclass -> calls .model_json_schema()
        - A dict -> deep-copied to prevent mutation of the original
        - A JSON string -> parsed via json.loads()

    Raises:
        TypeError: If schema is not one of the supported types.
        json.JSONDecodeError: If a string schema is not valid JSON.
        ValueError: If a string schema parses to a non-dict value.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return copy.deepcopy(schema)
    if isinstance(schema, str):
        result = json.loads(schema)
        if not isinstance(result, dict):
            raise ValueError(
                f"JSON schema string must parse to a dict (JSON object), "
                f"got {type(result).__name__}"
            )
        return result
    raise TypeError(f"Unsupported schema type: {type(schema).__name__}")


def find_model_family(model_id: str, known_families: Iterable[str]) -> str | None:
    """Find which known family a model ID belongs to via substring matching.

    Checks longest family names first to avoid ambiguity (e.g., "gpt-5-mini"
    matches before "gpt-5" when the model ID is "gpt-5-mini-2025-08-07").

    Returns the matching family string, or None if no match.
    """
    for family in sorted(known_families, key=len, reverse=True):
        if family in model_id:
            return family
    return None
