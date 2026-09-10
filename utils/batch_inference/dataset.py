"""Dataset model containing a collection of examples."""

import json

from pydantic import BaseModel, model_validator

from utils.batch_inference.example import Example


class Dataset(BaseModel):
    """A batch of examples to be submitted for inference.

    Validates that the list is non-empty and all custom_id values are unique.
    """

    examples: list[Example]

    @model_validator(mode="after")
    def validate_examples(self) -> "Dataset":
        if not self.examples:
            raise ValueError("examples must be non-empty")
        seen: set[str] = set()
        duplicates: list[str] = []
        for ex in self.examples:
            if ex.custom_id in seen:
                duplicates.append(ex.custom_id)
            seen.add(ex.custom_id)
        if duplicates:
            raise ValueError(f"Duplicate custom_id values: {duplicates}")
        return self


def parse_jsonl_bytes(raw: bytes) -> Dataset:
    """Parse uploaded JSONL bytes into a validated Dataset.

    Each line must be a JSON object matching the Example schema.
    Raises ValueError with descriptive messages for parse or validation errors.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"File is not valid UTF-8: {e}") from e
    lines = text.splitlines()
    examples: list[Example] = []
    errors: list[str] = []

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: invalid JSON — {e}")
            continue
        try:
            examples.append(Example(**data))
        except Exception as e:
            custom_id = data.get("custom_id", f"line {i}")
            errors.append(f"Example '{custom_id}': {e}")

    if errors:
        raise ValueError("\n".join(errors))
    if not examples:
        raise ValueError("File contains no valid examples")

    return Dataset(examples=examples)
