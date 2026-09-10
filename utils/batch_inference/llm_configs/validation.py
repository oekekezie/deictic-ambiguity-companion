"""Shared validation helpers for LLM configuration models."""


def validate_temperature(value: float | None, *, max_temp: float = 2.0) -> float | None:
    """Validate that temperature falls within [0.0, max_temp].

    Returns the value unchanged if valid or None.
    Raises ValueError if out of range.
    """
    if value is None:
        return None
    if not (0.0 <= value <= max_temp):
        raise ValueError(f"temperature must be between 0.0 and {max_temp}, got {value}")
    return value
