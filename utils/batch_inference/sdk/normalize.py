"""Normalize provider-specific batch status responses to JobRecord field updates.

Each provider's check_status function returns a raw dict with different key
names, status enum values, and timestamp formats. This module provides a
single dispatch function that maps any provider's status dict into a
consistent set of JobRecord-compatible field updates.
"""

from datetime import datetime, timezone
from typing import Any, Literal

# -- Status mapping tables ----------------------------------------------------

_OPENAI_STATUS: dict[str, str] = {
    "validating": "submitted",
    "in_progress": "in_progress",
    "finalizing": "in_progress",
    "completed": "completed",
    "failed": "failed",
    "expired": "expired",
    "cancelling": "cancelled",
    "cancelled": "cancelled",
}

# Gemini may return either JOB_STATE_ or BATCH_STATE_ prefixes;
# we strip the prefix and map the bare suffix.
_GEMINI_STATUS: dict[str, str] = {
    "UNSPECIFIED": "submitted",
    "QUEUED": "submitted",
    "PENDING": "submitted",
    "RUNNING": "in_progress",
    "PAUSED": "in_progress",
    "UPDATING": "in_progress",
    "SUCCEEDED": "completed",
    "PARTIALLY_SUCCEEDED": "completed",
    "FAILED": "failed",
    "EXPIRED": "expired",
    "CANCELLING": "cancelled",
    "CANCELLED": "cancelled",
}

_FIREWORKS_STATUS: dict[str, str] = {
    "UNSPECIFIED": "submitted",
    "CREATING": "submitted",
    "PENDING": "submitted",
    "VALIDATING": "submitted",
    "CREATING_INPUT_DATASET": "submitted",
    "RUNNING": "in_progress",
    "WRITING_RESULTS": "in_progress",
    "IDLE": "in_progress",
    "PAUSED": "in_progress",
    "RE_QUEUEING": "in_progress",
    "COMPLETED": "completed",
    "EARLY_STOPPED": "completed",
    "FAILED": "failed",
    "EXPIRED": "expired",
    "CANCELLING": "cancelled",
    "CANCELLED": "cancelled",
    "DELETING": "cancelled",
    "DELETING_CLEANING_UP": "cancelled",
}


# -- Shared helpers -----------------------------------------------------------


def _ensure_iso_string(value: datetime | str | None) -> str | None:
    """Convert a datetime to ISO 8601 string; pass through strings and None.

    The Gemini and Anthropic SDKs return datetime objects from model_dump(),
    but JobRecord.completed_at expects a plain string.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# -- Provider helpers ---------------------------------------------------------


def _normalize_openai(raw: dict[str, Any]) -> dict[str, Any]:
    """Map an OpenAI Batch status dict to JobRecord field updates."""
    provider_status = raw["status"]
    status = _OPENAI_STATUS.get(provider_status)
    if status is None:
        raise ValueError(f"Unknown OpenAI batch status: {provider_status!r}")

    # First non-None terminal timestamp, converted from Unix epoch
    completed_at = None
    for key in ("completed_at", "failed_at", "expired_at", "cancelled_at"):
        ts = raw.get(key)
        if ts is not None:
            completed_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            break

    # First error message, if any
    error_message = None
    errors = raw.get("errors")
    if errors is not None:
        data = errors.get("data")
        if data:
            error_message = data[0].get("message")

    return {"status": status, "completed_at": completed_at, "error_message": error_message}


def _normalize_anthropic(raw: dict[str, Any]) -> dict[str, Any]:
    """Map an Anthropic MessageBatch status dict to JobRecord field updates.

    Anthropic's processing_status has only 3 values (in_progress, canceling,
    ended). When ended, the actual outcome is inferred from request_counts.
    """
    processing_status = raw["processing_status"]

    if processing_status == "in_progress":
        status = "in_progress"
    elif processing_status == "canceling":
        status = "cancelled"
    elif processing_status == "ended":
        counts = raw["request_counts"]
        succeeded = counts["succeeded"]
        errored = counts["errored"]
        canceled = counts["canceled"]
        expired = counts["expired"]
        total_resolved = succeeded + errored + canceled + expired

        if total_resolved == 0:
            raise ValueError(
                f"Anthropic batch ended with all counts zero: {counts!r}"
            )
        elif errored == 0 and expired == 0 and succeeded == 0 and canceled > 0:
            status = "cancelled"
        elif errored == 0 and canceled == 0 and succeeded == 0 and expired > 0:
            status = "expired"
        elif succeeded == 0 and errored > 0:
            status = "failed"
        elif succeeded == 0:
            # Mix of canceled/expired with nothing succeeded → cancelled
            status = "cancelled"
        else:
            # Some succeeded (possibly mixed with errors/cancels/expired) → completed
            status = "completed"
    else:
        raise ValueError(f"Unknown Anthropic processing_status: {processing_status!r}")

    # Anthropic errors are per-request in the results stream, not at batch level
    return {
        "status": status,
        "completed_at": _ensure_iso_string(raw.get("ended_at")),
        "error_message": None,
    }


def _strip_state_prefix(state: str) -> str:
    """Strip JOB_STATE_ or BATCH_STATE_ prefix from a state enum string."""
    for prefix in ("BATCH_STATE_", "JOB_STATE_"):
        if state.startswith(prefix):
            return state[len(prefix):]
    return state


def _normalize_gemini(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a Gemini BatchJob status dict to JobRecord field updates."""
    bare = _strip_state_prefix(raw["state"])
    status = _GEMINI_STATUS.get(bare)
    if status is None:
        raise ValueError(f"Unknown Gemini batch state: {raw['state']!r}")

    error_message = None
    error = raw.get("error")
    if error is not None:
        error_message = error.get("message")

    return {
        "status": status,
        "completed_at": _ensure_iso_string(raw.get("end_time")),
        "error_message": error_message,
    }


def _normalize_fireworks(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a Fireworks batch job status dict to JobRecord field updates."""
    bare = _strip_state_prefix(raw["state"])
    status = _FIREWORKS_STATUS.get(bare)
    if status is None:
        raise ValueError(f"Unknown Fireworks batch state: {raw['state']!r}")

    # The Fireworks API has no dedicated completion timestamp; use updateTime
    # (last state-transition timestamp) only when the job has reached a
    # terminal state, so in-progress updateTime values don't leak through.
    is_terminal = status not in {"submitted", "in_progress"}

    # Extract error details from the gatewayStatus object (Google RPC Status
    # pattern: status.code + status.message). Only populate for non-success
    # terminal states — unlike OpenAI/Gemini where the error container is
    # absent on success, the Fireworks status object is always present
    # (it carries the status code), so we must gate explicitly.
    error_message = None
    if is_terminal and status != "completed":
        status_obj = raw.get("status")
        if status_obj is not None:
            error_message = status_obj.get("message")

    return {
        "status": status,
        "completed_at": raw.get("updateTime") if is_terminal else None,
        "error_message": error_message,
    }


# -- Public dispatch ----------------------------------------------------------

_PROVIDERS: dict[str, Any] = {
    "openai": _normalize_openai,
    "anthropic": _normalize_anthropic,
    "gemini": _normalize_gemini,
    "fireworks": _normalize_fireworks,
}


def normalize_status(
    provider: Literal["openai", "anthropic", "gemini", "fireworks"],
    raw: dict[str, Any],
) -> dict[str, Any]:
    """Map a raw provider status dict to normalized JobRecord field updates.

    Returns a dict suitable for ``record.model_copy(update=normalize_status(...))``.

    Guaranteed keys in the returned dict:
        - ``status``: one of the JobRecord status literals
        - ``completed_at``: ISO 8601 string or None
        - ``error_message``: string or None

    Raises ValueError for unknown provider names or unrecognized status values.
    """
    handler = _PROVIDERS.get(provider)
    if handler is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    return handler(raw)
