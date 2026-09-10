"""Retry logic for failed batch inference requests.

Identifies failed requests in a processed job, extracts the corresponding
original inputs, and prepares them for resubmission. The retry job's
JobRecord links back to the original via the ``retry_of`` field, enabling
transparent healing at analysis load time.

Two failure sources are considered:
- ``ProcessedResult`` entries where ``parse_error is not None``
  (envelope extraction or structural validation failures)
- ``BatchError`` entries in ``batch_errors`` (API-level failures from
  errors.jsonl that never produced a usable result)
"""

import json
from pathlib import Path
from typing import Any

from utils.batch_inference.dataset import Dataset
from utils.batch_inference.example import Example
from utils.batch_inference.processed import ProcessedJob


def identify_failed_custom_ids(processed: ProcessedJob) -> list[str]:
    """Extract custom_ids of all failed results and batch errors.

    Collects custom_ids from two sources: results where
    ``parse_error is not None`` (extraction/validation failures) and
    batch errors (API-level failures from errors.jsonl). Returns a
    sorted list for deterministic ordering. An empty list means all
    requests succeeded.
    """
    failed_from_results = [r.custom_id for r in processed.results if r.parse_error is not None]
    failed_from_errors = [e.custom_id for e in processed.batch_errors]
    return sorted(failed_from_results + failed_from_errors)


def load_original_inputs(job_dir: Path) -> list[dict[str, Any]]:
    """Load the input_original.jsonl from a job directory.

    Each line is a dict with keys matching the Example model fields
    (custom_id, messages, response_schema, system_prompt) — the
    provider-agnostic format stored before serialization.
    """
    path = job_dir / "input_original.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No input_original.jsonl found at {path}. "
            f"Cannot build retry input without original requests."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def filter_to_failed(
    original_inputs: list[dict[str, Any]],
    failed_custom_ids: list[str],
) -> list[dict[str, Any]]:
    """Filter original inputs to only those whose custom_id failed.

    Raises ValueError if any failed_custom_id is absent from the
    original inputs, indicating a data integrity problem.
    """
    failed_set = set(failed_custom_ids)
    filtered = [inp for inp in original_inputs if inp["custom_id"] in failed_set]

    found_ids = {inp["custom_id"] for inp in filtered}
    missing = failed_set - found_ids
    if missing:
        raise ValueError(
            f"{len(missing)} failed custom_id(s) not found in original inputs: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )

    return filtered


def build_retry_dataset(filtered_inputs: list[dict[str, Any]]) -> Dataset:
    """Convert filtered input dicts into a Dataset for re-serialization.

    Reconstructs Example objects from the stored dict representation.
    The resulting Dataset can be passed directly to ``serialize()``
    — no ``expand_trials()`` call needed since the custom_ids already
    carry the ``_trial_NNN`` suffix from the original expansion.
    """
    return Dataset(examples=[Example(**inp) for inp in filtered_inputs])


def summarize_failures(processed: ProcessedJob) -> dict[str, int]:
    """Group all failures by error description with counts.

    Includes both validation failures (from results where
    ``parse_error is not None``) and batch errors (from errors.jsonl).
    Batch errors are prefixed with ``"Batch "`` to distinguish them from
    extraction/validation failures. Returns counts in descending order
    (most common error first).
    """
    counts: dict[str, int] = {}
    for r in processed.results:
        if r.parse_error is not None:
            key = r.parse_error
            counts[key] = counts.get(key, 0) + 1
    for e in processed.batch_errors:
        key = f"Batch {e.error_type}: {e.detail}" if e.detail else f"Batch {e.error_type}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
