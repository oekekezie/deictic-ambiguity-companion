"""Anthropic SDK integration for batch inference.

Async wrappers around the Anthropic Python SDK for submitting message batches,
checking their status, and retrieving results.

For 4.5-era models with effort set, the beta namespace is required
(client.beta.messages.batches). For 4.6+ models and 4.5 models without
effort, the stable namespace is used (client.messages.batches).
"""

import json
import os
from pathlib import Path
from typing import Any

import anthropic

from utils.batch_inference.llm_configs.anthropic import (
    _ANTHROPIC_EFFORT_BETA_FAMILIES,
    _ANTHROPIC_KNOWN_FAMILIES,
)
from utils.batch_inference.schema_utils import find_model_family

_EFFORT_BETA_HEADER = "effort-2025-11-24"


def _get_client() -> anthropic.AsyncAnthropic:
    """Create an async Anthropic client, raising ValueError if the API key is missing."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required "
            "for Anthropic batch submission"
        )
    return anthropic.AsyncAnthropic(api_key=api_key)


def _requires_effort_beta(requests: list[dict[str, Any]]) -> bool:
    """Determine if requests need the effort beta header.

    Inspects the first request to check if the model is a 4.5-era family
    that requires the beta namespace for effort, and if effort is actually set.
    """
    if not requests:
        return False
    params = requests[0].get("params", {})
    model = params.get("model", "")
    effort = params.get("output_config", {}).get("effort")
    if effort is None:
        return False
    family = find_model_family(model, _ANTHROPIC_KNOWN_FAMILIES)
    return family in _ANTHROPIC_EFFORT_BETA_FAMILIES


async def submit_anthropic(input_path: Path) -> str:
    """Read JSONL and submit as a message batch.

    Anthropic's batch API accepts an in-memory list of requests,
    not a file upload. Returns the batch ID.
    """
    client = _get_client()

    requests = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]

    # 4.5-era models with effort require the beta namespace
    if _requires_effort_beta(requests):
        batch = await client.beta.messages.batches.create(
            betas=[_EFFORT_BETA_HEADER],
            requests=requests,
        )
    else:
        batch = await client.messages.batches.create(requests=requests)

    return batch.id


async def check_status_anthropic(job_id: str) -> dict[str, Any]:
    """Retrieve current batch status. Single API call, returns immediately."""
    client = _get_client()
    batch = await client.messages.batches.retrieve(job_id)
    return batch.model_dump()


async def cancel_anthropic(job_id: str) -> None:
    """Request cancellation of an in-progress message batch.

    The batch enters "canceling" state; in-progress non-interruptible requests
    may complete before the batch fully ends.
    """
    client = _get_client()
    await client.messages.batches.cancel(job_id)


async def retrieve_anthropic(job_id: str, output_path: Path, errors_path: Path) -> None:
    """Stream results from a completed or expired batch, separating successes from errors.

    Only results with type "succeeded" are written to output_path.
    All other types (errored, expired, canceled) go to errors_path.
    For fully expired batches, output_path will be empty and all
    entries land in errors_path.
    """
    client = _get_client()

    successes: list[str] = []
    errors: list[str] = []

    async for result in await client.messages.batches.results(job_id):
        line = json.dumps(result.model_dump(), ensure_ascii=False)
        if result.result.type == "succeeded":
            successes.append(line)
        else:
            errors.append(line)

    output_path.write_text("\n".join(successes) + "\n" if successes else "")
    if errors:
        errors_path.write_text("\n".join(errors) + "\n")
