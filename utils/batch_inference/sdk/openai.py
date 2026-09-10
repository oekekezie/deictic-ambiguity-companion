"""OpenAI SDK integration for batch inference.

Async wrappers around the OpenAI Python SDK for submitting batch jobs,
checking their status, and retrieving results.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


def _get_client() -> AsyncOpenAI:
    """Create an async OpenAI client, raising ValueError if the API key is missing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for OpenAI batch submission"
        )
    return AsyncOpenAI(api_key=api_key)


async def submit_openai(input_path: Path) -> str:
    """Upload the JSONL file and create a batch job.

    Returns the batch job ID.
    """
    client = _get_client()

    with input_path.open("rb") as f:
        uploaded_file = await client.files.create(file=f, purpose="batch")

    batch_job = await client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    return batch_job.id


async def check_status_openai(job_id: str) -> dict[str, Any]:
    """Retrieve current batch job status. Single API call, returns immediately."""
    client = _get_client()
    batch = await client.batches.retrieve(job_id)
    return batch.model_dump()


async def cancel_openai(job_id: str) -> None:
    """Request cancellation of an in-progress batch job.

    The batch transitions to "cancelling" and then "cancelled" (up to 10 min).
    Partial results may exist but are intentionally not retrieved.
    """
    client = _get_client()
    await client.batches.cancel(job_id)


async def retrieve_openai(job_id: str, output_path: Path, errors_path: Path) -> None:
    """Download output and error files from a completed or expired batch job.

    Downloads output and error files concurrently when both are available.
    For expired batches, output_file_id may be None (zero completions)
    while error_file_id contains expired requests with code "batch_expired".
    """
    client = _get_client()
    batch = await client.batches.retrieve(job_id)

    async def _download(file_id: str) -> bytes:
        content = await client.files.content(file_id)
        return content.read()

    # Download available files concurrently — gather manages all task
    # lifetimes atomically, preventing orphaned tasks if one fails.
    pending = {}
    if batch.output_file_id:
        pending["output"] = _download(batch.output_file_id)
    if batch.error_file_id:
        pending["errors"] = _download(batch.error_file_id)

    if pending:
        fetched = await asyncio.gather(*pending.values())
        results = dict(zip(pending.keys(), fetched))
    else:
        results = {}

    if "output" in results:
        output_path.write_bytes(results["output"])
    if "errors" in results:
        errors_path.write_bytes(results["errors"])
