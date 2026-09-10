"""Google Gemini SDK integration for batch inference.

Async wrappers around the google-genai SDK for submitting batch jobs,
checking their status, and retrieving results. Uses the ``client.aio``
property for non-blocking access to the Gemini API.
"""

import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai.types import UploadFileConfig


def _get_client() -> genai.Client:
    """Create a Gemini client, raising ValueError if the API key is missing."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required for Gemini batch submission"
        )
    return genai.Client(api_key=api_key)


async def submit_gemini(input_path: Path, model: str) -> str:
    """Upload the JSONL file and create a batch job.

    Gemini requires the model at job creation time (not in individual requests).
    Returns the batch job name (Gemini's native identifier).
    """
    client = _get_client()

    uploaded_file = await client.aio.files.upload(
        file=str(input_path),
        config=UploadFileConfig(
            display_name=input_path.name,
            mime_type="application/jsonl",
        ),
    )

    batch_job = await client.aio.batches.create(
        model=model,
        src=uploaded_file.name,  # pyright: ignore[reportArgumentType]
    )
    return batch_job.name  # pyright: ignore[reportReturnType]


async def check_status_gemini(job_name: str) -> dict[str, Any]:
    """Retrieve current batch job status. Single API call, returns immediately."""
    client = _get_client()
    batch = await client.aio.batches.get(name=job_name)
    return batch.model_dump()


async def cancel_gemini(job_name: str) -> None:
    """Request cancellation of an in-progress batch job.

    The job transitions to JOB_STATE_CANCELLED and stops processing new requests.
    """
    client = _get_client()
    await client.aio.batches.cancel(name=job_name)


async def retrieve_gemini(job_name: str, output_path: Path) -> None:
    """Download results from a completed or expired batch job.

    Gemini does not produce a separate errors file — errors are
    inline in the results (lines with a ``status`` field).
    For fully expired jobs, the API does not create a destination
    file — output_path will be empty.
    """
    client = _get_client()
    batch = await client.aio.batches.get(name=job_name)

    # Expired or failed jobs may have no destination file
    if batch.dest is None or batch.dest.file_name is None:
        output_path.write_bytes(b"")
        return

    content = await client.aio.files.download(file=batch.dest.file_name)  # pyright: ignore[reportArgumentType]
    output_path.write_bytes(content)
