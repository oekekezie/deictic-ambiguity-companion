"""Fireworks AI SDK integration for batch inference.

Async wrappers around the Fireworks HTTP API via httpx for submitting batch
inference jobs, checking their status, and retrieving results. File upload
uses multipart/form-data. Sampling parameters are set at the batch job level
via ``inferenceParameters``; reasoning parameters are set per-request in
the JSONL body (see ``serializers.fireworks``).
"""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

from utils.batch_inference.llm_configs.fireworks import FireworksLLMConfig

_BASE_URL = "https://api.fireworks.ai/v1"

# Generous read/write timeout for file upload and download operations.
# The connect timeout is tighter since connection setup should be fast.
_TRANSFER_TIMEOUT = httpx.Timeout(300.0, connect=30.0)

# Standard timeout for lightweight metadata calls (status checks, cancel).
_METADATA_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def _extract_short_id(resource_name: str) -> str:
    """Extract the short ID from a Fireworks full resource name.

    Fireworks API responses use hierarchical resource names like
    ``accounts/{account_id}/batchInferenceJobs/{job_id}``. The REST
    endpoints expect only the short ID in the URL path. Returns the
    final path segment, or the input unchanged if already a short ID.
    """
    return resource_name.rsplit("/", 1)[-1]


def _get_auth_headers() -> dict[str, str]:
    """Build authorization headers, raising ValueError if the API key is missing.

    Returns only the Authorization header. Content-Type is intentionally
    omitted — httpx sets it automatically for ``json=...`` (application/json)
    and ``files=...`` (multipart/form-data) kwargs.
    """
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError(
            "FIREWORKS_API_KEY environment variable is required "
            "for Fireworks batch submission"
        )
    return {"Authorization": f"Bearer {api_key}"}


def _get_account_id() -> str:
    """Get the Fireworks account ID from the environment."""
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not account_id:
        raise ValueError(
            "FIREWORKS_ACCOUNT_ID environment variable is required "
            "for Fireworks batch submission"
        )
    return account_id


def _raise_for_status(response: httpx.Response) -> None:
    """Raise an HTTPStatusError that includes the full response body.

    httpx's built-in raise_for_status() only captures the status code and URL,
    discarding the response body that contains the actual API error details.
    """
    if response.is_success:
        return
    raise httpx.HTTPStatusError(
        f"{response.status_code} {response.reason_phrase} for url "
        f"'{response.url}': {response.text}",
        request=response.request,
        response=response,
    )


def _join_byte_chunks(chunks: list[bytes]) -> bytes:
    """Join byte chunks, ensuring newline termination for JSONL integrity.

    Guarantees each chunk ends with ``\\n`` before joining, preventing
    JSON line fusion when concatenating multiple downloaded files.
    """
    if not chunks:
        return b""
    return b"".join(
        chunk if chunk.endswith(b"\n") else chunk + b"\n"
        for chunk in chunks
    )


def _count_jsonl_lines(path: Path) -> int:
    """Count non-blank lines in a JSONL file to determine the example count.

    Blank lines (empty or whitespace-only) are excluded to match the
    parsing convention in ``parse_jsonl_bytes`` and to avoid inflating
    the count from trailing newlines.
    """
    with path.open() as f:
        return sum(1 for line in f if line.strip())


async def _create_dataset(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    account_id: str,
    dataset_id: str,
    example_count: int,
) -> str:
    """Create a Fireworks dataset and return its resource name.

    The ``exampleCount`` field (string-encoded int64) is required on the
    nested ``dataset`` object per the Fireworks Create Dataset API spec.
    Accepts an ``httpx.AsyncClient`` for connection reuse and testability.
    """
    response = await client.post(
        f"{_BASE_URL}/accounts/{account_id}/datasets",
        headers=headers,
        json={
            "datasetId": dataset_id,
            "dataset": {
                "userUploaded": {},
                "exampleCount": str(example_count),
            },
        },
    )
    _raise_for_status(response)
    return response.json()["name"]


def _build_inference_parameters(llm_config: FireworksLLMConfig) -> dict[str, Any]:
    """Build the ``inferenceParameters`` object for the batch job submission.

    First-class fields (``maxTokens``, ``temperature``, ``topP``, ``topK``)
    map directly to camelCase keys per the Fireworks API spec. Non-first-class
    sampling parameters (``min_p``, ``presence_penalty``) are packed into the
    ``extraBody`` JSON string. Reasoning parameters (``reasoning_effort``,
    ``thinking``, ``reasoning_history``) are set per-request in the JSONL
    body — not here.
    """
    params: dict[str, Any] = {}

    # First-class inference parameters (camelCase per Fireworks API spec)
    if llm_config.max_tokens is not None:
        params["maxTokens"] = llm_config.max_tokens
    if llm_config.temperature is not None:
        params["temperature"] = llm_config.temperature
    if llm_config.top_p is not None:
        params["topP"] = llm_config.top_p
    if llm_config.top_k is not None:
        params["topK"] = llm_config.top_k

    # Non-first-class sampling params packed into extraBody as a JSON string
    extra: dict[str, Any] = {}
    if llm_config.min_p is not None:
        extra["min_p"] = llm_config.min_p
    if llm_config.presence_penalty is not None:
        extra["presence_penalty"] = llm_config.presence_penalty

    if extra:
        params["extraBody"] = json.dumps(extra)

    return params


async def submit_fireworks(input_path: Path, llm_config: FireworksLLMConfig) -> str:
    """Create the input dataset, upload the JSONL, and start a batch job.

    Sampling parameters are set at the batch job level via
    ``inferenceParameters`` on the ``batchInferenceJobs`` request.
    Reasoning parameters are already embedded per-request in the JSONL
    body by the serializer. Only the input dataset is pre-created (with
    data uploaded); the batch inference API creates the output dataset
    itself from the resource name supplied in ``outputDatasetId``.
    """
    headers = _get_auth_headers()
    account_id = _get_account_id()
    example_count = _count_jsonl_lines(input_path)

    # Unique dataset IDs to avoid collisions across jobs
    job_suffix = uuid.uuid4().hex[:8]

    async with httpx.AsyncClient(timeout=_TRANSFER_TIMEOUT) as client:
        # Create input dataset and upload file via multipart/form-data
        input_dataset_name = await _create_dataset(
            client, headers, account_id, f"batch-input-{job_suffix}", example_count
        )
        input_dataset_id = _extract_short_id(input_dataset_name)

        with input_path.open("rb") as f:
            upload_response = await client.post(
                f"{_BASE_URL}/accounts/{account_id}/datasets/{input_dataset_id}:upload",
                headers=headers,
                files={"file": (input_path.name, f, "application/octet-stream")},
            )
        _raise_for_status(upload_response)

        # Output dataset resource name — NOT pre-created; the batch job API
        # creates it when the job is submitted.
        output_dataset_name = (
            f"accounts/{account_id}/datasets/batch-output-{job_suffix}"
        )

        payload: dict[str, Any] = {
            "model": llm_config.model,
            "inputDatasetId": input_dataset_name,
            "outputDatasetId": output_dataset_name,
        }

        inference_params = _build_inference_parameters(llm_config)
        if inference_params:
            payload["inferenceParameters"] = inference_params

        job_response = await client.post(
            f"{_BASE_URL}/accounts/{account_id}/batchInferenceJobs",
            headers=headers,
            json=payload,
        )
        _raise_for_status(job_response)

    # The API returns a full resource name (accounts/.../batchInferenceJobs/{id});
    # extract just the short ID so it can be used directly in URL path segments.
    return _extract_short_id(job_response.json()["name"])


async def check_status_fireworks(job_id: str) -> dict[str, Any]:
    """Retrieve current batch job status. Single API call, returns immediately."""
    headers = _get_auth_headers()
    account_id = _get_account_id()
    short_id = _extract_short_id(job_id)

    async with httpx.AsyncClient(timeout=_METADATA_TIMEOUT) as client:
        response = await client.get(
            f"{_BASE_URL}/accounts/{account_id}/batchInferenceJobs/{short_id}",
            headers=headers,
        )
    _raise_for_status(response)
    return response.json()


async def cancel_fireworks(job_id: str) -> None:
    """Delete (cancel) a batch inference job via the Fireworks REST API.

    Fireworks uses HTTP DELETE rather than a dedicated cancel endpoint.
    The job transitions through DELETING/DELETING_CLEANING_UP states.
    """
    headers = _get_auth_headers()
    account_id = _get_account_id()
    short_id = _extract_short_id(job_id)

    async with httpx.AsyncClient(timeout=_METADATA_TIMEOUT) as client:
        response = await client.delete(
            f"{_BASE_URL}/accounts/{account_id}/batchInferenceJobs/{short_id}",
            headers=headers,
        )
    _raise_for_status(response)


async def retrieve_fireworks(
    job_id: str, output_path: Path, errors_path: Path | None = None
) -> None:
    """Download results and errors from a completed or expired batch job.

    The Fireworks batch output dataset contains a results file and an
    errors file. Files are routed by basename: those containing "error"
    go to ``errors_path`` (when provided), all others to ``output_path``.
    Signed-URL downloads run concurrently via ``asyncio.gather`` to
    minimize retrieval latency. Newline termination is enforced between
    concatenated chunks to preserve JSONL line boundaries.
    """
    headers = _get_auth_headers()
    account_id = _get_account_id()
    short_id = _extract_short_id(job_id)

    async with httpx.AsyncClient(timeout=_TRANSFER_TIMEOUT) as client:
        # Get the output dataset ID from the job
        job_response = await client.get(
            f"{_BASE_URL}/accounts/{account_id}/batchInferenceJobs/{short_id}",
            headers=headers,
        )
        _raise_for_status(job_response)
        # outputDatasetId may be a full resource name; extract just the short ID
        output_dataset_id = _extract_short_id(job_response.json()["outputDatasetId"])

        # The Fireworks getDownloadEndpoint API returns ``filenameToSignedUrls``,
        # a map of object-path → signed URL for each file in the dataset.
        download_response = await client.get(
            f"{_BASE_URL}/accounts/{account_id}/datasets/{output_dataset_id}:getDownloadEndpoint",
            headers=headers,
        )
        _raise_for_status(download_response)
        signed_urls: dict[str, str] = download_response.json()["filenameToSignedUrls"]

        if not signed_urls:
            raise ValueError(
                f"No download files available for dataset '{output_dataset_id}'"
            )

        # Download all files concurrently
        async def _download(url: str) -> httpx.Response:
            resp = await client.get(url)
            _raise_for_status(resp)
            return resp

        responses = await asyncio.gather(
            *[_download(url) for url in signed_urls.values()]
        )

    # Route files: "error" in basename → errors, else → results
    results_chunks: list[bytes] = []
    errors_chunks: list[bytes] = []

    for object_path, response in zip(signed_urls.keys(), responses):
        if not response.content:
            continue

        basename = object_path.rsplit("/", 1)[-1].lower()
        if "error" in basename and errors_path is not None:
            errors_chunks.append(response.content)
        else:
            results_chunks.append(response.content)

    output_path.write_bytes(_join_byte_chunks(results_chunks))
    if errors_path is not None and errors_chunks:
        errors_path.write_bytes(_join_byte_chunks(errors_chunks))
