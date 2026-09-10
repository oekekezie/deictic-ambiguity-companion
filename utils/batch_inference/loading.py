"""Job loading and retry healing for processed batch inference results.

Provides public functions to load ProcessedJob artifacts from disk with
automatic retry companion healing. Extracted from the experiment analysis
pipeline to serve as shared infrastructure for any consumer that needs
to load processed jobs — experiment analysis, rationale analysis, etc.
"""

from pathlib import Path

from pydantic import ValidationError

from utils.batch_inference.heal import heal_processed_jobs
from utils.batch_inference.job import JobRecord
from utils.batch_inference.processed import ProcessedJob
from utils.batch_inference.storage import load_index, resolve_job_dir


def build_record_map(
    base_dir: Path | None = None,
) -> dict[str, JobRecord]:
    """Load the storage index and return a job_id → JobRecord mapping.

    Wraps ``load_index()`` with ``prune=False`` (preserves all records
    including those whose directories may have been moved) and builds the
    lookup dict that ``read_processed_job`` and friends expect.
    """
    records = load_index(base_dir=base_dir, prune=False)
    return {r.job_id: r for r in records}


def read_processed_job(
    job_id: str,
    record_map: dict[str, JobRecord],
) -> ProcessedJob:
    """Load a ProcessedJob from disk using a pre-loaded index record map.

    Separates the per-job filesystem lookup from the index loading,
    so callers that process multiple jobs can load the index once.
    """
    if job_id not in record_map:
        raise ValueError(
            f"Job {job_id!r} not found in storage index. "
            f"Available: {sorted(record_map.keys())[:10]}..."
        )

    record = record_map[job_id]
    job_dir = resolve_job_dir(record.job_dir)
    processed_path = job_dir / "processed.json"

    if not processed_path.exists():
        raise FileNotFoundError(
            f"No processed.json found for job {job_id} at {processed_path}. "
            f"Run process_job_output first."
        )

    return ProcessedJob.model_validate_json(processed_path.read_text())


def find_retry_companions(
    job_id: str,
    record_map: dict[str, JobRecord],
) -> list[str]:
    """Find all retry job_ids that reference the given original job_id.

    Scans the record_map for entries whose ``retry_of`` field matches,
    returning them sorted by created_at for deterministic merge order
    (earliest retry applied first).
    """
    companions = [r for r in record_map.values() if r.retry_of == job_id]
    companions.sort(key=lambda r: r.created_at)
    return [r.job_id for r in companions]


def read_healed_job(
    job_id: str,
    record_map: dict[str, JobRecord],
) -> tuple[ProcessedJob, tuple[str, ...]]:
    """Load a ProcessedJob, applying any retry companion healing.

    If the job has retry companions (other jobs whose ``retry_of``
    points here), loads their ProcessedJobs and merges successful
    retry results into the original's failed results. Retries that
    cannot be loaded — not yet processed (missing processed.json) or
    with corrupted/unparseable data (e.g., Git LFS pointers) — are
    silently skipped so a single bad companion cannot block loading
    of the original job.

    Returns ``(job, contributing_retry_job_ids)``: the (possibly healed)
    job and the sorted job_ids of the retries whose results actually
    landed. The tuple is empty when the job has no loadable retry
    companion or none of them heal anything.
    """
    original = read_processed_job(job_id, record_map)

    retry_ids = find_retry_companions(job_id, record_map)
    if not retry_ids:
        return original, ()

    retry_jobs: list[ProcessedJob] = []
    for retry_id in retry_ids:
        try:
            retry_jobs.append(read_processed_job(retry_id, record_map))
        except (FileNotFoundError, ValidationError):
            # Retry not yet processed or data corrupted/LFS pointer — skip
            continue

    if not retry_jobs:
        return original, ()

    return heal_processed_jobs(original, retry_jobs)
