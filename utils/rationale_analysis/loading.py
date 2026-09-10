"""Load and flatten rationale analysis batch job results.

Provides the entry point for loading processed rationale analysis jobs
from disk, validating K-consistency, and flattening results into a single
list for downstream aggregation via ``build_rationale_analysis_results()``.
"""

from pathlib import Path

from pydantic import ValidationError

from utils.batch_inference.loading import build_record_map, read_healed_job
from utils.batch_inference.processed import ProcessedJob, ProcessedResult


def load_rationale_analysis_jobs(
    job_ids: list[str],
    base_dir: Path | None = None,
) -> tuple[list[ProcessedResult], int, list[ProcessedJob], tuple[str, ...]]:
    """Load rationale analysis batch jobs, validate K consistency, flatten results.

    Loads each job with retry healing applied, validates that all jobs
    share the same ``num_trials`` (K), and flattens all ``ProcessedResult``
    records into a single list suitable for ``build_rationale_analysis_results()``.

    Returns ``(all_results, k, jobs, merged_retry_job_ids)`` where:
    - ``all_results``: flattened list of all ProcessedResults across all jobs
    - ``k``: validated num_trials value (consistent across all jobs)
    - ``jobs``: list of ProcessedJob objects for metadata display
    - ``merged_retry_job_ids``: sorted, de-duplicated job_ids of the retry
      companions whose results were healed into these jobs (empty when no
      retry contributed) — recorded in snapshot provenance
    """
    if not job_ids:
        raise ValueError("At least one job ID is required")

    record_map = build_record_map(base_dir)

    jobs: list[ProcessedJob] = []
    merged_retry_job_ids: set[str] = set()
    for job_id in job_ids:
        try:
            job, contributing = read_healed_job(job_id, record_map)
        except ValidationError as exc:
            # Pydantic validation fails when processed.json is a Git LFS
            # pointer (the file content is a short text stub, not JSON)
            raise ValueError(
                f"Failed to parse processed.json for job {job_id!r}. "
                f"If this repository uses Git LFS, run `git lfs pull` "
                f"to download the actual file contents.\n"
                f"Original error: {exc}"
            ) from exc
        jobs.append(job)
        merged_retry_job_ids.update(contributing)

    # Validate K consistency across all jobs
    k_values = {job.num_trials for job in jobs}
    if len(k_values) != 1:
        per_job = ", ".join(f"{j.job_id}={j.num_trials}" for j in jobs)
        raise ValueError(
            f"All jobs must have the same num_trials (K), "
            f"but got mixed values: {per_job}"
        )

    k = k_values.pop()

    # Flatten results across all jobs
    all_results: list[ProcessedResult] = []
    for job in jobs:
        all_results.extend(job.results)

    return all_results, k, jobs, tuple(sorted(merged_retry_job_ids))
