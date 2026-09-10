"""Heal (merge) retry results into original job results.

Produces a merged ProcessedJob where failed results from the original
are replaced by successful results from the retry companions, and
batch errors (API-level failures from errors.jsonl) with successful
retries are promoted into the results list. The healed job retains
the original's job_id so it remains transparent to the manifest and
downstream analysis pipeline; alongside it, the merge reports which
retry job_ids actually contributed a landed result, so snapshot
provenance can record the retries that produced part of the output.

Multiple retry companions are supported: they are applied in list order
(typically sorted by created_at), with later retries overriding earlier
ones for the same custom_id.
"""

from utils.batch_inference.processed import BatchError, ProcessedJob, ProcessedResult


def heal_processed_jobs(
    original: ProcessedJob,
    retries: list[ProcessedJob],
) -> tuple[ProcessedJob, tuple[str, ...]]:
    """Replace failed original results with successful retry results.

    For each result in the original where ``parse_error is not None``,
    checks if any retry job produced a result with the same custom_id
    and ``parse_error is None``. If so, the retry's result replaces
    the original's.

    Batch errors whose custom_id was successfully retried are removed
    from ``batch_errors`` and the retry result is appended to ``results``
    (promoting the formerly-missing trial into the results list).

    Returns ``(healed_job, contributing_retry_job_ids)`` where the second
    element is the sorted, de-duplicated job_ids of the retries whose
    results actually landed — replacing a failure or promoting a batch
    error. When no retries are provided or none of them land a result,
    returns the original unchanged with an empty tuple.
    """
    if not retries:
        return original, ()

    # Build replacement map: custom_id -> (source retry job_id, best result).
    # Later retries in the list override earlier ones for the same id, so the
    # stored job_id is the one whose result actually wins (last-wins).
    replacement_map: dict[str, tuple[str, ProcessedResult]] = {}
    for retry in retries:
        for result in retry.results:
            if result.parse_error is None:
                replacement_map[result.custom_id] = (retry.job_id, result)

    # Retry job_ids whose result actually lands in the healed output.
    contributing: set[str] = set()

    # Swap failed originals for successful retries
    healed_results: list[ProcessedResult] = []
    results_healed = 0
    for result in original.results:
        if result.parse_error is not None and result.custom_id in replacement_map:
            source_job_id, replacement = replacement_map[result.custom_id]
            healed_results.append(replacement)
            contributing.add(source_job_id)
            results_healed += 1
        else:
            healed_results.append(result)

    # Heal batch errors: successful retries promote to results
    remaining_batch_errors: list[BatchError] = []
    for error in original.batch_errors:
        if error.custom_id in replacement_map:
            source_job_id, replacement = replacement_map[error.custom_id]
            healed_results.append(replacement)
            contributing.add(source_job_id)
        else:
            remaining_batch_errors.append(error)
    batch_errors_healed = len(original.batch_errors) - len(remaining_batch_errors)

    if results_healed == 0 and batch_errors_healed == 0:
        return original, ()

    # Recompute tallies after replacement
    successful = sum(1 for r in healed_results if r.parse_error is None)
    failed = sum(1 for r in healed_results if r.parse_error is not None)

    healed = original.model_copy(
        update={
            "results": healed_results,
            "batch_errors": remaining_batch_errors,
            "total_results": len(healed_results),
            "successful_parses": successful,
            "failed_parses": failed,
        }
    )
    return healed, tuple(sorted(contributing))
