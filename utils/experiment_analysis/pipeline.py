"""Pipeline orchestration — the entry points for analysis.

Composes loading, enrichment, accumulation, metrics, and comparisons
into cohesive workflows for both mid-experiment decision support and
post-experiment reporting.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from utils.batch_inference.loading import build_record_map, read_healed_job
from utils.batch_inference.processed import ProcessedJob, ProcessedResult
from utils.experiment_analysis.accumulation import (
    accumulate_batch,
    fold_outcomes,
    make_initial_accumulator,
)
from utils.experiment_analysis.comparisons import (
    AdjacentPairDegradation,
    CrossDatasetComparison,
    ModelSizeComparison,
    PairwiseComparison,
    compute_all_comparisons,
    compute_all_cross_dataset_comparisons,
    compute_all_degradation_tests,
    compute_all_model_size_tests,
)
from utils.experiment_analysis.config_identity import resolve_config_key
from utils.experiment_analysis.enrichment import enrich_job
from utils.experiment_analysis.manifest import ExperimentManifest
from utils.experiment_analysis.metrics import (
    cell_metrics,
    configuration_summary,
    ConfigurationSummary,
    CellMetrics,
)
from utils.experiment_analysis.models import (
    AccumulationParams,
    CellAccumulator,
    CellKey,
    TrialOutcome,
)


class ExperimentAnalysisResults(BaseModel):
    """Complete analysis output from a pipeline run.

    Contains everything needed for reporting: per-cell metrics,
    per-configuration summaries, and pairwise comparisons. The
    manifest is included for snapshot reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    manifest: ExperimentManifest | None
    cell_metrics_list: tuple[CellMetrics, ...]
    config_summaries: tuple[ConfigurationSummary, ...]
    comparisons: tuple[PairwiseComparison, ...]
    trial_outcomes: tuple[TrialOutcome, ...]
    degradation_tests: tuple[AdjacentPairDegradation, ...]
    model_size_comparisons: tuple[ModelSizeComparison, ...]


class BatchReport(BaseModel):
    """Report from processing a single new batch mid-experiment.

    Provides a targeted diff of current-standing statuses: which cells
    newly crossed each threshold, which reverted to active, which
    configurations are currently fully resolved, and which remain active.
    """

    model_config = ConfigDict(frozen=True)

    cells_newly_rejected: tuple[CellKey, ...]
    cells_newly_futile: tuple[CellKey, ...]
    cells_reverted_to_active: tuple[CellKey, ...]
    configs_fully_resolved: tuple[str, ...]
    configs_still_active: tuple[str, ...]


def load_experiment(
    manifest: ExperimentManifest,
    base_dir: Path | None = None,
) -> tuple[list[TrialOutcome], dict[tuple[str, int, str], ProcessedResult]]:
    """Load and enrich all trials from a manifest into TrialOutcome records.

    For each manifest entry, loads ProcessedJob files in declared list order
    (job_ids[0] = batch 0, job_ids[1] = batch 1, etc.), validates config_key
    consistency and created_at monotonicity, then enriches each job's results
    into TrialOutcome records. Entries with empty job_ids are skipped.

    Returns a tuple of (enriched outcomes, response lookup). The response
    lookup maps (config_key, batch_index, custom_id) → ProcessedResult,
    preserving the reasoning_text and raw_response_text that
    TrialOutcome does not carry. The composite key is necessary because
    custom_id alone is not globally unique — the same trial IDs are generated
    independently for each config and batch.
    """
    # Load the storage index once — O(M) — rather than per manifest entry
    record_map = build_record_map(base_dir)

    all_outcomes: list[TrialOutcome] = []
    response_lookup: dict[tuple[str, int, str], ProcessedResult] = {}

    for entry in manifest.entries:
        filled_ids = [jid for jid in entry.job_ids if jid]
        if not filled_ids:
            continue

        # Guard: retry jobs must not appear in manifest job_ids.
        # Including a retry alongside its original causes double-counting:
        # the retry is auto-healed into batch 0 AND emitted as a new batch,
        # inflating N by 1 per retry result.  A standalone retry (without
        # its original) is also invalid — it only contains the failed subset
        # of requests, producing wildly wrong N.  Pre-scan all IDs before
        # any file I/O so the guard fires regardless of list position.
        for job_id in filled_ids:
            if job_id in record_map and record_map[job_id].retry_of is not None:
                raise ValueError(
                    f"Job {job_id!r} in entry {entry.config_key!r} is a retry of "
                    f"{record_map[job_id].retry_of!r}. Retry job IDs must not appear "
                    f"in manifest job_ids — they are automatically merged as healing "
                    f"companions when the original job is loaded."
                )

        # Load and validate all jobs for this config (with retry healing)
        jobs: list[ProcessedJob] = []
        for job_id in filled_ids:
            job, _ = read_healed_job(job_id, record_map)
            derived_key = resolve_config_key(job.llm_config)
            if derived_key != entry.config_key:
                raise ValueError(
                    f"Config key mismatch for job {job_id!r}: "
                    f"manifest declares {entry.config_key!r} but "
                    f"job's llm_config resolves to {derived_key!r}"
                )
            jobs.append(job)

        # Validate created_at monotonicity in declared order
        for i in range(1, len(jobs)):
            if jobs[i].created_at < jobs[i - 1].created_at:
                raise ValueError(
                    f"Batch order violation for config {entry.config_key!r}: "
                    f"job {filled_ids[i]!r} "
                    f"(created_at={jobs[i].created_at!r}) precedes "
                    f"{filled_ids[i - 1]!r} "
                    f"(created_at={jobs[i - 1].created_at!r})"
                )

        # Enrich with batch_index from declared list position; preserve
        # the original ProcessedResult for response-level inspection
        for batch_index, job in enumerate(jobs):
            all_outcomes.extend(enrich_job(job, entry.config_key, batch_index))
            for result in job.results:
                response_lookup[
                    (entry.config_key, batch_index, result.custom_id)
                ] = result

    return all_outcomes, response_lookup


def find_dataset_jsonl(dataset_dir: Path, expected_sha256: str) -> Path:
    """Locate the assembled JSONL whose SHA-256 matches the manifest's dataset hash.

    Recursively searches dataset_dir for *_assembled_examples.jsonl files,
    computes each candidate's SHA-256, and returns the first match. This
    ensures the correct dataset variant (e.g. primary vs. ablation) is
    loaded based on the manifest's declared hash — not file position or
    naming convention.

    Raises FileNotFoundError with candidate paths and their hashes when
    no match is found, aiding diagnosis of missing or mismatched datasets.
    """
    candidates = sorted(dataset_dir.rglob("*_assembled_examples.jsonl"))
    seen: list[tuple[Path, str]] = []

    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == expected_sha256:
            return path
        seen.append((path, digest))

    # No match — build a diagnostic message
    if not seen:
        raise FileNotFoundError(
            f"No *_assembled_examples.jsonl files found under {dataset_dir}"
        )

    listing = "\n".join(f"  {p}  ({h[:16]}...)" for p, h in seen)
    raise FileNotFoundError(
        f"No assembled JSONL matches expected SHA-256 "
        f"{expected_sha256[:16]}...\n"
        f"Candidates found:\n{listing}"
    )


def analyze_experiment(
    outcomes: list[TrialOutcome],
    params: AccumulationParams,
    *,
    manifest: ExperimentManifest | None = None,
) -> ExperimentAnalysisResults:
    """Run the full analysis pipeline on enriched trial outcomes.

    1. Group outcomes by CellKey and fold across batches.
    2. Compute CellMetrics from each accumulator.
    3. Aggregate into configuration summaries.
    4. Run pairwise condition comparisons.
    5. Compute adjacent-pair reasoning effort level degradation tests.
    6. Compute within-provider model size comparison tests.

    Cells that do not cross the rejection or futility threshold remain
    "active", signaling that more data could resolve them.
    """
    # Fold all outcomes into per-cell accumulators
    all_accumulators = fold_outcomes(outcomes, params)
    config_keys = sorted({o.config_key for o in outcomes})

    # Compute per-cell metrics
    metrics_list = tuple(
        cell_metrics(acc, alpha=params.alpha)
        for acc in all_accumulators.values()
    )

    # Compute per-configuration summaries
    summaries = tuple(
        configuration_summary(all_accumulators, ck) for ck in config_keys
    )

    # Compute pairwise condition comparisons
    all_comparisons: list[PairwiseComparison] = []
    for config_key in config_keys:
        all_comparisons.extend(
            compute_all_comparisons(outcomes, config_key)
        )

    # Compute adjacent-pair reasoning effort level degradation tests
    degradation_tests = compute_all_degradation_tests(
        outcomes, alpha=params.alpha,
    )

    # Compute within-provider model size comparison tests
    model_size_comparisons = compute_all_model_size_tests(
        outcomes, alpha=params.alpha,
    )

    return ExperimentAnalysisResults(
        manifest=manifest,
        cell_metrics_list=metrics_list,
        config_summaries=summaries,
        comparisons=tuple(all_comparisons),
        trial_outcomes=tuple(outcomes),
        degradation_tests=tuple(degradation_tests),
        model_size_comparisons=tuple(model_size_comparisons),
    )


def process_new_batch(
    prior_accumulators: dict[CellKey, CellAccumulator],
    new_outcomes: list[TrialOutcome],
    params: AccumulationParams,
) -> tuple[dict[CellKey, CellAccumulator], BatchReport]:
    """Fold a new batch into existing accumulators (mid-experiment entry point).

    For each cell that received new outcomes, fold the new batch into
    the existing accumulator. Returns updated accumulators and a report
    diffing each cell's current standing against its prior one — newly
    crossed thresholds and reversions to active alike.

    Produces identical results to running analyze_experiment from scratch
    on the updated manifest — the fold is deterministic.

    Intentional scaffold surface: nothing calls this yet, because the
    notebooks recompute from scratch through ``analyze_experiment``. It is
    kept for mid-experiment what-changed reporting.
    """
    updated = dict(prior_accumulators)
    prior_statuses: dict[CellKey, str] = {
        k: v.status for k, v in prior_accumulators.items()
    }

    # Group new outcomes by cell
    by_cell: dict[CellKey, list[TrialOutcome]] = {}
    for o in new_outcomes:
        key = CellKey(example_id=o.example_id, config_key=o.config_key)
        by_cell.setdefault(key, []).append(o)

    for cell_key, cell_outcomes in by_cell.items():
        if cell_key not in updated:
            updated[cell_key] = make_initial_accumulator(
                cell_key, params.seed_alternative,
            )
        updated[cell_key] = accumulate_batch(
            prior=updated[cell_key],
            outcomes=tuple(cell_outcomes),
            p_null=params.p_null,
            alpha=params.alpha,
            futility_bound=params.futility_bound,
            clamp_epsilon=params.clamp_epsilon,
        )

    # Diff each cell's current standing against its prior one
    newly_rejected: list[CellKey] = []
    newly_futile: list[CellKey] = []
    reverted_to_active: list[CellKey] = []
    for key, acc in updated.items():
        old_status = prior_statuses.get(key, "active")
        if old_status != "rejected" and acc.status == "rejected":
            newly_rejected.append(key)
        elif old_status != "futile" and acc.status == "futile":
            newly_futile.append(key)
        elif old_status != "active" and acc.status == "active":
            reverted_to_active.append(key)

    # Determine which configs are fully resolved
    config_statuses: dict[str, list[str]] = {}
    for key, acc in updated.items():
        config_statuses.setdefault(key.config_key, []).append(acc.status)

    fully_resolved = sorted(
        ck for ck, statuses in config_statuses.items()
        if all(s != "active" for s in statuses)
    )
    still_active = sorted(
        ck for ck, statuses in config_statuses.items()
        if any(s == "active" for s in statuses)
    )

    report = BatchReport(
        cells_newly_rejected=tuple(newly_rejected),
        cells_newly_futile=tuple(newly_futile),
        cells_reverted_to_active=tuple(reverted_to_active),
        configs_fully_resolved=tuple(fully_resolved),
        configs_still_active=tuple(still_active),
    )

    return updated, report


# ---------------------------------------------------------------------------
# Cross-dataset ablation analysis
# ---------------------------------------------------------------------------


class CrossDatasetAnalysisResults(BaseModel):
    """Results from comparing accuracy between primary and ablation datasets.

    Contains two independently Bonferroni-corrected test families:
    condition-level (one test per config x condition pair, testing the
    per-condition accuracy difference) and config-level (one test per
    config, testing the class-reweighted balanced accuracy difference).
    """

    model_config = ConfigDict(frozen=True)

    condition_level_comparisons: tuple[CrossDatasetComparison, ...]
    config_level_comparisons: tuple[CrossDatasetComparison, ...]


def analyze_cross_dataset(
    primary_results: ExperimentAnalysisResults,
    ablation_results: ExperimentAnalysisResults,
    *,
    alpha: float = 0.05,
) -> CrossDatasetAnalysisResults:
    """Run cross-dataset significance tests between primary and ablation experiments.

    Extracts trial outcomes from each snapshot and delegates to
    ``compute_all_cross_dataset_comparisons`` for the two Bonferroni
    families (condition-level and config-level). The two experiments share
    a stimulus grid, so each comparison bets on the paired per-cell
    differences between them.
    """
    condition_level, config_level = compute_all_cross_dataset_comparisons(
        list(primary_results.trial_outcomes),
        list(ablation_results.trial_outcomes),
        alpha=alpha,
    )
    return CrossDatasetAnalysisResults(
        condition_level_comparisons=tuple(condition_level),
        config_level_comparisons=tuple(config_level),
    )