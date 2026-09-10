"""Snapshot provenance sourcing for rationale analysis.

Builds the self-describing ``RationaleAnalysisProvenance`` recorded in every
persisted snapshot: the probe model, the probe-facing contract SHAs, the
assembled-prompts SHA, the source job IDs, the healed-in retry job IDs, K,
the per-flag aggregation rule, and a build timestamp. The contract SHAs hash
the canonical repo artifacts
(``rationale_analysis/system_prompt.md`` / ``response_schema.json``) — the
exact bytes assembly embeds into every Example and the same files the
probe-audit fingerprint treats as authoritative, keeping provenance and
fingerprint mutually checkable.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from utils.batch_inference.processed import ProcessedJob
from utils.rationale_analysis.aggregation import flag_aggregation_method_names
from utils.rationale_analysis.models import (
    RationaleAnalysisAssemblyMetadata,
    RationaleAnalysisProvenance,
)

# Canonical probe-facing contract artifacts (the bytes assembly embeds and the
# fingerprint pins). Mirrors assembly.py's resolution from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RATIONALE_ANALYSIS_DIR = _REPO_ROOT / "rationale_analysis"
_SYSTEM_PROMPT_PATH = _RATIONALE_ANALYSIS_DIR / "system_prompt.md"
_RESPONSE_SCHEMA_PATH = _RATIONALE_ANALYSIS_DIR / "response_schema.json"
_STAGE_PATTERN = re.compile(r"stage_\d+")


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_snapshot_provenance(
    ra_jobs: list[ProcessedJob],
    k: int,
    assembly_metadata_path: Path,
    *,
    merged_retry_job_ids: tuple[str, ...],
    system_prompt_path: Path = _SYSTEM_PROMPT_PATH,
    response_schema_path: Path = _RESPONSE_SCHEMA_PATH,
) -> RationaleAnalysisProvenance:
    """Source a self-describing provenance record for one snapshot.

    Args:
        ra_jobs: The processed probe jobs the snapshot aggregates. All must
            share one ``model`` — a snapshot spanning probe models has no
            single provenance.
        k: The validated repetition count.
        assembly_metadata_path: The assembly sidecar the jobs were assembled
            from; its ``assembled_prompts_sha256`` pins the exact prompts the
            probe ran against.
        merged_retry_job_ids: Job IDs of the retry companions whose results
            were healed into ``ra_jobs`` (from ``load_rationale_analysis_jobs``);
            recorded separately from ``source_job_ids``, empty when none.
        system_prompt_path: Canonical system-prompt artifact to hash.
        response_schema_path: Canonical response-schema artifact to hash.

    Raises:
        ValueError: if ``ra_jobs`` is empty or spans more than one model.
    """
    models = {job.model for job in ra_jobs}
    if len(models) != 1:
        raise ValueError(
            f"Expected exactly one probe model across jobs, got {sorted(models)}"
        )
    probe_model = next(iter(models))

    assembly_metadata = RationaleAnalysisAssemblyMetadata.model_validate_json(
        assembly_metadata_path.read_text(encoding="utf-8")
    )

    return RationaleAnalysisProvenance(
        probe_model=probe_model,
        system_prompt_sha256=_sha256_file(system_prompt_path),
        response_schema_sha256=_sha256_file(response_schema_path),
        assembled_prompts_sha256=assembly_metadata.assembled_prompts_sha256,
        source_job_ids=tuple(sorted(job.job_id for job in ra_jobs)),
        merged_retry_job_ids=tuple(sorted(set(merged_retry_job_ids))),
        k=k,
        aggregation_method=flag_aggregation_method_names(),
        built_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def resolve_assembly_metadata_path(
    ra_jobs: list[ProcessedJob],
    rationale_analysis_dir: Path = _RATIONALE_ANALYSIS_DIR,
) -> Path:
    """Locate the assembly sidecar the snapshot's jobs were assembled from.

    Derives the stage from the jobs' shared ``submission_group`` (e.g.
    ``ra_stage_04_k2p6_v2`` -> ``stage_04``) and returns the unique
    ``*_assembly_metadata.json`` under ``rationale_analysis/{stage}/``.

    Raises:
        ValueError: if the jobs do not share one non-null submission_group,
            the group does not encode a stage, or the stage directory does
            not hold exactly one assembly-metadata sidecar.
    """
    groups = {job.submission_group for job in ra_jobs}
    if len(groups) != 1 or None in groups:
        raise ValueError(
            "Expected one non-null submission_group across jobs, got "
            f"{sorted(group or '<none>' for group in groups)}"
        )
    group = next(iter(groups))
    match = _STAGE_PATTERN.search(group)
    if match is None:
        raise ValueError(f"submission_group {group!r} does not encode a stage")
    stage_dir = rationale_analysis_dir / match.group()
    sidecars = sorted(stage_dir.glob("*_assembly_metadata.json"))
    if len(sidecars) != 1:
        raise ValueError(
            f"Expected exactly one assembly-metadata sidecar in {stage_dir}, "
            f"found {len(sidecars)}"
        )
    return sidecars[0]
