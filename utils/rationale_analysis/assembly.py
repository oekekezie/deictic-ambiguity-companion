"""Assembly pipeline for rationale analysis prompts.

Transforms experiment results into a JSONL dataset suitable for batch
inference submission to a rationale analyst model. The assembly step is
a self-contained transformation: one-or-more experiment manifest paths
plus one stimulus JSONL path in, one JSONL artifact (plus metadata
sidecar) out. When multiple manifests are provided, trials are unioned
and deduplicated by identity before assembly.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.batch_inference.example import Example
from utils.batch_inference.message import Message
from utils.experiment_analysis.ground_truth import parse_example_id
from utils.experiment_analysis.manifest import ExperimentManifest
from utils.experiment_analysis.models import TrialOutcome
from utils.experiment_analysis.pipeline import load_experiment
from utils.experiment_analysis.stimulus_loading import load_stimuli
from utils.experiment_analysis.stimulus_metadata import STIMULUS_METADATA_REGISTRY
from utils.rationale_analysis.custom_id import encode_rationale_analysis_custom_id
from utils.rationale_analysis.eligibility import filter_eligible_trials
from utils.rationale_analysis.models import (
    RationaleAnalysisAssemblyMetadata,
    RationaleAnalysisManifestProvenance,
)
from utils.rationale_analysis.template import (
    extract_grader_feedback,
    populate_user_prompt,
)

# Artifact paths resolved from the repository root via __file__,
# following the pattern in utils/experiment_analysis/stimulus_metadata.py
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_SYSTEM_PROMPT_PATH: Path = _REPO_ROOT / "rationale_analysis" / "system_prompt.md"
_RESPONSE_SCHEMA_PATH: Path = _REPO_ROOT / "rationale_analysis" / "response_schema.json"


def _load_system_prompt() -> str:
    """Load the rationale analyst system prompt from the artifact directory."""
    if not _SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Rationale analysis system prompt not found at {_SYSTEM_PROMPT_PATH}"
        )
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _load_response_schema() -> dict[str, Any]:
    """Load the rationale analyst response schema from the artifact directory.

    The seven-flag response schema is fully static — it carries no
    per-stimulus or per-trial content — so it is loaded once and embedded
    unchanged in every assembled ``Example``.
    """
    if not _RESPONSE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Rationale analysis response schema not found at {_RESPONSE_SCHEMA_PATH}"
        )
    return json.loads(_RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))


def _build_example(
    outcome: TrialOutcome,
    system_prompt: str,
    response_schema: dict[str, Any],
    grader_feedback_cache: dict[str, str],
) -> Example:
    """Build a single Example for one eligible trial.

    Populates the user prompt template from the trial's stimulus metadata
    and rationale, then wraps the result in the Example schema expected by
    the batch inference pipeline. The response schema is static (the same
    seven-flag schema for every trial) and is passed in pre-loaded.
    """
    base_example, condition = parse_example_id(outcome.example_id)
    metadata = STIMULUS_METADATA_REGISTRY[base_example]

    user_prompt = populate_user_prompt(
        condition=condition,
        metadata=metadata,
        grader_feedback=grader_feedback_cache[outcome.example_id],
        rationale=outcome.rationale,  # guaranteed non-None by parse-failure eligibility filter
        predicted_score=outcome.predicted_score,  # guaranteed non-None by eligibility
    )

    custom_id = encode_rationale_analysis_custom_id(
        example_id=outcome.example_id,
        config_key=outcome.config_key,
        batch_index=outcome.batch_index,
        trial=outcome.trial,
    )

    return Example(
        custom_id=custom_id,
        messages=[Message(role="user", content=user_prompt)],
        system_prompt=system_prompt,
        response_schema=response_schema,
    )


def _deduplicate_trial_outcomes(
    outcomes: list[TrialOutcome],
) -> list[TrialOutcome]:
    """Deduplicate trials by identity, first-occurrence wins.

    The identity key is ``(config_key, example_id, batch_index, trial)``.
    When multiple manifests share config entries (e.g., Stage 2 includes
    Stage 1 bound entries), the same trial appears in both. First
    occurrence is kept, matching the cumulative manifest convention
    established by ``_resolve_cumulative_entries()`` in
    ``generate_experiment_configs.py``.
    """
    seen: set[tuple[str, str, int, int]] = set()
    deduplicated: list[TrialOutcome] = []
    for outcome in outcomes:
        key = (outcome.config_key, outcome.example_id, outcome.batch_index, outcome.trial)
        if key not in seen:
            seen.add(key)
            deduplicated.append(outcome)
    return deduplicated


def assemble_rationale_analysis_dataset(
    manifest_paths: list[Path],
    stimulus_jsonl_path: Path,
    output_dir: Path | None = None,
) -> Path:
    """Assemble rationale analysis prompts from experiment results.

    Accepts one-or-more experiment manifest paths plus one stimulus JSONL
    path. When multiple manifests are provided, trials are unioned and
    deduplicated by ``(config_key, example_id, batch_index, trial)``
    identity before assembly — preventing redundant prompts when
    manifests share config entries (e.g., Stage 2 includes Stage 1 bound
    entries).

    Steps:
      1. Parse all manifests, validate ``dataset_sha256`` consistency,
         load experiments, and deduplicate combined trial outcomes
      2. Load stimuli via ``load_stimuli(stimulus_jsonl_path)``
      3. Extract ``grader_feedback`` for all unique example IDs upfront
      4. Filter eligible trials (exclude parse failures and correct trials)
      5. Build an ``Example`` per eligible trial with the rationale analyst's
         system prompt, populated user prompt, and the static response schema
      6. Write JSONL + metadata sidecar with SHA-256 hashes

    Returns the path to the written JSONL file.
    """
    # ── Step 1: Load experiments from all manifests, deduplicate ─────────
    if not manifest_paths:
        raise ValueError("manifest_paths must contain at least one manifest path")

    # Parse all manifests and validate dataset_sha256 consistency before
    # doing any load_experiment() I/O (fail-fast on mismatch)
    manifest_raw_bytes: list[bytes] = []
    parsed_manifests: list[ExperimentManifest] = []
    for mp in manifest_paths:
        raw = mp.read_bytes()
        manifest_raw_bytes.append(raw)
        parsed_manifests.append(
            ExperimentManifest.model_validate_json(raw.decode("utf-8"))
        )

    dataset_sha256s = {m.dataset_sha256 for m in parsed_manifests}
    if len(dataset_sha256s) > 1:
        detail = ", ".join(
            f"{mp}: {m.dataset_sha256}"
            for mp, m in zip(manifest_paths, parsed_manifests)
        )
        raise ValueError(
            f"All manifests must share the same dataset_sha256, "
            f"but got mismatched values: {detail}"
        )

    # Load trials from each manifest and union into a single list
    all_outcomes: list[TrialOutcome] = []
    for manifest in parsed_manifests:
        outcomes, _ = load_experiment(manifest)
        all_outcomes.extend(outcomes)

    # Deduplicate by trial identity (first-occurrence wins)
    outcomes = _deduplicate_trial_outcomes(all_outcomes)

    # Build per-manifest provenance records
    experiment_manifests = tuple(
        RationaleAnalysisManifestProvenance(
            path=str(mp),
            sha256=hashlib.sha256(raw).hexdigest(),
            stage=m.stage,
        )
        for mp, raw, m in zip(manifest_paths, manifest_raw_bytes, parsed_manifests)
    )

    # ── Step 2: Load stimuli ─────────────────────────────────────────────
    stimuli = load_stimuli(stimulus_jsonl_path)

    # ── Step 3: Extract grader_feedback upfront (fail-fast validation) ───
    unique_example_ids = sorted({o.example_id for o in outcomes})
    grader_feedback_cache: dict[str, str] = {}

    for example_id in unique_example_ids:
        if example_id not in stimuli:
            raise ValueError(
                f"Example {example_id!r} from experiment outcomes not found "
                f"in stimulus JSONL {stimulus_jsonl_path}"
            )
        try:
            grader_feedback_cache[example_id] = extract_grader_feedback(
                stimuli[example_id]["user_content"]
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to extract grader_feedback for {example_id!r}: {e}"
            ) from e

    # ── Step 4: Filter eligible trials (exclude parse failures and correct trials)
    eligibility = filter_eligible_trials(outcomes)

    # ── Step 5: Build Examples ───────────────────────────────────────────
    system_prompt = _load_system_prompt()
    response_schema = _load_response_schema()

    examples: list[Example] = []
    for outcome in eligibility.eligible:
        examples.append(
            _build_example(
                outcome,
                system_prompt,
                response_schema,
                grader_feedback_cache,
            )
        )

    # ── Step 6: Write JSONL + metadata sidecar ───────────────────────────
    # Default output_dir uses the first manifest's stage (single-manifest
    # case is unchanged; multi-manifest case follows the same convention)
    if output_dir is None:
        output_dir = _REPO_ROOT / "rationale_analysis" / parsed_manifests[0].stage
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"{timestamp}_assembled_prompts.jsonl"
    metadata_path = output_dir / f"{timestamp}_assembly_metadata.json"

    # Write JSONL
    jsonl_lines: list[str] = []
    for example in examples:
        line = json.dumps(
            example.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
        )
        jsonl_lines.append(line)
    jsonl_content = "\n".join(jsonl_lines) + "\n" if jsonl_lines else ""
    jsonl_path.write_text(jsonl_content, encoding="utf-8")

    # Compute SHA-256 hashes for provenance
    assembled_sha256 = hashlib.sha256(jsonl_content.encode("utf-8")).hexdigest()
    stimulus_sha256 = hashlib.sha256(
        stimulus_jsonl_path.read_bytes()
    ).hexdigest()

    # Write metadata sidecar
    assembly_metadata = RationaleAnalysisAssemblyMetadata(
        experiment_manifests=experiment_manifests,
        stimulus_jsonl_path=str(stimulus_jsonl_path),
        stimulus_jsonl_sha256=stimulus_sha256,
        assembled_prompts_sha256=assembled_sha256,
        total_eligible_trials=len(examples),
        excluded_parse_failure=eligibility.excluded_parse_failure,
        excluded_correct_trial=eligibility.excluded_correct_trial,
        assembled_at=datetime.now(timezone.utc).isoformat(),
    )
    metadata_path.write_text(
        assembly_metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return jsonl_path
