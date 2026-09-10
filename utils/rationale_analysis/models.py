"""Data models for the rationale analysis layer.

Defines all Pydantic models used across the rationale analysis pipeline:
assembly metadata, parsed responses, per-trial aggregated records, and
eligibility results. All models are frozen and use tuples for sequence
fields to match the immutability convention in experiment_analysis/models.py.
"""

from pydantic import BaseModel, ConfigDict

from utils.experiment_analysis.models import TrialOutcome


# The seven binary flags the rationale analysis probe classifies, in the
# order the response schema lists them: five misattribution flags followed
# by the two deictic flags. Single-sourced here so the response parser, the
# aggregator, and the probe-audit layer cannot drift from one another.
RATIONALE_ANALYSIS_FLAG_KEYS: tuple[str, ...] = (
    "misattributed_scenario_current_value",
    "misattributed_scenario_historical_value",
    "misattributed_scenario_proposed_value",
    "misattributed_assistant_fallback_value",
    "misattributed_grader_claim",
    "articulated_operational_interpretation",
    "operational_interpretation_governed_judgment",
)


# The five misattribution flag keys, derived from RATIONALE_ANALYSIS_FLAG_KEYS
# (the leading misattribution partition) so a reordering of the canonical tuple
# cannot make them drift. Consumed by the failure mode aggregators and the
# misattribution-rate tables in the experiment analysis layer.
MISATTRIBUTION_FLAG_KEYS: tuple[str, ...] = tuple(
    k for k in RATIONALE_ANALYSIS_FLAG_KEYS if k.startswith("misattributed_")
)


# Reader-facing display names for all seven flags. The single source for the
# misattribution-rate table headers (experiment analysis) and the cell
# deep-dive per-flag breakdown, so both render the same labels.
RATIONALE_ANALYSIS_FLAG_DISPLAY_NAMES: dict[str, str] = {
    "misattributed_scenario_current_value": "Misattributed Current Value",
    "misattributed_scenario_historical_value": "Misattributed Historical Value",
    "misattributed_scenario_proposed_value": "Misattributed Proposed Value",
    "misattributed_assistant_fallback_value": "Misattributed Assistant Fallback Citation",
    "misattributed_grader_claim": "Misattributed Grader Claim",
    "articulated_operational_interpretation": "Articulated Operational Interpretation",
    "operational_interpretation_governed_judgment": "Operational Interpretation Governed Judgment",
}


# ── Assembly metadata (Section 3.11) ─────────────────────────────────────


class RationaleAnalysisExcludedTrial(BaseModel):
    """Identity of a trial excluded from rationale analysis assembly.

    Carries the four dimensions needed to join back to the experiment's
    TrialOutcome for diagnosing exclusion patterns by configuration,
    stimulus, or condition.
    """

    model_config = ConfigDict(frozen=True)

    example_id: str
    config_key: str
    batch_index: int
    trial: int


class RationaleAnalysisManifestProvenance(BaseModel):
    """Provenance record for one experiment manifest consumed during assembly.

    When multiple manifests are passed (e.g., Stages 1 + 2), one record
    is created per manifest so the sidecar captures the full input set.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    stage: str


class RationaleAnalysisAssemblyMetadata(BaseModel):
    """Provenance sidecar for an assembled rationale analysis dataset.

    Written alongside the assembled JSONL to record the full provenance
    chain from experiment data to rationale analysis dataset, enabling
    reproducibility and staleness detection.
    """

    model_config = ConfigDict(frozen=True)

    experiment_manifests: tuple[RationaleAnalysisManifestProvenance, ...]
    stimulus_jsonl_path: str
    stimulus_jsonl_sha256: str
    assembled_prompts_sha256: str
    total_eligible_trials: int
    excluded_parse_failure: tuple[RationaleAnalysisExcludedTrial, ...]
    excluded_correct_trial: tuple[RationaleAnalysisExcludedTrial, ...]
    assembled_at: str


# ── Eligibility filtering result ─────────────────────────────────────────


class RationaleAnalysisEligibilityResult(BaseModel):
    """Result of partitioning trials into eligible and excluded sets.

    Two filters are applied in order, producing three mutually exclusive
    categories: (1) parse failures (envelope, structural, or domain) go
    to ``excluded_parse_failure``; (2) correct trials
    (``matches_ground_truth is True``) go to ``excluded_correct_trial``;
    (3) all remaining trials — incorrect with a successful parse — are
    ``eligible`` for rationale analysis.
    """

    model_config = ConfigDict(frozen=True)

    eligible: tuple[TrialOutcome, ...]
    excluded_parse_failure: tuple[RationaleAnalysisExcludedTrial, ...]
    excluded_correct_trial: tuple[RationaleAnalysisExcludedTrial, ...]


# ── Per-repetition parsed response (Section 3.12, Layer 2) ──────────────


class RationaleAnalysisFlagResult(BaseModel):
    """One flag's parsed output from a single rationale analyst repetition.

    Contains the deliberation structure the response schema enforces:
    an analysis citing the relevant evidence, then the binary classification.
    """

    model_config = ConfigDict(frozen=True)

    analysis: str
    classification: bool


class RationaleAnalysisParsedResponse(BaseModel):
    """Parsed output from one rationale analyst repetition.

    Carries the seven binary flag results — five misattribution flags and
    two deictic flags, named by ``RATIONALE_ANALYSIS_FLAG_KEYS`` — and an
    optional reasoning trace. The flags are populated by the response
    parser; ``reasoning_text`` is attached later during aggregation from
    the ``ProcessedResult`` (extended thinking from models that support it).
    """

    model_config = ConfigDict(frozen=True)

    misattributed_scenario_current_value: RationaleAnalysisFlagResult
    misattributed_scenario_historical_value: RationaleAnalysisFlagResult
    misattributed_scenario_proposed_value: RationaleAnalysisFlagResult
    misattributed_assistant_fallback_value: RationaleAnalysisFlagResult
    misattributed_grader_claim: RationaleAnalysisFlagResult
    articulated_operational_interpretation: RationaleAnalysisFlagResult
    operational_interpretation_governed_judgment: RationaleAnalysisFlagResult
    # Extended thinking from the RA analyst LLM (attached during aggregation)
    reasoning_text: str | None = None


# ── Per-trial aggregated record (Section 3.10) ──────────────────────────


class RationaleAnalysisTrialRecord(BaseModel):
    """Aggregated rationale analysis output for one experiment trial.

    Combines per-flag aggregated values, per-flag vote tallies, and the K
    individual parsed responses (audit trail) for a single trial. The five
    misattribution flags and the articulation flag aggregate by
    any-affirmative; the governing flag aggregates by majority vote. A flag
    value is None when fewer than ceil(K/2) repetitions parsed.
    """

    model_config = ConfigDict(frozen=True)

    # Trial identity (joins back to TrialOutcome)
    config_key: str
    example_id: str
    batch_index: int
    trial: int

    # Aggregated flag values
    misattributed_scenario_current_value: bool | None
    misattributed_scenario_historical_value: bool | None
    misattributed_scenario_proposed_value: bool | None
    misattributed_assistant_fallback_value: bool | None
    misattributed_grader_claim: bool | None
    articulated_operational_interpretation: bool | None
    operational_interpretation_governed_judgment: bool | None

    # Per-flag vote tallies
    misattributed_scenario_current_value_votes_for: int
    misattributed_scenario_current_value_votes_against: int
    misattributed_scenario_historical_value_votes_for: int
    misattributed_scenario_historical_value_votes_against: int
    misattributed_scenario_proposed_value_votes_for: int
    misattributed_scenario_proposed_value_votes_against: int
    misattributed_assistant_fallback_value_votes_for: int
    misattributed_assistant_fallback_value_votes_against: int
    misattributed_grader_claim_votes_for: int
    misattributed_grader_claim_votes_against: int
    articulated_operational_interpretation_votes_for: int
    articulated_operational_interpretation_votes_against: int
    operational_interpretation_governed_judgment_votes_for: int
    operational_interpretation_governed_judgment_votes_against: int

    # Audit trail: K parsed responses (None entries = parse failures)
    repetitions: tuple[RationaleAnalysisParsedResponse | None, ...]


# ── Snapshot provenance (Section 3.13) ───────────────────────────────────


class RationaleAnalysisProvenance(BaseModel):
    """Self-describing provenance for one rationale analysis snapshot.

    Records what produced the snapshot — the probe model, the probe-facing
    contract (system prompt + response schema), the assembled prompts the
    probe ran against, the original source jobs (and any retry jobs healed
    into them), K, and the per-flag aggregation rule — so a snapshot on disk
    is unambiguous about its own origin without
    relying on filename, mtime, or external sidecars. The auditor model and
    audit pool identity are deliberately excluded: those belong to the
    probe-audit layer, downstream of the snapshot.
    """

    model_config = ConfigDict(frozen=True)

    probe_model: str
    system_prompt_sha256: str
    response_schema_sha256: str
    assembled_prompts_sha256: str
    # Declared/original source jobs; retries are forbidden here and recorded
    # separately in merged_retry_job_ids.
    source_job_ids: tuple[str, ...]
    # Retry jobs whose healed results landed in this snapshot (empty when no
    # retry contributed) — distinct from source_job_ids so the snapshot is
    # honest about originals vs. healing companions.
    merged_retry_job_ids: tuple[str, ...]
    k: int
    # Per-flag (flag_key, policy_name) pairs — honest about the heterogeneous
    # aggregation (six flags any-affirmative, the governing flag majority vote).
    aggregation_method: tuple[tuple[str, str], ...]
    built_at: str


# ── Aggregated results container ─────────────────────────────────────────


class RationaleAnalysisResults(BaseModel):
    """All rationale analysis trial records from one pipeline run.

    Carries a required ``provenance`` block so every persisted snapshot is
    self-describing and no snapshot can be constructed, serialized, or
    loaded without recording what produced it.
    """

    model_config = ConfigDict(frozen=True)

    trial_records: tuple[RationaleAnalysisTrialRecord, ...]
    provenance: RationaleAnalysisProvenance
