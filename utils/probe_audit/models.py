"""Pydantic data models for probe-audit.

Split out from the subcommand modules so they can be imported without
bringing in the full CLI surface. Every sequence field is a tuple to
match the immutability convention in ``utils/experiment_analysis/models.py``
and ``utils/rationale_analysis/models.py``.

The probe classifies meta-evaluator rationales on seven binary flags —
five misattribution flags and two deictic flags, named by
``RATIONALE_ANALYSIS_FLAG_KEYS``. Every per-flag model field below uses
the flag key verbatim as its name (or as a ``{flag}_*`` prefix) so the
audit cannot drift from the probe it audits.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from utils.experiment_analysis.ground_truth import Condition
from utils.probe_audit.args import (
    BatchAllocation,
    ReplicationEvidence,
    SampleArgs,
    replication_evidence_for_k,
)


class RepReading(BaseModel):
    """One sub-agent rep's independent reading before seeing probe output.

    Carries the rep's own classification of all seven probe flags — five
    misattribution flags and two deictic flags, named by
    ``RATIONALE_ANALYSIS_FLAG_KEYS``. Each flag is a ``{flag}`` bool plus
    a ``{flag}_reasoning`` free-text justification. The reading is formed
    before the rep sees the probe's output, so it is an independent
    re-read of the same rationale the probe classified.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rep_index: int
    rationale_bundle_sha256: str

    misattributed_scenario_current_value: bool
    misattributed_scenario_current_value_reasoning: str
    misattributed_scenario_historical_value: bool
    misattributed_scenario_historical_value_reasoning: str
    misattributed_scenario_proposed_value: bool
    misattributed_scenario_proposed_value_reasoning: str
    misattributed_assistant_fallback_value: bool
    misattributed_assistant_fallback_value_reasoning: str
    misattributed_grader_claim: bool
    misattributed_grader_claim_reasoning: str
    articulated_operational_interpretation: bool
    articulated_operational_interpretation_reasoning: str
    operational_interpretation_governed_judgment: bool
    operational_interpretation_governed_judgment_reasoning: str

    recorded_at: str
    sub_agent_session_id: str | None = None
    sub_agent_model: str


class RepJudgment(BaseModel):
    """One sub-agent rep's judgment of whether the probe was correct.

    For each of the seven probe flags the rep records a
    ``{flag}_probe_correct`` bool — whether the probe's classification
    matched the rep's independent reading — a ``{flag}_reasoning``
    explanation, and an optional ``{flag}_corrected_classification``: the
    rep's own bool, supplied only when the probe was judged wrong.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rep_index: int
    probe_output_bundle_sha256: str

    misattributed_scenario_current_value_probe_correct: bool
    misattributed_scenario_current_value_reasoning: str
    misattributed_scenario_current_value_corrected_classification: bool | None = None
    misattributed_scenario_historical_value_probe_correct: bool
    misattributed_scenario_historical_value_reasoning: str
    misattributed_scenario_historical_value_corrected_classification: bool | None = None
    misattributed_scenario_proposed_value_probe_correct: bool
    misattributed_scenario_proposed_value_reasoning: str
    misattributed_scenario_proposed_value_corrected_classification: bool | None = None
    misattributed_assistant_fallback_value_probe_correct: bool
    misattributed_assistant_fallback_value_reasoning: str
    misattributed_assistant_fallback_value_corrected_classification: bool | None = None
    misattributed_grader_claim_probe_correct: bool
    misattributed_grader_claim_reasoning: str
    misattributed_grader_claim_corrected_classification: bool | None = None
    articulated_operational_interpretation_probe_correct: bool
    articulated_operational_interpretation_reasoning: str
    articulated_operational_interpretation_corrected_classification: bool | None = None
    operational_interpretation_governed_judgment_probe_correct: bool
    operational_interpretation_governed_judgment_reasoning: str
    operational_interpretation_governed_judgment_corrected_classification: (
        bool | None
    ) = None

    recorded_at: str
    sub_agent_session_id: str | None = None
    sub_agent_model: str


DisagreementCategory = Literal[
    "stable_agreement",
    "stable_disagreement",
    "shifted_against_probe",
    "shifted_to_probe",
]


class AuditTrialRecord(BaseModel):
    """Aggregated per-trial audit output over K sub-agent reps.

    Carries seven-flag forms of the aggregated independent reading, the
    aggregated probe-correct judgment, both vote tallies, and the
    disagreement category — one field per flag in
    ``RATIONALE_ANALYSIS_FLAG_KEYS``.

    The audit-wide replication shape (single-rep vs. replicated) is
    NOT carried per record — it lives once on
    :class:`ProbeAuditProvenance.replication_evidence` because every
    record in a single audit shares the same K. Consumers iterating
    records without provenance should re-derive via
    :func:`utils.probe_audit.args.replication_evidence_for_k` rather
    than expecting a per-record copy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_key: tuple[str, str, int, int]
    k: int
    rep_readings: tuple[RepReading, ...]
    rep_judgments: tuple[RepJudgment, ...]

    aggregated_independent_reading_misattributed_scenario_current_value: bool | None
    aggregated_independent_reading_misattributed_scenario_historical_value: bool | None
    aggregated_independent_reading_misattributed_scenario_proposed_value: bool | None
    aggregated_independent_reading_misattributed_assistant_fallback_value: bool | None
    aggregated_independent_reading_misattributed_grader_claim: bool | None
    aggregated_independent_reading_articulated_operational_interpretation: bool | None
    aggregated_independent_reading_operational_interpretation_governed_judgment: (
        bool | None
    )

    aggregated_probe_correct_misattributed_scenario_current_value: bool | None
    aggregated_probe_correct_misattributed_scenario_historical_value: bool | None
    aggregated_probe_correct_misattributed_scenario_proposed_value: bool | None
    aggregated_probe_correct_misattributed_assistant_fallback_value: bool | None
    aggregated_probe_correct_misattributed_grader_claim: bool | None
    aggregated_probe_correct_articulated_operational_interpretation: bool | None
    aggregated_probe_correct_operational_interpretation_governed_judgment: bool | None

    independent_reading_votes_misattributed_scenario_current_value: tuple[int, int]
    independent_reading_votes_misattributed_scenario_historical_value: tuple[int, int]
    independent_reading_votes_misattributed_scenario_proposed_value: tuple[int, int]
    independent_reading_votes_misattributed_assistant_fallback_value: tuple[int, int]
    independent_reading_votes_misattributed_grader_claim: tuple[int, int]
    independent_reading_votes_articulated_operational_interpretation: tuple[int, int]
    independent_reading_votes_operational_interpretation_governed_judgment: tuple[
        int, int
    ]

    probe_correct_votes_misattributed_scenario_current_value: tuple[int, int]
    probe_correct_votes_misattributed_scenario_historical_value: tuple[int, int]
    probe_correct_votes_misattributed_scenario_proposed_value: tuple[int, int]
    probe_correct_votes_misattributed_assistant_fallback_value: tuple[int, int]
    probe_correct_votes_misattributed_grader_claim: tuple[int, int]
    probe_correct_votes_articulated_operational_interpretation: tuple[int, int]
    probe_correct_votes_operational_interpretation_governed_judgment: tuple[int, int]

    disagreement_category_misattributed_scenario_current_value: DisagreementCategory
    disagreement_category_misattributed_scenario_historical_value: DisagreementCategory
    disagreement_category_misattributed_scenario_proposed_value: DisagreementCategory
    disagreement_category_misattributed_assistant_fallback_value: DisagreementCategory
    disagreement_category_misattributed_grader_claim: DisagreementCategory
    disagreement_category_articulated_operational_interpretation: DisagreementCategory
    disagreement_category_operational_interpretation_governed_judgment: (
        DisagreementCategory
    )


class ProbeAuditProvenance(BaseModel):
    """Everything a reviewer needs to reproduce or scrutinize a run.

    ``replication_evidence`` is a derived label over ``k`` (see
    :func:`utils.probe_audit.args.replication_evidence_for_k`). Persisting
    it explicitly — alongside a model-validator that asserts the
    derivation — gives reviewers a self-documenting signal without
    requiring them to memorize the K=1 → "none" mapping, and converts
    a hand-edited snapshot.json (where K and the label disagree) from
    a silent provenance lie into a load-time refusal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_fingerprint: str
    fingerprint_method: Literal["git-blob-then-sha256", "raw-bytes-sha256"]
    skill_human_label: str | None
    skill_fingerprint_inputs: tuple[str, ...]
    skill_per_file_hashes: dict[str, str]
    git_commit_sha: str | None
    git_tree_dirty: bool
    canonical_args_json: str
    args_digest: str
    k: int
    # The parametric draw algorithm this audit was sampled under. Part of the
    # pooling identity: pooled statistics must not mix allocation modes, so
    # ``pool_compatibility`` treats a mismatch as a hard error.
    batch_allocation: BatchAllocation
    replication_evidence: ReplicationEvidence
    sampling_seed: str
    probe_snapshot_path: str
    probe_snapshot_sha256: str
    rationale_source_path: str
    rationale_source_sha256: str
    stimulus_source_path: str
    stimulus_source_sha256: str
    auditor_model_family: str
    auditor_model_versions: tuple[str, ...]
    claude_session_id: str | None
    run_started_at: str
    run_completed_at: str | None
    # The flags the coverage-convergent draw classified as rare and floored
    # (carried verbatim from the sample manifest). The reporting layer reads
    # this — the source of truth for what the draw actually floored — rather
    # than recomputing a rare set from the probe payload, which is built from a
    # potentially different population than the joined eligible frame. ``None``
    # for the stratified draw (no floor). Defaulted so existing stratified
    # snapshots load without the field.
    classified_rare_flags: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _classified_rare_flags_matches_allocation(self) -> "ProbeAuditProvenance":
        """``classified_rare_flags`` is populated iff the draw was convergent.

        Mirrors the same iff-invariant on ``SampleManifest`` so a hand-edited
        snapshot cannot misrepresent (or hide) which flags were floored.
        """
        populated = self.classified_rare_flags is not None
        expected = self.batch_allocation == "coverage_convergent"
        if populated != expected:
            raise ValueError(
                f"classified_rare_flags must be populated iff "
                f"batch_allocation == 'coverage_convergent'; got "
                f"batch_allocation={self.batch_allocation!r}, "
                f"classified_rare_flags="
                f"{'populated' if populated else 'None'}"
            )
        return self

    @model_validator(mode="after")
    def _replication_evidence_matches_k(self) -> "ProbeAuditProvenance":
        """Reject snapshots whose ``k`` and ``replication_evidence`` disagree.

        The label is a pure function of K, so any mismatch is either a
        hand-edit (tampering) or a constructor bug. Failing at load time
        means the marimo notebook and CLI report can trust the field
        without re-deriving — and a tampered snapshot cannot silently
        misrepresent its replication shape to a reviewer.
        """
        expected = replication_evidence_for_k(self.k)
        if self.replication_evidence != expected:
            raise ValueError(
                f"replication_evidence={self.replication_evidence!r} disagrees "
                f"with k={self.k} (expected {expected!r}); the label is a "
                "derived function of K, so a mismatch is either tampering "
                "or a constructor bug"
            )
        return self


class ProbeAuditResults(BaseModel):
    """Full aggregated output of one audit run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provenance: ProbeAuditProvenance
    sample_args: SampleArgs
    records: tuple[AuditTrialRecord, ...]


class RationaleBundle(BaseModel):
    """Payload printed to the auditor sub-agent by ``fetch-rationale``.

    Includes the seven scenario context fields the original Kimi K2.6
    probe received so the auditor can disambiguate referential anchors
    (e.g. what "previous" resolves to) before forming an independent
    reading. The fields are for referential disambiguation only; the
    auditor still classifies what the rationale text states.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_key: tuple[str, str, int, int]
    condition: Condition
    rationale: str
    predicted_score: str | None
    ground_truth_score: str
    current_value: str
    proposed_value: str
    historical_value: str
    domain_noun: str
    draft_fallback_value: str
    grader_feedback: str
    ground_truth_description: str


class ProbeOutputBundle(BaseModel):
    """Payload printed to the auditor sub-agent by ``fetch-probe-output``.

    Carries the probe's aggregated classification and vote tally for each
    of the seven flags in ``RATIONALE_ANALYSIS_FLAG_KEYS``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_key: tuple[str, str, int, int]

    aggregated_misattributed_scenario_current_value: bool | None
    aggregated_misattributed_scenario_historical_value: bool | None
    aggregated_misattributed_scenario_proposed_value: bool | None
    aggregated_misattributed_assistant_fallback_value: bool | None
    aggregated_misattributed_grader_claim: bool | None
    aggregated_articulated_operational_interpretation: bool | None
    aggregated_operational_interpretation_governed_judgment: bool | None

    votes_misattributed_scenario_current_value: tuple[int, int]
    votes_misattributed_scenario_historical_value: tuple[int, int]
    votes_misattributed_scenario_proposed_value: tuple[int, int]
    votes_misattributed_assistant_fallback_value: tuple[int, int]
    votes_misattributed_grader_claim: tuple[int, int]
    votes_articulated_operational_interpretation: tuple[int, int]
    votes_operational_interpretation_governed_judgment: tuple[int, int]
