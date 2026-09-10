"""Sampling arguments and their canonical serialization.

``SampleArgs`` holds the per-run sampling-algorithm parameters:
``sample_size``, ``stratify_by``, ``k``, ``probe_snapshot_path``,
``pinned_trial_keys``, and ``exclude_prior_audits``. It does NOT carry
the rationale-source path — ``manifest.rationale_source`` is the sole
manifest authority for the stage-N snapshot's path and raw-bytes SHA.
Likewise it does NOT carry the stimulus-source path —
``manifest.stimulus_source`` is the sole manifest authority for the
assembled-stimulus JSONL's path and raw-bytes SHA. Keeping both off
``SampleArgs`` ensures a single authority per source; divergent
manifests cannot validate when only the manifest fields are consulted.

``k`` is constrained to a positive odd integer. ``K = 1`` is a first-
class single-rep audit mode in which both aggregation policies
(any-affirmative for articulation reading, majority vote for governing
reading + both probe-correct flags) collapse to the identity of the
single rep's classification — no replication evidence is gathered.
``K >= 3`` runs that many independent auditor sub-agents per trial and
aggregates their judgments. The two-state shape is reified by
:func:`replication_evidence_for_k`, whose output is persisted on
``ProbeAuditProvenance.replication_evidence`` so reviewers don't have
to re-derive it from K.

``exclude_prior_audits`` is a seed-relevant boolean: when ``True``, the
sampler queries the index for prior audits under the same byte-aware
pooling identity tuple — ``(skill_fingerprint, probe_snapshot_sha256,
rationale_source_path, rationale_source_sha256, stimulus_source_path,
stimulus_source_sha256, batch_allocation)`` — and subtracts their trial
keys from the joined eligible frame before the draw. The byte-aware check
matters: a path-only filter would let the sampler subtract trials from
audits whose auditors saw different rationale or stimulus content (same
path, mutated bytes), silently shrinking or biasing the new sample.
Two runs that differ only in this flag draw from different pools and
therefore produce different samples; the flag enters ``canonical_json``
accordingly. The exclusion SET itself is not in the seed (it is a
deterministic function of the current index state); the list of
contributing prior ``session_ids`` lives on ``SampleManifest`` as
provenance.

``probe_snapshot_path`` carries sample identity only — the CLI
normalizes ``--probe-snapshot`` to a repo-relative form via
``relative_to(repo_root)`` before constructing ``SampleArgs``, and
``canonical_json`` emits that repo-relative string verbatim. The
absolute path used for byte-I/O flows separately, as a keyword-only
arg to :func:`~utils.probe_audit.sampling.draw_sample`, and never
enters sample identity. This mirrors the rationale-source split below
(``rationale_source_repo_relative`` vs. ``rationale_source_path``) so
the same conceptual snapshot read from different absolute locations
yields the same ``args_digest`` and sampling seed.

``canonical_json(args)`` produces a byte-stable JSON over the seed-
relevant fields of ``SampleArgs``. The sampling seed is:

    sha256(
        skill_fingerprint
        || canonical_json(sample_args)
        || probe_snapshot_sha256
        || rationale_source_repo_relative
        || rationale_source_sha256
    )

``rationale_source_repo_relative`` and ``rationale_source_sha256`` are
passed explicitly into :func:`~utils.probe_audit.sampling.seed_from_inputs`;
they do NOT live on ``SampleArgs`` and do NOT appear in the canonical-
JSON output. Using the repo-relative string (not the absolute reading
path) is load-bearing: two runs on the same repo from different
absolute locations must seed identically for the same conceptual
inputs.

The set-valued list field ``stratify_by`` is sorted before
serialization so that reordering does not drift the seed — a reviewer
who reorders the CLI flags still gets the same sample draw.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Whitelist of stratum dimensions accepted by the sampler. Keeping this
# closed (rather than free-form strings) means a typo fails fast at
# argument-validation time instead of silently producing a degenerate
# stratification.
#
# All SEVEN rationale analysis flags are admissible so the machinery treats
# them uniformly: ``articulation`` / ``governing`` are the historical short
# aliases for the two deictic flags (``articulated_operational_interpretation``
# / ``operational_interpretation_governed_judgment``), and the five
# ``misattributed_*`` full keys are the misattribution flags. Exposing a flag
# as a dimension is a *capability*; whether a given run stratifies by it is a
# separate recipe choice. ``split_vote`` reads the two deictic flags' vote
# tallies only (it is degenerate at probe K=1 regardless).
StratumName = Literal[
    "condition",
    "articulation",
    "governing",
    "split_vote",
    "config",
    "misattributed_scenario_current_value",
    "misattributed_scenario_historical_value",
    "misattributed_scenario_proposed_value",
    "misattributed_assistant_fallback_value",
    "misattributed_grader_claim",
]


# The parametric batch-allocation algorithm. ``stratified`` is the legacy
# equal-allocation draw (``_stratified_draw``); ``coverage_convergent`` is the
# iterative breadth-then-converge draw (``_coverage_convergent_draw``) used for
# pooled, without-replacement audit campaigns. The choice changes which trials
# are drawn, so it folds into ``canonical_json`` (and therefore the seed and
# ``args_digest``).
BatchAllocation = Literal["stratified", "coverage_convergent"]


# Two-state label persisted on ``ProbeAuditProvenance.replication_evidence``.
# ``"none"`` means K=1 (single-rep audit; no replication evidence).
# ``"replicated"`` means K>=3 (multiple independent auditor reps per trial).
# Frames the *meaning* rather than the count, so a future K-tier
# expansion (e.g., a per-tier noise-characteristic label) can extend the
# enum without renaming the field.
ReplicationEvidence = Literal["none", "replicated"]


def validate_auditor_rep_count(k: int) -> int:
    """Return ``k`` after asserting it is a positive odd integer.

    Single source of truth for the K-shape contract used by both the
    ``SampleArgs.k`` field validator and :func:`replication_evidence_for_k`.
    Raises ``ValueError`` (not ``ValidationError``) so callers outside
    pydantic (e.g. provenance builders) get a vanilla exception they
    can catch without depending on pydantic's error type.
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer; got {k}")
    if k % 2 == 0:
        raise ValueError(f"k must be odd so majority vote cannot tie; got {k}")
    return k


def replication_evidence_for_k(k: int) -> ReplicationEvidence:
    """Map ``k`` → ``"none"`` (K=1) or ``"replicated"`` (K>=3).

    Calls :func:`validate_auditor_rep_count` first so that an invalid
    K (zero, negative, or even) propagates the same ``ValueError``
    rather than silently falling through to ``"replicated"``.
    """
    validate_auditor_rep_count(k)
    return "none" if k == 1 else "replicated"


class SampleArgs(BaseModel):
    """Inputs to the sampler that must round-trip into the sampling seed.

    ``k`` is deliberately excluded from the seed's args-digest portion
    (see :func:`canonical_json`): bumping K should not re-roll the sample.
    We still validate it here because it lives with the user-facing args,
    and the CLI should reject K=0, negative, or even K up front. K=1 is
    accepted and represents a single-rep audit (no replication evidence).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_size: int = Field(gt=0)
    stratify_by: tuple[StratumName, ...]
    k: int = Field(gt=0)
    probe_snapshot_path: Path
    # Empty tuple means parametric stratified draw; non-empty means explicit
    # trial pinning (bypasses the stratified sampler). See the matching
    # ``canonical_json`` branch below — the field participates in the seed
    # only when non-empty, so historical parametric digests stay valid.
    pinned_trial_keys: tuple[tuple[str, str, int, int], ...] = ()
    # Required: no Python default. When True, the sampler subtracts every
    # trial_key drawn by prior audits under the same byte-aware pooling
    # identity tuple — (skill_fingerprint, probe_snapshot_sha256,
    # rationale_source_path/_sha256, stimulus_source_path/_sha256,
    # batch_allocation) — from the eligible frame before the draw. The full
    # tuple
    # (rather than just fingerprint + probe SHA) catches priors whose
    # rationale or stimulus bytes differ from the current run; see the
    # module docstring for the byte-aware exclusion rationale. Mutually
    # exclusive with ``pinned_trial_keys`` (pinning bypasses the
    # eligibility computation entirely) — enforced by the model-level
    # validator below.
    exclude_prior_audits: bool
    # Required, no default: every draw declares its allocation algorithm
    # explicitly (mirrors ``exclude_prior_audits``). The CLI supplies an
    # argparse default of ``"stratified"`` so existing invocations are
    # unchanged, but the model itself takes no silent default.
    batch_allocation: BatchAllocation
    # Coverage-convergent parameters. ``None`` for the stratified draw;
    # required (and positive) when ``batch_allocation == "coverage_convergent"``,
    # enforced by ``_coverage_convergent_invariants`` below. ``coverage_floor``
    # is the per-rare-flag early-coverage target F (a research decision tied to
    # the error-finding goal); ``coverage_rare_max_count`` is the pool-TRUE-count
    # threshold C at or below which a flag is "rare" (and floored). Both fold
    # into ``canonical_json`` because they change which trials are drawn.
    coverage_floor: int | None = None
    coverage_rare_max_count: int | None = None

    @field_validator("k")
    @classmethod
    def _k_must_be_positive_odd(cls, value: int) -> int:
        """K must be a positive odd integer.

        Delegates to :func:`validate_auditor_rep_count`, which is the
        single source of truth for the K-shape contract shared with
        :func:`replication_evidence_for_k`. ``K = 1`` is allowed and
        represents a single-rep audit; both aggregation policies
        collapse to identity, and provenance records
        ``replication_evidence == "none"`` to surface the absence of
        replication evidence.
        """
        return validate_auditor_rep_count(value)

    @field_validator("pinned_trial_keys")
    @classmethod
    def _pinned_trial_keys_must_be_unique(
        cls, v: tuple[tuple[str, str, int, int], ...]
    ) -> tuple[tuple[str, str, int, int], ...]:
        """Duplicates would collapse K reps across multiple positions.

        Failing fast here — at argument-validation time — produces a more
        localized, actionable message than waiting for the downstream
        ``SampleManifest.trial_keys`` validator to refuse the draw.
        """
        if len(v) != len(set(v)):
            dup = [t for t in set(v) if sum(1 for x in v if x == t) > 1]
            raise ValueError(
                f"pinned_trial_keys must be unique; found {len(dup)} duplicate(s), "
                f"first: {dup[0] if dup else None}"
            )
        return v

    @model_validator(mode="after")
    def _pinning_invariants(self) -> "SampleArgs":
        """Cross-field rules for explicit pinning.

        A ``field_validator`` on ``pinned_trial_keys`` cannot see
        ``sample_size`` in Pydantic v2, so this invariant lives on the
        whole model. Also enforces that ``exclude_prior_audits`` is not
        combined with pinning — pinning bypasses the eligibility
        computation entirely, so a pinned draw with ``exclude_prior_audits
        =True`` would be a provenance lie (the flag says "exclusion was
        applied" when no eligibility filtering ran at all).
        """
        if self.pinned_trial_keys:
            if self.sample_size != len(self.pinned_trial_keys):
                raise ValueError(
                    f"sample_size ({self.sample_size}) must equal "
                    f"len(pinned_trial_keys) ({len(self.pinned_trial_keys)}) "
                    "under explicit pinning"
                )
            if self.exclude_prior_audits:
                raise ValueError(
                    "exclude_prior_audits must be False under explicit "
                    "pinning — pinning bypasses the eligibility filter, so "
                    "prior-audit exclusion would be inert and its presence "
                    "on the manifest would misrepresent how the draw was made"
                )
        return self

    @model_validator(mode="after")
    def _coverage_convergent_invariants(self) -> "SampleArgs":
        """Cross-field rules tying the coverage params to the allocation mode.

        ``coverage_convergent`` is the iterative without-replacement campaign
        draw: it floors rare flag-TRUE classes early then converges to the pool
        over a fixed ``condition`` partition. That contract requires the floor
        parameters to be set, prior-audit exclusion to be on (the WOR mechanism
        that makes batches disjoint and cumulatively convergent), no explicit
        pinning (which bypasses the eligible frame entirely), and the condition
        partition. The ``stratified`` draw must NOT carry coverage params, so a
        manifest can't misrepresent a stratified draw as a floored one.
        """
        if self.batch_allocation == "coverage_convergent":
            if self.coverage_floor is None or self.coverage_rare_max_count is None:
                raise ValueError(
                    "coverage_floor and coverage_rare_max_count are required "
                    "when batch_allocation='coverage_convergent'"
                )
            if self.coverage_floor <= 0 or self.coverage_rare_max_count <= 0:
                raise ValueError(
                    "coverage_floor and coverage_rare_max_count must be "
                    f"positive; got coverage_floor={self.coverage_floor}, "
                    f"coverage_rare_max_count={self.coverage_rare_max_count}"
                )
            if not self.exclude_prior_audits:
                raise ValueError(
                    "batch_allocation='coverage_convergent' requires "
                    "exclude_prior_audits=True — the without-replacement "
                    "exclusion is the mechanism that makes successive batches "
                    "disjoint and the cumulative sample convergent"
                )
            if self.pinned_trial_keys:
                raise ValueError(
                    "batch_allocation='coverage_convergent' cannot be combined "
                    "with explicit pinning — pinning bypasses the eligible "
                    "frame the convergent draw operates over"
                )
            if self.stratify_by != ("condition",):
                raise ValueError(
                    "batch_allocation='coverage_convergent' uses condition as "
                    "the fixed convergence partition; stratify_by must be "
                    f"('condition',), got {self.stratify_by!r}"
                )
        else:
            if (
                self.coverage_floor is not None
                or self.coverage_rare_max_count is not None
            ):
                raise ValueError(
                    "coverage_floor and coverage_rare_max_count are only valid "
                    "when batch_allocation='coverage_convergent'"
                )
        return self


def canonical_json(args: SampleArgs) -> str:
    """Byte-stable JSON over the seed-relevant fields of ``args``.

    ``k`` is excluded because changing K must not change which trials are
    sampled — only how many reps run per trial. Set-valued list fields
    are sorted so that reordering the user-facing arguments does not
    drift the seed. ``pinned_trial_keys`` participates only when
    non-empty, so parametric runs keep their historical ``args_digest``.
    ``exclude_prior_audits`` is always emitted because flipping it
    changes the eligible pool and must re-roll the sample.
    ``batch_allocation`` is always emitted because the two allocation
    algorithms draw different trials; ``coverage_floor`` /
    ``coverage_rare_max_count`` are emitted only for the
    ``coverage_convergent`` draw, where they govern the floor and must
    re-roll the sample when changed.
    """
    payload: dict = {
        "probe_snapshot_path": str(args.probe_snapshot_path),
        "sample_size": args.sample_size,
        "stratify_by": sorted(args.stratify_by),
        "exclude_prior_audits": args.exclude_prior_audits,
        # Always emitted — the allocation algorithm changes which trials are
        # drawn, so the two modes must seed distinctly and never collide on
        # ``args_digest``.
        "batch_allocation": args.batch_allocation,
    }
    # Coverage params are emitted only for the convergent draw, where they are
    # required and govern the draw. They are ``None`` (and absent) for the
    # stratified draw — semantic absence, not a backward-compat omission.
    if args.batch_allocation == "coverage_convergent":
        payload["coverage_floor"] = args.coverage_floor
        payload["coverage_rare_max_count"] = args.coverage_rare_max_count
    if args.pinned_trial_keys:
        # Sorted elementwise — tuples sort lexicographically, producing a
        # canonical order that does not depend on the CLI input order.
        payload["pinned_trial_keys"] = sorted(args.pinned_trial_keys)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def args_digest(args: SampleArgs) -> str:
    """12-hex-char prefix of ``sha256(canonical_json(args))``.

    Used as a directory-name component under
    ``analysis_outputs/probe_audit/<skill_fingerprint_short>/<args_digest>/``
    so two runs that differ only in (say) sample size sit in sibling
    directories rather than overwriting each other.
    """
    digest = hashlib.sha256(canonical_json(args).encode("utf-8")).hexdigest()
    return digest[:12]
