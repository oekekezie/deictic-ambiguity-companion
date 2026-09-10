"""Deterministic stratified sampler for the probe audit.

The sampling seed is

    sha256(
        skill_fingerprint
        || canonical_json(sample_args)
        || probe_snapshot_sha256
        || rationale_source_repo_relative
        || rationale_source_sha256
    )

Five UTF-8 segments concatenated as raw bytes with ``|`` separators
before the final SHA-256. Any of the five inputs changing reshuffles
the draw. ``rationale_source_repo_relative`` is a repo-relative string
(not an absolute ``Path``) so the same conceptual inputs seed
identically on different machines or checkouts — the absolute
filesystem path is used only for reading bytes.

Population is the **joined eligible frame**: probe ``trial_records``
intersected with stage ``trial_outcomes`` filtered to ``not
matches_ground_truth and parse_error is None``. Within a given seed,
the draw is a stratified random sample over that frame — trials are
bucketed into strata defined by ``stratify_by``, a per-stratum target
is computed, and each stratum is drawn from using a seeded Mersenne
Twister.

When ``sample_args.exclude_prior_audits`` is ``True``, the sampler
subtracts every trial_key drawn by prior audits under the same
byte-aware pooling identity tuple — ``(skill_fingerprint,
probe_snapshot_sha256, rationale_source_path,
rationale_source_sha256, stimulus_source_path,
stimulus_source_sha256, batch_allocation)`` — from the eligible frame
before the draw. ``batch_allocation`` is part of the identity so a
coverage-convergent campaign excludes/counts only its own prior batches
(independent campaigns; see Q3). The byte-aware check rejects priors
whose source bytes have been mutated since they ran (same path, different SHA);
auditors who saw different rationale or stimulus content cannot
contribute exclusion keys. The set of contributing prior sessions is
recorded on ``SampleManifest.excluded_prior_audit_session_ids`` as
provenance; the flag itself is folded into ``canonical_json(args)`` so
two runs that differ only in this setting receive different seeds.
"""

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from utils.experiment_analysis.stimulus_loading import parse_stimuli_jsonl
from utils.probe_audit.args import SampleArgs, canonical_json
from utils.probe_audit.scenario_context import (
    build_scenario_context,
    condition_from_example_id,
)
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS


SelectionMode = Literal["stratified", "pinned_from_manifest", "pinned_inline"]

# The five misattribution flag keys (the subset of the seven-flag schema that
# is NOT the two deictic flags). ``JoinedRecord`` carries their values so the
# coverage-convergent floor can read rare flag-TRUE membership, and they are
# admissible ``_record_stratum`` dimensions. Filtered by prefix rather than
# sliced so a reordering of ``RATIONALE_ANALYSIS_FLAG_KEYS`` cannot drift it.
_MISATTRIBUTION_FLAG_KEYS: tuple[str, ...] = tuple(
    k for k in RATIONALE_ANALYSIS_FLAG_KEYS if k.startswith("misattributed_")
)


class SampledTrial(BaseModel):
    """One trial selected by the sampler, with its stratum tag.

    ``trial_key`` is ``(config_key, example_id, batch_index, trial)`` —
    the same four-tuple used as the primary join key everywhere else in
    the rationale analysis pipeline. ``stratum`` is the pipe-joined
    composite key (e.g., ``"correct|True|False"``) recorded for
    downstream per-stratum slicing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_key: tuple[str, str, int, int]
    stratum: str


class SelectionSource(BaseModel):
    """Pointer to the prior ``sample_manifest.json`` that a pinned run was drawn from.

    ``path`` is repo-relative (so the index stays portable across clones),
    ``sha256`` is the raw-bytes SHA-256 of the source manifest file so any
    later edit to it is detectable. Only populated for
    ``selection_mode == "pinned_from_manifest"``; inline-pinned runs carry
    no ``SelectionSource`` because there is no file to point at.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str


class RationaleSource(BaseModel):
    """Pointer to the stage-N snapshot whose ``trial_outcomes`` were classified.

    The probe under audit classified rationales drawn from this file;
    pinning ``(path, sha256)`` on the manifest is how the orchestrator
    and every rep agree on the canonical upstream source without the
    orchestrator having to pass it on the CLI each time. ``path`` is
    repo-relative; ``sha256`` is the raw-bytes SHA-256 of the file so a
    later edit is detectable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str


class StimulusSource(BaseModel):
    """Pointer to the assembled stimulus JSONL backing scenario context.

    The auditor's ``RationaleBundle`` includes the seven scenario context
    fields the original Kimi K2.6 probe received; those fields are derived
    from this JSONL (via ``load_stimuli`` + ``extract_grader_feedback``)
    plus the static ``STIMULUS_METADATA_REGISTRY``. Pinning
    ``(path, sha256)`` on the manifest enforces byte-level reproducibility
    of what the auditor saw — a later edit to the stimulus file is
    detectable on every fetch and aggregate.

    Distinct from :class:`RationaleSource` because the two files play
    different roles: rationale_source defines the eligible trial frame
    (and is part of the sampling seed); stimulus_source augments scenario
    context only and is deliberately excluded from the seed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str


class SkillForensics(BaseModel):
    """Forensic metadata about the Skill that produced a sample.

    Captured at sample time from :func:`compute_fingerprint` and
    :func:`git_describe_label` so the aggregated snapshot's provenance
    panel can render the per-input-file blob SHAs, the dirty-tree flag,
    and the exact fingerprint method that was used. An empty
    ``SkillForensics()`` is allowed for tests that do not need these
    fields — production sample runs always populate them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint_method: Literal["git-blob-then-sha256", "raw-bytes-sha256"] | None = (
        None
    )
    fingerprint_inputs: tuple[str, ...] = ()
    per_file_hashes: dict[str, str] = {}
    human_label: str | None = None
    git_commit_sha: str | None = None
    git_tree_dirty: bool = False


class SampleManifest(BaseModel):
    """Provenance-rich manifest of a sample draw.

    The fields here are the inputs a reviewer needs to reproduce the
    sample — seed, Skill fingerprint, probe snapshot hash, and the exact
    ``SampleArgs`` used. Together with the probe snapshot file they
    uniquely determine the draw.

    ``forensics`` carries the additional Skill metadata used by the
    provenance panel (per-file blob SHAs, dirty-tree flag, etc.).

    ``selection_mode`` and ``selection_source`` together record how the
    trial set was chosen: ``"stratified"`` for a parametric draw,
    ``"pinned_from_manifest"`` with a populated ``SelectionSource`` when
    ``--trials-from`` reused a prior manifest's keys, or
    ``"pinned_inline"`` when ``--trial-keys`` provided the list directly.
    Both fields are required — absent values on disk are a schema gap,
    not a default, and fail to load.

    ``rationale_source`` pins the stage-N snapshot whose ``trial_outcomes``
    the probe classified. It is required: without it the orchestrator has
    no deterministic way to tell reps which upstream file to read, and
    the gap is what let two reps silently substitute a different source
    in a prior smoke test. Absent values on disk fail to load.

    ``excluded_prior_audit_session_ids`` records which prior audits (if
    any) had their trial keys subtracted from the eligible frame before
    the stratified draw. ``None`` when ``sample_args.exclude_prior_audits``
    was ``False`` (the flag was off) or the path bypassed eligibility
    (pinning); an empty tuple when the flag was on but no prior audits
    existed under the same byte-aware pooling identity tuple
    (``skill_fingerprint``, ``probe_snapshot_sha256``,
    ``rationale_source_path/_sha256``, ``stimulus_source_path/_sha256``,
    ``batch_allocation``);
    a populated tuple listing the contributing session IDs otherwise. The
    three states are semantically distinct — ``None`` says "exclusion not
    performed", ``()`` says "exclusion performed, no priors found".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_fingerprint: str
    probe_snapshot_sha256: str
    sample_args: SampleArgs
    seed: str
    trial_keys: tuple[tuple[str, str, int, int], ...]
    sampled_trials: tuple[SampledTrial, ...]
    stratification_counts: dict[str, int]
    forensics: SkillForensics = SkillForensics()
    selection_mode: SelectionMode
    selection_source: SelectionSource | None
    rationale_source: RationaleSource
    stimulus_source: StimulusSource
    excluded_prior_audit_session_ids: tuple[str, ...] | None
    # The flags the coverage-convergent draw classified as rare (pool-TRUE
    # count <= coverage_rare_max_count) and therefore floored — recorded for
    # auditability. Populated iff ``sample_args.batch_allocation ==
    # "coverage_convergent"``; ``None`` for the stratified / pinned draws.
    classified_rare_flags: tuple[str, ...] | None = None

    @field_validator("trial_keys")
    @classmethod
    def _trial_keys_must_be_unique(
        cls, v: tuple[tuple[str, str, int, int], ...]
    ) -> tuple[tuple[str, str, int, int], ...]:
        """M10: duplicate trial_keys would double-count rep slots.

        ``status`` computes ``total = len(trial_keys) * k`` and never
        observes the dedup via the "valid_pairs" set; a duplicate would
        silently inflate ``total`` and block the ``(complete)`` label
        forever. Fail at manifest construction time instead.
        """
        if len(v) != len(set(v)):
            dup = [t for t in set(v) if sum(1 for x in v if x == t) > 1]
            raise ValueError(
                f"trial_keys must be unique; found {len(dup)} duplicate(s), "
                f"first: {dup[0] if dup else None}"
            )
        return v

    @model_validator(mode="after")
    def _selection_source_matches_mode(self) -> "SampleManifest":
        """``selection_source`` is populated iff ``selection_mode`` is pinned-from-manifest.

        Stratified and inline-pinned draws have no source file to hash,
        so carrying a ``SelectionSource`` would be a provenance lie.
        Inverse: a pinned-from-manifest draw without a source would erase
        the pointer to the manifest its trial set came from.
        """
        has_source = self.selection_source is not None
        wants_source = self.selection_mode == "pinned_from_manifest"
        if has_source != wants_source:
            raise ValueError(
                f"selection_source must be populated iff selection_mode is "
                f"'pinned_from_manifest'; got mode={self.selection_mode!r}, "
                f"source={'present' if has_source else 'None'}"
            )
        return self

    @model_validator(mode="after")
    def _excluded_prior_matches_flag(self) -> "SampleManifest":
        """``excluded_prior_audit_session_ids`` is populated iff the flag was on.

        ``sample_args.exclude_prior_audits == True`` means the sampler
        consulted prior audits; the manifest must carry the contributing
        session list (possibly empty if none existed). ``False`` means the
        sampler did not consult priors at all; carrying a tuple would
        misrepresent the draw's provenance. Pinned mode forces the flag to
        ``False`` via ``SampleArgs._pinning_invariants``, so this rule
        collapses to a single iff-check on the ``SampleArgs`` field.
        """
        populated = self.excluded_prior_audit_session_ids is not None
        expected = self.sample_args.exclude_prior_audits
        if populated != expected:
            raise ValueError(
                f"excluded_prior_audit_session_ids must be populated iff "
                f"sample_args.exclude_prior_audits is True; got "
                f"exclude_prior_audits={expected!r}, "
                f"excluded_prior_audit_session_ids="
                f"{'populated' if populated else 'None'}"
            )
        return self

    @model_validator(mode="after")
    def _classified_rare_flags_matches_allocation(self) -> "SampleManifest":
        """``classified_rare_flags`` is populated iff the draw was convergent.

        The coverage-convergent draw records which flags it floored; the
        stratified and pinned draws have no floor, so carrying the field would
        misrepresent how the sample was made. A single iff-check on
        ``sample_args.batch_allocation`` keeps the manifest honest.
        """
        populated = self.classified_rare_flags is not None
        expected = self.sample_args.batch_allocation == "coverage_convergent"
        if populated != expected:
            raise ValueError(
                f"classified_rare_flags must be populated iff "
                f"sample_args.batch_allocation == 'coverage_convergent'; got "
                f"batch_allocation={self.sample_args.batch_allocation!r}, "
                f"classified_rare_flags="
                f"{'populated' if populated else 'None'}"
            )
        return self


def compute_probe_snapshot_sha256(path: Path) -> str:
    """SHA-256 of the probe snapshot file's raw bytes.

    Used by the rep-time and aggregate-time drift checks that need a
    SHA without parsing the file. The sample-time path goes through
    :func:`prepare_sample_sources` instead, which derives the SHA and
    the parsed payload from a single read to close the TOCTOU window
    between hashing and parsing.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SampleSourcePayloads(BaseModel):
    """Single-read snapshot of every byte input to ``draw_sample``.

    The sample-time TOCTOU contract is "bytes in, parsed payloads
    through": each source file is read exactly once, the SHA-256 is
    derived from those exact bytes, and the parsed payload is derived
    from those same bytes. Downstream consumers (the manifest-drift
    refusal check, the prior-audit lookup, and ``draw_sample``'s
    eligibility / coverage / scenario context validation) receive the
    parsed payloads — they never re-read the source files. A mutation
    on disk between :func:`prepare_sample_sources` and any later step
    therefore cannot let the manifest's pinned SHA disagree with the
    bytes the trial selection or validation observed.

    Treated as immutable by convention: the model itself is frozen
    (no field reassignment), and consumers must not mutate the nested
    ``probe_snapshot_payload`` / ``rationale_source_payload`` /
    ``stimuli`` containers in place. ``arbitrary_types_allowed`` lets
    the parsed JSON dicts pass through without per-field schema
    enforcement, since their internal shape is enforced by the helpers
    that read them (e.g., :func:`build_joined_eligible_frame_from_payloads`).
    """

    model_config = ConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid"
    )

    probe_snapshot_payload: dict
    probe_snapshot_sha256: str
    rationale_source_payload: dict
    rationale_source_sha256: str
    stimuli: dict[str, dict[str, str]]
    stimulus_source_sha256: str


def _read_labeled_bytes(path: Path, label: str) -> bytes:
    """Read raw bytes; on missing/IO error wrap with a sample-source label.

    The CLI catches ``FileNotFoundError`` from
    :func:`prepare_sample_sources` and prints its message verbatim, so
    surfacing which source file was unreadable (``probe snapshot`` /
    ``rationale source`` / ``stimulus source``) keeps the diagnostic
    actionable without exposing the internal source-of-error to the
    operator.
    """
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} missing at {path}: {exc}") from exc


def prepare_sample_sources(
    *,
    probe_snapshot_path: Path,
    rationale_source_path: Path,
    stimulus_source_path: Path,
) -> SampleSourcePayloads:
    """Read each source file once; hash AND parse from those exact bytes.

    Returns a frozen :class:`SampleSourcePayloads` whose SHA fields and
    parsed payloads were derived from the same byte snapshot per source.
    All sample-time consumers (manifest-drift refusal, prior-audit
    lookup, :func:`draw_sample`) thread this object through, so any
    mutation to the underlying files between this call and the manifest
    write cannot create a SHA-vs-content mismatch — the bytes captured
    here are what flow through.

    Raises:
        FileNotFoundError: if any path is missing; the message labels
            which source file (``probe snapshot``, ``rationale source``,
            ``stimulus source``) was unreadable.
        json.JSONDecodeError: if the probe snapshot or rationale source
            cannot be parsed as JSON.
        ValueError: from :func:`parse_stimuli_jsonl` when the stimulus
            JSONL is malformed (missing required keys, duplicate
            ``custom_id``, etc.).
    """
    probe_bytes = _read_labeled_bytes(probe_snapshot_path, "probe snapshot")
    rationale_bytes = _read_labeled_bytes(
        rationale_source_path, "rationale source"
    )
    stimulus_bytes = _read_labeled_bytes(
        stimulus_source_path, "stimulus source"
    )
    return SampleSourcePayloads(
        probe_snapshot_payload=json.loads(probe_bytes),
        probe_snapshot_sha256=hashlib.sha256(probe_bytes).hexdigest(),
        rationale_source_payload=json.loads(rationale_bytes),
        rationale_source_sha256=hashlib.sha256(rationale_bytes).hexdigest(),
        stimuli=parse_stimuli_jsonl(stimulus_bytes.decode("utf-8")),
        stimulus_source_sha256=hashlib.sha256(stimulus_bytes).hexdigest(),
    )


def seed_from_inputs(
    skill_fingerprint: str,
    args: SampleArgs,
    probe_snapshot_sha256: str,
    rationale_source_repo_relative: str,
    rationale_source_sha256: str,
) -> str:
    """Combine the five seed inputs into a single 64-hex seed.

    The seed depends on:

    - ``skill_fingerprint`` — pins the Skill body version
    - ``canonical_json(args)`` — pins the sampling-algorithm parameters
      (``sample_size``, ``stratify_by``, ``probe_snapshot_path``,
      ``pinned_trial_keys``)
    - ``probe_snapshot_sha256`` — pins the probe snapshot bytes
    - ``rationale_source_repo_relative`` — pins which stage-N snapshot
      file was used, as a **repo-relative string**
    - ``rationale_source_sha256`` — pins the stage-N snapshot bytes

    Passing ``rationale_source_repo_relative`` as a ``str`` (not a
    ``Path``) is deliberate: the absolute filesystem path can differ
    between machines for the same conceptual file (CI vs. laptop;
    worktree A vs. worktree B), but the repo-relative form is stable
    across checkouts. The CLI is the sole producer of the repo-relative
    form via ``relative_to(repo_root)`` before any identity computation
    runs. The absolute path is used only for reading bytes and
    computing ``rationale_source_sha256``.

    Including both the path and the SHA catches every useful change:
    re-pointing to a different file (path changes → seed changes) and
    mutating the same file in place (SHA changes → seed changes).

    Inputs are concatenated as raw UTF-8 bytes of their hex/JSON forms
    before the final SHA-256 so the seed shifts with any change.
    """
    payload = (
        skill_fingerprint.encode("utf-8")
        + b"|"
        + canonical_json(args).encode("utf-8")
        + b"|"
        + probe_snapshot_sha256.encode("utf-8")
        + b"|"
        + rationale_source_repo_relative.encode("utf-8")
        + b"|"
        + rationale_source_sha256.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _record_stratum(
    record: "JoinedRecord", stratify_by: tuple[str, ...]
) -> str:
    """Compute the composite stratum key for one joined record.

    Reads fields off a :class:`JoinedRecord`. Callers in the pinned-mode
    path convert raw probe dicts to :class:`JoinedRecord` via
    :func:`_joined_record_from_probe_dict` so both draw paths share one
    type at this seam.
    """
    parts: list[str] = []
    for dim in stratify_by:
        if dim == "condition":
            parts.append(condition_from_example_id(record.example_id))
        elif dim == "articulation":
            parts.append(str(record.articulated_operational_interpretation))
        elif dim == "governing":
            parts.append(
                str(record.operational_interpretation_governed_judgment)
            )
        elif dim == "split_vote":
            articulation_split = (
                record.articulated_operational_interpretation_votes_for > 0
                and record.articulated_operational_interpretation_votes_against
                > 0
            )
            governing_split = (
                record.operational_interpretation_governed_judgment_votes_for > 0
                and record.operational_interpretation_governed_judgment_votes_against
                > 0
            )
            parts.append(
                "split" if (articulation_split or governing_split) else "unanimous"
            )
        elif dim == "config":
            parts.append(record.config_key)
        elif dim in _MISATTRIBUTION_FLAG_KEYS:
            # The five misattribution dimensions are named by their full flag
            # key, which is also the ``JoinedRecord`` attribute name.
            parts.append(str(getattr(record, dim)))
        else:
            # Defensive: SampleArgs validation should have caught this.
            raise ValueError(f"Unknown stratum dimension: {dim!r}")
    return "|".join(parts)


class JoinedRecord(BaseModel):
    """One trial in the joined eligible audit frame.

    Carries every probe-side field ``_record_stratum`` reads, plus the
    four-tuple trial identity. Eligibility filtering (``not
    matches_ground_truth and parse_error is None``) is enforced when the
    frame is built, so consumers can treat the frame as "already
    filtered" — no secondary re-filter pass required at the draw seam.
    The type is frozen so a joined frame returned by the helper cannot be
    mutated in place by a downstream caller.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_key: str
    example_id: str
    batch_index: int
    trial: int
    articulated_operational_interpretation: bool | None
    operational_interpretation_governed_judgment: bool | None
    # The five misattribution flag values — carried so the coverage-convergent
    # floor can read rare flag-TRUE membership and so all seven flags are
    # uniformly stratifiable. ``None`` means the probe could not determine the
    # flag (quorum failure); it is distinct from ``False``.
    misattributed_scenario_current_value: bool | None
    misattributed_scenario_historical_value: bool | None
    misattributed_scenario_proposed_value: bool | None
    misattributed_assistant_fallback_value: bool | None
    misattributed_grader_claim: bool | None
    articulated_operational_interpretation_votes_for: int
    articulated_operational_interpretation_votes_against: int
    operational_interpretation_governed_judgment_votes_for: int
    operational_interpretation_governed_judgment_votes_against: int

    @property
    def trial_key(self) -> tuple[str, str, int, int]:
        """The four-tuple identity used as the primary join key everywhere."""
        return (self.config_key, self.example_id, self.batch_index, self.trial)


def _joined_record_from_probe_dict(rec: dict) -> JoinedRecord:
    """Materialize a :class:`JoinedRecord` from a raw probe-side dict.

    Single source of truth for the probe-dict → :class:`JoinedRecord`
    mapping. Both the stratified path (via
    :func:`build_joined_eligible_frame`) and the pinned path (via
    :func:`_pinned_draw`) delegate here so a future field addition has
    exactly one call site to update.
    """
    return JoinedRecord(
        config_key=rec["config_key"],
        example_id=rec["example_id"],
        batch_index=int(rec["batch_index"]),
        trial=int(rec["trial"]),
        articulated_operational_interpretation=rec.get(
            "articulated_operational_interpretation"
        ),
        operational_interpretation_governed_judgment=rec.get(
            "operational_interpretation_governed_judgment"
        ),
        # Strict reads (``rec[key]``, not ``.get``): the five misattribution
        # flags are part of the seven-flag probe schema, so an absent key is a
        # malformed record, not an acceptable silent-False. A present ``None``
        # value (quorum failure) is preserved.
        **{key: rec[key] for key in _MISATTRIBUTION_FLAG_KEYS},
        articulated_operational_interpretation_votes_for=int(
            rec.get("articulated_operational_interpretation_votes_for", 0)
        ),
        articulated_operational_interpretation_votes_against=int(
            rec.get("articulated_operational_interpretation_votes_against", 0)
        ),
        operational_interpretation_governed_judgment_votes_for=int(
            rec.get("operational_interpretation_governed_judgment_votes_for", 0)
        ),
        operational_interpretation_governed_judgment_votes_against=int(
            rec.get(
                "operational_interpretation_governed_judgment_votes_against", 0
            )
        ),
    )


def build_joined_eligible_frame(
    *,
    probe_snapshot_payload: dict,
    rationale_source_payload: dict,
) -> tuple[JoinedRecord, ...]:
    """Return the eligible audit frame: probe ∩ stage, filtered by eligibility.

    Operates on parsed payloads — never on file paths — so the bytes
    used for hashing (in :func:`prepare_sample_sources`) and the bytes
    used for eligibility filtering are guaranteed to be the same.
    Inner-joins on the ``(config_key, example_id, batch_index, trial)``
    tuple, keeps only trials where the stage side has
    ``matches_ground_truth == False`` and ``parse_error is None``, and
    materializes each survivor as a :class:`JoinedRecord` carrying every
    probe-side field needed by ``_record_stratum``. Trials present in
    only one payload are silently dropped — the intersection is the
    sampler's population, so an orphan in either payload is not part of
    the audit. The rationale source payload is shape-validated via
    :func:`check_rationale_source_shape` so a probe-snapshot-shaped
    payload passed by mistake produces a named diagnostic instead of a
    ``KeyError``.
    """
    rs_payload = check_rationale_source_shape(payload=rationale_source_payload)

    # Strict per-record validation: the joined eligible frame's filter
    # is only meaningful if every outcome carries the fields it reads.
    # Permissive ``.get()`` access would silently treat missing
    # ``matches_ground_truth`` / ``parse_error`` as eligible, re-widening
    # the pool back toward "all overlapping trials". Fail loudly with
    # the trial identity so the operator can point at exactly which
    # stage-side record is malformed. ``parse_error: None`` is valid
    # (it means "parsed successfully"); field-absent is the error.
    required_fields = (
        "config_key",
        "example_id",
        "batch_index",
        "trial",
        "matches_ground_truth",
        "parse_error",
    )
    eligible_keys: set[tuple[str, str, int, int]] = set()
    for outcome in rs_payload["trial_outcomes"]:
        missing = [f for f in required_fields if f not in outcome]
        if missing:
            raise ValueError(
                f"rationale source trial_outcomes record is missing "
                f"required field(s) {missing}; offending record: "
                f"config_key={outcome.get('config_key')!r}, "
                f"example_id={outcome.get('example_id')!r}, "
                f"batch_index={outcome.get('batch_index')!r}, "
                f"trial={outcome.get('trial')!r}"
            )
        if outcome["matches_ground_truth"] is True:
            continue
        if outcome["parse_error"] is not None:
            continue
        eligible_keys.add(
            (
                outcome["config_key"],
                outcome["example_id"],
                int(outcome["batch_index"]),
                int(outcome["trial"]),
            )
        )

    frame: list[JoinedRecord] = []
    for rec in probe_snapshot_payload["trial_records"]:
        key = (
            rec["config_key"],
            rec["example_id"],
            int(rec["batch_index"]),
            int(rec["trial"]),
        )
        if key not in eligible_keys:
            continue
        frame.append(_joined_record_from_probe_dict(rec))
    return tuple(frame)


def eligible_trial_keys(
    *,
    probe_snapshot_payload: dict,
    rationale_source_payload: dict,
) -> frozenset[tuple[str, str, int, int]]:
    """Thin derivative of ``build_joined_eligible_frame`` returning just keys.

    Useful for consumers (e.g., the pinned-key warning path) that only
    need to check membership in the eligible set and don't need the
    probe-side record fields.
    """
    frame = build_joined_eligible_frame(
        probe_snapshot_payload=probe_snapshot_payload,
        rationale_source_payload=rationale_source_payload,
    )
    return frozenset(r.trial_key for r in frame)


def _stratified_draw(
    pool: list[tuple[tuple[str, str, int, int], str]],
    sample_size: int,
    rng: random.Random,
) -> list[tuple[tuple[str, str, int, int], str]]:
    """Draw ``sample_size`` items from a (trial_key, stratum)-annotated pool.

    Per-stratum target is ``floor(sample_size / n_strata)`` with the
    remainder distributed via a seeded shuffle of the stratum list. When
    a stratum is smaller than its target, the shortfall rolls over to
    remaining strata in the same pass — so the final count is
    ``min(sample_size, len(pool))`` without silently dropping.
    """
    if sample_size >= len(pool):
        return sorted(pool, key=lambda item: item[0])

    by_stratum: dict[str, list[tuple[tuple[str, str, int, int], str]]] = {}
    for item in pool:
        by_stratum.setdefault(item[1], []).append(item)

    strata = sorted(by_stratum.keys())
    base_target = sample_size // len(strata)
    remainder = sample_size - base_target * len(strata)

    # Seeded shuffle decides which strata get the extra +1 from the remainder.
    remainder_order = list(strata)
    rng.shuffle(remainder_order)
    boosted = set(remainder_order[:remainder])

    selected: list[tuple[tuple[str, str, int, int], str]] = []
    shortfall = 0
    for stratum in strata:
        target = base_target + (1 if stratum in boosted else 0)
        stratum_items = sorted(by_stratum[stratum], key=lambda item: item[0])
        if len(stratum_items) <= target:
            selected.extend(stratum_items)
            shortfall += target - len(stratum_items)
        else:
            draws = rng.sample(stratum_items, target)
            selected.extend(draws)

    # Fill any shortfall from trials not yet selected, deterministically.
    if shortfall > 0:
        chosen_keys = {item[0] for item in selected}
        remaining = sorted(
            (item for item in pool if item[0] not in chosen_keys),
            key=lambda item: item[0],
        )
        if remaining:
            rng.shuffle(remaining)
            selected.extend(remaining[:shortfall])

    return sorted(selected, key=lambda item: item[0])


def _pinned_draw(
    pinned: tuple[tuple[str, str, int, int], ...],
    records: list[dict],
    stratify_by: tuple[str, ...],
) -> list[tuple[tuple[str, str, int, int], str]]:
    """Validate each pinned key against the pool and tag it with its stratum.

    Bypasses ``_stratified_draw`` entirely — no RNG consumption — because
    explicit pinning names the trials to use. Eligibility filtering is
    also bypassed: pinning is an explicit user choice to audit specific
    trials, including (possibly) ones outside the joined eligible frame.
    Missing keys in the probe snapshot still raise so the researcher
    sees exactly what's absent rather than getting a silently smaller
    sample. Callers responsible for warning about pinned-but-ineligible
    trials live at the ``draw_sample`` level, not here.
    """
    by_key: dict[tuple[str, str, int, int], JoinedRecord] = {
        (
            rec["config_key"],
            rec["example_id"],
            int(rec["batch_index"]),
            int(rec["trial"]),
        ): _joined_record_from_probe_dict(rec)
        for rec in records
    }
    missing = [key for key in pinned if key not in by_key]
    if missing:
        raise ValueError(
            f"pinned_trial_keys not present in probe snapshot: {missing}"
        )
    tagged = [
        (key, _record_stratum(by_key[key], stratify_by)) for key in pinned
    ]
    # Sort by trial_key to match the canonical ordering ``_stratified_draw``
    # emits — downstream aggregation and reporting assume sorted order.
    return sorted(tagged, key=lambda item: item[0])


def compute_rationale_source_sha256(path: Path) -> str:
    """SHA-256 of the rationale source file's raw bytes.

    Kept separate from ``compute_probe_snapshot_sha256`` so the two
    provenance hashes have distinct call sites — easier to trace on
    grep when a provenance mismatch is being debugged.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_stimulus_source_sha256(path: Path) -> str:
    """SHA-256 of the stimulus source JSONL's raw bytes.

    Kept separate from ``compute_rationale_source_sha256`` so the two
    provenance hashes have distinct call sites and the stimulus pin is
    independently traceable. The stimulus file augments scenario context
    on the auditor's bundle and is pinned for byte-level reproducibility,
    not for sample identity (it is excluded from the sampling seed).
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_rationale_source_shape(
    payload: dict,
    *,
    source_label: str | None = None,
) -> dict:
    """Assert ``payload`` has ``trial_outcomes`` at the top level.

    Returns the payload unchanged when valid so callers can chain the
    check ahead of further processing. Raises a named ``ValueError``
    when the payload is a probe-snapshot (has ``trial_records`` only)
    or lacks both keys — these are the two orchestrator-error modes
    the diagnostic was introduced to catch.

    ``source_label`` is folded into the error message when provided so
    the operator sees which file or context the malformed payload came
    from; sample-time callers (which work from in-memory payloads)
    typically omit it, while rep-time callers
    (:func:`build_rationale_bundle`) pass the on-disk path.
    """
    label = f" at {source_label}" if source_label else ""
    if "trial_outcomes" not in payload:
        if "trial_records" in payload:
            raise ValueError(
                f"rationale source{label} lacks 'trial_outcomes'; "
                f"top-level key is 'trial_records' — this looks like a "
                f"probe-snapshot, not a rationale-source"
            )
        raise ValueError(
            f"rationale source{label} lacks both 'trial_outcomes' and "
            f"'trial_records'"
        )
    return payload


def _validate_rationale_source(
    rationale_source_payload: dict,
    chosen_trial_keys: tuple[tuple[str, str, int, int], ...],
) -> None:
    """Check the rationale source covers every chosen trial key.

    Fails fast at sample time with a named ``ValueError`` rather than
    letting a mismatch surface later as a rep-side ``KeyError``. The
    top-level-shape check is delegated to
    :func:`check_rationale_source_shape` so the rep-side bundle builder
    and the sample-side validator raise the same diagnostic wording.
    Operates on a parsed payload (no file I/O) so the bytes hashed at
    :func:`prepare_sample_sources` time are the bytes validated here.
    """
    payload = check_rationale_source_shape(rationale_source_payload)
    present = {
        (
            o["config_key"],
            o["example_id"],
            int(o["batch_index"]),
            int(o["trial"]),
        )
        for o in payload["trial_outcomes"]
    }
    missing = [key for key in chosen_trial_keys if key not in present]
    if missing:
        raise ValueError(
            f"rationale source is missing trial_outcomes entries for "
            f"{len(missing)} selected trial_key(s); first: {missing[0]}"
        )


def _validate_stimulus_source(
    stimuli: dict[str, dict[str, str]],
    chosen_trial_keys: tuple[tuple[str, str, int, int], ...],
) -> None:
    """Check the stimulus mapping covers every selected ``example_id`` AND
    that scenario context can be built for each.

    Operates on the parsed stimulus mapping (no file I/O) so the bytes
    hashed at :func:`prepare_sample_sources` time are the bytes validated
    here. Asserts each distinct ``example_id`` referenced by
    ``chosen_trial_keys`` has a corresponding stimulus record AND that
    :func:`build_scenario_context` can extract grader-analysis text from
    that record's ``user_content``. Raises a named ``ValueError`` at
    sample time so a coverage gap or a malformed Grader's Feedback
    section surfaces here instead of at rep-time bundle construction
    (where every rep would otherwise fail deep inside
    ``extract_grader_feedback``).

    The scenario context cross-validation parallels
    :func:`_validate_rationale_source`'s shape check: sampling fails
    fast on inputs that would make every later ``fetch-rationale``
    abort, rather than letting a malformed file pass the manifest
    write and poison the audit at rep time.
    """
    needed = {trial_key[1] for trial_key in chosen_trial_keys}
    missing = sorted(eid for eid in needed if eid not in stimuli)
    if missing:
        raise ValueError(
            f"stimulus source is missing {len(missing)} example_id(s) "
            f"referenced by selected trial_key(s); first: {missing[0]!r}"
        )
    # Scenario context buildability: catches malformed Grader's Feedback
    # sections, missing ``analysis`` fields, or unparseable JSON inside
    # the stimulus's ``user_content`` — failures the auditor's
    # ``RationaleBundle`` builder would otherwise hit at every rep.
    for example_id in sorted(needed):
        try:
            build_scenario_context(example_id, stimuli)
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"stimulus source cannot build scenario context for "
                f"example_id={example_id!r}: {exc}"
            ) from exc


def _derive_selection_mode(
    pinned: tuple[tuple[str, str, int, int], ...],
    source: "SelectionSource | None",
) -> SelectionMode:
    """``selection_mode`` is a function of ``pinned`` and ``source``.

    Keeping the derivation in one place (rather than taking ``selection_
    mode`` as a caller arg) avoids the possibility of the caller passing
    a mode that contradicts the other two values. The ``@model_validator``
    on ``SampleManifest`` re-checks the iff-invariant, so a programmer
    error here fails at manifest construction rather than downstream.
    """
    if not pinned:
        return "stratified"
    if source is None:
        return "pinned_inline"
    return "pinned_from_manifest"


# ── coverage-convergent batch allocation ─────────────────────────────────


class CampaignCumulative(BaseModel):
    """Cumulative composition of a coverage-convergent campaign's prior batches.

    Reconstructed from the prior-audit trial-key union intersected with the
    current eligible frame (cell membership is a pure function of each probe
    record), so the allocator needs no persisted state beyond the manifests
    prior batches already wrote. ``total`` is the count of prior-drawn trials,
    ``condition_counts`` their per-condition tally, and ``rare_flag_true_counts``
    the per-rare-flag TRUE tally — the three quantities the floor and the
    proportional top-up consult.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int
    condition_counts: dict[str, int]
    rare_flag_true_counts: dict[str, int]


def pool_flag_true_counts(frame: tuple[JoinedRecord, ...]) -> dict[str, int]:
    """Per-flag count of TRUE classifications over the full eligible frame.

    Keyed by the seven full flag keys (``RATIONALE_ANALYSIS_FLAG_KEYS``), each
    of which is a ``JoinedRecord`` attribute. ``None`` (probe quorum failure)
    is not counted as TRUE.
    """
    counts = {flag: 0 for flag in RATIONALE_ANALYSIS_FLAG_KEYS}
    for record in frame:
        for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
            if getattr(record, flag) is True:
                counts[flag] += 1
    return counts


def classify_rare_flags(
    frame: tuple[JoinedRecord, ...], rare_max_count: int
) -> tuple[str, ...]:
    """The flags whose pool-TRUE count is at or below ``rare_max_count``.

    A count threshold (not a prevalence threshold) is used because the rare
    cluster and the common cluster are separated by a wide, dataset-stable gap;
    a fixed prevalence threshold would straddle a flag's primary-vs-ablation
    rate. Empty classes (0 TRUE) are included — the floor is a no-op for them.
    Returned sorted for deterministic iteration.
    """
    counts = pool_flag_true_counts(frame)
    return tuple(
        sorted(flag for flag, n in counts.items() if n <= rare_max_count)
    )


def coverage_convergent_pool_stats(
    frame: tuple[JoinedRecord, ...], rare_max_count: int
) -> tuple[dict[str, float], dict[str, int], tuple[str, ...]]:
    """Return ``(condition_fractions, class_sizes, rare_flags)`` for ``frame``.

    ``condition_fractions`` is the pool's condition distribution (the
    convergence target); ``class_sizes`` is the per-flag pool-TRUE count (caps
    the floor); ``rare_flags`` is the floored subset (count ≤ ``rare_max_count``).
    """
    class_sizes = pool_flag_true_counts(frame)
    rare_flags = classify_rare_flags(frame, rare_max_count)
    n = len(frame)
    cond_counts: dict[str, int] = {}
    for record in frame:
        cond = condition_from_example_id(record.example_id)
        cond_counts[cond] = cond_counts.get(cond, 0) + 1
    condition_fractions = {c: cnt / n for c, cnt in cond_counts.items()} if n else {}
    return condition_fractions, class_sizes, rare_flags


def campaign_cumulative_counts(
    prior_union: frozenset[tuple[str, str, int, int]],
    frame: tuple[JoinedRecord, ...],
    rare_flags: tuple[str, ...],
) -> CampaignCumulative:
    """Tally the prior campaign batches by condition and rare-flag TRUE.

    Only prior keys that are still in the current eligible frame are counted
    (a same-campaign invariant; the intersection is defensive). Cell membership
    is read off the frame's :class:`JoinedRecord` for each prior key.
    """
    by_key = {record.trial_key: record for record in frame}
    cond_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {flag: 0 for flag in rare_flags}
    total = 0
    for key in prior_union:
        record = by_key.get(key)
        if record is None:
            continue
        total += 1
        cond = condition_from_example_id(record.example_id)
        cond_counts[cond] = cond_counts.get(cond, 0) + 1
        for flag in rare_flags:
            if getattr(record, flag) is True:
                class_counts[flag] += 1
    return CampaignCumulative(
        total=total,
        condition_counts=cond_counts,
        rare_flag_true_counts=class_counts,
    )


def _largest_remainder_allocation(
    weights: dict[str, float], total: int
) -> dict[str, int]:
    """Apportion ``total`` integer units across keys proportional to ``weights``.

    Hamilton's largest-remainder method: floor each ideal share, then hand the
    leftover units to the largest fractional remainders (ties broken by key,
    ascending, for determinism). When every weight is zero the units are spread
    evenly so nothing is silently dropped.
    """
    keys = sorted(weights)
    if not keys:
        return {}
    weight_sum = sum(weights[k] for k in keys)
    if weight_sum <= 0.0:
        shares = {k: total / len(keys) for k in keys}
    else:
        shares = {k: total * weights[k] / weight_sum for k in keys}
    floored = {k: int(math.floor(shares[k])) for k in keys}
    leftover = total - sum(floored.values())
    ranked = sorted(keys, key=lambda k: (-(shares[k] - math.floor(shares[k])), k))
    for k in ranked[:leftover]:
        floored[k] += 1
    return floored


def _coverage_convergent_draw(
    frame: tuple[JoinedRecord, ...],
    sample_size: int,
    *,
    stratify_by: tuple[str, ...],
    prior_cumulative: CampaignCumulative,
    rare_flags: tuple[str, ...],
    floor_target: int,
    pool_condition_fractions: dict[str, float],
    pool_class_sizes: dict[str, int],
    rng: random.Random,
) -> list[tuple[tuple[str, str, int, int], str]]:
    """Breadth-first floor + proportional-convergence draw, without replacement.

    ``frame`` is the RESIDUAL eligible frame (prior-campaign keys already
    subtracted). Returns ``(trial_key, stratum)`` pairs sorted by trial_key,
    matching :func:`_stratified_draw`'s contract.

    (A) Coverage floor: each rare flag is drawn up to ``min(floor_target, pool
    class size)`` net of prior coverage, **uniformly within that flag's TRUE
    set** (so the per-value-class estimate stays valid) and **deduplicated by
    counting** — a drawn trial is credited to every rare flag it is TRUE on, so
    one draw can retire multiple floors. (B) Proportional top-up: the remaining
    slots are apportioned across conditions by largest-remainder toward the
    pool condition fractions, debiting floor picks from the condition ledger so
    a condition cannot be pushed past its cumulative target; draws within a
    condition are uniform-WOR with shortfall redistribution.

    Terminal batch: if the residual is no larger than ``sample_size`` the whole
    residual is returned (the convergence endpoint). The caller compares the
    returned count to the requested size to surface a short-batch note.
    """
    by_key = {record.trial_key: record for record in frame}
    if sample_size >= len(frame):
        return [
            (key, _record_stratum(by_key[key], stratify_by))
            for key in sorted(by_key)
        ]

    picked_keys: set[tuple[str, str, int, int]] = set()
    picked_class: dict[str, int] = {flag: 0 for flag in rare_flags}
    picked_cond: dict[str, int] = {}

    def _credit(record: JoinedRecord) -> None:
        picked_keys.add(record.trial_key)
        cond = condition_from_example_id(record.example_id)
        picked_cond[cond] = picked_cond.get(cond, 0) + 1
        for flag in rare_flags:
            if getattr(record, flag) is True:
                picked_class[flag] += 1

    # (A) Coverage floor — round-robin across the binding rare flags (one draw
    # per flag per pass) so a tight floor budget is shared fairly rather than
    # consumed in flag-name order. When the budget is ample — the common case,
    # since total floor demand is small — every flag still reaches
    # ``min(F, pool class size)``. Each draw is uniform within that flag's TRUE
    # set and credited by dedup, so one draw can retire multiple floors and the
    # next pass re-reads the (reduced) per-flag need.
    remaining = sample_size

    def _floor_need(flag: str) -> int:
        target = min(floor_target, pool_class_sizes.get(flag, 0))
        already = (
            prior_cumulative.rare_flag_true_counts.get(flag, 0)
            + picked_class[flag]
        )
        return target - already

    while remaining > 0:
        progressed = False
        for flag in rare_flags:
            if remaining <= 0:
                break
            if _floor_need(flag) <= 0:
                continue
            candidates = sorted(
                (
                    record
                    for record in frame
                    if getattr(record, flag) is True
                    and record.trial_key not in picked_keys
                ),
                key=lambda record: record.trial_key,
            )
            if not candidates:
                continue
            _credit(rng.choice(candidates))
            remaining -= 1
            progressed = True
        if not progressed:
            break

    # (B) Proportional top-up over conditions.
    if remaining > 0:
        target_total = prior_cumulative.total + sample_size
        conditions = sorted(pool_condition_fractions)
        deficit = {
            cond: max(
                0.0,
                pool_condition_fractions[cond] * target_total
                - (
                    prior_cumulative.condition_counts.get(cond, 0)
                    + picked_cond.get(cond, 0)
                ),
            )
            for cond in conditions
        }
        alloc = _largest_remainder_allocation(deficit, remaining)
        shortfall = 0
        for cond in conditions:
            candidates = sorted(
                (
                    record
                    for record in frame
                    if condition_from_example_id(record.example_id) == cond
                    and record.trial_key not in picked_keys
                ),
                key=lambda record: record.trial_key,
            )
            want = alloc.get(cond, 0)
            take = min(want, len(candidates))
            for record in rng.sample(candidates, take):
                _credit(record)
            shortfall += want - take
        if shortfall > 0:
            leftover = sorted(
                (
                    record
                    for record in frame
                    if record.trial_key not in picked_keys
                ),
                key=lambda record: record.trial_key,
            )
            for record in rng.sample(leftover, min(shortfall, len(leftover))):
                _credit(record)

    return sorted(
        ((key, _record_stratum(by_key[key], stratify_by)) for key in picked_keys),
        key=lambda item: item[0],
    )


def draw_sample(
    skill_fingerprint: str,
    args: SampleArgs,
    forensics: SkillForensics | None = None,
    *,
    source_payloads: SampleSourcePayloads,
    rationale_source_repo_relative: str,
    stimulus_source_repo_relative: str,
    selection_source: "SelectionSource | None" = None,
    prior_audit_keys: frozenset[tuple[str, str, int, int]] | None = None,
    prior_audit_session_ids: tuple[str, ...] | None = None,
) -> SampleManifest:
    """Produce a deterministic sample of probe trials from the joined eligible frame.

    Consumes a single :class:`SampleSourcePayloads` snapshot — never
    reads files — so the manifest's pinned SHAs and the bytes used for
    eligibility filtering, prior-audit subtraction, and coverage
    validation are derived from one read per source. Callers (the CLI
    layer) build the snapshot via :func:`prepare_sample_sources` once
    and thread it through both the prior-audit lookup and this call,
    closing the TOCTOU window between SHA hash and parsed-content
    consumption.

    The stratified path draws from the joined eligible frame —
    intersection of probe ``trial_records`` with stage
    ``trial_outcomes`` filtered to ``not matches_ground_truth and
    parse_error is None``. Population is the intersection, not just
    whichever trials happen to live in the probe snapshot. Changing
    ``skill_fingerprint``, any field of ``SampleArgs``, the probe
    snapshot bytes, the rationale source path, or the rationale source
    bytes re-rolls the sample; changing only ``k`` does not, because K
    is excluded from the args portion of the seed.

    When ``args.pinned_trial_keys`` is non-empty, the sampler bypasses
    both the stratified draw and the eligibility filter — explicit
    pinning names the trials to audit, including ones outside the
    eligible frame. A stderr warning is emitted when any pinned key
    falls outside the joined eligible frame, **as a best-effort
    check**: if the rationale source is missing the eligibility fields
    ``build_joined_eligible_frame`` requires (e.g., a partial stage
    snapshot that still covers the pinned ``trial_keys`` for rep-time
    fetching), the warning is skipped with a stderr note and the draw
    proceeds. Pinned mode's contract is "explicit pinning wins"; a
    best-effort advisory must never convert a valid pinned run into a
    refusal. ``selection_source`` is recorded on the manifest when a
    prior ``sample_manifest.json`` provided the pinned keys.

    The repo-relative path kwargs carry sample identity but no byte
    I/O: they populate ``RationaleSource.path`` / ``StimulusSource.path``
    on the manifest and feed :func:`seed_from_inputs` for the rationale
    source. The CLI is the sole producer of the repo-relative form via
    its ``relative_to(repo_root)`` normalization.

    ``prior_audit_keys`` and ``prior_audit_session_ids`` feed the
    ``exclude_prior_audits`` path. They are produced by
    :func:`~utils.probe_audit.storage.prior_audit_trial_keys` at the CLI
    layer (the sampler itself is kept free of storage-layer coupling) and
    must be supplied iff ``args.exclude_prior_audits`` is ``True``. When
    supplied, the keys in the set are subtracted from the joined eligible
    frame before ``_stratified_draw``; a reduced frame smaller than
    ``args.sample_size`` raises ``ValueError``. The ``session_ids`` list
    flows onto ``SampleManifest.excluded_prior_audit_session_ids`` as
    provenance.

    Raises:
        ValueError: if the joined eligible frame is empty (stratified
            path only — pinned mode never raises here because it
            bypasses the filter), if any pinned key is absent from the
            probe snapshot, if a stage-side outcome record is missing
            an eligibility field on the **stratified** draw path, or if
            ``exclude_prior_audits`` reduces the eligible frame below
            ``sample_size``. Pinned mode swallows the eligibility-field
            error class for the best-effort eligibility-advisory check
            (see above).
    """
    _exclusion_requested = args.exclude_prior_audits
    _exclusion_supplied = prior_audit_keys is not None
    if _exclusion_requested != _exclusion_supplied:
        raise ValueError(
            f"prior_audit_keys must be supplied iff "
            f"args.exclude_prior_audits is True; got "
            f"exclude_prior_audits={_exclusion_requested!r}, "
            f"prior_audit_keys="
            f"{'supplied' if _exclusion_supplied else 'None'}"
        )
    if (prior_audit_session_ids is None) != (prior_audit_keys is None):
        raise ValueError(
            "prior_audit_session_ids and prior_audit_keys must both be "
            "supplied or both be None"
        )
    probe_sha = source_payloads.probe_snapshot_sha256
    rationale_sha = source_payloads.rationale_source_sha256
    # The stimulus SHA is captured for byte-level provenance only — it
    # augments per-trial context on each bundle but does not change which
    # trials are eligible for sampling, so it is deliberately excluded
    # from ``seed_from_inputs``.
    stimulus_sha = source_payloads.stimulus_source_sha256
    seed_hex = seed_from_inputs(
        skill_fingerprint,
        args,
        probe_sha,
        rationale_source_repo_relative,
        rationale_sha,
    )

    # Populated only by the coverage-convergent draw (the flags whose pool-TRUE
    # count is at or below ``coverage_rare_max_count``); ``None`` otherwise.
    classified_rare_flags: tuple[str, ...] | None = None

    if args.pinned_trial_keys:
        records = list(source_payloads.probe_snapshot_payload["trial_records"])
        chosen = _pinned_draw(args.pinned_trial_keys, records, args.stratify_by)
        # Pinning bypasses the eligibility filter by design, but surface a
        # stderr warning when any pinned key falls outside the joined
        # eligible frame so the researcher notices. Silent acceptance
        # here is the path of least resistance and the most dangerous;
        # refusing would contradict "explicit pinning wins" — warning is
        # the middle ground that preserves both properties.
        #
        # The eligibility computation is BEST-EFFORT: ``build_joined_
        # eligible_frame`` is strict on ``matches_ground_truth`` /
        # ``parse_error``, so a partial rationale source that still
        # covers the pinned ``trial_keys`` for rep-time fetching would
        # otherwise convert a valid pinned run into a hard refusal.
        # Catch ``ValueError`` here, emit a ``note:`` explaining what
        # was skipped, and continue — the pinned draw itself never
        # reads eligibility fields.
        try:
            eligible = eligible_trial_keys(
                probe_snapshot_payload=source_payloads.probe_snapshot_payload,
                rationale_source_payload=source_payloads.rationale_source_payload,
            )
        except ValueError as exc:
            print(
                f"note: skipping pinned-key eligibility check — rationale "
                f"source cannot be evaluated for eligibility ({exc}). "
                f"Pinning proceeds; rep-time rationale fetching still "
                f"requires every pinned trial_key to be covered.",
                file=sys.stderr,
            )
            eligible = None
        if eligible is None:
            ineligible_pinned = []
        else:
            ineligible_pinned = [
                key for key in args.pinned_trial_keys if key not in eligible
            ]
        if ineligible_pinned:
            print(
                f"warning: {len(ineligible_pinned)} pinned trial key(s) fall "
                f"outside the joined eligible frame "
                f"(matches_ground_truth=True or parse_error set): "
                f"{ineligible_pinned}",
                file=sys.stderr,
            )
    else:
        full_frame = build_joined_eligible_frame(
            probe_snapshot_payload=source_payloads.probe_snapshot_payload,
            rationale_source_payload=source_payloads.rationale_source_payload,
        )
        if not full_frame:
            raise ValueError(
                "joined eligible frame is empty; check that the rationale "
                "source was produced from the same stage-N outputs as the "
                "probe snapshot and that some trials have "
                "matches_ground_truth=False and parse_error=None"
            )
        # Subtract the prior-audit trial keys when the flag is on. The
        # subtraction runs after ``build_joined_eligible_frame`` so the
        # eligibility filter still runs exactly once; the resulting residual
        # frame is what the draw sees.
        if args.exclude_prior_audits:
            # Guarded by the contract check at the top of this function,
            # so ``prior_audit_keys`` is known non-None here.
            residual = tuple(
                record for record in full_frame
                if record.trial_key not in prior_audit_keys
            )
        else:
            residual = full_frame
        rng = random.Random(int(seed_hex, 16))
        if args.batch_allocation == "coverage_convergent":
            # ``coverage_convergent`` requires ``exclude_prior_audits`` (the
            # SampleArgs validator), so ``residual`` is the prior-subtracted
            # frame and ``prior_audit_keys`` is non-None. The floor + convergent
            # top-up draw against the FULL pool's stats and this campaign's
            # cumulative composition; the terminal batch (residual <=
            # sample_size) draws all remaining rather than erroring — that is
            # the convergence endpoint, not a shortfall.
            (
                cond_fractions,
                class_sizes,
                rare_flags,
            ) = coverage_convergent_pool_stats(
                full_frame, args.coverage_rare_max_count
            )
            prior_cumulative = campaign_cumulative_counts(
                prior_audit_keys, full_frame, rare_flags
            )
            chosen = _coverage_convergent_draw(
                residual,
                args.sample_size,
                stratify_by=args.stratify_by,
                prior_cumulative=prior_cumulative,
                rare_flags=rare_flags,
                floor_target=args.coverage_floor,
                pool_condition_fractions=cond_fractions,
                pool_class_sizes=class_sizes,
                rng=rng,
            )
            classified_rare_flags = rare_flags
        else:
            # Stratified draw: a below-sample-size residual is a hard error
            # (the caller asked for N trials of NEW evidence and would
            # otherwise quietly get fewer). The convergent draw deliberately
            # does NOT share this — its terminal batch draws what remains.
            if args.exclude_prior_audits and len(residual) < args.sample_size:
                raise ValueError(
                    f"after excluding {len(prior_audit_keys)} prior-audit "
                    f"trial key(s), the reduced eligible frame has "
                    f"{len(residual)} trial(s), which is smaller than "
                    f"--sample-size ({args.sample_size}). Either re-run "
                    "with --allow-prior-audit-overlap to draw from the "
                    "full frame, reduce --sample-size, or wait for more "
                    "upstream trials to become eligible."
                )
            pool: list[tuple[tuple[str, str, int, int], str]] = [
                (record.trial_key, _record_stratum(record, args.stratify_by))
                for record in residual
            ]
            chosen = _stratified_draw(pool, args.sample_size, rng)

    chosen_keys = tuple(trial_key for trial_key, _ in chosen)
    _validate_rationale_source(
        source_payloads.rationale_source_payload, chosen_keys
    )
    _validate_stimulus_source(source_payloads.stimuli, chosen_keys)
    rationale_source = RationaleSource(
        path=rationale_source_repo_relative,
        sha256=rationale_sha,
    )
    stimulus_source = StimulusSource(
        path=stimulus_source_repo_relative,
        sha256=stimulus_sha,
    )

    sampled_trials = tuple(
        SampledTrial(trial_key=trial_key, stratum=stratum) for trial_key, stratum in chosen
    )
    stratification_counts: dict[str, int] = {}
    for _, stratum in chosen:
        stratification_counts[stratum] = stratification_counts.get(stratum, 0) + 1

    return SampleManifest(
        skill_fingerprint=skill_fingerprint,
        probe_snapshot_sha256=probe_sha,
        sample_args=args,
        seed=seed_hex,
        trial_keys=chosen_keys,
        sampled_trials=sampled_trials,
        forensics=forensics or SkillForensics(),
        stratification_counts=stratification_counts,
        selection_mode=_derive_selection_mode(args.pinned_trial_keys, selection_source),
        selection_source=selection_source,
        rationale_source=rationale_source,
        stimulus_source=stimulus_source,
        excluded_prior_audit_session_ids=prior_audit_session_ids,
        classified_rare_flags=classified_rare_flags,
    )
