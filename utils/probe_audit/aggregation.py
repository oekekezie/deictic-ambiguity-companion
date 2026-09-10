"""K-rep aggregation for probe-audit trials.

Each of the seven probe flags has its independent reading aggregated
with the same per-flag policy the probe uses
(``_FLAG_AGGREGATION_POLICIES`` — any-affirmative for the five
misattribution flags and articulation, majority vote for governing),
so the audit's reading stays behaviorally identical to the probe.
Probe-correctness per flag is majority-voted across K reps. Below
``ceil(K/2)`` parsed reps per signal, the aggregated flag resolves to
``None`` — the same semantics the probe itself uses.
"""

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from utils.probe_audit.args import args_digest as _args_digest
from utils.probe_audit.args import canonical_json as _canon
from utils.probe_audit.args import (
    replication_evidence_for_k,
    validate_auditor_rep_count,
)
from utils.probe_audit.disagreement import categorize_disagreement
from utils.probe_audit.log import LogValidationError, read_log, validate_log
from utils.probe_audit.models import (
    AuditTrialRecord,
    ProbeAuditProvenance,
    ProbeAuditResults,
    RepJudgment,
    RepReading,
)
from utils.probe_audit.sampling import (
    SampleManifest,
    compute_probe_snapshot_sha256,
    compute_rationale_source_sha256,
    compute_stimulus_source_sha256,
)
from utils.rationale_analysis.aggregation import (
    _FLAG_AGGREGATION_POLICIES,
    _majority_vote,
)
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS


class AggregateError(Exception):
    """Raised when aggregate refuses to produce a snapshot.

    Surfaces both log-chain defects (``LogValidationError`` re-raised as
    this type) and sample-coverage defects (missing reps) through a
    single exception class so CLI callers have one catch point.
    """


def aggregate_trial_k_reps(
    trial_key: tuple[str, str, int, int],
    rep_readings: tuple[RepReading, ...],
    rep_judgments: tuple[RepJudgment, ...],
    k: int,
    probe_flags: dict[str, bool | None],
) -> AuditTrialRecord:
    """Build one ``AuditTrialRecord`` from its K readings and K judgments.

    ``probe_flags`` maps each of the seven flag keys in
    ``RATIONALE_ANALYSIS_FLAG_KEYS`` to the probe's own aggregated
    classification for this trial, looked up from the probe snapshot by
    the directory-level aggregator. They are required inputs to the
    disagreement classifier — without them the classifier cannot
    distinguish ``stable_disagreement`` from ``shifted_against_probe``
    (and likewise ``shifted_to_probe`` from plain ``stable_agreement``).
    A ``None`` value for a flag (the probe did not resolve it) collapses
    those two category pairs for that flag.

    ``k`` must satisfy the same positive-odd contract enforced at the
    ``SampleArgs`` boundary. Production aggregation from manifests is
    already protected, but this helper is exported and directly
    importable; without the guard a caller passing ``k=2`` would slip
    through into ``_majority_vote``, where 1-1 ties silently collapse
    to ``False`` (the ``votes_for > votes_against`` predicate). One
    rule, two enforcement points — both delegate to
    :func:`validate_auditor_rep_count`.

    Each flag's independent reading is aggregated with the same per-flag
    policy the probe uses (``_FLAG_AGGREGATION_POLICIES``); each flag's
    probe-correct judgment is majority-voted across the K reps.
    """
    validate_auditor_rep_count(k)

    record_fields: dict[str, Any] = {
        "trial_key": trial_key,
        "k": k,
        "rep_readings": tuple(rep_readings),
        "rep_judgments": tuple(rep_judgments),
    }

    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        reading_policy = _FLAG_AGGREGATION_POLICIES[flag]
        reading_classifications = [getattr(r, flag) for r in rep_readings]
        agg_reading, reading_for, reading_against = reading_policy(
            reading_classifications, k
        )

        judgment_correct = [
            getattr(j, f"{flag}_probe_correct") for j in rep_judgments
        ]
        agg_pc, pc_for, pc_against = _majority_vote(judgment_correct, k)

        disagreement = categorize_disagreement(
            probe_value=probe_flags[flag],
            independent_reading=agg_reading,
            aggregated_probe_correct=agg_pc,
        )

        record_fields[f"aggregated_independent_reading_{flag}"] = agg_reading
        record_fields[f"aggregated_probe_correct_{flag}"] = agg_pc
        record_fields[f"independent_reading_votes_{flag}"] = (
            reading_for,
            reading_against,
        )
        record_fields[f"probe_correct_votes_{flag}"] = (pc_for, pc_against)
        record_fields[f"disagreement_category_{flag}"] = disagreement

    return AuditTrialRecord(**record_fields)


def _events_by_type_and_rep(events: list[dict]) -> dict:
    """Index events for fast lookup by (trial_id, rep, event_type)."""
    idx: dict = {}
    for event in events:
        key = (event["trial_id"], int(event["rep_index"]), event["event_type"])
        idx.setdefault(key, []).append(event)
    return idx


def _reading_from_event(event: dict, rationale_bundle_sha256: str) -> RepReading:
    """Build a ``RepReading`` from an ``independent_reading_recorded`` event.

    The rationale bundle SHA-256 lives on the sibling ``rationale_fetched``
    event, not on the reading event — the caller looks it up and passes it
    in so the reconstructed struct carries the exact bytes the auditor
    saw at reading time (empty string when the fetch event is missing).
    ``sub_agent_model`` is captured per-rep-event by the recording CLIs;
    we surface it here so downstream provenance can verify which model
    produced the reading without consulting the skill fingerprint alone.

    The payload carries one ``{flag}`` bool and one ``{flag}_reasoning``
    string per flag in ``RATIONALE_ANALYSIS_FLAG_KEYS``.
    """
    payload = event["payload"]
    reading_fields: dict[str, Any] = {
        "rep_index": int(event["rep_index"]),
        "rationale_bundle_sha256": rationale_bundle_sha256,
        "recorded_at": event["recorded_at"],
        "sub_agent_session_id": event.get("sub_agent_session_id"),
        "sub_agent_model": event["sub_agent_model"],
    }
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        reading_fields[flag] = bool(payload[flag])
        reading_fields[f"{flag}_reasoning"] = payload[f"{flag}_reasoning"]
    return RepReading(**reading_fields)


def _judgment_from_event(event: dict, probe_output_bundle_sha256: str) -> RepJudgment:
    """Build a ``RepJudgment`` from a ``probe_judgment_recorded`` event.

    The probe-output bundle SHA-256 lives on the sibling
    ``probe_output_fetched`` event; the caller threads it in by rep so
    the reconstructed struct preserves the bytes the auditor saw at
    judgment time.

    The payload carries one ``{flag}_probe_correct`` bool, one
    ``{flag}_reasoning`` string, and an optional
    ``{flag}_corrected_classification`` per flag in
    ``RATIONALE_ANALYSIS_FLAG_KEYS``.
    """
    payload = event["payload"]
    judgment_fields: dict[str, Any] = {
        "rep_index": int(event["rep_index"]),
        "probe_output_bundle_sha256": probe_output_bundle_sha256,
        "recorded_at": event["recorded_at"],
        "sub_agent_session_id": event.get("sub_agent_session_id"),
        "sub_agent_model": event["sub_agent_model"],
    }
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        judgment_fields[f"{flag}_probe_correct"] = bool(
            payload[f"{flag}_probe_correct"]
        )
        judgment_fields[f"{flag}_reasoning"] = payload[f"{flag}_reasoning"]
        judgment_fields[f"{flag}_corrected_classification"] = payload.get(
            f"{flag}_corrected_classification"
        )
    return RepJudgment(**judgment_fields)


_REQUIRED_REP_EVENTS = (
    "rationale_fetched",
    "independent_reading_recorded",
    "probe_output_fetched",
    "probe_judgment_recorded",
)


def _find_incomplete_reps(
    manifest: SampleManifest, idx: dict
) -> list[tuple[str, int, list[str]]]:
    """List ``(trial_id, rep, missing_events)`` for any rep lacking all four events."""
    incomplete: list[tuple[str, int, list[str]]] = []
    for trial_key in manifest.trial_keys:
        tid = "|".join(str(p) for p in trial_key)
        for rep in range(1, manifest.sample_args.k + 1):
            missing = [
                et
                for et in _REQUIRED_REP_EVENTS
                if (tid, rep, et) not in idx
            ]
            if missing:
                incomplete.append((tid, rep, missing))
    return incomplete


def _load_probe_flags_by_trial(
    probe_snapshot_path: str,
) -> dict[tuple[str, str, int, int], dict[str, bool | None]]:
    """Build a lookup ``trial_key → {flag: probe_classification}``.

    The probe snapshot is the ground truth for what the probe said. We
    read each record, extract its seven aggregated flag classifications,
    and key by the four-tuple the audit pipeline uses everywhere.
    Trials in the probe snapshot but not in the sample are simply
    unreachable via this map; trials in the sample but not in the
    snapshot would be a manifest integrity bug upstream.
    """
    payload = json.loads(Path(probe_snapshot_path).read_text())
    result: dict[tuple[str, str, int, int], dict[str, bool | None]] = {}
    for rec in payload["trial_records"]:
        key = (
            rec["config_key"],
            rec["example_id"],
            int(rec["batch_index"]),
            int(rec["trial"]),
        )
        result[key] = {
            flag: rec.get(flag) for flag in RATIONALE_ANALYSIS_FLAG_KEYS
        }
    return result


def aggregate_probe_audit_records(
    audit_dir: Path,
    auditor_model_version: str,
    allow_incomplete: bool = False,
    repo_root: Path | None = None,
) -> ProbeAuditResults:
    """Walk the audit log + manifest and produce ``ProbeAuditResults``.

    Builds one ``AuditTrialRecord`` per trial in the manifest. Bundle
    SHAs from the fetch events (``rationale_fetched``,
    ``probe_output_fetched``) are threaded into the reconstructed
    ``RepReading``/``RepJudgment`` so the audit trail preserves the
    exact bytes the auditor saw. Probe classifications for each trial
    are loaded from the probe snapshot and passed to the disagreement
    classifier.

    Before aggregating, this function:

    - Runs :func:`validate_log` on the audit log and re-raises any
      ``LogValidationError`` as :class:`AggregateError` so callers see a
      single failure mode.
    - Verifies that every ``(trial_id, rep_index)`` in the sample
      manifest has a complete five-event trace
      (``rationale_fetched``, ``independent_reading_recorded``,
      ``probe_output_fetched``, ``probe_judgment_recorded``). Incomplete
      traces are a hard refusal unless ``allow_incomplete=True``.
    """
    log_path = audit_dir / "audit_log.jsonl"
    try:
        validate_log(log_path)
    except LogValidationError as exc:
        raise AggregateError(f"audit log validation failed: {exc}") from exc

    manifest = SampleManifest.model_validate_json(
        (audit_dir / "sample_manifest.json").read_text()
    )

    # H6: re-hash the probe snapshot bytes and verify against the hash
    # persisted in the manifest at sample time. If they diverge, the
    # snapshot has been mutated since sampling and the probe flags we'd
    # load to drive categorize_disagreement no longer correspond to
    # what the auditor's reps actually saw at fetch-probe-output time.
    # Silently proceeding would collapse stable_disagreement evidence into
    # shifted_against_probe, hiding the strongest disagreement bucket.
    # ``manifest.sample_args.probe_snapshot_path`` is repo-relative
    # (normalized by the CLI at sample time); anchor against
    # ``repo_root`` so the re-hash reads the same bytes regardless of
    # where aggregation is invoked from. Callers that pass
    # ``repo_root=None`` get CWD, which is only correct if invoked from
    # the repo root.
    anchor = repo_root if repo_root is not None else Path.cwd()
    probe_snapshot_path = anchor / manifest.sample_args.probe_snapshot_path
    try:
        current_sha = compute_probe_snapshot_sha256(probe_snapshot_path)
    except FileNotFoundError as exc:
        raise AggregateError(
            f"probe snapshot missing at {probe_snapshot_path}"
        ) from exc
    if current_sha != manifest.probe_snapshot_sha256:
        raise AggregateError(
            f"probe snapshot has been mutated since sample time. Manifest "
            f"records SHA-256 {manifest.probe_snapshot_sha256[:12]}… but "
            f"{probe_snapshot_path} now hashes to {current_sha[:12]}…. "
            "Refusing to aggregate against a mutated snapshot."
        )

    # Mirror of the probe-snapshot guard above, for the rationale source.
    # Once the sampler's joined eligible frame depends on rationale bytes
    # and the seed folds in the rationale SHA, a post-sample mutation of
    # the rationale file invalidates downstream aggregation the same way
    # a probe-snapshot mutation does: reps fetched rationales against
    # bytes different from the ones the audit was drawn from.
    # ``manifest.rationale_source.path`` is the sole manifest authority
    # for the stage-N snapshot location — the same field ``_cmd_fetch_
    # rationale`` reads — so aggregation re-hashes against that single
    # canonical source. It is repo-relative, so anchor against the same
    # ``anchor`` used above for the probe snapshot.
    rationale_source_path = anchor / manifest.rationale_source.path
    try:
        current_rationale_sha = compute_rationale_source_sha256(
            rationale_source_path
        )
    except FileNotFoundError as exc:
        raise AggregateError(
            f"rationale source pinned by manifest is missing at "
            f"{rationale_source_path}"
        ) from exc
    if current_rationale_sha != manifest.rationale_source.sha256:
        raise AggregateError(
            f"rationale source has been mutated since sample time. Manifest "
            f"records SHA-256 {manifest.rationale_source.sha256[:12]}… but "
            f"{rationale_source_path} now hashes to "
            f"{current_rationale_sha[:12]}…. Refusing to aggregate against "
            "a mutated rationale source."
        )

    # Same drift refusal for the stimulus source. The auditor's bundle's
    # scenario context fields are derived from this file plus the static
    # registry, so a post-sample mutation would silently feed the auditor
    # different referential anchors than the audit was drawn against.
    stimulus_source_path = anchor / manifest.stimulus_source.path
    try:
        current_stimulus_sha = compute_stimulus_source_sha256(
            stimulus_source_path
        )
    except FileNotFoundError as exc:
        raise AggregateError(
            f"stimulus source pinned by manifest is missing at "
            f"{stimulus_source_path}"
        ) from exc
    if current_stimulus_sha != manifest.stimulus_source.sha256:
        raise AggregateError(
            f"stimulus source has been mutated since sample time. Manifest "
            f"records SHA-256 {manifest.stimulus_source.sha256[:12]}… but "
            f"{stimulus_source_path} now hashes to "
            f"{current_stimulus_sha[:12]}…. Refusing to aggregate against "
            "a mutated stimulus source."
        )

    events = read_log(log_path)
    idx = _events_by_type_and_rep(events)

    if not allow_incomplete:
        incomplete = _find_incomplete_reps(manifest, idx)
        if incomplete:
            sample = ", ".join(
                f"{tid} rep={rep} missing={missing}"
                for tid, rep, missing in incomplete[:3]
            )
            suffix = (
                f" (and {len(incomplete) - 3} more)"
                if len(incomplete) > 3
                else ""
            )
            raise AggregateError(
                f"{len(incomplete)} rep trace(s) are incomplete — "
                f"pass --allow-incomplete to aggregate anyway. Examples: "
                f"{sample}{suffix}"
            )

    # Reuse the anchored probe-snapshot path resolved above for the
    # drift check — both call sites must read the same bytes, and using
    # the repo-relative manifest field without anchoring would yield a
    # CWD-relative read that doesn't round-trip across invocation sites.
    probe_flags = _load_probe_flags_by_trial(str(probe_snapshot_path))

    # First ``sample_drawn`` event marks run start; last event of any
    # type marks run end. Empty log falls back to "now" for both.
    # Filter sample_drawn events to those matching the current manifest's
    # seed so an orphan from a crashed prior run (manifest wrote before
    # event, or event wrote before manifest in the old order) cannot
    # hijack run_started_at.
    sample_drawn_events = [
        e
        for e in events
        if e["event_type"] == "sample_drawn"
        and e.get("payload", {}).get("seed") == manifest.seed
    ]
    run_started_at = (
        sample_drawn_events[0]["recorded_at"]
        if sample_drawn_events
        else _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    run_completed_at = (
        events[-1]["recorded_at"]
        if events
        else _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    records: list[AuditTrialRecord] = []
    for trial_key in manifest.trial_keys:
        tid = "|".join(str(p) for p in trial_key)
        readings: list[RepReading] = []
        judgments: list[RepJudgment] = []
        for rep in range(1, manifest.sample_args.k + 1):
            read_events = idx.get((tid, rep, "independent_reading_recorded"), [])
            judge_events = idx.get((tid, rep, "probe_judgment_recorded"), [])
            rationale_events = idx.get((tid, rep, "rationale_fetched"), [])
            probe_fetch_events = idx.get((tid, rep, "probe_output_fetched"), [])
            rationale_sha = (
                rationale_events[0]["payload"].get("bundle_sha256", "")
                if rationale_events
                else ""
            )
            probe_sha = (
                probe_fetch_events[0]["payload"].get("bundle_sha256", "")
                if probe_fetch_events
                else ""
            )
            if read_events:
                readings.append(_reading_from_event(read_events[0], rationale_sha))
            if judge_events:
                judgments.append(_judgment_from_event(judge_events[0], probe_sha))
        trial_probe_flags = probe_flags.get(
            tuple(trial_key),
            {flag: None for flag in RATIONALE_ANALYSIS_FLAG_KEYS},
        )
        records.append(
            aggregate_trial_k_reps(
                tuple(trial_key),
                tuple(readings),
                tuple(judgments),
                k=manifest.sample_args.k,
                probe_flags=trial_probe_flags,
            )
        )

    # Derive auditor_model_versions from the sub_agent_model values that
    # rep events actually recorded. The per-event consistency check in
    # log.append_event guarantees each (trial_id, rep) reports a single
    # model, so this set is the ground truth for "what models produced
    # the judgments in this audit." Fall back to the CLI-supplied label
    # only when the log contains zero rep events (an empty audit).
    observed_models = sorted({
        e["sub_agent_model"]
        for e in events
        if e.get("event_type") != "sample_drawn"
        and e.get("sub_agent_model") is not None
    })
    auditor_model_versions = (
        tuple(observed_models) if observed_models else (auditor_model_version,)
    )

    # Cross-check the rationale and stimulus source paths recorded on
    # ``rationale_fetched`` events against the manifest pins. The
    # write-time drift checks in ``log.append_event`` guarantee one
    # unique value per source per run; this is a defensive
    # belt-and-suspenders check at the aggregation layer so a tampered
    # log cannot surface conflicting source paths silently.
    #
    # Events carry path strings only — the manifest is the sole authority
    # for the SHAs, which were re-validated against on-disk bytes above.
    # Skip the cross-check when no rep events were recorded (the
    # ``allow_incomplete`` path with zero rep work yet); the manifest is
    # the sole authority in that case.
    observed_rationale_sources = sorted({
        e["rationale_source"]
        for e in events
        if e.get("event_type") == "rationale_fetched"
        and e.get("rationale_source") is not None
    })
    if len(observed_rationale_sources) > 1:
        raise AggregateError(
            f"rationale_source drift across rep events: {observed_rationale_sources}"
        )
    if (
        observed_rationale_sources
        and observed_rationale_sources[0] != manifest.rationale_source.path
    ):
        raise AggregateError(
            f"rationale_source drift between manifest and rep events: "
            f"manifest pins {manifest.rationale_source.path!r} but events "
            f"report {observed_rationale_sources[0]!r}"
        )
    observed_stimulus_sources = sorted({
        e["stimulus_source"]
        for e in events
        if e.get("event_type") == "rationale_fetched"
        and e.get("stimulus_source") is not None
    })
    if len(observed_stimulus_sources) > 1:
        raise AggregateError(
            f"stimulus_source drift across rep events: {observed_stimulus_sources}"
        )
    if (
        observed_stimulus_sources
        and observed_stimulus_sources[0] != manifest.stimulus_source.path
    ):
        raise AggregateError(
            f"stimulus_source drift between manifest and rep events: "
            f"manifest pins {manifest.stimulus_source.path!r} but events "
            f"report {observed_stimulus_sources[0]!r}"
        )
    provenance = ProbeAuditProvenance(
        skill_fingerprint=manifest.skill_fingerprint,
        fingerprint_method=manifest.forensics.fingerprint_method
        or "git-blob-then-sha256",
        skill_human_label=manifest.forensics.human_label,
        skill_fingerprint_inputs=manifest.forensics.fingerprint_inputs,
        skill_per_file_hashes=dict(manifest.forensics.per_file_hashes),
        git_commit_sha=manifest.forensics.git_commit_sha,
        git_tree_dirty=manifest.forensics.git_tree_dirty,
        canonical_args_json=_canon(manifest.sample_args),
        args_digest=_args_digest(manifest.sample_args),
        k=manifest.sample_args.k,
        batch_allocation=manifest.sample_args.batch_allocation,
        classified_rare_flags=manifest.classified_rare_flags,
        replication_evidence=replication_evidence_for_k(manifest.sample_args.k),
        sampling_seed=manifest.seed,
        probe_snapshot_path=str(manifest.sample_args.probe_snapshot_path),
        probe_snapshot_sha256=manifest.probe_snapshot_sha256,
        rationale_source_path=manifest.rationale_source.path,
        rationale_source_sha256=manifest.rationale_source.sha256,
        stimulus_source_path=manifest.stimulus_source.path,
        stimulus_source_sha256=manifest.stimulus_source.sha256,
        auditor_model_family="claude",
        auditor_model_versions=auditor_model_versions,
        claude_session_id=None,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
    )

    return ProbeAuditResults(
        provenance=provenance,
        sample_args=manifest.sample_args,
        records=tuple(records),
    )
