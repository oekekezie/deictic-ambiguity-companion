"""Storage functions for probe-audit session index persistence.

Manages a SQLite index over audit-session directories under a
configurable base directory (``$PROBE_AUDIT_DIR`` or
``./analysis_outputs/probe_audit``).

The on-disk audit directories are the canonical source of truth. SQLite
serves as a fast index over those directories, with each session's full
provenance still recorded in ``sample_manifest.json`` (sample time) and
``snapshot.json`` (after aggregation). Pruning keeps the index
consistent with the filesystem when audit directories are removed
externally.

The module's design intentionally mirrors ``utils/batch_inference/
storage.py`` — same two-column DDL, same ``INSERT OR REPLACE`` upsert,
same ``contextmanager``-based connection idiom — so researchers moving
between the batch-inference and probe-audit UIs encounter the same
ergonomics.
"""

import datetime as _dt
import hashlib
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from utils.probe_audit.args import BatchAllocation, args_digest
from utils.probe_audit.log import read_log
from utils.probe_audit.models import ProbeAuditResults
from utils.probe_audit.sampling import SampleManifest, SelectionMode


AuditStatus = Literal["sampled", "in_progress", "aggregated"]

_DB_FILENAME = "probe_audits.db"


def resolve_base_dir(
    explicit: Path | None = None,
    *,
    fallback: Path | None = None,
) -> Path:
    """Resolve the probe-audit index base directory.

    Precedence: ``explicit`` > ``$PROBE_AUDIT_DIR`` > ``fallback`` >
    ``./analysis_outputs/probe_audit``. Exported so the CLI, storage
    internals, and the marimo review notebook share this single source
    of truth rather than duplicating the env-var lookup. The ``fallback``
    kwarg lets a caller supply a context-specific default (e.g., the
    notebook passes a ``REPO_ROOT``-based absolute path so the index
    resolves correctly regardless of CWD) without losing the env-var
    override path.
    """
    if explicit is not None:
        return explicit
    env_val = os.environ.get("PROBE_AUDIT_DIR")
    if env_val:
        return Path(env_val)
    if fallback is not None:
        return fallback
    return Path("./analysis_outputs/probe_audit")


def _get_db_path(base_dir: Path) -> Path:
    return base_dir / _DB_FILENAME


def _init_db(conn: sqlite3.Connection) -> None:
    """Create the single-table schema if it doesn't already exist."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS audits ("
        "session_id TEXT PRIMARY KEY, "
        "data TEXT NOT NULL"
        ")"
    )


@contextmanager
def _get_db_connection(
    base_dir: Path,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for a SQLite connection with auto-init on first open.

    10-second timeout matches the batch-inference convention, handling
    the rare case where a concurrent write holds the database lock.
    ``mkdir(parents=True, exist_ok=True)`` ensures the base directory
    exists; tests relying on a fresh tmp_path benefit directly.

    ``isolation_level=None`` switches the sqlite3 driver into autocommit
    mode so callers can explicitly bracket read-then-write sequences with
    ``BEGIN IMMEDIATE`` / ``COMMIT``. The downgrade guard in
    ``update_audit_status`` relies on this: Python's default deferred
    isolation would let a concurrent writer commit ``"aggregated"``
    between the SELECT and the INSERT OR REPLACE, silently bypassing
    the guard.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = _get_db_path(base_dir)
    conn = sqlite3.connect(str(db_path), timeout=10.0, isolation_level=None)
    try:
        _init_db(conn)
        yield conn
    finally:
        conn.close()


def transition_record(
    record: "ProbeAuditIndexRecord",
    **updates: object,
) -> "ProbeAuditIndexRecord":
    """Copy ``record`` with field ``updates``, rerunning model validators.

    ``BaseModel.model_copy(update=...)`` in Pydantic v2 does NOT rerun
    ``@model_validator`` hooks, which would silently let a caller build
    a record that violates an invariant — e.g., ``audit_status =
    "aggregated"`` with only two of the three completion fields set, or
    ``selection_mode = "pinned_from_manifest"`` alongside
    ``source_manifest_sha256 = None``. Routing every transition through
    ``model_validate`` (on the dumped dict) re-runs the validators at
    write time, so the SQLite row never holds corrupt data.
    """
    return ProbeAuditIndexRecord.model_validate(
        {**record.model_dump(), **updates}
    )


def compute_trial_set_sha(
    trial_keys: tuple[tuple[str, str, int, int], ...],
) -> str:
    """Deterministic 64-hex SHA-256 over the trial-key set.

    Sorting inside the helper makes the sha invariant over caller-side
    ordering — two draws whose final ``trial_keys`` differ only in order
    (they shouldn't, but defensively) still collide into the same
    ``trial_set_sha``, giving the marimo discovery tab a stable filter
    key for cross-skill-version re-audits.

    Note: ``trial_set_sha`` is a **discovery** affordance, not the
    pooling compatibility key. The probe-audit review notebook's pooled-
    CS view uses the full byte-aware pooling identity (``skill_fingerprint``,
    ``probe_snapshot_sha256``, the ``rationale_source`` path+SHA pair, the
    ``stimulus_source`` path+SHA pair, and ``batch_allocation``) plus an
    explicit per-
    ``trial_key`` intersection (via ``pool_compatibility`` in
    ``utils/probe_audit/review_data.py``) to decide whether two audits are
    poolable; ``trial_set_sha`` just lets a human eyeball same-trials
    re-audits in the tracked-audits table.
    """
    payload = json.dumps(
        [list(k) for k in sorted(trial_keys)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ProbeAuditIndexRecord(BaseModel):
    """One row in the ``audits`` table. Canonical provenance lives in the
    session directory's ``sample_manifest.json`` and ``snapshot.json``;
    this record is a fast-query summary.

    Three invariants are enforced by the ``@model_validator``: trial-set
    SHA format, the ``selection_mode`` ↔ ``source_manifest_sha256`` iff
    rule, and the ``audit_status == "aggregated"`` ↔ all-three-
    completion-fields-populated rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    session_dir: str
    trial_set_sha: str
    fp_short: str
    args_digest: str
    skill_fingerprint: str
    probe_snapshot_sha256: str
    k: int
    selection_mode: SelectionMode
    # Part of the byte-aware pooling identity: a coverage-convergent campaign
    # excludes/counts only its own prior batches, so prior-audit exclusion and
    # pool compatibility key on the allocation mode alongside the source SHAs.
    batch_allocation: BatchAllocation
    source_manifest_sha256: str | None
    sample_timestamp: str
    audit_status: AuditStatus
    snapshot_path: str | None = None
    auditor_model_versions: tuple[str, ...] | None = None
    audit_completion_timestamp: str | None = None
    rationale_source_path: str
    rationale_source_sha256: str
    stimulus_source_path: str
    stimulus_source_sha256: str

    @model_validator(mode="after")
    def _validate_invariants(self) -> "ProbeAuditIndexRecord":
        if len(self.trial_set_sha) != 64:
            raise ValueError(
                f"trial_set_sha must be a 64-hex SHA-256 string; "
                f"got length {len(self.trial_set_sha)}"
            )
        try:
            int(self.trial_set_sha, 16)
        except ValueError:
            raise ValueError(
                f"trial_set_sha must be hexadecimal; got {self.trial_set_sha!r}"
            ) from None
        has_source = self.source_manifest_sha256 is not None
        wants_source = self.selection_mode == "pinned_from_manifest"
        if has_source != wants_source:
            raise ValueError(
                f"source_manifest_sha256 must be populated iff selection_mode "
                f"is 'pinned_from_manifest'; got mode={self.selection_mode!r}, "
                f"source={'present' if has_source else 'None'}"
            )
        completion = (
            self.snapshot_path,
            self.auditor_model_versions,
            self.audit_completion_timestamp,
        )
        all_populated = all(v is not None for v in completion)
        all_empty = all(v is None for v in completion)
        if self.audit_status == "aggregated" and not all_populated:
            raise ValueError(
                "audit_status == 'aggregated' requires snapshot_path, "
                "auditor_model_versions, and audit_completion_timestamp to "
                "all be populated"
            )
        if self.audit_status != "aggregated" and not all_empty:
            raise ValueError(
                f"audit_status == {self.audit_status!r} requires all "
                "completion fields (snapshot_path, auditor_model_versions, "
                "audit_completion_timestamp) to be None"
            )
        return self


def persist_audit_draw(
    record: ProbeAuditIndexRecord,
    base_dir: Path | None = None,
) -> None:
    """Write or replace a row at ``audit_status='sampled'``.

    Replay-safe via ``INSERT OR REPLACE``. Calling this again on the
    same ``session_id`` overwrites the prior row — the only sensible
    semantic for a draw-time record (subsequent status transitions go
    through ``update_audit_status``).
    """
    base = resolve_base_dir(base_dir)
    with _get_db_connection(base) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO audits (session_id, data) VALUES (?, ?)",
            (record.session_id, record.model_dump_json()),
        )


def update_audit_status(
    record: ProbeAuditIndexRecord,
    base_dir: Path | None = None,
) -> None:
    """Transition an existing row; refuses ``aggregated`` → earlier states.

    The check-then-write is wrapped in ``BEGIN IMMEDIATE`` / ``COMMIT``
    so the SELECT and the INSERT OR REPLACE execute as a single atomic
    transaction. Without this, two concurrent callers could both observe
    ``existing.audit_status != "aggregated"`` before either commits,
    then both write — silently bypassing the guard. ``IMMEDIATE``
    specifically acquires a RESERVED lock at ``BEGIN`` time, blocking
    other writers from committing between our SELECT and our INSERT.
    """
    base = resolve_base_dir(base_dir)
    with _get_db_connection(base) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT data FROM audits WHERE session_id = ?",
                (record.session_id,),
            ).fetchone()
            if row is not None:
                existing = ProbeAuditIndexRecord.model_validate_json(row[0])
                if (
                    existing.audit_status == "aggregated"
                    and record.audit_status != "aggregated"
                ):
                    raise ValueError(
                        f"cannot downgrade {record.session_id!r} from "
                        f"'aggregated' to {record.audit_status!r}"
                    )
            conn.execute(
                "INSERT OR REPLACE INTO audits (session_id, data) VALUES (?, ?)",
                (record.session_id, record.model_dump_json()),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def prune_missing_audits(base_dir: Path | None = None) -> list[str]:
    """Delete rows whose ``session_dir`` is absent from disk; return their ids.

    Reconciliation pattern: the index is a query surface, the filesystem
    is canonical. A researcher who ``rm -rf``s an old audit directory
    shouldn't then see a dangling row in the marimo discovery tab.
    """
    base = resolve_base_dir(base_dir)
    pruned: list[str] = []
    with _get_db_connection(base) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT session_id, data FROM audits"
            ).fetchall()
            for session_id, data in rows:
                # Per-row parse guard: a single corrupt JSON blob (e.g.,
                # from a manual DB edit or an incompatible schema change)
                # must not block pruning of the healthy rows around it.
                # Leave the corrupt row in place so a human can inspect
                # it via ``sqlite3`` CLI rather than deleting evidence.
                try:
                    record = ProbeAuditIndexRecord.model_validate_json(data)
                except ValidationError as exc:
                    print(
                        f"warning: prune skipped {session_id!r} — row data "
                        f"did not parse as ProbeAuditIndexRecord: {exc}",
                        file=sys.stderr,
                    )
                    continue
                session_dir = base / record.session_dir
                if not session_dir.exists():
                    conn.execute(
                        "DELETE FROM audits WHERE session_id = ?",
                        (session_id,),
                    )
                    pruned.append(session_id)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    return pruned


def _derive_sample_timestamp(audit_dir: Path, manifest_path: Path) -> str:
    """sample_drawn event's ``recorded_at`` > manifest file mtime (UTC).

    Used by the backfill path to populate ``sample_timestamp`` on rows
    seeded from on-disk audits whose index row is absent. The audit log is the
    authoritative timestamp source because it was written at sample
    time; the filesystem mtime is a fallback for audits whose log is
    missing or corrupt.
    """
    log_path = audit_dir / "audit_log.jsonl"
    if log_path.exists():
        try:
            events = read_log(log_path)
        except Exception:
            events = []
        for event in events:
            if event.get("event_type") == "sample_drawn":
                ts = event.get("recorded_at")
                if ts:
                    return str(ts)
    mtime = _dt.datetime.fromtimestamp(
        manifest_path.stat().st_mtime, tz=_dt.UTC
    )
    return mtime.strftime("%Y-%m-%dT%H:%M:%SZ")


def _derive_audit_status_and_completion(
    audit_dir: Path, session_dir: str
) -> tuple[str, str | None, tuple[str, ...] | None, str | None]:
    """Determine ``audit_status`` + completion fields from on-disk state.

    Returns ``(audit_status, snapshot_path, auditor_model_versions,
    audit_completion_timestamp)``. The three completion fields are non-
    None iff ``audit_status == "aggregated"`` — matches the model
    invariant enforced on ``ProbeAuditIndexRecord``.
    """
    snapshot_path_on_disk = audit_dir / "snapshot.json"
    if snapshot_path_on_disk.exists():
        results = ProbeAuditResults.model_validate_json(
            snapshot_path_on_disk.read_text()
        )
        return (
            "aggregated",
            f"{session_dir}/snapshot.json",
            results.provenance.auditor_model_versions,
            results.provenance.run_completed_at,
        )
    # No snapshot.json: walk the audit log. Any event beyond the
    # initial ``sample_drawn`` means at least one rep was attempted.
    log_path = audit_dir / "audit_log.jsonl"
    if log_path.exists():
        try:
            events = read_log(log_path)
        except Exception:
            events = []
        has_post_sample_event = any(
            e.get("event_type") != "sample_drawn" for e in events
        )
        if has_post_sample_event:
            return ("in_progress", None, None, None)
    return ("sampled", None, None, None)


def backfill_from_filesystem(base_dir: Path | None = None) -> list[str]:
    """Seed the index from on-disk audit sessions missing rows.

    Scans ``base_dir`` (``$PROBE_AUDIT_DIR`` / default fallback) for
    every ``sample_manifest.json`` and, for sessions not already in the
    index, constructs a ``ProbeAuditIndexRecord`` from the manifest plus
    (when present) the ``snapshot.json``. Idempotent: sessions already
    indexed are skipped, not overwritten.

    Returns the list of newly-seeded ``session_id`` values for the
    caller to display. Malformed manifests, snapshots, or records are
    skipped with a one-line stderr warning rather than aborting the
    scan — a single bad session should not block a researcher from
    surfacing the healthy ones.
    """
    base = resolve_base_dir(base_dir)
    if not base.is_dir():
        return []
    seeded: list[str] = []
    for manifest_path in sorted(base.rglob("sample_manifest.json")):
        audit_dir = manifest_path.parent
        session_id = audit_dir.name
        if load_audit_record(session_id, base_dir=base) is not None:
            continue
        try:
            manifest = SampleManifest.model_validate_json(
                manifest_path.read_text()
            )
        except (OSError, ValidationError) as exc:
            # ``OSError`` covers permission-denied, I/O error, stale NFS
            # handle; ``ValidationError`` covers schema mismatches. Both
            # mean "skip this session, log why, keep scanning" — we
            # never want one bad file to abort the whole backfill.
            print(
                f"warning: backfill skipped {manifest_path} — cannot read "
                f"as SampleManifest ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            continue
        try:
            session_dir = str(audit_dir.resolve().relative_to(base.resolve()))
        except ValueError:
            continue
        try:
            audit_status, snapshot_rel, models_tup, completion_ts = (
                _derive_audit_status_and_completion(audit_dir, session_dir)
            )
        except (OSError, ValidationError) as exc:
            # Same pattern: snapshot.json might be unreadable or malformed.
            print(
                f"warning: backfill skipped {session_id} — snapshot.json "
                f"failed to parse ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            continue
        try:
            record = ProbeAuditIndexRecord(
                session_id=session_id,
                session_dir=session_dir,
                trial_set_sha=compute_trial_set_sha(manifest.trial_keys),
                fp_short=manifest.skill_fingerprint[:12],
                args_digest=args_digest(manifest.sample_args),
                skill_fingerprint=manifest.skill_fingerprint,
                probe_snapshot_sha256=manifest.probe_snapshot_sha256,
                k=manifest.sample_args.k,
                selection_mode=manifest.selection_mode,
                batch_allocation=manifest.sample_args.batch_allocation,
                source_manifest_sha256=(
                    manifest.selection_source.sha256
                    if manifest.selection_source is not None
                    else None
                ),
                sample_timestamp=_derive_sample_timestamp(
                    audit_dir, manifest_path
                ),
                audit_status=audit_status,
                snapshot_path=snapshot_rel,
                auditor_model_versions=models_tup,
                audit_completion_timestamp=completion_ts,
                rationale_source_path=manifest.rationale_source.path,
                rationale_source_sha256=manifest.rationale_source.sha256,
                stimulus_source_path=manifest.stimulus_source.path,
                stimulus_source_sha256=manifest.stimulus_source.sha256,
            )
        except (OSError, ValidationError) as exc:
            # ``_derive_sample_timestamp`` can raise ``OSError`` when the
            # log file was deleted between rglob and read, or when mtime
            # can't be read; ``ValidationError`` would fire if the record
            # invariants reject the combination.
            print(
                f"warning: backfill could not build index record for "
                f"{session_id} ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            continue
        persist_audit_draw(record, base_dir=base)
        seeded.append(session_id)
    return seeded


def _manifest_identity_mismatch_reason(
    manifest: SampleManifest,
    *,
    skill_fingerprint: str,
    probe_snapshot_sha256: str,
    rationale_source_path: str,
    rationale_source_sha256: str,
    stimulus_source_path: str,
    stimulus_source_sha256: str,
    batch_allocation: BatchAllocation,
) -> str | None:
    """Return a mismatch-reason string, or ``None`` if the manifest agrees.

    Centralizes the canonical-manifest-vs-request identity comparison
    used by ``prior_audit_trial_keys`` so warning messages can name the
    exact disagreeing field without duplicating six branches at the
    call site. Returns the first mismatch found in a fixed order
    (skill_fingerprint → probe_snapshot_sha256 → rationale path/SHA →
    stimulus path/SHA); each comparison is independent, so the order is
    only relevant for which name surfaces in the warning.
    """
    if manifest.skill_fingerprint != skill_fingerprint:
        return (
            f"skill_fingerprint disagrees "
            f"(manifest={manifest.skill_fingerprint[:12]}…, "
            f"requested={skill_fingerprint[:12]}…)"
        )
    if manifest.probe_snapshot_sha256 != probe_snapshot_sha256:
        return (
            f"probe_snapshot_sha256 disagrees "
            f"(manifest={manifest.probe_snapshot_sha256[:12]}…, "
            f"requested={probe_snapshot_sha256[:12]}…)"
        )
    if manifest.rationale_source.path != rationale_source_path:
        return (
            f"rationale_source.path disagrees "
            f"(manifest={manifest.rationale_source.path!r}, "
            f"requested={rationale_source_path!r})"
        )
    if manifest.rationale_source.sha256 != rationale_source_sha256:
        return (
            f"rationale_source.sha256 disagrees "
            f"(manifest={manifest.rationale_source.sha256[:12]}…, "
            f"requested={rationale_source_sha256[:12]}…)"
        )
    if manifest.stimulus_source.path != stimulus_source_path:
        return (
            f"stimulus_source.path disagrees "
            f"(manifest={manifest.stimulus_source.path!r}, "
            f"requested={stimulus_source_path!r})"
        )
    if manifest.stimulus_source.sha256 != stimulus_source_sha256:
        return (
            f"stimulus_source.sha256 disagrees "
            f"(manifest={manifest.stimulus_source.sha256[:12]}…, "
            f"requested={stimulus_source_sha256[:12]}…)"
        )
    if manifest.sample_args.batch_allocation != batch_allocation:
        return (
            f"batch_allocation disagrees "
            f"(manifest={manifest.sample_args.batch_allocation!r}, "
            f"requested={batch_allocation!r})"
        )
    return None


def prior_audit_trial_keys(
    *,
    skill_fingerprint: str,
    probe_snapshot_sha256: str,
    rationale_source_path: str,
    rationale_source_sha256: str,
    stimulus_source_path: str,
    stimulus_source_sha256: str,
    batch_allocation: BatchAllocation,
    base_dir: Path | None = None,
    exclude_session_id: str | None = None,
) -> tuple[frozenset[tuple[str, str, int, int]], tuple[str, ...]]:
    """Union of trial keys drawn by prior audits under the same pool identity.

    Filters ``load_audit_index`` by exact equality on the full byte-aware
    pooling identity tuple — ``skill_fingerprint``,
    ``probe_snapshot_sha256``, ``rationale_source_path`` AND
    ``rationale_source_sha256``, ``stimulus_source_path`` AND
    ``stimulus_source_sha256``, and ``batch_allocation`` (so a
    coverage-convergent campaign subtracts only its own prior batches) —
    then reads each matching session's
    ``sample_manifest.json`` from disk to extract its ``trial_keys``. The
    index row alone carries a rolled-up ``trial_set_sha`` but not the
    individual keys, so the on-disk manifest is canonical for the per-key
    data the sampler needs.

    The byte-aware identity matters: a path-only filter would silently
    accept priors whose source bytes had been mutated (same path, new
    bytes), and would let ``--exclude-prior-audits`` subtract trials
    from audits whose auditors saw different rationale or stimulus
    content than the current run. Both halves of each source pair must
    match — same path with mutated bytes is a different audit context
    and must not contribute exclusion keys.

    ``exclude_session_id`` drops that ``session_id`` from the matching
    set before reading manifests. The CLI's ``_cmd_sample`` passes
    ``audit_dir.name`` so that re-running ``sample`` in an existing
    ``--output-dir`` does not self-exclude the current session's trial
    keys from its own reduced eligible frame — which would otherwise
    silently change the draw on re-entry (the seed is unchanged, but
    the effective pool would be smaller). ``None`` (the default) keeps
    every matching row, which is the right behavior for first-time
    invocations where no row with the caller's prospective session_id
    exists yet.

    Returns ``(trial_keys_union, contributing_session_ids)``:

    - ``trial_keys_union`` — ``frozenset`` of every four-tuple key drawn
      across all matching prior sessions. Union semantics: a trial key
      that appeared in two prior sessions counts once.
    - ``contributing_session_ids`` — sorted tuple of the session_ids
      whose manifests contributed at least one key. Populated on the new
      ``SampleManifest.excluded_prior_audit_session_ids`` so provenance
      captures exactly which priors were subtracted.

    Skips (with stderr warning) any session whose manifest cannot be read
    or parsed — same reconciliation pattern as ``load_audit_index`` and
    ``prune_missing_audits``. A single bad manifest must not block the
    sampler from seeing the healthy priors around it.
    """
    base = resolve_base_dir(base_dir)
    index_rows = load_audit_index(base_dir=base)
    matching = [
        r
        for r in index_rows
        if r.skill_fingerprint == skill_fingerprint
        and r.probe_snapshot_sha256 == probe_snapshot_sha256
        and r.rationale_source_path == rationale_source_path
        and r.rationale_source_sha256 == rationale_source_sha256
        and r.stimulus_source_path == stimulus_source_path
        and r.stimulus_source_sha256 == stimulus_source_sha256
        and r.batch_allocation == batch_allocation
        and r.session_id != exclude_session_id
    ]
    all_keys: set[tuple[str, str, int, int]] = set()
    contributing: list[str] = []
    for row in matching:
        manifest_path = base / row.session_dir / "sample_manifest.json"
        try:
            manifest = SampleManifest.model_validate_json(manifest_path.read_text())
        except (OSError, ValidationError) as exc:
            print(
                f"warning: prior_audit_trial_keys skipped {row.session_id!r} — "
                f"sample_manifest.json could not be read or validated "
                f"({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            continue
        # Cross-check the full byte-aware identity tuple against the
        # canonical manifest. The SQLite index is a fast-lookup affordance;
        # the ``sample_manifest.json`` is the authoritative provenance
        # source. A stale or hand-edited index row whose ANY identity
        # field disagrees with its on-disk manifest would otherwise let
        # exclusion subtract trial keys from a manifest with different
        # provenance, silently violating the byte-aware exclusion
        # contract. The cross-check covers the same six fields the
        # SQLite filter uses (skill_fingerprint, probe_snapshot_sha256,
        # rationale_source.path/_sha256, stimulus_source.path/_sha256)
        # via a centralized helper so the warning names the exact
        # disagreeing field. Skip+warn mirrors the reconciliation pattern
        # used elsewhere in this module.
        mismatch = _manifest_identity_mismatch_reason(
            manifest,
            skill_fingerprint=skill_fingerprint,
            probe_snapshot_sha256=probe_snapshot_sha256,
            rationale_source_path=rationale_source_path,
            rationale_source_sha256=rationale_source_sha256,
            stimulus_source_path=stimulus_source_path,
            stimulus_source_sha256=stimulus_source_sha256,
            batch_allocation=batch_allocation,
        )
        if mismatch is not None:
            print(
                f"warning: prior_audit_trial_keys skipped {row.session_id!r} — "
                f"canonical sample_manifest.json identity disagrees with the "
                f"SQLite index row: {mismatch}. The manifest is canonical; "
                f"the index row is stale.",
                file=sys.stderr,
            )
            continue
        keys = tuple(tuple(k) for k in manifest.trial_keys)
        if keys:
            all_keys.update(keys)
            contributing.append(row.session_id)
    return (frozenset(all_keys), tuple(sorted(contributing)))


def load_audit_record(
    session_id: str,
    base_dir: Path | None = None,
) -> ProbeAuditIndexRecord | None:
    """Targeted single-row lookup. ``None`` when the session is absent or malformed.

    Used by ``_cmd_next_rep`` and ``_cmd_aggregate`` to read the current
    status before transitioning — avoids pulling the full index just to
    look up one row.

    A row whose serialized data fails Pydantic validation under the
    current schema is treated as "absent" (returns ``None`` with a
    stderr warning), mirroring the skip+warn reconciliation pattern in
    ``load_audit_index``. This keeps targeted reads safe in the
    presence of pre-hard-break index rows that lack the required
    rationale/stimulus source fields — ``backfill_from_filesystem`` can
    re-seed them from the on-disk manifest, and ``next-rep`` /
    ``aggregate`` can no-op the index transition without aborting the
    primary operation.
    """
    base = resolve_base_dir(base_dir)
    with _get_db_connection(base) as conn:
        row = conn.execute(
            "SELECT data FROM audits WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        return ProbeAuditIndexRecord.model_validate_json(row[0])
    except ValidationError as exc:
        print(
            f"warning: load_audit_record skipped {session_id!r} — row "
            f"data did not parse as ProbeAuditIndexRecord: {exc}",
            file=sys.stderr,
        )
        return None


def load_audit_index(
    base_dir: Path | None = None,
    *,
    prune: bool = False,
) -> list[ProbeAuditIndexRecord]:
    """Load all rows, sorted by ``sample_timestamp`` descending.

    Python-side sort — ``sample_timestamp`` lives inside the JSON blob,
    so SQL-level ``ORDER BY`` would require ``json_extract`` (SQLite
    specific) or a separate indexed column (violates the single-blob-
    column convention). Python-side is the simpler choice at the
    constant-factor scale of this index.
    """
    base = resolve_base_dir(base_dir)
    if prune:
        prune_missing_audits(base_dir=base)
    with _get_db_connection(base) as conn:
        rows = conn.execute("SELECT session_id, data FROM audits").fetchall()
    # Per-row parse guard, same reasoning as ``prune_missing_audits``:
    # one corrupt row must not break the whole index load (and, via the
    # marimo notebook, the entire review UI). Skip + warn so healthy
    # rows still render.
    records: list[ProbeAuditIndexRecord] = []
    for session_id, data in rows:
        try:
            records.append(ProbeAuditIndexRecord.model_validate_json(data))
        except ValidationError as exc:
            print(
                f"warning: load_audit_index skipped {session_id!r} — row "
                f"data did not parse as ProbeAuditIndexRecord: {exc}",
                file=sys.stderr,
            )
    return sorted(records, key=lambda r: r.sample_timestamp, reverse=True)
