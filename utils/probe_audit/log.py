"""Append-only audit log for probe-audit runs.

Each event is a canonical JSON object on its own line. Events carry a
``sha256_chain`` field whose value is the SHA-256 of the previous
event's serialized form (or the empty string for the first event) —
rewriting any past line breaks the chain, so :func:`validate_log` can
detect tampering even when the underlying filesystem has no tamper-proof
guarantees.

Writes are line-buffered, flushed, and fsync'd so a mid-write crash
leaves a truncated final line rather than corrupting earlier records.
:func:`read_log` silently skips the trailing corrupt line (consistent
with append-semantic invariants) but :func:`validate_log` refuses to
accept one.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

ALLOWED_EVENT_TYPES: tuple[str, ...] = (
    "sample_drawn",
    "rationale_fetched",
    "independent_reading_recorded",
    "probe_output_fetched",
    "probe_judgment_recorded",
)

# Events that must carry (trial_id, rep_index). ``sample_drawn`` is the
# only exception — it is written once per run at sample time and uses
# sentinel values ("-", 0) for trial_id/rep_index.
_REP_EVENT_TYPES = frozenset(ALLOWED_EVENT_TYPES) - {"sample_drawn"}


class LogValidationError(Exception):
    """Raised by :func:`validate_log` when the chain is broken or malformed."""


class EventRefused(Exception):
    """Raised by :func:`append_event` when a guard callback refuses the write.

    Callers pass a ``guard`` to enforce per-event preconditions (e.g.,
    "no prior independent_reading_recorded for this ``(trial_id, rep)``")
    atomically against the locked log. The guard's refusal reason is
    surfaced as the exception message so CLI handlers can print it
    verbatim to stderr.
    """


def _canonical(event: dict[str, Any]) -> str:
    """Sorted-key separator-compact JSON for a single event."""
    return json.dumps(event, sort_keys=True, separators=(",", ":"))


def _validate_event(event: dict[str, Any]) -> None:
    """Reject events that don't meet the schema invariants."""
    et = event.get("event_type")
    if et not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"Unknown event type: {et!r}")
    if et in _REP_EVENT_TYPES:
        if "trial_id" not in event:
            raise ValueError(f"Event {et!r} must carry trial_id")
        if "rep_index" not in event:
            raise ValueError(f"Event {et!r} must carry rep_index")
        # Rep-scoped events must identify the model that produced the
        # judgment so a post-hoc reviewer can verify which model was
        # active without reading the audit skill's frontmatter. Missing
        # field = hard refusal (not silent-default to "unknown"): the
        # reproducibility story relies on this being present.
        model = event.get("sub_agent_model")
        if not isinstance(model, str) or not model:
            raise ValueError(
                f"Event {et!r} must carry a non-empty sub_agent_model string"
            )
    if et == "rationale_fetched":
        # The rationale and stimulus source paths are only known at fetch
        # time and are the same for every rep in a run. Persisting them on
        # the fetch event means ``ProbeAuditProvenance``'s source-path
        # fields can be cross-checked against the log without a researcher
        # having to re-pick the files after the audit is persisted.
        source = event.get("rationale_source")
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"Event {et!r} must carry a non-empty rationale_source string"
            )
        stim = event.get("stimulus_source")
        if not isinstance(stim, str) or not stim:
            raise ValueError(
                f"Event {et!r} must carry a non-empty stimulus_source string"
            )


def _chain_from_text(text: str) -> str:
    """SHA-256 of the last valid JSON line in ``text``, or empty-bytes hash.

    Walks backward over non-empty lines and picks the first that parses
    as JSON, so a crash-truncated tail line does not become an illegal
    predecessor. Returns the SHA-256 of the raw line bytes (without the
    trailing newline, since ``splitlines()`` strips it).
    """
    if not text:
        return hashlib.sha256(b"").hexdigest()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return hashlib.sha256(b"").hexdigest()
    for line in reversed(lines):
        try:
            json.loads(line)
            return hashlib.sha256(line.encode("utf-8")).hexdigest()
        except json.JSONDecodeError:
            continue
    return hashlib.sha256(b"").hexdigest()


def _parse_events(text: str) -> list[dict[str, Any]]:
    """Parse the valid JSON lines in ``text`` into event dicts.

    Corrupt/truncated lines are skipped (matching :func:`read_log`) so
    the guard callback sees the same event history an external reader
    would.
    """
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _sub_agent_model_drift(
    event: dict[str, Any], prior_events: list[dict[str, Any]]
) -> str | None:
    """Return a refusal reason when a rep's model disagrees with prior events.

    Each ``(trial_id, rep_index)`` pair represents a single sub-agent's
    four-event trace. All four events must report the same
    ``sub_agent_model`` because a single sub-agent runs a single model.
    A mid-trace model change would indicate either (a) the orchestrator
    spawned two different sub-agents for one rep, which is a protocol
    violation, or (b) a malicious actor editing the log. Either way it
    invalidates provenance, so we refuse at append time.

    ``sample_drawn`` is exempt (it carries no ``sub_agent_model``) and is
    compared against nothing.
    """
    if event.get("event_type") == "sample_drawn":
        return None
    trial_id = event.get("trial_id")
    rep_index = event.get("rep_index")
    new_model = event.get("sub_agent_model")
    # Missing-field cases are covered by ``_validate_event``; this
    # helper only checks agreement between events that each pass
    # schema validation.
    if new_model is None or trial_id is None or rep_index is None:
        return None
    try:
        new_rep = int(rep_index)
    except (TypeError, ValueError):
        return None
    for prior in prior_events:
        if prior.get("event_type") == "sample_drawn":
            continue
        if prior.get("trial_id") != trial_id:
            continue
        prior_rep = prior.get("rep_index")
        if prior_rep is None:
            continue
        try:
            if int(prior_rep) != new_rep:
                continue
        except (TypeError, ValueError):
            continue
        prior_model = prior.get("sub_agent_model")
        if prior_model is not None and prior_model != new_model:
            return (
                f"sub_agent_model drift within (trial_id={trial_id!r}, "
                f"rep={rep_index}): prior events recorded {prior_model!r} "
                f"but this event reports {new_model!r}"
            )
    return None


def _stimulus_source_drift(
    event: dict[str, Any], prior_events: list[dict[str, Any]]
) -> str | None:
    """Return a refusal reason when a run's stimulus source disagrees.

    Mirrors :func:`_rationale_source_drift` for the stimulus JSONL pinned
    on the manifest. Two reps reporting different ``stimulus_source``
    paths in a single run indicates either a protocol violation or log
    tampering — either case invalidates the byte-level provenance of
    the auditor's scenario context, so refuse at append time.

    Applies only to ``rationale_fetched`` events — that is the only
    event type that carries the field.
    """
    if event.get("event_type") != "rationale_fetched":
        return None
    new_source = event.get("stimulus_source")
    if not isinstance(new_source, str):
        return None
    for prior in prior_events:
        if prior.get("event_type") != "rationale_fetched":
            continue
        prior_source = prior.get("stimulus_source")
        if prior_source is not None and prior_source != new_source:
            return (
                f"stimulus_source drift within run: prior rationale_fetched "
                f"events recorded {prior_source!r} but this event reports "
                f"{new_source!r}"
            )
    return None


def _rationale_source_drift(
    event: dict[str, Any], prior_events: list[dict[str, Any]]
) -> str | None:
    """Return a refusal reason when a run's rationale source disagrees.

    Unlike ``sub_agent_model`` which is per-(trial_id, rep_index), the
    rationale source path is a single value for the entire audit run —
    the sample manifest pins ``probe_snapshot_path`` analogously, and
    the rationale source is its pair for the audit's input provenance.
    Two reps reporting different ``rationale_source`` paths in a single
    run indicates either a protocol violation (the orchestrator spawned
    reps against different rationale files) or log tampering. Either
    case invalidates provenance; refuse at append time.

    Applies only to ``rationale_fetched`` events — that is the only
    event type that carries the field.
    """
    if event.get("event_type") != "rationale_fetched":
        return None
    new_source = event.get("rationale_source")
    if not isinstance(new_source, str):
        # Missing-field case is covered by ``_validate_event``; this
        # helper only checks agreement between events that each pass
        # schema validation.
        return None
    for prior in prior_events:
        if prior.get("event_type") != "rationale_fetched":
            continue
        prior_source = prior.get("rationale_source")
        if prior_source is not None and prior_source != new_source:
            return (
                f"rationale_source drift within run: prior rationale_fetched "
                f"events recorded {prior_source!r} but this event reports "
                f"{new_source!r}"
            )
    return None


def append_event(
    path: Path,
    event: dict[str, Any],
    *,
    guard: Callable[[list[dict[str, Any]]], str | None] | None = None,
) -> None:
    """Append a validated, chain-linked event to ``path``.

    The read-compute-write cycle is serialized by an exclusive advisory
    lock on the log file (``fcntl.flock(LOCK_EX)``), so concurrent
    sub-agent writers produced by the K-rep orchestration model cannot
    both observe the same predecessor line and clobber each other's
    chain link. The lock is process-scoped at the kernel level — a
    second writer that calls ``append_event`` on the same path in a
    separate process (or on a different file descriptor in the same
    process) blocks until the current writer releases the lock on close.

    ``guard`` is an optional callback evaluated INSIDE the lock window
    against the parsed existing events. Returning a non-None string
    atomically refuses the append and raises :class:`EventRefused` with
    that message. This is how duplicate-event and protocol-ordering
    invariants stay tight under concurrent writers — reading the log
    outside the lock would race against another writer's append.

    The event dict is copied before mutation so callers that reuse the
    same dict across calls do not see the ``sha256_chain`` field leak
    in from a prior invocation.
    """
    _validate_event(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open ``a+`` and use the same fd for both the predecessor read and
    # the append. Seek to 0 before reading (``a+`` leaves the position
    # undefined on some platforms) and rely on the append flag so the
    # subsequent ``write`` lands at EOF regardless of the seek.
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            existing = f.read()
            parsed_existing = _parse_events(existing)
            if guard is not None:
                refusal = guard(parsed_existing)
                if refusal is not None:
                    raise EventRefused(refusal)
            drift = _sub_agent_model_drift(event, parsed_existing)
            if drift is not None:
                raise EventRefused(drift)
            source_drift = _rationale_source_drift(event, parsed_existing)
            if source_drift is not None:
                raise EventRefused(source_drift)
            stim_drift = _stimulus_source_drift(event, parsed_existing)
            if stim_drift is not None:
                raise EventRefused(stim_drift)
            chain = _chain_from_text(existing)
            event_with_chain = dict(event)
            event_with_chain["sha256_chain"] = chain
            line = _canonical(event_with_chain) + "\n"
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_log(path: Path) -> list[dict[str, Any]]:
    """Return parsed events from ``path``; missing file → empty list.

    Corrupt tail lines (e.g., a crash-truncated final record) are
    silently skipped so callers can still see the full history up to
    the crash point. Use :func:`validate_log` to refuse corrupt logs.
    """
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip a crash-truncated tail — validate_log catches it.
            continue
    return events


def validate_log(path: Path) -> None:
    """Raise :class:`LogValidationError` on any chain or parse defect.

    Each line must be valid JSON, carry a ``sha256_chain`` linking to
    the prior line's canonical SHA-256, and satisfy :func:`_validate_event`.
    """
    if not path.exists():
        return
    prev = hashlib.sha256(b"").hexdigest()
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogValidationError(f"Line {line_no}: unparseable JSON") from exc
        try:
            _validate_event(event)
        except ValueError as exc:
            raise LogValidationError(f"Line {line_no}: {exc}") from exc
        if event.get("sha256_chain") != prev:
            raise LogValidationError(
                f"Line {line_no}: chain mismatch — expected {prev[:8]}…, "
                f"got {str(event.get('sha256_chain'))[:8]}…"
            )
        prev = hashlib.sha256(line.encode("utf-8")).hexdigest()
