"""Pool resolution for the probe-audit summary CLI.

Translates a permutation of CLI filters into a validated pool of
:class:`~utils.probe_audit.storage.ProbeAuditIndexRecord` records and
their loaded :class:`~utils.probe_audit.models.ProbeAuditResults`.
Each refusal path raises a distinct exception type so the CLI surface
can map the diagnosis to a stderr message:

- :class:`EmptyPoolError` — filter narrowed the pool to zero matches.
- :class:`NonAggregatedAuditsError` — at least one matching record is
  still ``"sampled"`` or ``"in_progress"``; the script needs aggregated
  snapshots only.
- :class:`PoolCompatibilityError` — every matching record was loaded,
  but :func:`pool_compatibility` returned identity-field disagreements.

The gates are layered in a fixed order so refusal messages name the
narrowest correct diagnosis (an empty match is reported before the
script tries to call :func:`pool_compatibility`, which would otherwise
emit a confusing "at least one ProbeAuditResults" error).
"""

from pathlib import Path

from utils.probe_audit.models import ProbeAuditResults
from utils.probe_audit.review_data import PoolCompatibility, pool_compatibility
from utils.probe_audit.storage import (
    ProbeAuditIndexRecord,
    load_audit_index,
)


class EmptyPoolError(ValueError):
    """No audit-index record matched the supplied filter combination."""


class NonAggregatedAuditsError(ValueError):
    """At least one matched record has ``audit_status != 'aggregated'``."""

    def __init__(
        self,
        message: str,
        *,
        offenders: tuple[ProbeAuditIndexRecord, ...],
    ) -> None:
        super().__init__(message)
        self.offenders = offenders


class PoolCompatibilityError(ValueError):
    """``pool_compatibility`` rejected the pool with one or more errors."""

    def __init__(
        self,
        message: str,
        *,
        errors: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.errors = errors


def _filter_records(
    records: list[ProbeAuditIndexRecord],
    *,
    fingerprint_prefix: str | None,
    args_digest_prefix: str | None,
    probe_snapshot_sha256: str | None,
    session_ids: tuple[str, ...] | None,
) -> list[ProbeAuditIndexRecord]:
    """Apply each filter as a conjunctive predicate. ``None`` filters skip."""
    if session_ids is not None:
        wanted = frozenset(session_ids)
        return [r for r in records if r.session_id in wanted]
    out = list(records)
    if fingerprint_prefix is not None:
        out = [r for r in out if r.skill_fingerprint.startswith(fingerprint_prefix)]
    if args_digest_prefix is not None:
        out = [r for r in out if r.args_digest.startswith(args_digest_prefix)]
    if probe_snapshot_sha256 is not None:
        out = [
            r
            for r in out
            if r.probe_snapshot_sha256 == probe_snapshot_sha256
        ]
    return out


def resolve_pool(
    *,
    audit_dir: Path,
    fingerprint_prefix: str | None,
    args_digest_prefix: str | None,
    probe_snapshot_sha256: str | None,
    session_ids: tuple[str, ...] | None,
) -> tuple[
    tuple[ProbeAuditIndexRecord, ...],
    tuple[ProbeAuditResults, ...],
    PoolCompatibility,
]:
    """Filter the audit index, load matching snapshots, and validate compatibility.

    Returns ``(records, results, compatibility)`` on success — the
    ``PoolCompatibility`` object is returned alongside the records and
    results so callers can read its ``warnings`` (and shared-identity
    fields) without recomputing trial-key overlap detection across all
    inputs. Raises :class:`EmptyPoolError`,
    :class:`NonAggregatedAuditsError`, or :class:`PoolCompatibilityError`
    on the corresponding refusal.

    ``audit_dir`` is the resolved base directory (caller must invoke
    :func:`utils.probe_audit.storage.resolve_base_dir` first if a CLI
    override is in play). All other arguments come from CLI flags.
    """
    index = load_audit_index(base_dir=audit_dir)
    matched = _filter_records(
        index,
        fingerprint_prefix=fingerprint_prefix,
        args_digest_prefix=args_digest_prefix,
        probe_snapshot_sha256=probe_snapshot_sha256,
        session_ids=session_ids,
    )
    if not matched:
        raise EmptyPoolError(
            "no audit-index records matched the supplied filters; "
            "run with --list to see tracked sessions"
        )
    non_aggregated = tuple(
        r for r in matched if r.audit_status != "aggregated"
    )
    if non_aggregated:
        offenders_summary = ", ".join(
            f"{r.session_id} ({r.audit_status})" for r in non_aggregated
        )
        raise NonAggregatedAuditsError(
            f"refusing pool: {len(non_aggregated)} matching session(s) are "
            f"not aggregated yet — {offenders_summary}. Aggregate them via "
            "the probe-audit CLI before summarizing.",
            offenders=non_aggregated,
        )
    results: list[ProbeAuditResults] = []
    for record in matched:
        if record.snapshot_path is None:
            raise NonAggregatedAuditsError(
                f"record {record.session_id!r} is marked aggregated but has "
                "no snapshot_path — index is corrupt; re-run "
                "backfill-from-filesystem",
                offenders=(record,),
            )
        snapshot_full = audit_dir / record.snapshot_path
        results.append(
            ProbeAuditResults.model_validate_json(snapshot_full.read_text())
        )
    compat = pool_compatibility(results)
    if compat.errors:
        joined = "; ".join(compat.errors)
        raise PoolCompatibilityError(
            f"pool_compatibility refused the pool: {joined}",
            errors=compat.errors,
        )
    return tuple(matched), tuple(results), compat
