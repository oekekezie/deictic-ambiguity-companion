"""SQLite persistence layer for probe-audit annotations.

The storage layer mirrors ``utils/probe_audit/storage.py`` ergonomically
— same ``contextmanager`` connection idiom, ``isolation_level=None``
autocommit with explicit ``BEGIN IMMEDIATE`` / ``COMMIT`` brackets,
``CREATE TABLE IF NOT EXISTS`` self-init on first connection — so a
researcher moving between the two tools encounters the same shape.

Schema is hybrid: typed columns for filter dimensions
(``trial_key_str``, ``flag``, ``annotator_id``, ``criterion_sha256``,
``skill_fingerprint``, ``disagreement_category``, ``annotated_at``)
plus a JSON-blob ``data`` column carrying the full
``TrialAnnotation.model_dump_json()``. Filter queries read indexed
typed columns; reconstruction uses the JSON blob as the source of truth.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from utils.probe_audit_annotation.models import (
    AnnotationQuery,
    Flag,
    TrialAnnotation,
)

_DB_FILENAME = "probe_audit_annotations.db"
_ENV_VAR = "PROBE_AUDIT_ANNOTATION_DIR"
_DEFAULT_BASE_DIR = Path("./analysis_outputs/probe_audit_annotations")


def resolve_base_dir(
    explicit: Path | None = None,
    *,
    fallback: Path | None = None,
) -> Path:
    """Resolve the annotation base directory.

    Precedence: ``explicit`` > ``$PROBE_AUDIT_ANNOTATION_DIR`` >
    ``fallback`` > ``./analysis_outputs/probe_audit_annotations``.
    Mirrors ``utils.probe_audit.storage.resolve_base_dir`` so the two
    tools share the same env-var lookup ergonomics — only the variable
    name differs.
    """
    if explicit is not None:
        return explicit
    env_val = os.environ.get(_ENV_VAR)
    if env_val:
        return Path(env_val)
    if fallback is not None:
        return fallback
    return _DEFAULT_BASE_DIR


def _trial_key_to_str(trial_key: tuple[str, str, int, int]) -> str:
    """Pipe-join the four-tuple into a canonical scalar string.

    The pipe is the separator because neither config keys nor example
    IDs contain pipes in the project's naming convention. Components
    that DO contain a pipe trigger ``ValueError`` rather than silent
    corruption — pipes inside fields would make the canonical form
    ambiguous.
    """
    config_key, example_id, batch_index, trial = trial_key
    for piece in (config_key, example_id):
        if "|" in piece:
            raise ValueError(
                f"trial_key components must not contain '|'; got {trial_key!r}"
            )
    return f"{config_key}|{example_id}|{batch_index}|{trial}"


def _db_path(base_dir: Path) -> Path:
    return base_dir / _DB_FILENAME


def _create_schema(conn: sqlite3.Connection) -> None:
    """Idempotent schema init: table + four named indices.

    ``IF NOT EXISTS`` makes repeated calls safe. The UNIQUE constraint
    on ``(trial_key_str, flag, annotator_id, criterion_sha256)`` is
    what enforces "one annotation per (trial, flag, annotator, epoch)";
    SQLite auto-creates an index for it which the four named indices
    below complement for non-unique filter dimensions.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            annotation_id TEXT PRIMARY KEY,
            trial_key_str TEXT NOT NULL,
            flag TEXT NOT NULL,
            annotator_id TEXT NOT NULL,
            criterion_sha256 TEXT NOT NULL,
            skill_fingerprint TEXT NOT NULL,
            disagreement_category TEXT NOT NULL,
            annotated_at TEXT NOT NULL,
            data TEXT NOT NULL,
            UNIQUE (trial_key_str, flag, annotator_id, criterion_sha256)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_annotations_annotator_flag "
        "ON annotations (annotator_id, flag)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_annotations_criterion "
        "ON annotations (criterion_sha256)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_annotations_trial_key "
        "ON annotations (trial_key_str)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_annotations_skill_fingerprint "
        "ON annotations (skill_fingerprint)"
    )


@contextmanager
def _connection(base_dir: Path) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with auto-init on first open.

    ``isolation_level=None`` puts the driver in autocommit mode so the
    save path can explicitly ``BEGIN IMMEDIATE`` and ``COMMIT`` /
    ``ROLLBACK``. The 10-second timeout absorbs brief lock contention
    from a concurrent reader without surfacing it to the caller.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(_db_path(base_dir)), timeout=10.0, isolation_level=None
    )
    try:
        _create_schema(conn)
        yield conn
    finally:
        conn.close()


def init_schema(*, base_dir: Path | None = None) -> Path:
    """Eagerly create the database file and schema. Returns the DB path.

    Useful when a caller wants to materialize the file before the first
    save (e.g., to grant filesystem permissions). The save and load
    paths invoke ``_create_schema`` themselves, so explicit init is
    optional in normal use.
    """
    resolved = resolve_base_dir(base_dir)
    with _connection(resolved):
        pass
    return _db_path(resolved)


def save_annotation(
    annotation: TrialAnnotation,
    *,
    base_dir: Path | None = None,
) -> None:
    """Idempotent upsert of one annotation.

    The UNIQUE constraint on
    ``(trial_key_str, flag, annotator_id, criterion_sha256)`` provides
    epoch-scoped one-row-per-trial semantics. ``INSERT OR REPLACE``
    against that constraint atomically deletes any prior row sharing
    the same 4-tuple and inserts the new one; the ``annotation_id``
    primary key changes on each save (it is a row identity, not a
    stable external identifier).

    Wrapped in ``BEGIN IMMEDIATE`` / ``COMMIT`` with ``ROLLBACK`` on
    exception so a mid-statement failure leaves no partial write.
    """
    resolved = resolve_base_dir(base_dir)
    payload = annotation.model_dump_json()
    with _connection(resolved) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT OR REPLACE INTO annotations (
                    annotation_id, trial_key_str, flag, annotator_id,
                    criterion_sha256, skill_fingerprint,
                    disagreement_category, annotated_at, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation.annotation_id,
                    _trial_key_to_str(annotation.trial_key),
                    annotation.flag,
                    annotation.annotator_id,
                    annotation.criterion_sha256,
                    annotation.skill_fingerprint,
                    annotation.disagreement_category,
                    annotation.annotated_at,
                    payload,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _query_clauses(query: AnnotationQuery) -> tuple[list[str], list[object]]:
    """Translate ``AnnotationQuery`` into SQL ``WHERE`` clauses + params.

    Each ``None`` field contributes nothing; everything else becomes a
    parameterized ``column = ?`` predicate. ``trial_key`` is converted
    to its canonical string before matching. ``pooled_audit_session_ids``
    is not a typed column — filtering on it requires a JSON-blob scan,
    handled by the caller after fetching candidate rows.
    """
    clauses: list[str] = []
    params: list[object] = []
    if query.annotator_id is not None:
        clauses.append("annotator_id = ?")
        params.append(query.annotator_id)
    if query.flag is not None:
        clauses.append("flag = ?")
        params.append(query.flag)
    if query.criterion_sha256 is not None:
        clauses.append("criterion_sha256 = ?")
        params.append(query.criterion_sha256)
    if query.skill_fingerprint is not None:
        clauses.append("skill_fingerprint = ?")
        params.append(query.skill_fingerprint)
    if query.trial_key is not None:
        clauses.append("trial_key_str = ?")
        params.append(_trial_key_to_str(query.trial_key))
    return clauses, params


def load_annotations(
    query: AnnotationQuery,
    *,
    base_dir: Path | None = None,
) -> list[TrialAnnotation]:
    """Load annotations matching ``query``. ``None`` filters are unfiltered.

    Results are sorted by ``annotated_at`` ascending — the natural
    chronological order for replay. The ``pooled_audit_session_ids``
    filter, when set, is applied Python-side against the deserialized
    rows (it is not a typed column).
    """
    resolved = resolve_base_dir(base_dir)
    clauses, params = _query_clauses(query)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(resolved) as conn:
        cursor = conn.execute(
            f"SELECT data FROM annotations {where} ORDER BY annotated_at ASC",
            params,
        )
        rows = [
            TrialAnnotation.model_validate_json(row[0]) for row in cursor.fetchall()
        ]
    if query.pooled_audit_session_ids is not None:
        rows = [
            r
            for r in rows
            if r.pooled_audit_session_ids == query.pooled_audit_session_ids
        ]
    return rows


def load_annotation_for(
    *,
    annotator_id: str,
    trial_key: tuple[str, str, int, int],
    flag: Flag,
    criterion_sha256: str,
    base_dir: Path | None = None,
) -> TrialAnnotation | None:
    """Targeted single-row lookup by the UNIQUE-constraint 4-tuple.

    Returns ``None`` when no annotation exists for the given key. This
    is the lookup the notebook uses to pre-populate the annotation form
    when an annotator returns to a previously-saved row.
    """
    resolved = resolve_base_dir(base_dir)
    with _connection(resolved) as conn:
        cursor = conn.execute(
            "SELECT data FROM annotations WHERE trial_key_str = ? "
            "AND flag = ? AND annotator_id = ? AND criterion_sha256 = ?",
            (
                _trial_key_to_str(trial_key),
                flag,
                annotator_id,
                criterion_sha256,
            ),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return TrialAnnotation.model_validate_json(row[0])


def annotation_exists(
    *,
    annotator_id: str,
    trial_key: tuple[str, str, int, int],
    flag: Flag,
    criterion_sha256: str,
    base_dir: Path | None = None,
) -> bool:
    """Fast existence check without deserializing the JSON blob."""
    resolved = resolve_base_dir(base_dir)
    with _connection(resolved) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM annotations WHERE trial_key_str = ? "
            "AND flag = ? AND annotator_id = ? AND criterion_sha256 = ? LIMIT 1",
            (
                _trial_key_to_str(trial_key),
                flag,
                annotator_id,
                criterion_sha256,
            ),
        )
        return cursor.fetchone() is not None
