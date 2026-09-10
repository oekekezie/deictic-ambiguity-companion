"""Storage functions for batch inference job persistence.

Manages the on-disk layout of job metadata, input artifacts, and results
under a configurable base directory ($BATCH_INFERENCE_DIR or ./batch_inference/).

The on-disk job directories are the canonical source of truth. SQLite serves
as a fast index over those directories, with each job's full metadata also
stored as metadata.json for human readability. Pruning keeps the index
consistent with the filesystem when job directories are removed externally.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from utils.batch_inference.job import JobRecord

logger = logging.getLogger(__name__)


def _get_base_dir() -> Path:
    """Resolve the base directory for batch inference storage."""
    return Path(os.environ.get("BATCH_INFERENCE_DIR", "./batch_inference"))


def _get_db_path(base_dir: Path) -> Path:
    """Return the path to the SQLite database file."""
    return base_dir / "jobs.db"


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema if not present."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)
    conn.commit()


@contextmanager
def _get_db_connection(base_dir: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connection with auto-init."""
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = _get_db_path(base_dir)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        _init_db(conn)
        yield conn
    finally:
        conn.close()


def resolve_job_dir(job_dir: str, *, mkdir: bool = False) -> Path:
    """Resolve a relative job_dir to its absolute path under the base directory.

    JobRecord.job_dir stores a relative path (e.g. "jobs/openai/{id}").
    All storage functions internally prepend _get_base_dir(); this function
    exposes the same resolution for callers that need the full filesystem path.

    When mkdir=True, creates the directory tree if it doesn't exist.
    """
    full_path = _get_base_dir() / job_dir
    if mkdir:
        full_path.mkdir(parents=True, exist_ok=True)
    return full_path


def sanitize_model_for_path(model: str) -> str:
    """Extract a filesystem-safe directory name from a model identifier.

    Strips any hierarchical prefix by taking the segment after the last
    forward slash (e.g., "accounts/fireworks/models/kimi-k2p5" → "kimi-k2p5").
    Model names without slashes pass through unchanged.
    """
    sanitized = model.rsplit("/", 1)[-1]
    if not sanitized:
        raise ValueError(f"Model name resolves to empty string: {model!r}")
    return sanitized


def _write_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    """Write a list of dicts as newline-delimited JSON."""
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _upsert_job(conn: sqlite3.Connection, record: JobRecord) -> None:
    """Insert or replace a job record in the database."""
    conn.execute(
        "INSERT OR REPLACE INTO jobs (job_id, data) VALUES (?, ?)",
        (record.job_id, record.model_dump_json()),
    )
    conn.commit()


def persist_job(
    record: JobRecord,
    original_jsonl: list[dict[str, Any]],
    transformed_jsonl: list[dict[str, Any]],
    base_dir: Path | None = None,
) -> Path:
    """Write job metadata and input artifacts to disk.

    Creates the job directory, writes metadata.json, input_original.jsonl,
    and input_transformed.jsonl, then inserts into the SQLite index.
    Returns the absolute path to the job directory.
    """
    base = base_dir or _get_base_dir()
    job_dir = base / record.job_dir
    job_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "metadata.json").write_text(record.model_dump_json(indent=2))
    _write_jsonl(job_dir / "input_original.jsonl", original_jsonl)
    _write_jsonl(job_dir / "input_transformed.jsonl", transformed_jsonl)

    with _get_db_connection(base) as conn:
        _upsert_job(conn, record)

    return job_dir


def update_job_status(
    record: JobRecord,
    base_dir: Path | None = None,
) -> None:
    """Update job metadata in both metadata.json and the SQLite index.

    Uses INSERT OR REPLACE for atomic in-place updates.
    """
    base = base_dir or _get_base_dir()
    job_dir = base / record.job_dir

    (job_dir / "metadata.json").write_text(record.model_dump_json(indent=2))

    with _get_db_connection(base) as conn:
        _upsert_job(conn, record)


def save_results(
    job_dir: Path,
    output_data: str,
    errors_data: str | None = None,
) -> None:
    """Write output and optional error data to the job directory."""
    (job_dir / "output.jsonl").write_text(output_data)
    if errors_data is not None:
        (job_dir / "errors.jsonl").write_text(errors_data)


def prune_missing_jobs(base_dir: Path | None = None) -> list[str]:
    """Remove stale jobs from the SQLite index.

    A job is stale when its stored JSON is corrupted, its on-disk directory
    no longer exists, or its metadata no longer validates against the current
    schema (e.g., a model family was renamed after the job was persisted).

    Corrupted rows are caught via lightweight ``json.loads`` first, then
    surviving rows undergo full Pydantic validation so schema-incompatible
    records are pruned rather than left as zombie rows.

    Returns the pruned job_ids for caller visibility.
    """
    base = base_dir or _get_base_dir()
    db_path = _get_db_path(base)

    if not db_path.exists():
        return []

    pruned_ids: list[str] = []

    with _get_db_connection(base) as conn:
        # Materialize all rows before mutating the table
        rows = conn.execute("SELECT job_id, data FROM jobs").fetchall()
        for job_id, data in rows:
            try:
                parsed = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Pruning job %s: corrupted JSON", job_id)
                pruned_ids.append(job_id)
                continue

            job_dir = parsed.get("job_dir") if isinstance(parsed, dict) else None
            if not isinstance(job_dir, str) or not job_dir or not (base / job_dir).is_dir():
                pruned_ids.append(job_id)
                continue

            # Directory exists — verify the record still validates
            try:
                JobRecord.model_validate_json(data)
            except Exception as exc:
                logger.warning("Pruning job %s: schema validation failed (%s)", job_id, exc)
                pruned_ids.append(job_id)

        if pruned_ids:
            placeholders = ",".join("?" for _ in pruned_ids)
            conn.execute(
                f"DELETE FROM jobs WHERE job_id IN ({placeholders})",
                pruned_ids,
            )
            conn.commit()

    return pruned_ids


def load_index(base_dir: Path | None = None, *, prune: bool = False) -> list[JobRecord]:
    """Load all job records from the SQLite index.

    When prune=True, first removes entries whose job directories no longer
    exist on disk, keeping the index consistent with the filesystem.

    Records that fail schema validation (e.g., unrecognized model families
    from earlier submissions) are skipped and logged as warnings so one bad
    row cannot crash the entire index load.

    Returns an empty list if the database does not exist.
    """
    base = base_dir or _get_base_dir()

    if prune:
        prune_missing_jobs(base_dir=base)

    db_path = _get_db_path(base)

    if not db_path.exists():
        return []

    records: list[JobRecord] = []
    with _get_db_connection(base) as conn:
        cursor = conn.execute("SELECT job_id, data FROM jobs ORDER BY job_id")
        for job_id, data in cursor:
            try:
                records.append(JobRecord.model_validate_json(data))
            except Exception as exc:
                logger.warning("Skipping job %s: validation failed (%s)", job_id, exc)
    return records


def load_job(job_dir: Path) -> JobRecord:
    """Load a single JobRecord from its metadata.json file."""
    return JobRecord.model_validate_json(
        (job_dir / "metadata.json").read_text()
    )
