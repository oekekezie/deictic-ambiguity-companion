"""Skill-fingerprint computation.

A Skill fingerprint is a SHA-256 digest of a canonicalized list of
``(relative-path, per-file-hash)`` pairs. The preferred per-file hash is
the Git blob SHA-1 (via ``git hash-object``) so every byte sequence that
participated in the fingerprint remains independently addressable from
Git history. When ``.git`` is absent (e.g., the Skill is extracted
standalone), we fall back to raw-bytes SHA-256 with explicit CRLF→LF
normalization — the method is recorded in the result so the notebook can
flag cross-method comparisons as not apples-to-apples.
"""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FingerprintMethod = Literal["git-blob-then-sha256", "raw-bytes-sha256"]


@dataclass(frozen=True)
class FingerprintResult:
    """Structured output of :func:`compute_fingerprint`.

    ``fingerprint`` is the outer SHA-256 (64 hex chars). ``per_file_hashes``
    is the deterministic mapping ``path -> git-blob-sha1 | raw-bytes-sha256``
    that the outer digest summarizes, and ``resolved_inputs`` is the sorted
    tuple of on-disk paths that the manifest globs expanded to.
    """

    fingerprint: str
    method: FingerprintMethod
    per_file_hashes: dict[str, str]
    resolved_inputs: tuple[str, ...]


def _is_git_repo(repo_root: Path) -> bool:
    """True when ``repo_root`` is a Git working tree (has ``.git``)."""
    return (repo_root / ".git").exists()


def _resolve_manifest_inputs(
    repo_root: Path, manifest_inputs: tuple[str, ...]
) -> tuple[str, ...]:
    """Expand globs and literal paths into a sorted list of relative files.

    Raises ``FileNotFoundError`` when a manifest entry matches zero files
    on disk — a silent empty expansion would let a deleted file disappear
    from the fingerprint without anyone noticing.
    """
    resolved: set[str] = set()
    for pattern in manifest_inputs:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(repo_root.glob(pattern))
            if not matches:
                raise FileNotFoundError(
                    f"Manifest pattern resolved to zero files: {pattern!r}"
                )
            for match in matches:
                if match.is_file():
                    resolved.add(str(match.relative_to(repo_root)))
        else:
            literal = repo_root / pattern
            if not literal.exists() or not literal.is_file():
                raise FileNotFoundError(
                    f"Manifest input not found on disk: {pattern!r}"
                )
            resolved.add(pattern)
    return tuple(sorted(resolved))


def _git_blob_hash(repo_root: Path, relpath: str) -> str:
    """Git blob SHA-1 of the current on-disk bytes of ``relpath``.

    ``git hash-object`` reads the on-disk bytes (uncommitted changes
    included) and applies the repo's ``.gitattributes`` normalization —
    exactly the bytes a ``git commit`` would store. This keeps
    normalization policy as Git's responsibility.
    """
    completed = subprocess.run(
        ["git", "hash-object", "--", relpath],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _raw_bytes_hash(repo_root: Path, relpath: str) -> str:
    """SHA-256 of the file's bytes with CRLF→LF normalization."""
    raw = (repo_root / relpath).read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _combine(per_file_hashes: dict[str, str]) -> str:
    """Outer SHA-256 over ``"<path>\\n<hash>\\n"`` lines in sorted-path order."""
    body = "".join(
        f"{path}\n{per_file_hashes[path]}\n" for path in sorted(per_file_hashes)
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def compute_fingerprint(
    repo_root: Path,
    manifest_inputs: tuple[str, ...],
) -> FingerprintResult:
    """Compute a deterministic fingerprint over ``manifest_inputs``.

    Globs in ``manifest_inputs`` (``**`` and ``*``) are expanded relative
    to ``repo_root`` and sorted before hashing so the result is
    path-order-independent. When a ``.git`` directory is present, each
    file is hashed via ``git hash-object`` (Git blob SHA-1); otherwise the
    raw-bytes SHA-256 fallback is used with CRLF→LF normalization.
    """
    resolved = _resolve_manifest_inputs(repo_root, manifest_inputs)
    method: FingerprintMethod
    if _is_git_repo(repo_root):
        method = "git-blob-then-sha256"
        per_file = {path: _git_blob_hash(repo_root, path) for path in resolved}
    else:
        method = "raw-bytes-sha256"
        per_file = {path: _raw_bytes_hash(repo_root, path) for path in resolved}
    return FingerprintResult(
        fingerprint=_combine(per_file),
        method=method,
        per_file_hashes=per_file,
        resolved_inputs=resolved,
    )


def git_describe_label(repo_root: Path) -> str | None:
    """Return ``git describe --always --dirty --long`` or ``None`` outside Git.

    Graceful degradation: a missing ``.git`` directory, an unborn branch,
    or any other describe failure returns ``None`` so callers can record
    "unknown" without blowing up.
    """
    if not _is_git_repo(repo_root):
        return None
    try:
        completed = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--long"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return completed.stdout.strip() or None
