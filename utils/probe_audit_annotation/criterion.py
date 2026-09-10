"""Criterion-text loading + SHA-256 computation.

The criterion is ``rationale_analysis/system_prompt.md`` — the same
seven-flag classification spec the probe consumes and the byte-equality
test in ``tests/probe_audit/test_agent_body_consistency.py`` pins
verbatim into the rep-agent. The annotation tool surfaces the criterion
inline (the per-flag ``### `` section next to the meta-evaluator's
rationale being judged) so the annotator does not have to recall a
flag's definition from memory.

``criterion_sha256`` on each annotation is computed at annotation time
from the bytes of this file. That hash is the annotation's epoch
identifier — distinct from the audit's ``skill_per_file_hashes``
(git-blob SHA-1) because the annotator's epoch is "what was the
criterion when this human reviewed it," not "what was the criterion
when the audit ran."
"""

import hashlib
from pathlib import Path

_CRITERION_RELPATH = Path("rationale_analysis") / "system_prompt.md"

# The file-level horizontal rule separating the seven-flag spec from the
# JSON-output template. It is the end marker for the final flag's section.
_TRAILING_DELIMITER = "\n---\n"


def _default_repo_root() -> Path:
    """Resolve the repo root from this file's location.

    ``utils/probe_audit_annotation/criterion.py`` sits two ``parents``
    away from the repo root. Callers in tests pass ``repo_root``
    explicitly to point into ``tmp_path``; production callers in the
    notebook pass the notebook's ``REPO_ROOT`` constant.
    """
    return Path(__file__).resolve().parents[2]


def resolve_criterion_path(repo_root: Path | None = None) -> Path:
    """Return the absolute path to ``rationale_analysis/system_prompt.md``."""
    base = repo_root if repo_root is not None else _default_repo_root()
    return base / _CRITERION_RELPATH


def compute_criterion_sha256(repo_root: Path | None = None) -> str:
    """Return ``hashlib.sha256(bytes).hexdigest()`` for the criterion file.

    Raises ``FileNotFoundError`` if the file is absent — the notebook's
    cell-5 hard-stop displays the resolved path so the researcher can
    diagnose a wrong working directory or a missing file.
    """
    target = resolve_criterion_path(repo_root)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_criterion_text(repo_root: Path | None = None) -> str:
    """Read and return the criterion file as text (UTF-8)."""
    target = resolve_criterion_path(repo_root)
    return target.read_text(encoding="utf-8")


def flag_excerpt(criterion_text: str, flag: str) -> str:
    """Section text for one flag's ``### `` heading in the criterion.

    Each of the seven flags has its own ``### <Label> (`<flag_key>`)``
    heading in ``rationale_analysis/system_prompt.md``. The excerpt runs
    from that heading to the next ``### `` heading, or — for the final
    flag — to the file-level ``---`` rule that separates the flag spec
    from the JSON-output template.

    Section parsing is anchored on the literal ``(`<flag_key>`)`` marker
    so it stays stable against whitespace edits elsewhere in the file. A
    missing heading raises ``ValueError`` rather than returning an empty
    string — silent failure here would mean a future edit to
    ``system_prompt.md`` could ship empty excerpts to annotators without
    anyone noticing.
    """
    marker = f"(`{flag}`)"
    marker_idx = criterion_text.find(marker)
    if marker_idx == -1:
        raise ValueError(
            f"criterion text does not contain a heading for flag {flag!r} "
            f"(expected a '### ' line carrying the marker {marker!r})"
        )
    # Walk back to the start of the heading's line and confirm it is a
    # '### ' heading — a stray inline mention of the marker must not be
    # mistaken for the section heading.
    line_start = criterion_text.rfind("\n", 0, marker_idx) + 1
    if not criterion_text.startswith("### ", line_start):
        raise ValueError(
            f"the {marker!r} marker for flag {flag!r} is not on a '### ' "
            "heading line"
        )
    # The section ends at the next '### ' heading, or — for the final
    # flag — at the file-level '---' rule before the JSON-output template.
    next_heading = criterion_text.find("\n### ", marker_idx)
    trailing = criterion_text.find(_TRAILING_DELIMITER, marker_idx)
    candidates = [idx for idx in (next_heading, trailing) if idx != -1]
    if not candidates:
        raise ValueError(
            f"criterion text has no section terminator (next '### ' heading "
            f"or trailing '---' rule) after the {flag!r} heading"
        )
    return criterion_text[line_start : min(candidates)].rstrip()
