"""Stimulus loading — parse the assembled stimulus JSONL into per-example content.

These functions are the seam between the assembled synthetic-dataset JSONL and
every consumer that renders stimulus content: the analysis notebooks, the
rationale analysis bundle assembly, and the probe-audit sampling and
input-bundling paths.

This module is deliberately separate from ``pipeline.py``. It is one of the
probe-audit skill's fingerprint inputs (see
``.claude/skills/probe-audit/fingerprint_manifest.json``) because its output
shapes the scenario context bytes the auditor sees; keeping it free of
orchestration code means routine pipeline edits do not drift the audit
fingerprint.
"""

import json
from pathlib import Path


def parse_stimuli_jsonl(text: str) -> dict[str, dict[str, str]]:
    """Parse stimulus JSONL text into the same shape ``load_stimuli`` returns.

    Splits the text into lines, decodes each non-blank line as a JSON
    record, and returns a dict keyed by ``custom_id``. Callers that have
    already read the file's bytes (for SHA verification) use this helper
    to avoid a second filesystem read — eliminating the TOCTOU window
    where a SHA-verified file could be swapped between the SHA check
    and the parse.

    Validates each record's required shape (``custom_id``,
    ``system_prompt``, and ``messages[0].content``) and refuses
    duplicate ``custom_id`` values. Direct dict indexing without these
    guards would raise ``KeyError`` / ``IndexError`` and escape the
    ``except (json.JSONDecodeError, ValueError)`` catches in
    ``_cmd_sample`` and ``build_rationale_bundle`` as a traceback;
    explicit ``ValueError``s with line numbers and offending IDs keep
    diagnostics actionable. Duplicates are refused (rather than
    silently overwriting) because ``custom_id`` is the lookup key for
    scenario context — the second record would otherwise win and the
    audit would render the wrong ``grader_feedback`` for the trial.

    Raises ``ValueError`` if no records parse — empty input is treated
    as a load failure rather than a silently-empty mapping.
    """
    stimuli: dict[str, dict[str, str]] = {}
    line_number = 0
    for line in text.splitlines():
        line_number += 1
        if not line.strip():
            continue
        record = json.loads(line)
        if "custom_id" not in record:
            raise ValueError(
                f"stimulus record on line {line_number} is missing "
                f"'custom_id'"
            )
        custom_id = record["custom_id"]
        if "system_prompt" not in record:
            raise ValueError(
                f"stimulus record on line {line_number} (custom_id="
                f"{custom_id!r}) is missing 'system_prompt'"
            )
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"stimulus record on line {line_number} (custom_id="
                f"{custom_id!r}) is missing or has empty 'messages' list"
            )
        first_message = messages[0]
        if not isinstance(first_message, dict) or "content" not in first_message:
            raise ValueError(
                f"stimulus record on line {line_number} (custom_id="
                f"{custom_id!r}) is missing 'messages[0].content'"
            )
        if custom_id in stimuli:
            raise ValueError(
                f"stimulus record on line {line_number} has duplicate "
                f"custom_id={custom_id!r} — each custom_id must be unique "
                f"because it is the lookup key for scenario context; a "
                f"silent overwrite would render the wrong grader_feedback "
                f"for an audited trial"
            )
        stimuli[custom_id] = {
            "system_prompt": record["system_prompt"],
            "user_content": first_message["content"],
        }

    if not stimuli:
        raise ValueError("No stimuli parsed from JSONL text")

    return stimuli


def load_stimuli(jsonl_path: Path) -> dict[str, dict[str, str]]:
    """Load stimulus content from the assembled JSONL, keyed by example_id.

    Returns a dict mapping each example_id (e.g., 'example_01_correct')
    to a dict with two keys:
      - 'system_prompt': the meta-evaluator instructions the model received
      - 'user_content': the full stimulus markdown (system prompt, user
        prompt, initial draft, grader feedback) the model evaluated

    Thin wrapper over ``parse_stimuli_jsonl`` that handles the file
    read. Callers that need to SHA-verify the bytes before parsing
    should call ``parse_stimuli_jsonl`` directly with their already-read
    text to avoid a second filesystem read (TOCTOU).
    """
    try:
        return parse_stimuli_jsonl(jsonl_path.read_text())
    except ValueError as exc:
        # Re-raise with the file path so the caller sees which file was empty.
        raise ValueError(f"No stimuli loaded from {jsonl_path}") from exc
