"""Disagreement extraction from already-validated audit sessions.

Companion to ``utils.probe_audit.review_data`` — this module accepts a
list of ``(session_id, ProbeAuditResults)`` tuples that the notebook
has already validated through ``pool_compatibility``, plus the parsed
probe payload, and yields one ``DisagreementRow`` per
``(trial_key, flag)`` pair where the auditor's reading and the probe's
classification disagreed at any phase of review
(``disagreement_category_<flag> != "stable_agreement"``). All three
non-trivial DisagreementCategory values are queue-eligible:

- ``stable_disagreement``: pre-view and post-view both opposed the probe
- ``shifted_against_probe``: pre-view aligned with the probe (or probe
  value None at aggregate time); post-view shifted to oppose
- ``shifted_to_probe``: pre-view opposed the probe; post-view shifted to
  align with the probe — possible anchoring artifact (the auditor's
  independent reading was overwritten after exposure to the probe).
  These cases are essential for honest SM5 probe-reliability
  calibration; treating them as "aligned" because the post-view aligned
  with the probe would bake the anchoring artifact into the alignment
  rate.

The probe's per-trial value is read exactly once here, via
``probe_flags_by_trial`` — downstream code (notebook save handler,
export) consumes it from the row and never re-reads the snapshot. The
schema's ``probe_classification: bool`` makes ``None`` an invalid
state, so we raise ``ProbeValueMissingError`` atomically whenever a
queue-eligible trial has a None probe value. Per CLAUDE.md, raise
rather than silently filter; investigate the source audit if this
fires.
"""

from typing import NamedTuple

from utils.probe_audit.models import (
    DisagreementCategory,
    ProbeAuditResults,
)
from utils.probe_audit.review_data import (
    disagreement_category_descriptions,
    probe_flags_by_trial,
)
from utils.probe_audit.scenario_context import condition_from_example_id
from utils.probe_audit_annotation.models import Flag, TrialAnnotation
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS


class ProbeValueMissingError(RuntimeError):
    """The probe's per-flag value is ``None`` for a queue-eligible trial.

    Surfaced from ``disagreement_records`` when a trial satisfies the
    queue predicate (any non-trivial disagreement category) but the
    probe's own per-flag value did not parse. Under normal flow this
    arises only for ``shifted_against_probe`` rows where the probe's K
    reps fell below the parse threshold at probe-output time and the
    auditor still judged the probe wrong post-view. For
    ``stable_disagreement`` and ``shifted_to_probe`` rows the
    disagreement category itself is only computable from a concrete
    probe value at aggregate time (per
    ``utils/probe_audit/disagreement.categorize_disagreement``), so a
    None probe value at queue-construction time implies probe-snapshot
    drift between aggregate and queue construction. The annotator
    cannot meaningfully compare against a missing probe value, so we
    refuse to construct a queue containing such a trial. Investigate
    the source audit; do not silently filter.
    """

    def __init__(
        self,
        trial_key: tuple[str, str, int, int],
        flag: Flag,
        session_id: str,
    ) -> None:
        self.trial_key = trial_key
        self.flag = flag
        self.session_id = session_id
        super().__init__(
            f"Probe value is None for {trial_key=!r}, {flag=!r}, "
            f"{session_id=!r}. The disagreement category for this row "
            "is queue-eligible, but the probe snapshot does not carry "
            "a True/False value to display alongside it. Investigate "
            "the audit before annotating; the queue cannot be "
            "constructed safely."
        )


class DisagreementRow(NamedTuple):
    """One row of the annotation queue.

    Each row is a ``(trial_key, flag)`` pair where the auditor's
    reading and the probe's classification disagreed at any phase of
    review (``disagreement_category != "stable_agreement"``). All three
    non-trivial DisagreementCategory values are queue-eligible.
    ``probe_classification`` is the probe's own True/False — read once
    during queue construction from the probe snapshot via
    ``probe_flags_by_trial`` — and is the single source of truth for
    the probe value at save time.
    """

    trial_key: tuple[str, str, int, int]
    flag: Flag
    session_id: str
    condition: str
    config_key: str
    example_id: str
    batch_index: int
    trial: int
    disagreement_category: DisagreementCategory
    probe_classification: bool
    skill_fingerprint: str


def _disagreement_category(
    record,
    flag: Flag,
) -> DisagreementCategory:
    return getattr(record, f"disagreement_category_{flag}")


def disagreement_records(
    sessions: list[tuple[str, ProbeAuditResults]],
    probe_payload: dict,
    *,
    flags: tuple[Flag, ...] = RATIONALE_ANALYSIS_FLAG_KEYS,
) -> list[DisagreementRow]:
    """Extract ``DisagreementRow``s from validated sessions.

    Predicate: ``record.disagreement_category_<flag> != "stable_agreement"``.
    Captures all three non-trivial DisagreementCategory values:

    - ``stable_disagreement``: pre-view and post-view both opposed the
      probe's classification.
    - ``shifted_against_probe``: pre-view aligned with the probe (or
      probe value None at aggregate time), post-view shifted to oppose
      after seeing the probe's reasoning.
    - ``shifted_to_probe``: pre-view opposed the probe, post-view
      shifted to align after seeing the probe's reasoning. This is the
      "possible anchoring artifact" case — important to surface for
      SM5 probe-reliability calibration; treating it as alignment
      because the post-view aligned would bake the anchoring artifact
      into the alignment rate.

    Trials whose category is ``stable_agreement`` (consistent
    probe-auditor reading at both phases, including the
    indeterminate-fallback case where probe value or independent
    reading was None at aggregate time) are excluded. ``flags``
    controls which flags contribute rows; default emits all seven.

    The ``sessions`` list must already be pool-compatible — this module
    does not call ``pool_compatibility`` itself; the notebook gates on
    compat before calling here. The single ``probe_payload`` dict is
    the parsed probe snapshot whose bytes the audit already pinned via
    drift refusal at the data-load cell.

    Raises ``ProbeValueMissingError`` atomically when a queue-eligible
    trial has ``probe_<flag>`` value ``None``. No partial list is
    returned on failure.
    """
    probe_flags = probe_flags_by_trial(probe_payload)
    rows: list[DisagreementRow] = []
    for session_id, results in sessions:
        skill_fingerprint = results.provenance.skill_fingerprint
        for record in results.records:
            for flag in flags:
                if _disagreement_category(record, flag) == "stable_agreement":
                    continue
                config_key, example_id, batch_index, trial = record.trial_key
                probe_value = probe_flags.get(record.trial_key, {}).get(flag)
                if probe_value is None:
                    raise ProbeValueMissingError(
                        trial_key=record.trial_key,
                        flag=flag,
                        session_id=session_id,
                    )
                rows.append(
                    DisagreementRow(
                        trial_key=record.trial_key,
                        flag=flag,
                        session_id=session_id,
                        condition=condition_from_example_id(example_id),
                        config_key=config_key,
                        example_id=example_id,
                        batch_index=batch_index,
                        trial=trial,
                        disagreement_category=_disagreement_category(record, flag),
                        probe_classification=probe_value,
                        skill_fingerprint=skill_fingerprint,
                    )
                )
    rows.sort(
        key=lambda r: (
            r.condition,
            r.example_id,
            r.batch_index,
            r.trial,
            r.flag,
        )
    )
    return rows


_FLAG_DISPLAY: dict[Flag, str] = {
    flag: flag.replace("_", " ").capitalize()
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS
}


# Single source of truth for the four interpretive disagreement-category
# labels: ``utils/probe_audit/review_data.disagreement_category_descriptions``.
# Both this module and the review notebook consume the canonical mapping
# so the labels stay synchronized — a researcher moving between the
# annotation queue and the review notebook sees the same vocabulary.
_CATEGORY_LABEL_FOR: dict[DisagreementCategory, str] = {
    d.category: d.short_label for d in disagreement_category_descriptions()
}


def disagreement_queue_rows(
    rows: list[DisagreementRow],
    *,
    annotation_lookup: dict[
        tuple[tuple[str, str, int, int], Flag], TrialAnnotation | None
    ],
    conditions: tuple[str, ...] = (),
    session_ids: tuple[str, ...] = (),
    completed_only: bool | None = None,
) -> list[dict]:
    """Project ``DisagreementRow``s into display dicts for ``mo.ui.table``.

    ``annotation_lookup`` is keyed by ``(trial_key, flag)``; a value of
    ``None`` (or absence) indicates "not yet annotated." The status
    column carries "saved" / "pending" so the annotator can scan the
    queue. ``completed_only=True`` keeps only saved rows;
    ``completed_only=False`` keeps only pending rows; ``None`` returns
    everything. ``conditions`` and ``session_ids`` are conjunctive
    filters; an empty tuple means "no filter."
    """
    out: list[dict] = []
    for row in rows:
        if conditions and row.condition not in conditions:
            continue
        if session_ids and row.session_id not in session_ids:
            continue
        annotation = annotation_lookup.get((row.trial_key, row.flag))
        is_saved = annotation is not None
        if completed_only is True and not is_saved:
            continue
        if completed_only is False and is_saved:
            continue
        # Column order is workflow-first: status + flag (the queue's
        # primary scan axes) lead, then the review-notebook trial-
        # identity cluster verbatim ([Source Audit (pooled),] Config
        # Key, Example ID, Batch, Trial), then analytical context.
        # The notebook drops "Source Audit" in single-audit mode via
        # a comprehension on these display rows; the dict shape is
        # the same for both modes. ``mo.ui.table`` honors dict
        # insertion order for column display order.
        #
        # Status values use a symbol-prefixed Title-Case form so the
        # state pops at a glance even mid-horizontal-scroll. Bare
        # ``saved`` / ``pending`` lowercase strings are too quiet to
        # read in a queue.
        out.append(
            {
                "Annotation Status": "✓ Saved" if is_saved else "○ Pending",
                "Flag": _FLAG_DISPLAY[row.flag],
                "Source Audit": row.session_id,
                "Config Key": row.config_key,
                "Example ID": row.example_id,
                "Batch": row.batch_index,
                "Trial": row.trial,
                "Condition": row.condition,
                "Disagreement Category": _CATEGORY_LABEL_FOR[
                    row.disagreement_category
                ],
                "Probe Classification": "Yes" if row.probe_classification else "No",
            }
        )
    return out
