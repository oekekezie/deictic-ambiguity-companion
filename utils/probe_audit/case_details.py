"""Case-level disagreement join — probe (rationale analysis) ⨝ auditor.

The reasoning-carrying sibling of
``utils.probe_audit_annotation.loading.disagreement_records``. Where that
module emits a thin annotation-queue row (probe value + category) per
``(trial, flag)`` disagreement, this analyst-facing helper joins the full
PROBE side — the rationale analysis snapshot's aggregated classification
and per-repetition analysis text — with the full AUDITOR side — each rep's
independent reading + reasoning, post-view probe-correct judgment +
reasoning, and corrected classification. The result feeds a case-level
typology cell in ``cell_deep_dive.py``.

This module lives beside ``review_data`` (analyst-facing, pure functions
over ``ProbeAuditResults``) rather than in the annotation layer: it does
not edit ``loading.py`` and shares only the ``!= "stable_agreement"``
queue predicate.

The probe's per-trial classification and per-rep analyses come from the
rationale analysis snapshot (``RationaleAnalysisResults``), keyed by the
same four-tuple the audit pipeline uses everywhere. A queue-eligible
audit trial with no matching RA record is provenance drift between the
audit and the snapshot it was built against — we raise rather than
silently drop the row, per the raise-over-silent convention.
"""

from typing import NamedTuple

from utils.probe_audit.models import (
    DisagreementCategory,
    ProbeAuditResults,
)
from utils.probe_audit.review_data import Flag
from utils.probe_audit.scenario_context import condition_from_example_id
from utils.rationale_analysis.models import (
    RationaleAnalysisResults,
    RationaleAnalysisTrialRecord,
)


class DisagreementCaseDetail(NamedTuple):
    """One ``(trial, flag)`` disagreement joined across probe and auditor.

    The probe side (``probe_classification``, ``probe_analyses``) is read
    from the rationale analysis snapshot; the auditor side (aggregated
    reading + probe-correct judgment, plus the per-rep reasoning/corrected
    tuples) is read from the audit record. ``direction`` summarizes the
    flag-level mismatch between the probe's classification and the
    auditor's independent reading, independent of the disagreement category.

    Per-rep fields are tuples so the struct is K-agnostic. ``probe_analyses``
    drops parse-failure (None) repetitions; the auditor reading/judgment
    tuples preserve one entry per recorded rep in rep order.
    """

    trial_key: tuple[str, str, int, int]
    condition: str
    config_key: str
    example_id: str
    flag: Flag
    disagreement_category: DisagreementCategory
    probe_classification: bool | None
    probe_analyses: tuple[str, ...]
    auditor_independent_reading: bool | None
    auditor_reading_reasonings: tuple[str, ...]
    auditor_probe_correct: bool | None
    auditor_corrected_classifications: tuple[bool | None, ...]
    auditor_judgment_reasonings: tuple[str, ...]
    direction: str


def _direction(probe_classification: bool | None, reading: bool | None) -> str:
    """Flag-level mismatch label between the probe call and auditor reading.

    ``over_flag`` — the probe flagged (True) where the auditor's
    independent reading did not (False); ``under_flag`` — the probe did
    not flag where the auditor did. Any other combination (a None on
    either side, or agreement) is ``indeterminate``: there is no clean
    over/under direction to report.
    """
    if probe_classification is True and reading is False:
        return "over_flag"
    if probe_classification is False and reading is True:
        return "under_flag"
    return "indeterminate"


def _probe_analyses(
    ra_record: RationaleAnalysisTrialRecord, flag: Flag
) -> tuple[str, ...]:
    """Per-repetition probe analysis text for ``flag``, skipping None reps.

    A None repetition is a parse failure carrying no analysis for any
    flag, so it is omitted from the analysis tuple (the auditor reasoning
    tuples, by contrast, preserve one entry per recorded rep).
    """
    return tuple(
        getattr(rep, flag).analysis
        for rep in ra_record.repetitions
        if rep is not None
    )


def disagreement_case_details(
    pool_results: list[ProbeAuditResults],
    ra_results: RationaleAnalysisResults,
    *,
    flag: Flag,
    conditions: tuple[str, ...] = (),
) -> list[DisagreementCaseDetail]:
    """Join every ``(trial, flag)`` disagreement to its probe + auditor detail.

    For each audit record across ``pool_results`` whose
    ``disagreement_category_<flag> != "stable_agreement"`` (and, when
    ``conditions`` is non-empty, whose condition is in ``conditions``),
    look up the matching rationale analysis record by trial key and emit
    one :class:`DisagreementCaseDetail`.

    The probe side comes from the RA snapshot: ``probe_classification`` is
    its aggregated per-flag value and ``probe_analyses`` is the per-rep
    analysis text (None reps dropped). The auditor side comes from the
    audit record: the aggregated independent reading and probe-correct
    judgment, plus per-rep reasoning and corrected-classification tuples.

    Raises ``KeyError`` naming the trial key and flag when a queue-eligible
    audit trial has no matching RA record — that is provenance drift
    between the audit and the snapshot it was built against, and silently
    skipping it would understate the disagreement set.

    Rows sort deterministically by
    ``(condition, example_id, batch_index, trial, flag)``; ``flag`` is
    fixed per call, so the effective order is by condition then trial
    identity.
    """
    ra_by_key: dict[tuple[str, str, int, int], RationaleAnalysisTrialRecord] = {
        (r.config_key, r.example_id, r.batch_index, r.trial): r
        for r in ra_results.trial_records
    }

    details: list[DisagreementCaseDetail] = []
    for results in pool_results:
        for record in results.records:
            if getattr(record, f"disagreement_category_{flag}") == "stable_agreement":
                continue
            config_key, example_id, batch_index, trial = record.trial_key
            condition = condition_from_example_id(example_id)
            if conditions and condition not in conditions:
                continue

            ra_record = ra_by_key.get(record.trial_key)
            if ra_record is None:
                raise KeyError(
                    f"No rationale analysis record for trial_key="
                    f"{record.trial_key!r}, flag={flag!r}. The audit marks "
                    "this trial as a disagreement, but the rationale analysis "
                    "snapshot has no matching record — provenance drift "
                    "between the audit and the snapshot it was built against. "
                    "Refusing to silently drop the case."
                )

            probe_classification = getattr(ra_record, flag)
            auditor_reading = getattr(
                record, f"aggregated_independent_reading_{flag}"
            )
            details.append(
                DisagreementCaseDetail(
                    trial_key=record.trial_key,
                    condition=condition,
                    config_key=config_key,
                    example_id=example_id,
                    flag=flag,
                    disagreement_category=getattr(
                        record, f"disagreement_category_{flag}"
                    ),
                    probe_classification=probe_classification,
                    probe_analyses=_probe_analyses(ra_record, flag),
                    auditor_independent_reading=auditor_reading,
                    auditor_reading_reasonings=tuple(
                        getattr(rep, f"{flag}_reasoning")
                        for rep in record.rep_readings
                    ),
                    auditor_probe_correct=getattr(
                        record, f"aggregated_probe_correct_{flag}"
                    ),
                    auditor_corrected_classifications=tuple(
                        getattr(j, f"{flag}_corrected_classification")
                        for j in record.rep_judgments
                    ),
                    auditor_judgment_reasonings=tuple(
                        getattr(j, f"{flag}_reasoning")
                        for j in record.rep_judgments
                    ),
                    direction=_direction(probe_classification, auditor_reading),
                )
            )

    details.sort(
        key=lambda d: (
            d.condition,
            d.example_id,
            d.trial_key[2],
            d.trial_key[3],
            d.flag,
        )
    )
    return details
