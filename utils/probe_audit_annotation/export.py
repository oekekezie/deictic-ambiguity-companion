"""CSV export + summary statistics for stored annotations.

Two surfaces:

- ``export_annotations_csv`` writes a UTF-8 CSV whose column order is
  pinned by ``ExportRow.__annotations__``. The output filename embeds
  the skill fingerprint prefix and a UTC timestamp so a researcher can
  archive multiple exports without filename collisions and trace each
  CSV back to the audit epoch it was extracted under.

- ``annotation_summary`` returns a flat dict of counts (total, by flag,
  by annotator adjudication, by disagreement category) suitable for
  the SM5 prose narrative — "of N disagreements reviewed, M fit pattern
  X with reasoning Y."
"""

import csv
import datetime as _dt
from pathlib import Path

from utils.probe_audit.scenario_context import condition_from_example_id
from utils.probe_audit_annotation.models import (
    AnnotatorAdjudication,
    ExportRow,
    Flag,
    TrialAnnotation,
)
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS


def _format_row(annotation: TrialAnnotation) -> dict[str, object]:
    """Project ``TrialAnnotation`` onto the display-label CSV row.

    The dict is keyed by display labels (``"Annotation ID"``,
    ``"Trial Key (Batch Index)"``, ...) so ``csv.DictWriter`` emits the
    headers verbatim from ``ExportRow.__annotations__.keys()``. Pooled
    session ids are pipe-joined for round-trip parsability — a downstream
    consumer can split on ``"|"`` to recover the tuple.
    """
    config_key, example_id, batch_index, trial = annotation.trial_key
    return {
        "Annotation ID": annotation.annotation_id,
        "Trial Key (Config Key)": config_key,
        "Trial Key (Example ID)": example_id,
        "Trial Key (Batch Index)": batch_index,
        "Trial Key (Trial Number)": trial,
        "Flag": annotation.flag,
        "Condition": condition_from_example_id(example_id),
        "Disagreement Category": annotation.disagreement_category,
        "Probe Classification": annotation.probe_classification,
        "Annotator ID": annotation.annotator_id,
        "Annotator Adjudication": annotation.annotator_adjudication,
        "Annotator Reasoning": annotation.annotator_reasoning,
        "Pooled Audit Session IDs": "|".join(annotation.pooled_audit_session_ids),
        "Skill Fingerprint": annotation.skill_fingerprint,
        "Criterion SHA-256": annotation.criterion_sha256,
        "Tool Version": annotation.tool_version,
        "Annotated At (UTC)": annotation.annotated_at,
    }


def _build_filename(
    skill_fingerprint: str | None,
    *,
    now: _dt.datetime | None = None,
) -> str:
    """Return ``annotations_<fpPrefix>_<YYYYMMDDTHHMMSSZ>.csv``.

    The fingerprint prefix lets a researcher identify which audit
    epoch produced the export at a glance; ``fpUNKNOWN`` covers the
    legitimate case where a caller exports annotations spanning
    multiple skill fingerprints (the caller is then responsible for
    reading the per-row ``Skill Fingerprint`` column).
    """
    when = now if now is not None else _dt.datetime.now(_dt.UTC)
    fp_prefix = (
        f"fp{skill_fingerprint[:12]}" if skill_fingerprint else "fpUNKNOWN"
    )
    timestamp = when.strftime("%Y%m%dT%H%M%SZ")
    return f"annotations_{fp_prefix}_{timestamp}.csv"


def export_annotations_csv(
    annotations: list[TrialAnnotation],
    *,
    output_dir: Path,
    skill_fingerprint: str | None,
    now: _dt.datetime | None = None,
) -> Path:
    """Write ``annotations`` as CSV under ``output_dir``. Returns the file path.

    Empty input still writes a headers-only file — the empty CSV is a
    valid artifact a researcher can attach to a "no annotations under
    epoch X" note. The column order is pinned by
    ``ExportRow.__annotations__.keys()``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _build_filename(skill_fingerprint, now=now)
    fieldnames = list(ExportRow.__annotations__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for annotation in annotations:
            writer.writerow(_format_row(annotation))
    return output_path


def annotation_summary(
    annotations: list[TrialAnnotation],
) -> dict[str, object]:
    """Count annotations across three orthogonal dimensions.

    Returns ``total`` plus three sub-dicts (``by_flag``,
    ``by_adjudication``, ``by_disagreement_category``). All keys are
    pre-populated with zero counts so consumer code can index without
    a missing-key check, regardless of input distribution.
    """
    by_flag: dict[Flag, int] = {flag: 0 for flag in RATIONALE_ANALYSIS_FLAG_KEYS}
    by_adjudication: dict[AnnotatorAdjudication, int] = {
        "probe": 0,
        "auditor": 0,
        "inconclusive": 0,
    }
    by_category: dict[str, int] = {
        "stable_agreement": 0,
        "stable_disagreement": 0,
        "shifted_against_probe": 0,
        "shifted_to_probe": 0,
    }
    for ann in annotations:
        by_flag[ann.flag] += 1
        by_adjudication[ann.annotator_adjudication] += 1
        by_category[ann.disagreement_category] += 1
    return {
        "total": len(annotations),
        "by_flag": by_flag,
        "by_adjudication": by_adjudication,
        "by_disagreement_category": by_category,
    }
