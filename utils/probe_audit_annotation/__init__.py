"""Probe-audit-disagreement annotation tool — schemas, storage, loading, export.

The stable public surface re-exported here is what the marimo annotation
notebook (``marimo_notebooks/probe_audit_annotation.py``) and any
downstream consumer of stored annotations should import. Everything else
in this package is an implementation detail.
"""

from utils.probe_audit_annotation.criterion import (
    compute_criterion_sha256,
    flag_excerpt,
    load_criterion_text,
    resolve_criterion_path,
)
from utils.probe_audit_annotation.export import (
    annotation_summary,
    export_annotations_csv,
)
from utils.probe_audit_annotation.loading import (
    DisagreementRow,
    ProbeValueMissingError,
    disagreement_queue_rows,
    disagreement_records,
)
from utils.probe_audit_annotation.models import (
    TOOL_VERSION,
    AnnotationQuery,
    AnnotatorAdjudication,
    ExportRow,
    Flag,
    TrialAnnotation,
)
from utils.probe_audit_annotation.storage import (
    annotation_exists,
    init_schema,
    load_annotation_for,
    load_annotations,
    resolve_base_dir,
    save_annotation,
)

__all__ = [
    "TOOL_VERSION",
    "AnnotationQuery",
    "AnnotatorAdjudication",
    "DisagreementRow",
    "ExportRow",
    "Flag",
    "ProbeValueMissingError",
    "TrialAnnotation",
    "annotation_exists",
    "annotation_summary",
    "compute_criterion_sha256",
    "disagreement_queue_rows",
    "disagreement_records",
    "export_annotations_csv",
    "flag_excerpt",
    "init_schema",
    "load_annotation_for",
    "load_annotations",
    "load_criterion_text",
    "resolve_base_dir",
    "resolve_criterion_path",
    "save_annotation",
]
