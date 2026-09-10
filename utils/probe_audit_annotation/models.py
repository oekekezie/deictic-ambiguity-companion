"""Pydantic schemas for the probe-audit annotation tool.

The schemas here are the data contract for stored annotations and the
filter spec for loading them back. They are deliberately minimal — only
fields that carry annotation-time provenance or annotator-supplied
content live here. Aggregation rules (e.g., majority-vote of auditor
``*_corrected_classification`` across reps) are analysis-layer concerns
that operate on the audit snapshots referenced by
``pooled_audit_session_ids``; baking such rules into the schema would
make every existing row stale the day the rule changes.

``annotator_adjudication`` carries the side-taking adjudication
directly — ``adjudication == "probe"`` means "annotator sided with the
probe", ``adjudication == "auditor"`` means "annotator sided with the
auditor". Downstream code compares the value against the Literal
options directly; no coercion helper exists.
"""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.probe_audit.models import DisagreementCategory

TOOL_VERSION = "0.1.0"

# One member per flag in ``RATIONALE_ANALYSIS_FLAG_KEYS``. A unit test
# pins this Literal to that tuple so the annotation tool's flag surface
# cannot drift from the probe's seven-flag schema.
Flag = Literal[
    "misattributed_scenario_current_value",
    "misattributed_scenario_historical_value",
    "misattributed_scenario_proposed_value",
    "misattributed_assistant_fallback_value",
    "misattributed_grader_claim",
    "articulated_operational_interpretation",
    "operational_interpretation_governed_judgment",
]
AnnotatorAdjudication = Literal["probe", "auditor", "inconclusive"]


class TrialAnnotation(BaseModel):
    """One human annotation of a probe-auditor disagreement.

    ``disagreement_category`` is annotator-time provenance — captured
    against the audit pool the annotator was reviewing when they saved.
    It is not a recomputable derived value. Aggregation rules belong in
    the analysis layer, where they can evolve without orphaning stored
    rows.

    Schema is K-agnostic: identical for K=1 audits (single rep is the
    classification) and K>=3 audits (any aggregation rule the analysis
    layer wants is computable from ``pooled_audit_session_ids`` plus the
    on-disk audit snapshots).

    ``probe_classification`` is the probe's own True/False output for
    ``flag``, loaded once during queue construction in ``loading.py``
    from the audit-pinned probe snapshot. The schema is ``bool`` by
    design; loading raises ``ProbeValueMissingError`` when the probe's
    per-flag value is ``None``, so the schema never has to encode that
    edge case.

    ``annotator_adjudication`` is the annotator's adjudication of which
    side's reasoning is more defensible per the criterion text. The
    annotator reads the rationale, the criterion excerpt, the probe's
    reasoning, and the auditor's reasoning, then judges the relative
    quality of two existing applications of the criterion. The annotator
    does not produce a third application themselves. Each
    ``AnnotatorAdjudication`` value names the side the annotator chose:

    - ``"probe"`` — annotator finds the probe's reasoning more
      defensible than the auditor's. Which auditor view counts as "the
      auditor" varies by ``disagreement_category``: under
      ``stable_disagreement``, both pre-view and post-view (consistent);
      under ``shifted_against_probe``, the post-view shift; under
      ``shifted_to_probe``, the auditor's independent pre-view.
    - ``"auditor"`` — annotator finds the auditor's reasoning more
      defensible than the probe's. Same per-category referent for which
      auditor view "the auditor" names.
    - ``"inconclusive"`` — annotator cannot decisively pick a side:
      either both readings are defensible per the criterion, or neither
      is sufficiently defensible.

    SM5 narrative computes probe-reliability metrics — for example,
    "M of N disagreements where the annotator found the probe's
    reasoning more defensible" (``annotator_adjudication == "probe"``)
    — directly from the value distribution. The categorical context
    (which auditor view aligned with the probe at which phase) is
    preserved per-row by ``disagreement_category``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    annotation_id: str
    trial_key: tuple[str, str, int, int]
    flag: Flag
    pooled_audit_session_ids: tuple[str, ...]
    skill_fingerprint: str
    probe_classification: bool
    disagreement_category: DisagreementCategory
    annotator_id: str
    annotator_adjudication: AnnotatorAdjudication
    annotator_reasoning: Annotated[str, Field(min_length=1)]
    criterion_sha256: str
    tool_version: str
    annotated_at: str

    @field_validator("pooled_audit_session_ids")
    @classmethod
    def _ids_sorted_and_nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty tuples and non-canonical (unsorted) orderings.

        The sort guarantee makes any two annotations referencing the
        same pool produce byte-identical row payloads, which keeps the
        UNIQUE-constraint key stable even though the pool ids are not
        themselves part of that key.
        """
        if not v:
            raise ValueError("pooled_audit_session_ids must be non-empty")
        if list(v) != sorted(v):
            raise ValueError(
                "pooled_audit_session_ids must be sorted (canonical form)"
            )
        return v


class AnnotationQuery(BaseModel):
    """All-optional filter spec for ``load_annotations``.

    A field of ``None`` means "do not filter on this dimension." Mixing
    multiple filters is conjunctive: each filter narrows the result set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    annotator_id: str | None = None
    flag: Flag | None = None
    criterion_sha256: str | None = None
    skill_fingerprint: str | None = None
    trial_key: tuple[str, str, int, int] | None = None
    pooled_audit_session_ids: tuple[str, ...] | None = None

    @field_validator("pooled_audit_session_ids")
    @classmethod
    def _ids_sorted_when_set(
        cls, v: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """When a query specifies a pool, require the canonical sort.

        Empty tuples remain meaningful here — a caller could plausibly
        ask for "annotations with no pool ids." We let that through and
        only enforce the sort.
        """
        if v is None:
            return v
        if list(v) != sorted(v):
            raise ValueError(
                "pooled_audit_session_ids must be sorted (canonical form)"
            )
        return v


# Reader-facing CSV row schema. Column order is part of the contract.
# Keys follow the CLAUDE.md full-spelled-out convention (parentheses,
# hyphens, spaces). The functional ``TypedDict`` syntax is used so the
# keys themselves carry the display labels — ``ExportRow.__annotations__.keys()``
# yields the exact CSV header row in the correct order.
ExportRow = TypedDict(
    "ExportRow",
    {
        "Annotation ID": str,
        "Trial Key (Config Key)": str,
        "Trial Key (Example ID)": str,
        "Trial Key (Batch Index)": int,
        "Trial Key (Trial Number)": int,
        "Flag": str,
        "Condition": str,
        "Disagreement Category": str,
        "Probe Classification": bool,
        "Annotator ID": str,
        "Annotator Adjudication": str,
        "Annotator Reasoning": str,
        "Pooled Audit Session IDs": str,
        "Skill Fingerprint": str,
        "Criterion SHA-256": str,
        "Tool Version": str,
        "Annotated At (UTC)": str,
    },
)


