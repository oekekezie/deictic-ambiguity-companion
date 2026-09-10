"""Markdown rendering and annotation matching for the summary CLI.

The renderers consume already-computed analytics (``pool_compatibility``
output, ``pooled_per_condition_*`` rows, ``pooled_disagreement_category_
counts_by_condition`` rows, plus the matched-annotations dict from
:func:`match_annotations_to_pool`) and return markdown strings. The
renderer module never reads from disk except via
:func:`match_annotations_to_pool`'s annotation-DB guard.

Per-flag output loops the seven flags in ``RATIONALE_ANALYSIS_FLAG_KEYS``
— one alignment table, one annotations breakdown, and one drill-down
section per flag.

All cell formatting goes through :mod:`utils.formatting`'s canonical
formatters; ad-hoc f-string format specs are not allowed.
"""

import math
import re
from pathlib import Path

from utils.formatting import format_accuracy, format_percent
from utils.probe_audit.models import ProbeAuditResults
from utils.probe_audit.review_data import (
    PoolCompatibility,
    disagreement_category_descriptions,
    pooled_disagreement_category_counts_by_condition,
    pooled_per_condition_per_value_class_probe_correctness,
    pooled_per_condition_probe_correctness,
    provenance_entries,
)
from utils.probe_audit.scenario_context import condition_from_example_id
from utils.probe_audit.storage import ProbeAuditIndexRecord
from utils.probe_audit_annotation.models import (
    AnnotationQuery,
    AnnotatorAdjudication,
    Flag,
    TrialAnnotation,
)
from utils.probe_audit_annotation.storage import load_annotations
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS


_ADJUDICATIONS: tuple[AnnotatorAdjudication, ...] = (
    "probe",
    "auditor",
    "inconclusive",
)
_DISAGREEMENT_CATEGORIES_FOR_BREAKDOWN: tuple[str, ...] = (
    "stable_disagreement",
    "shifted_against_probe",
    "shifted_to_probe",
)
_CONDITION_DISPLAY: dict[str, str] = {
    "correct": "correct-draft",
    "transparent": "incorrect-transparent",
    "opaque": "incorrect-opaque",
}


_ATX_HEADING_RE = re.compile(r"(?m)^(#{1,6}) ")


def _demote_headings(markdown: str, heading_offset: int) -> str:
    """Add ``heading_offset`` levels to every ATX heading in a rendered block.

    A behavior-preserving seam for embedding a renderer's output beneath a
    higher-level section heading (for example, inside a notebook section).
    With the default ``heading_offset=0`` the text is returned unchanged, so
    the CLI output stays byte-identical. A positive offset prefixes that many
    ``#`` to each ``^#{1,6} `` heading line and touches nothing else — table
    rows start with ``|`` and list items with ``-``, so neither is matched.
    """
    if heading_offset <= 0:
        return markdown
    return _ATX_HEADING_RE.sub(
        lambda m: "#" * (len(m.group(1)) + heading_offset) + " ", markdown,
    )


def _flag_title(flag: str) -> str:
    """Human-readable title for a flag key.

    ``misattributed_grader_claim`` → ``Misattributed grader claim``.
    Used for every per-flag section heading so the seven flags read as
    prose rather than identifiers.
    """
    return flag.replace("_", " ").capitalize()


def _label_for(category: str) -> str:
    """Canonical short_label for a DisagreementCategory string."""
    for d in disagreement_category_descriptions():
        if d.category == category:
            return d.short_label
    raise KeyError(f"unknown disagreement category: {category!r}")


def _format_pct_or_dash(rate: float) -> str:
    """Wrap ``format_percent`` so NaN renders as the em-dash sentinel."""
    if math.isnan(rate):
        return format_percent(None)
    return format_percent(rate)


def _format_acc_or_dash(value: float) -> str:
    """Wrap ``format_accuracy`` so NaN renders as the em-dash sentinel."""
    if math.isnan(value):
        return format_accuracy(None)
    return format_accuracy(value)


def render_pool_table(
    records: tuple[ProbeAuditIndexRecord, ...], *, heading_offset: int = 0
) -> str:
    """One markdown row per audit in the pool.

    Session IDs are truncated to the first 12 characters so the table
    fits in a reasonable column width; the header explicitly says
    "first 12" so a reader knows the prefix is partial.
    """
    lines = [
        "## Pool",
        "",
        "| # | Session ID (first 12) | Run Started | Run Completed | K | Args Digest | Auditor Models |",
        "| - | --------------------- | ----------- | ------------- | - | ----------- | -------------- |",
    ]
    for idx, r in enumerate(records, start=1):
        models = (
            ", ".join(r.auditor_model_versions)
            if r.auditor_model_versions
            else "—"
        )
        lines.append(
            f"| {idx} | `{r.session_id[:12]}` | {r.sample_timestamp} | "
            f"{r.audit_completion_timestamp or '—'} | {r.k} | "
            f"`{r.args_digest}` | {models} |"
        )
    return _demote_headings("\n".join(lines), heading_offset)


def render_compatibility_section(compat: PoolCompatibility) -> str:
    """Pool compatibility status — passes here implies ``compat.errors == ()``."""
    lines = [
        "## Pool Compatibility",
        "",
        "Status: PASSED",
        "",
        "Shared identity fields:",
        "",
        "- skill_fingerprint",
        "- probe_snapshot_sha256",
        "- rationale_source path + SHA-256",
        "- stimulus_source path + SHA-256",
    ]
    if compat.warnings:
        lines.extend(["", "Warnings:", ""])
        for w in compat.warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


def render_alignment_table(
    *,
    pcc_rows: list[dict],
    cat_rows: list[dict],
    flag_label: str,
    suppress_cs: bool = False,
    heading_offset: int = 0,
) -> str:
    """Per-condition alignment table for one flag.

    Ten columns: condition, n_valid, aligned, misaligned, alignment rate, a
    one-sided lower confidence sequence bound, plus the four
    DisagreementCategory counts. The
    aligned/misaligned columns are derivable from the four category
    counts (Stable Agreement + Shifted Toward Probe = Aligned; Stable Disagreement +
    Shifted Against Probe = Misaligned), but pre-computing them keeps the table
    readable without making the agent reader re-derive the relationship.

    When ``suppress_cs`` is set (a flag the coverage-convergent draw floored),
    the pooled lower bound is rendered as the em-dash null sentinel: that
    pooled rate mixes the probe's TRUE/FALSE classes at an enriched ratio and
    is NOT a pool rate, so it must not be read as one. The counts and category
    columns remain (the error-finding view); the certifiable rate for these
    flags is the per-value-class table that follows.
    """
    pcc_by_cond = {r["condition"]: r for r in pcc_rows}
    cat_by_cond = {r["condition"]: r for r in cat_rows}
    header = f"### {flag_label}"
    if suppress_cs:
        header += (
            " — rare/floored flag: pooled bound suppressed (enriched, not a "
            "pool rate); see per-value-class table below"
        )
    lines = [
        header,
        "",
        "| Condition | N Valid | Aligned | Misaligned | Alignment Rate (%) | "
        "Confidence Sequence Lower Bound | Stable Agreement | Shifted Toward Probe | "
        "Stable Disagreement | Shifted Against Probe |",
        "| --------- | ------- | ------- | ---------- | ------------------ | "
        "-------- | ---------------- | -------------------- | "
        "------------------- | --------------------- |",
    ]
    for cond in ("correct", "transparent", "opaque"):
        pcc = pcc_by_cond[cond]
        cat = cat_by_cond[cond]
        cs_cell = _format_acc_or_dash(
            float("nan") if suppress_cs else pcc["cs_lower"]
        )
        lines.append(
            f"| {_CONDITION_DISPLAY[cond]} | {pcc['n']} | "
            f"{pcc['probe_correct']} | {pcc['probe_incorrect']} | "
            f"{_format_pct_or_dash(pcc['probe_correctness_rate'])} | "
            f"{cs_cell} | "
            f"{cat['stable_agreement']} | {cat['shifted_to_probe']} | "
            f"{cat['stable_disagreement']} | {cat['shifted_against_probe']} |"
        )
    return _demote_headings("\n".join(lines), heading_offset)


_PROBE_VALUE_DISPLAY = {True: "Probe TRUE", False: "Probe FALSE"}


def render_value_class_alignment_table(
    *,
    pvc_rows: list[dict],
    flag_label: str,
    heading_offset: int = 0,
) -> str:
    """Per-(condition × probe-classification-value) alignment table for one flag.

    The honest estimand for a rare/floored flag: alignment conditioned on the
    probe's own classification value de-pools the base-rate inflation, and the
    within-class draw is uniform-WOR, so the lower bound is valid (and exact at
    campaign exhaustion, when the small positive classes are censused). Six
    rows — three conditions × {Probe TRUE, Probe FALSE}.
    """
    pvc_by_key = {(r["condition"], r["probe_value"]): r for r in pvc_rows}
    lines = [
        f"#### {flag_label} — per probe-classification value",
        "",
        "| Condition | Probe Classification | N Valid | Aligned | Misaligned | "
        "Alignment Rate (%) | Confidence Sequence Lower Bound |",
        "| --------- | -------------------- | ------- | ------- | ---------- | "
        "------------------ | ------------------------------- |",
    ]
    for cond in ("correct", "transparent", "opaque"):
        for probe_value in (True, False):
            row = pvc_by_key[(cond, probe_value)]
            lines.append(
                f"| {_CONDITION_DISPLAY[cond]} | "
                f"{_PROBE_VALUE_DISPLAY[probe_value]} | {row['n']} | "
                f"{row['probe_correct']} | {row['probe_incorrect']} | "
                f"{_format_pct_or_dash(row['probe_correctness_rate'])} | "
                f"{_format_acc_or_dash(row['cs_lower'])} |"
            )
    return _demote_headings("\n".join(lines), heading_offset)


def _build_pool_disagreement_keys(
    pool_results: tuple[ProbeAuditResults, ...],
    flag: Flag,
) -> frozenset[tuple[str, str, int, int]]:
    """Trial keys whose ``flag`` disagreement category is non-stable_agreement."""
    keys: set[tuple[str, str, int, int]] = set()
    for r in pool_results:
        for record in r.records:
            if (
                getattr(record, f"disagreement_category_{flag}")
                != "stable_agreement"
            ):
                keys.add(record.trial_key)
    return frozenset(keys)


def _empty_breakdown() -> dict[str, dict[str, int]]:
    """Pre-populated zero counts so the renderer never indexes into a missing key."""
    return {
        category: {adj: 0 for adj in _ADJUDICATIONS}
        for category in _DISAGREEMENT_CATEGORIES_FOR_BREAKDOWN
    }


def _per_category_breakdown(
    annotations: list[TrialAnnotation],
) -> dict[str, dict[str, int]]:
    """Count adjudications per disagreement category for one flag."""
    breakdown = _empty_breakdown()
    for a in annotations:
        if a.disagreement_category in breakdown:
            breakdown[a.disagreement_category][a.annotator_adjudication] += 1
    return breakdown


def match_annotations_to_pool(
    *,
    pool_results: tuple[ProbeAuditResults, ...],
    skill_fingerprint: str,
    annotation_base_dir: Path,
    current_criterion_sha256: str,
) -> dict:
    """Match annotations to the pool's disagreement set under the current criterion.

    Returns a dict consumed by :func:`render_annotations_section`. The
    ``annotation_db_present`` sentinel lets the renderer surface a
    "database not found" notice when the file is absent rather than
    silently triggering schema creation via the storage layer's
    self-init connection.

    Filters annotations by ``skill_fingerprint`` at the SQL level
    (``AnnotationQuery.skill_fingerprint``) but partitions by
    ``criterion_sha256`` Python-side so the dropped-epoch count can be
    surfaced. Then intersects the current-epoch annotations with the
    pool's per-flag disagreement set, one set per flag in
    ``RATIONALE_ANALYSIS_FLAG_KEYS``.
    """
    db_path = annotation_base_dir / "probe_audit_annotations.db"
    pool_disagreements: dict[str, frozenset[tuple[str, str, int, int]]] = {
        flag: _build_pool_disagreement_keys(pool_results, flag=flag)
        for flag in RATIONALE_ANALYSIS_FLAG_KEYS
    }
    pool_disagreement_counts = {
        flag: len(keys) for flag, keys in pool_disagreements.items()
    }
    if not db_path.is_file():
        return {
            "annotation_db_present": False,
            "annotation_db_path": db_path,
            "pool_disagreements": pool_disagreement_counts,
            "current_epoch_total": 0,
            "dropped_epoch_total": 0,
            "dropped_epoch_distinct_shas": (),
            "matched": {flag: () for flag in RATIONALE_ANALYSIS_FLAG_KEYS},
            "by_flag_category": {
                flag: _empty_breakdown()
                for flag in RATIONALE_ANALYSIS_FLAG_KEYS
            },
        }
    all_annotations = load_annotations(
        AnnotationQuery(skill_fingerprint=skill_fingerprint),
        base_dir=annotation_base_dir,
    )
    current = [
        a
        for a in all_annotations
        if a.criterion_sha256 == current_criterion_sha256
    ]
    dropped = [
        a
        for a in all_annotations
        if a.criterion_sha256 != current_criterion_sha256
    ]
    matched: dict[str, tuple[TrialAnnotation, ...]] = {
        flag: tuple(
            a
            for a in current
            if a.flag == flag and a.trial_key in pool_disagreements[flag]
        )
        for flag in RATIONALE_ANALYSIS_FLAG_KEYS
    }
    return {
        "annotation_db_present": True,
        "annotation_db_path": db_path,
        "pool_disagreements": pool_disagreement_counts,
        "current_epoch_total": len(current),
        "dropped_epoch_total": len(dropped),
        "dropped_epoch_distinct_shas": tuple(
            sorted({a.criterion_sha256[:12] for a in dropped})
        ),
        "matched": matched,
        "by_flag_category": {
            flag: _per_category_breakdown(list(matched[flag]))
            for flag in RATIONALE_ANALYSIS_FLAG_KEYS
        },
    }


def _render_per_flag_breakdown(
    *,
    flag_label: str,
    pool_disagreements: int,
    matched: tuple[TrialAnnotation, ...],
    breakdown: dict[str, dict[str, int]],
) -> list[str]:
    """One ``### Flag Adjudications by Disagreement Category`` block."""
    lines = [
        "",
        f"### {flag_label} — Adjudications by Disagreement Category",
        "",
        f"- Pool disagreements ({flag_label.lower()}): {pool_disagreements}",
        f"- Annotated: {len(matched)}; un-annotated: "
        f"{max(pool_disagreements - len(matched), 0)}",
    ]
    for category in _DISAGREEMENT_CATEGORIES_FOR_BREAKDOWN:
        cat_label = _label_for(category)
        cat_counts = breakdown[category]
        annotated = sum(cat_counts.values())
        # Per-category trial count is the count of matched annotations
        # in this category (since each matched annotation came from a
        # disagreement of that category).
        lines.extend(
            [
                "",
                f"#### {cat_label} ({category})",
                f"- Annotated: {annotated}",
                f"  - probe = {cat_counts['probe']}",
                f"  - auditor = {cat_counts['auditor']}",
                f"  - inconclusive = {cat_counts['inconclusive']}",
            ]
        )
    return lines


def render_annotations_section(
    matched: dict,
    *,
    current_criterion_sha256: str,
    heading_offset: int = 0,
) -> str:
    """Annotations section. Emits the DB-not-found notice when applicable."""
    lines = ["## Annotations under Current Criterion", ""]
    if not matched["annotation_db_present"]:
        lines.append(
            f"Annotation database not found at "
            f"`{matched['annotation_db_path']}` — section omitted."
        )
        return _demote_headings("\n".join(lines), heading_offset)
    lines.append(
        f"Current criterion SHA-256: `{current_criterion_sha256[:12]}`"
    )
    lines.append(
        f"Annotations matching pool's skill fingerprint under current "
        f"criterion: {matched['current_epoch_total']}"
    )
    if matched["dropped_epoch_total"] > 0:
        joined = ", ".join(matched["dropped_epoch_distinct_shas"])
        lines.append(
            f"Annotations dropped (different criterion epoch): "
            f"{matched['dropped_epoch_total']} (epochs: {joined})"
        )
    pool_dis: dict[str, int] = matched["pool_disagreements"]
    summary = ", ".join(
        f"{_flag_title(flag)} = {pool_dis[flag]}"
        for flag in RATIONALE_ANALYSIS_FLAG_KEYS
    )
    lines.append(f"Pool disagreements: {summary}")
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        lines.extend(
            _render_per_flag_breakdown(
                flag_label=_flag_title(flag),
                pool_disagreements=pool_dis[flag],
                matched=matched["matched"][flag],
                breakdown=matched["by_flag_category"][flag],
            )
        )
    return _demote_headings("\n".join(lines), heading_offset)


def _drill_down_lines_for_flag(
    *,
    records: tuple[ProbeAuditIndexRecord, ...],
    results: tuple[ProbeAuditResults, ...],
    flag: Flag,
) -> list[str]:
    """Per-flag drill-down: one ``#### Category`` block listing trial keys."""
    show_session = len(results) > 1
    session_by_audit_index = [r.session_id for r in records]
    grouped: dict[str, list[tuple[str, tuple[str, str, int, int]]]] = {
        c: [] for c in _DISAGREEMENT_CATEGORIES_FOR_BREAKDOWN
    }
    for audit_idx, r in enumerate(results):
        session = session_by_audit_index[audit_idx]
        for record in r.records:
            category = getattr(record, f"disagreement_category_{flag}")
            if category in grouped:
                grouped[category].append((session, record.trial_key))
    lines = [f"### {_flag_title(flag)}"]
    for category in _DISAGREEMENT_CATEGORIES_FOR_BREAKDOWN:
        cat_label = _label_for(category)
        items = sorted(
            grouped[category],
            key=lambda pair: (
                condition_from_example_id(pair[1][1]),
                pair[1][1],
                pair[1][2],
                pair[1][3],
            ),
        )
        lines.extend(["", f"#### {cat_label} ({len(items)} trials)"])
        if not items:
            lines.append("- (none)")
            continue
        for session, trial_key in items:
            cfg, ex, batch, trial = trial_key
            base = f"- `{cfg}|{ex}|{batch}|{trial}`"
            if show_session:
                base = f"{base}  [session {session[:12]}]"
            lines.append(base)
    return lines


def render_drill_down_appendix(
    *,
    records: tuple[ProbeAuditIndexRecord, ...],
    results: tuple[ProbeAuditResults, ...],
) -> str:
    """Appendix listing every disagreement trial key per ``(flag, category)``.

    Sorted deterministically by ``(condition, example_id, batch_index,
    trial)`` so re-runs produce identical output. The session_id is
    appended only when the pool has more than one audit; for a
    single-audit pool every trial belongs to the same session.
    """
    lines = ["## Per-Trial Drill-Down Pointers (Appendix)", ""]
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        lines.extend(
            _drill_down_lines_for_flag(
                records=records, results=results, flag=flag
            )
        )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def render_implications_and_limits(
    pool_results: tuple[ProbeAuditResults, ...], *, heading_offset: int = 0
) -> str:
    """Static implications/limits prose qualified with per-condition counts.

    The ``does NOT support`` bullet about absolute probe correctness is
    qualified per flag and per condition (per ``paper/CLAUDE.md``: never
    pool an alignment rate across conditions). Replication evidence is
    read off ``provenance.replication_evidence`` so the prose stays
    honest about K=1 limitations without re-deriving from K.
    """
    pcc_by_flag: dict[str, dict[str, dict]] = {
        flag: {
            r["condition"]: r
            for r in pooled_per_condition_probe_correctness(
                list(pool_results), flag=flag
            )
        }
        for flag in RATIONALE_ANALYSIS_FLAG_KEYS
    }

    def _frac(row: dict) -> str:
        return f"{row['probe_incorrect']}/{row['n']}"

    any_singletons = any(
        r.provenance.replication_evidence == "none" for r in pool_results
    )
    lines = [
        "## Implications and Limits",
        "",
        "**The data supports:**",
        "",
        "- The probe applies the criterion as written: per-discriminator-cell "
        "alignment rates above. Each rate is paired with its condition; "
        "rates are not pooled across conditions.",
        "- Stable Disagreement counts (when annotated and adjudicated "
        "\"auditor\"): third-reader confirmation of probe error.",
        "- Shifted Toward Probe counts (when annotated and adjudicated "
        "\"auditor\"): third-reader-confirmed anchoring exposure.",
        "",
        "**The data does NOT support:**",
        "",
        "- Absolute probe correctness — under the same criterion text the "
        "auditor disagreed with the probe at some phase on the following "
        "fractions of trials (per flag, per condition):",
    ]
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        by_cond = pcc_by_flag[flag]
        lines.append(
            f"  - {_flag_title(flag)}: "
            f"{_frac(by_cond['correct'])} correct-draft, "
            f"{_frac(by_cond['transparent'])} incorrect-transparent, "
            f"{_frac(by_cond['opaque'])} incorrect-opaque"
        )
    lines.append(
        "- Coverage of matching-verdict trials — the audit is keyed on "
        "rationale analysis mismatch trials only; no probe-correctness "
        "signal exists on trials whose verdict matched ground truth."
    )
    lines.append(
        "- Rare-positive flags — when the probe rarely classifies a flag "
        "positive, probe-auditor agreement is dominated by the shared negative "
        "class, so the pooled one-sided lower bound is base-rate-driven and does "
        "not establish reliability on the probe's positive calls. For "
        "coverage-convergent pools this pooled bound is suppressed for the "
        "floored (rare) flags and a per-value-class table is reported instead, "
        "which de-pools the effect by conditioning on the probe's classification "
        "value (alignment among the probe's positive calls is reported "
        "separately, and is exact once those small positive classes are "
        "censused at campaign exhaustion)."
    )
    if any_singletons:
        lines.append(
            "- Within-trial replication for K=1 audits — replication_evidence "
            "is \"none\" per the provenance, so per-trial inter-rater "
            "reliability is not measured. A single rep's internally "
            "consistent but wrong judgment is therefore unguardable (the "
            "record-time checks catch only contradictory verdicts); this "
            "residual is mitigated by replication (K≥3), not validation."
        )
    lines.append(
        "- Generalization beyond the criterion text and probe model in this pool."
    )
    return _demote_headings("\n".join(lines), heading_offset)


def render_summary(
    *,
    records: tuple[ProbeAuditIndexRecord, ...],
    results: tuple[ProbeAuditResults, ...],
    compatibility: PoolCompatibility,
    matched_annotations: dict,
    current_criterion_sha256: str,
    generated_at: str,
    probe_flags: dict[tuple[str, str, int, int], dict[str, bool | None]] | None = None,
    rare_flags: tuple[str, ...] = (),
) -> str:
    """Top-level renderer composing every section into the final markdown.

    ``rare_flags`` are the flags a coverage-convergent draw floored (empty for
    a stratified pool, which has no floor and no enrichment). For those flags
    the pooled per-condition bound is suppressed and a per-value-class table is
    rendered instead — ``probe_flags`` (the :func:`probe_flags_by_trial` map
    over the probe payload) supplies the probe's per-trial classification used
    to split by value-class. A stratified pool passes ``rare_flags=()`` and is
    rendered exactly as before.
    """
    fp = results[0].provenance.skill_fingerprint
    fp12 = fp[:12]
    n_audits = len(records)
    total_trials = sum(len(r.records) for r in results)
    sections: list[str] = [
        f"# Probe Audit Summary — `{fp12}` (pool of {n_audits})",
        "",
        f"Generated: {generated_at}",
        f"Skill fingerprint: `{fp}`",
        f"Pool size: {n_audits} audits, {total_trials} total trials",
        "",
        render_pool_table(records),
        "",
        render_compatibility_section(compatibility),
        "",
        "## Per-Condition × Per-Flag Probe-Auditor Alignment",
        "",
    ]
    # One alignment table per flag — never pooled across conditions. A
    # coverage-convergent floored flag suppresses its pooled bound (enriched,
    # not a pool rate) and appends a per-value-class table that de-pools the
    # base-rate inflation; a stratified pool (rare_flags=()) is unaffected.
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        pcc = pooled_per_condition_probe_correctness(list(results), flag=flag)
        cat = pooled_disagreement_category_counts_by_condition(
            list(results), flag=flag
        )
        enriched = flag in rare_flags and probe_flags is not None
        sections.append(
            render_alignment_table(
                pcc_rows=pcc,
                cat_rows=cat,
                flag_label=_flag_title(flag),
                suppress_cs=enriched,
            )
        )
        sections.append("")
        if enriched:
            pvc = pooled_per_condition_per_value_class_probe_correctness(
                list(results), flag=flag, probe_flags=probe_flags
            )
            sections.append(
                render_value_class_alignment_table(
                    pvc_rows=pvc, flag_label=_flag_title(flag)
                )
            )
            sections.append("")
    sections.extend(
        [
            render_annotations_section(
                matched_annotations,
                current_criterion_sha256=current_criterion_sha256,
            ),
            "",
            render_implications_and_limits(results),
            "",
            render_drill_down_appendix(records=records, results=results),
            "",
            "## Provenance (single-audit pools)",
            "",
        ]
    )
    if len(results) == 1:
        for label, value in provenance_entries(results[0]):
            sections.append(f"- **{label}:** {value}")
    else:
        sections.append(
            "(omitted for multi-audit pools — see per-audit `snapshot.json` "
            "files for full provenance)"
        )
    return "\n".join(sections) + "\n"
