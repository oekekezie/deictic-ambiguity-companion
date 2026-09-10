"""Per-trial drill-down hydration + journal export rendering.

This module is the I/O-touching sibling of ``review_data``. It reads the
stage-snapshot rationale payload and the probe-output snapshot payload
(both already loaded into dicts by the notebook's data-load cell) and
assembles one ``TrialDetail`` per selected trial. It also renders the
compact markdown journal-export template the researcher pastes into
their Google Doc experiment journal.

Kept separate from ``review_data`` so the pure-function invariant there
(no I/O, no external schemas) remains intact. The two external payloads
are pre-loaded by the notebook once per session rather than re-opened
per trial selection — a 20 MB stage snapshot parsed on every click
would kill the UX.

Per-flag data is keyed by the flag names in ``RATIONALE_ANALYSIS_FLAG_KEYS``
throughout: ``TrialDetail`` and ``RepAgreementCell`` carry flag-keyed
dicts and the journal renderer loops the seven flags rather than naming
each one.
"""

from dataclasses import dataclass
from typing import Literal

from utils.probe_audit.models import (
    AuditTrialRecord,
    DisagreementCategory,
    ProbeAuditResults,
    RepJudgment,
    RepReading,
)
from utils.probe_audit.review_data import disagreement_category_descriptions
from utils.probe_audit.scenario_context import (
    build_scenario_context,
    condition_from_example_id,
)
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS

_DISAGREEMENT_CATEGORY_LABELS: dict[DisagreementCategory, str] = {
    d.category: d.short_label for d in disagreement_category_descriptions()
}


def flag_label(flag: str) -> str:
    """Human-readable label for a flag key.

    ``misattributed_grader_claim`` → ``Misattributed grader claim``;
    ``articulated_operational_interpretation`` → ``Articulated
    operational interpretation``. Used for journal section headings and
    table rows so the seven flags read as prose rather than identifiers.
    """
    return flag.replace("_", " ").capitalize()


@dataclass(frozen=True)
class RepAgreementCell:
    """One row of the per-rep probe-vs-auditor agreement matrix.

    ``probe_aggregated`` maps each flag in ``RATIONALE_ANALYSIS_FLAG_KEYS``
    to the probe's single (aggregated) classification for this trial;
    ``auditor`` maps each flag to the individual rep's pre-view reading;
    ``agrees`` maps each flag to the strict comparison, or ``None`` when
    the probe's aggregated value is ``None`` (insufficient parsed reps).
    """

    rep_index: int
    probe_aggregated: dict[str, bool | None]
    auditor: dict[str, bool]
    agrees: dict[str, bool | None]


@dataclass(frozen=True)
class TrialDetail:
    """Everything the per-trial drill-down cell needs to render.

    The drill-down shows five panels: trial header (derived from trial_key
    + rationale payload), rationale card (from rationale_text), probe
    output card (aggregated + per-rep breakdown), auditor rep tabs
    (from audit_record), and the agreement matrix (per_rep_agreement).

    ``probe_aggregated`` maps each flag to the probe's aggregated
    classification; ``probe_rep_classifications`` and
    ``probe_rep_analyses`` map each flag to its per-rep tuple — the
    ``classification`` / ``analysis`` sub-fields of the probe's
    ``{"classification", "analysis"}`` schema. ``probe_rep_reasonings``
    carries the probe's free-form per-rep reasoning trace, rendered
    separately from the flag-specific analyses.
    """

    trial_key: tuple[str, str, int, int]
    condition: str
    meta_evaluator_predicted_score: str | None
    meta_evaluator_ground_truth_score: str | None
    meta_evaluator_matches_ground_truth: bool | None
    rationale_text: str
    probe_aggregated: dict[str, bool | None]
    probe_rep_classifications: dict[str, tuple[bool, ...]]
    probe_rep_analyses: dict[str, tuple[str, ...]]
    probe_rep_reasonings: tuple[str, ...]
    audit_record: AuditTrialRecord
    per_rep_agreement: tuple[RepAgreementCell, ...]
    # Scenario context — same seven fields the auditor sub-agent
    # received on its bundle, surfaced here so the journal entry can
    # render the same referential anchors a human reviewer needs.
    current_value: str
    proposed_value: str
    historical_value: str
    domain_noun: str
    draft_fallback_value: str
    grader_feedback: str
    ground_truth_description: str


def _audit_record_for(
    results: ProbeAuditResults, trial_key: tuple[str, str, int, int]
) -> AuditTrialRecord:
    for record in results.records:
        if tuple(record.trial_key) == trial_key:
            return record
    raise KeyError(f"trial_key {trial_key!r} not in results.records")


def _probe_record_for(
    probe_payload: dict, trial_key: tuple[str, str, int, int]
) -> dict:
    for rec in probe_payload.get("trial_records", []):
        key = (
            rec["config_key"],
            rec["example_id"],
            int(rec["batch_index"]),
            int(rec["trial"]),
        )
        if key == trial_key:
            return rec
    raise KeyError(
        f"trial_key {trial_key!r} not in probe_payload['trial_records']"
    )


def _rationale_outcome_for(
    rationale_payload: dict, trial_key: tuple[str, str, int, int]
) -> dict:
    for outcome in rationale_payload.get("trial_outcomes", []):
        key = (
            outcome["config_key"],
            outcome["example_id"],
            int(outcome["batch_index"]),
            int(outcome["trial"]),
        )
        if key == trial_key:
            return outcome
    raise KeyError(
        f"trial_key {trial_key!r} not in rationale_payload['trial_outcomes']"
    )


def _agreement(probe: bool | None, auditor: bool) -> bool | None:
    if probe is None:
        return None
    return probe == auditor


def build_trial_detail(
    results: ProbeAuditResults,
    trial_key: tuple[str, str, int, int],
    rationale_payload: dict,
    probe_payload: dict,
    stimuli: dict[str, dict[str, str]],
) -> TrialDetail:
    """Hydrate one trial's drill-down view from the four input sources.

    Raises ``KeyError`` when ``trial_key`` is absent from any of the inputs —
    drilling into a trial the audit didn't cover, or that the probe/rationale
    snapshots don't contain, or whose example_id is missing from ``stimuli``,
    is a data-integrity bug the caller should surface loudly rather than
    silently render an empty card.

    ``stimuli`` is the mapping produced by :func:`load_stimuli` — its
    ``user_content`` is parsed by ``extract_grader_feedback`` to populate
    the scenario context fields the journal renders for the reviewer.
    """
    record = _audit_record_for(results, trial_key)
    probe_rec = _probe_record_for(probe_payload, trial_key)
    rationale_outcome = _rationale_outcome_for(rationale_payload, trial_key)

    rationale_text = rationale_outcome["rationale"]
    predicted_score = rationale_outcome.get("predicted_score")
    ground_truth_score = rationale_outcome["ground_truth_score"]
    matches_gt: bool | None
    if predicted_score is None or ground_truth_score is None:
        matches_gt = None
    else:
        matches_gt = predicted_score == ground_truth_score

    probe_aggregated: dict[str, bool | None] = {
        flag: probe_rec.get(flag) for flag in RATIONALE_ANALYSIS_FLAG_KEYS
    }
    repetitions = tuple(probe_rec.get("repetitions", ()))
    # Per-rep flag entries are ``{"classification": bool, "analysis": str}``
    # objects — the probe's own schema. A naive ``bool(r[flag])`` would
    # return True for any non-empty dict, collapsing the per-rep
    # distinction. Read the sub-fields explicitly, per flag.
    probe_rep_classifications: dict[str, tuple[bool, ...]] = {}
    probe_rep_analyses: dict[str, tuple[str, ...]] = {}
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        probe_rep_classifications[flag] = tuple(
            bool(r[flag]["classification"]) for r in repetitions
        )
        probe_rep_analyses[flag] = tuple(
            str(r[flag]["analysis"]) for r in repetitions
        )
    probe_rep_reasonings = tuple(
        str(r.get("reasoning_text", "")) for r in repetitions
    )

    per_rep_agreement = tuple(
        RepAgreementCell(
            rep_index=reading.rep_index,
            probe_aggregated=probe_aggregated,
            auditor={
                flag: getattr(reading, flag)
                for flag in RATIONALE_ANALYSIS_FLAG_KEYS
            },
            agrees={
                flag: _agreement(probe_aggregated[flag], getattr(reading, flag))
                for flag in RATIONALE_ANALYSIS_FLAG_KEYS
            },
        )
        for reading in record.rep_readings
    )

    scenario, _condition = build_scenario_context(trial_key[1], stimuli)

    return TrialDetail(
        trial_key=trial_key,
        condition=condition_from_example_id(trial_key[1]),
        meta_evaluator_predicted_score=predicted_score,
        meta_evaluator_ground_truth_score=ground_truth_score,
        meta_evaluator_matches_ground_truth=matches_gt,
        rationale_text=rationale_text,
        probe_aggregated=probe_aggregated,
        probe_rep_classifications=probe_rep_classifications,
        probe_rep_analyses=probe_rep_analyses,
        probe_rep_reasonings=probe_rep_reasonings,
        audit_record=record,
        per_rep_agreement=per_rep_agreement,
        current_value=scenario.current_value,
        proposed_value=scenario.proposed_value,
        historical_value=scenario.historical_value,
        domain_noun=scenario.domain_noun,
        draft_fallback_value=scenario.draft_fallback_value,
        grader_feedback=scenario.grader_feedback,
        ground_truth_description=scenario.ground_truth_description,
    )


# ── journal-export rendering ────────────────────────────────────────────


def _probe_correct_symbol(probe_correct: bool | None) -> str:
    if probe_correct is True:
        return "✓"
    if probe_correct is False:
        return "✗"
    return "—"


def _bool_cell(value: bool | None) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return "—"


def blockquote(text: str) -> str:
    """Turn multi-paragraph prose into a markdown blockquote.

    Markdown blockquotes require ``>`` on every line; a bare ``> ...``
    only quotes the first line and subsequent lines render as normal
    paragraphs — a common foot-gun when pasting rationale text with
    multiple paragraphs. Prefix every line (including empty ones) so
    the whole body renders as one coherent quote in Google Docs.
    """
    lines = text.splitlines() or [""]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def yes_no(value: bool) -> str:
    """Human-readable Yes/No for boolean flag displays.

    Used in the journal + drill-down to spell out classifications
    instead of showing bare ``True`` / ``False`` — less cognitive load
    for researchers scanning per-flag bullets.
    """
    return "Yes" if value else "No"


def auditor_classification_display(
    probe_correct: bool,
    corrected: bool | None,
    probe_aggregated: bool | None,
) -> tuple[str, str]:
    """Derive ``(classification, alignment)`` display strings for a rep's flag.

    The auditor sees the probe's aggregated classification (built by
    ``build_probe_output_bundle``) and records two values per flag:
    ``probe_correct`` (did the auditor agree?) and
    ``corrected_classification`` (the auditor's own classification when
    they disagreed; ``None`` when they agreed since the probe's value
    is already authoritative).

    Derivation:
      * ``corrected`` specified → use it verbatim.
      * ``corrected=None`` + ``probe_correct=True`` → the auditor agreed
        with the probe, so the auditor's classification equals the probe's
        aggregated value. If that too is ``None`` (insufficient parsed
        reps below the majority threshold) the classification cannot be
        established; return an explicit sentinel rather than a misleading
        default.
      * ``corrected=None`` + ``probe_correct=False`` → invariant violation.
        The rep CLI (``_cmd_record_judgment``) rejects this combination at
        record time; reaching this branch means the audit log has been
        corrupted. Raise ``ValueError`` loudly rather than silently
        render a fallback.

    Alignment is driven by ``probe_correct`` directly — the CLI treats
    that flag as the authoritative agreement signal — so alignment is
    independent of the classification derivation branch above.
    """
    if corrected is not None:
        classification = yes_no(corrected)
    elif probe_correct:
        if probe_aggregated is None:
            classification = "cannot be established (probe classification indeterminate)"
        else:
            classification = yes_no(probe_aggregated)
    else:
        raise ValueError(
            "Invariant violation: a flag's corrected_classification is None "
            "but its probe_correct is False. The rep CLI "
            "(_cmd_record_judgment) rejects this combination at record "
            "time; reaching this branch indicates the audit log has been "
            "corrupted."
        )
    alignment = (
        "aligned (auditor agrees with the probe)"
        if probe_correct
        else "not aligned (auditor disagrees with the probe)"
    )
    return classification, alignment


def _auditor_rep_block(
    rep_index: int,
    reading: RepReading,
    judgment: RepJudgment,
    probe_aggregated: dict[str, bool | None],
) -> str:
    """Render one auditor rep's full block — pre-view reading + post-view judgment.

    Mirrors the UI ``_rep_card`` shape in the probe-audit review
    notebook: a bold rep header, a pre-view section (independent reading
    prose for every flag, before the rep saw the probe's output), and a
    post-view section (probe vs. auditor classification + alignment +
    judgment reasoning per flag). The journal uses ``blockquote`` (rather
    than the UI's ``probe-analysis-body`` div) because it pastes into
    Google Docs which renders markdown blockquotes natively.

    Probe-classification display sentinels follow the UI: when the
    probe's aggregated classification is ``None`` (insufficient parsed
    reps below ceil(K/2)), the bullet reads "cannot be established"
    rather than a misleading default.
    """
    lines: list[str] = [
        f"**Auditor Rep {rep_index}**",
        "",
        "**Pre-view reading** (independent, before probe output shown)",
        "",
    ]
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        lines.append(
            f"- {flag_label(flag)}: **{yes_no(getattr(reading, flag))}**"
        )
        lines.append("")
        lines.append(blockquote(getattr(reading, f"{flag}_reasoning")))
        lines.append("")
    lines.append("**Post-view judgment** (after probe output shown)")
    lines.append("")
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        probe_correct = getattr(judgment, f"{flag}_probe_correct")
        corrected = getattr(judgment, f"{flag}_corrected_classification")
        classification, alignment = auditor_classification_display(
            probe_correct=probe_correct,
            corrected=corrected,
            probe_aggregated=probe_aggregated[flag],
        )
        probe_display = (
            yes_no(probe_aggregated[flag])
            if probe_aggregated[flag] is not None
            else "cannot be established"
        )
        lines.append(f"**{flag_label(flag)}**")
        lines.append("")
        lines.append(f"- The probe's classification: **{probe_display}**")
        lines.append(f"- The auditor's classification: **{classification}**")
        lines.append(f"- Alignment: **{alignment}**")
        lines.append("")
        lines.append(blockquote(getattr(judgment, f"{flag}_reasoning")))
        lines.append("")
    return "\n".join(lines)


def _aggregate_bullet(
    flag_label: str,
    aggregated: bool | None,
    votes: tuple[int, int],
    k: int,
) -> str:
    """Render one per-flag bullet in the probe-auditor-alignment-aggregate block.

    Splits phrasing by status: ``True`` → "auditors judged the probe
    correct"; ``False`` → "auditors judged the probe not correct"; ``None``
    → "auditor aggregate is indeterminate". The indeterminate phrasing
    differs because semantically the audit's verdict — not the probe —
    is what becomes indeterminate when fewer than ceil(K/2) reps parse.

    Inline vote counts ``(votes F for / A against, K=K)`` make the margin
    visible to a journal reader without scanning elsewhere — this bullet
    is the only place in the journal where vote counts surface (the
    verdict-comparison table carries only the aggregate boolean and a
    ✓/✗/— symbol). ``F + A ≤ K`` because reps that fail to parse are
    counted in neither.

    Per ``paper/CLAUDE.md``'s rule reserving "incorrect" for stimulus
    conditions (incorrect-draft / incorrect-transparent / incorrect-opaque);
    this bullet uses "not correct" instead.
    """
    f, a = votes
    parens = f"(votes {f} for / {a} against, K={k})"
    if aggregated is True:
        body = "auditors judged the probe **correct**"
    elif aggregated is False:
        body = "auditors judged the probe **not correct**"
    else:
        body = "auditor aggregate is **indeterminate**"
    return f"- {flag_label}: {body} {parens}"


def _probe_rep_block_rep_major(
    rep_index: int,
    classifications: dict[str, bool],
    analyses: dict[str, str],
    reasoning: str,
    include_reasoning_text: bool,
) -> str:
    """Render one probe rep's full block (rep-major layout).

    Within-rep order — one analysis section per flag in
    ``RATIONALE_ANALYSIS_FLAG_KEYS``, then (optionally) the probe's
    free-form reasoning trace — mirrors the probe's own justification
    chain: classify each flag with its analysis, then explain. Each flag
    value surfaces as a ``Yes/No`` bullet UNDER its matching analysis
    section heading, keeping the flag visually adjacent to the analysis
    it refers to. Bodies route through ``blockquote`` so multi-paragraph
    prose renders as one coherent quote in Google Docs.

    ``include_reasoning_text`` gates the trailing ``**Reasoning**``
    subsection. The probe's ``reasoning_text`` field is the reasoning
    trace named in ``paper/CLAUDE.md``'s terminology table — the only
    one in the audit data
    model. When ``False``, only the per-flag analyses render; the
    reasoning trace is suppressed.
    """
    lines: list[str] = [f"**Probe Rep {rep_index}**", ""]
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        lines.append(f"**{flag_label(flag)} analysis**")
        lines.append("")
        lines.append(
            f"- {flag_label(flag)}: **{yes_no(classifications[flag])}**"
        )
        lines.append("")
        lines.append(blockquote(analyses[flag]))
        lines.append("")
    head = "\n".join(lines)
    if include_reasoning_text:
        return head + f"\n**Reasoning**\n\n{blockquote(reasoning)}\n"
    return head


def _probe_flag_sections_flag_major(
    detail: TrialDetail,
    selected_probe_reps: tuple[int, ...],
    include_reasoning_text: bool,
) -> str:
    """Render selected probe reps grouped by flag (flag-major layout).

    One section per flag in ``RATIONALE_ANALYSIS_FLAG_KEYS`` plus an
    optional reasoning section, each iterating the reps in
    ``sorted(selected_probe_reps)``. Each rep subheader is a plain
    ``**Rep N**`` (with the original 1-indexed number, preserved across
    non-contiguous subsets) and the per-rep classification surfaces as a
    ``Yes/No`` bullet below it. Designed for cross-rep diagnosis (e.g.,
    "why did probes over-classify articulation?") where a reader wants
    every rep's analysis for one flag side-by-side rather than
    interleaved with the other flags.

    Empty ``selected_probe_reps`` is handled at the call site (in
    ``render_journal_template``), which emits a placeholder and skips
    invoking this helper; this helper assumes a non-empty selection.

    ``include_reasoning_text`` gates the trailing ``#### Reasoning
    (across reps)`` section. The probe's ``reasoning_text`` field is
    the reasoning trace named in ``paper/CLAUDE.md``'s terminology
    table. When ``False``, only
    the per-flag sections render.
    """
    rep_nums = sorted(selected_probe_reps)
    lines: list[str] = []

    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        lines.append(f"#### {flag_label(flag)} (across reps)")
        lines.append("")
        for rep_num in rep_nums:
            i = rep_num - 1
            lines.append(f"**Rep {rep_num}**")
            lines.append("")
            lines.append(
                f"- {flag_label(flag)}: "
                f"**{yes_no(detail.probe_rep_classifications[flag][i])}**"
            )
            lines.append("")
            lines.append(blockquote(detail.probe_rep_analyses[flag][i]))
            lines.append("")

    if include_reasoning_text:
        lines.append("#### Reasoning (across reps)")
        lines.append("")
        for rep_num in rep_nums:
            i = rep_num - 1
            lines.append(f"**Rep {rep_num}**")
            lines.append("")
            lines.append(blockquote(detail.probe_rep_reasonings[i]))
            lines.append("")

    return "\n".join(lines)


def render_journal_template(
    detail: TrialDetail,
    results: ProbeAuditResults,
    audit_snapshot_path: str,
    selected_probe_reps: tuple[int, ...],
    selected_auditor_reps: tuple[int, ...],
    include_probe_reasoning_text: bool,
    layout: Literal["rep_major", "flag_major"],
) -> str:
    """Render the compact markdown entry the researcher pastes into the journal.

    The journal assembles a per-flag probe-auditor-alignment-aggregate
    block (correct / not correct / indeterminate per flag, with inline
    vote counts and K from ``record.probe_correct_votes_{flag}`` and
    ``record.k``) immediately after the verdict-comparison table; an
    auditor-reps section iterating ``selected_auditor_reps`` (each rep's
    pre-view reading and post-view judgment); a probe per-rep section
    filtered by ``selected_probe_reps`` whose trailing ``reasoning_text``
    subsection is gated by ``include_probe_reasoning_text``.

    The probe per-rep block obeys ``layout``: ``rep_major`` groups by
    rep (each rep's per-flag analyses and reasoning in sequence);
    ``flag_major`` groups by flag (one section per flag iterating
    selected reps). Every other section is identical across layouts.

    Raises:
        ValueError: ``selected_probe_reps`` or ``selected_auditor_reps``
            contains duplicates or values outside ``1..K`` for the
            relevant K. The notebook UI cells produce only valid values,
            so this guard fires only at test/integration boundaries —
            silent acceptance of bad inputs would mask UI bugs as silent
            misrenders.
    """
    p = results.provenance
    record = detail.audit_record
    config_key, example_id, batch_index, trial = detail.trial_key

    # Validate rep-selection inputs. The function is exported and tests
    # call it directly, so bad inputs must surface as ValueError, not as
    # silent misrenders or IndexError. The notebook UI cells produce only
    # valid values, so this guard fires only at test/integration
    # boundaries.
    k_probe = len(detail.probe_rep_reasonings)
    k_auditor = len(record.rep_readings)
    if len(set(selected_probe_reps)) != len(selected_probe_reps):
        raise ValueError(
            f"selected_probe_reps contains duplicate rep numbers: "
            f"{selected_probe_reps!r}"
        )
    if len(set(selected_auditor_reps)) != len(selected_auditor_reps):
        raise ValueError(
            f"selected_auditor_reps contains duplicate rep numbers: "
            f"{selected_auditor_reps!r}"
        )
    for n in selected_probe_reps:
        if not 1 <= n <= k_probe:
            raise ValueError(
                f"selected_probe_reps contains out-of-range rep number "
                f"{n!r}; valid range is 1..{k_probe} (probe K)"
            )
    for n in selected_auditor_reps:
        if not 1 <= n <= k_auditor:
            raise ValueError(
                f"selected_auditor_reps contains out-of-range rep number "
                f"{n!r}; valid range is 1..{k_auditor} (auditor K)"
            )

    matches_gt_label = (
        "**Yes**"
        if detail.meta_evaluator_matches_ground_truth is True
        else "**No**"
        if detail.meta_evaluator_matches_ground_truth is False
        else "—"
    )

    lines: list[str] = []
    lines.append(
        f"### Probe audit trial — {config_key} · {example_id} · {batch_index} · {trial}"
    )
    lines.append("")
    lines.append("**Pointers**")
    lines.append(f"- Audit snapshot: `{audit_snapshot_path}`")
    lines.append(f"- Rationale source: `{p.rationale_source_path or '—'}`")
    lines.append(f"- Skill fingerprint: `{p.skill_fingerprint}`")
    lines.append(f"- Probe snapshot SHA-256: `{p.probe_snapshot_sha256}`")
    lines.append(
        f"- Auditor: `{', '.join(p.auditor_model_versions)}` · K = {p.k}"
    )
    lines.append(
        f"- Run: {p.run_started_at} → {p.run_completed_at or '—'}"
    )
    lines.append("")
    lines.append("**Context**")
    lines.append(f"- Condition: `{detail.condition}`")
    lines.append(
        f"- Meta-evaluator predicted score: "
        f"`{detail.meta_evaluator_predicted_score or '—'}`"
    )
    lines.append(
        f"- Ground truth score: `{detail.meta_evaluator_ground_truth_score or '—'}`"
    )
    lines.append(f"- Meta-evaluator verdict matches ground truth: {matches_gt_label}")
    lines.append("")
    # Scenario context — the seven referential-disambiguation fields the
    # auditor sub-agent received on its bundle. Surfaced here so a human
    # reviewer reading the journal can resolve "previous" against the
    # same anchors the auditor saw, without having to flip back to the
    # stimulus file.
    lines.append("**Scenario context**")
    lines.append(f"- Current value: {detail.current_value}")
    lines.append(f"- Proposed value: {detail.proposed_value}")
    lines.append(f"- Historical value: {detail.historical_value}")
    lines.append(f"- Domain noun: {detail.domain_noun}")
    lines.append(f"- Draft fallback value: {detail.draft_fallback_value}")
    lines.append(f"- Ground truth description: {detail.ground_truth_description}")
    lines.append("")
    # The grader's feedback is multi-line structured JSON, so it renders
    # as a fenced ``json`` block under its own subheading rather than as
    # an inline bullet — matches the shape the auditor saw in its bundle
    # and keeps the journal scannable when pasted into Google Docs.
    lines.append("**Grader feedback (verbatim)**")
    lines.append("")
    lines.append("```json")
    lines.append(detail.grader_feedback)
    lines.append("```")
    lines.append("")
    # The rationale is the object under audit — the journal entry is
    # most useful to a future reader if the actual text they're
    # reasoning about is embedded, not just pointed to. Multi-paragraph
    # blockquoting via ``blockquote`` so pasted content in Google Docs
    # renders as a single coherent quote.
    lines.append("**Meta-evaluator rationale (the text under audit)**")
    lines.append("")
    lines.append(blockquote(detail.rationale_text))
    lines.append("")
    lines.append("**Verdict comparison**")
    lines.append("")
    # "Auditor aggregated" rather than "Auditor majority" because the
    # misattribution flags + articulation use any-affirmative while
    # governing uses majority-vote — both are "aggregation," only the
    # last is a majority vote. Mirroring that exactly in the column label
    # avoids misleading a reader looking at a 1-for/2-against row.
    lines.append(
        f"| Flag | Probe aggregated | Auditor aggregated (K={p.k}) | "
        "Probe-auditor alignment? | Disagreement category |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        reading_aggregated = getattr(
            record, f"aggregated_independent_reading_{flag}"
        )
        probe_correct = getattr(record, f"aggregated_probe_correct_{flag}")
        category = getattr(record, f"disagreement_category_{flag}")
        lines.append(
            f"| {flag_label(flag)} "
            f"| {_bool_cell(detail.probe_aggregated[flag])} "
            f"| {_bool_cell(reading_aggregated)} "
            f"| {_probe_correct_symbol(probe_correct)} "
            f"| {_DISAGREEMENT_CATEGORY_LABELS[category]} |"
        )
    lines.append("")

    # Per-flag probe-auditor alignment aggregate. Sits immediately
    # after the verdict-comparison table because the table and the
    # per-flag aggregate together are the audit's headline finding.
    # ``True``/``False``/``None`` map to "correct"/"not correct"/
    # "indeterminate" via ``_aggregate_bullet`` — independently per flag.
    # Inline vote counts make the margin visible without claiming
    # unanimity (``aggregated_probe_correct_*`` use majority-vote).
    lines.append("**Probe-auditor alignment aggregate**")
    lines.append("")
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        lines.append(
            _aggregate_bullet(
                flag_label(flag),
                getattr(record, f"aggregated_probe_correct_{flag}"),
                getattr(record, f"probe_correct_votes_{flag}"),
                record.k,
            )
        )
    lines.append("")

    # Probe's per-rep reasoning is the third leg of the triangle (the
    # rationale text and the auditor reasoning are the other two), and
    # it's what a researcher reaches for when diagnosing a probe
    # failure mode. Rendered for the selected probe reps; the trailing
    # reasoning_text subsection is gated by ``include_probe_reasoning_text``
    # (the only true reasoning trace in the audit data model, as
    # paper/CLAUDE.md's terminology table defines that term).
    lines.append("**Probe per-rep reasoning**")
    lines.append("")
    # Empty-selection short-circuit at the call site: skip both layout
    # helpers entirely and emit a placeholder. Without this guard,
    # ``flag_major`` would render the empty per-flag headings followed
    # by the placeholder — confusing UX.
    if not selected_probe_reps:
        lines.append("_(no probe reps selected)_")
        lines.append("")
    elif layout == "rep_major":
        # Iterate the selected reps directly so original 1-indexed
        # numbers survive a non-contiguous selection (e.g., reps (1, 3)
        # render as ``Probe Rep 1`` and ``Probe Rep 3``, not renumbered
        # 1 and 2). ``sorted`` ensures stable ascending output regardless
        # of the order the UI multi-select widget yielded the selection.
        for rep_num in sorted(selected_probe_reps):
            i = rep_num - 1
            lines.append(
                _probe_rep_block_rep_major(
                    rep_index=rep_num,
                    classifications={
                        flag: detail.probe_rep_classifications[flag][i]
                        for flag in RATIONALE_ANALYSIS_FLAG_KEYS
                    },
                    analyses={
                        flag: detail.probe_rep_analyses[flag][i]
                        for flag in RATIONALE_ANALYSIS_FLAG_KEYS
                    },
                    reasoning=detail.probe_rep_reasonings[i],
                    include_reasoning_text=include_probe_reasoning_text,
                )
            )
    else:
        lines.append(
            _probe_flag_sections_flag_major(
                detail,
                selected_probe_reps=selected_probe_reps,
                include_reasoning_text=include_probe_reasoning_text,
            )
        )

    # Auditor-reps section. Sits AFTER probe per-rep reasoning so the
    # narrative reads "probe argues, then auditor responds" — pre-view
    # reading + post-view judgment are bundled per rep, with the
    # temporal (before/after probe output) framing surfaced as bold
    # subheadings inside each rep block. Empty selection short-circuits
    # to a placeholder so the section heading still anchors the
    # researcher's mental map. Iteration order is the sorted
    # ``selected_auditor_reps`` for stable journal output regardless of
    # how the UI multi-select widget yielded them.
    lines.append("**Auditor reps**")
    lines.append("")
    if not selected_auditor_reps:
        lines.append("_(no auditor reps selected)_")
        lines.append("")
    else:
        for rep_num in sorted(selected_auditor_reps):
            i = rep_num - 1  # 1-indexed → 0-indexed array index
            lines.append(
                _auditor_rep_block(
                    rep_index=rep_num,
                    reading=record.rep_readings[i],
                    judgment=record.rep_judgments[i],
                    probe_aggregated=detail.probe_aggregated,
                )
            )

    lines.append("**Researcher notes**")
    lines.append("- ")
    lines.append("- ")
    lines.append("- ")

    return "\n".join(lines)
