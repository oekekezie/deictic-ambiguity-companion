"""Notebook-facing data preparation — pure functions over ``ProbeAuditResults``.

Kept in a standalone module so the marimo notebook is a thin presenter
and all derivations are testable from pytest. No matplotlib / plotly /
pandas imports here: the marimo cells can import pandas if they want.
"""

import math
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from utils.experiment_analysis.confidence_sequences import bernoulli_cs_bounds
from utils.probe_audit.models import DisagreementCategory, ProbeAuditResults
from utils.probe_audit.scenario_context import condition_from_example_id
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS

# One member per flag in ``RATIONALE_ANALYSIS_FLAG_KEYS``. A unit test
# pins this Literal to that tuple so the notebook-facing flag surface
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


class DisagreementCategoryDescription(NamedTuple):
    """One row of the disagreement-category legend rendered above the
    Trials Explorer filters.

    The triple is the canonical source of UI metadata for the four
    ``DisagreementCategory`` enum values. ``short_label`` is the
    human-readable variant shown in the multiselect dropdown;
    ``description`` is the explanatory prose copied verbatim from the
    canonical module docstring at ``utils/probe_audit/disagreement.py``.
    """

    category: DisagreementCategory
    short_label: str
    description: str


def disagreement_category_descriptions() -> tuple[DisagreementCategoryDescription, ...]:
    """Canonical (category, short_label, description) tuple for UI use.

    The triple is the seam where descriptive enum keys (storage layer)
    map to interpretive UI labels and reader-facing prose. Reader-facing
    surfaces — legend table, dropdown, tooltip, journal markdown, count
    tables — must consume ``short_label`` and ``description``; only
    storage-adjacent code (filter predicates, JSON serialization) reads
    ``category`` directly.

    Order matches ``typing.get_args(DisagreementCategory)`` and the init
    order of ``disagreement_category_counts``. ``short_label`` and
    ``description`` deliberately speak in the auditor's voice — "auditor
    judged the probe correct/incorrect" rather than "probe correct" —
    because the probe's classification has no ground truth and the
    auditor's post-view judgment is the only label available. The
    Stable Agreement row carries the indeterminate-fallback caveat in-row so
    skim readers do not miss it.

    A doc-sync test in ``tests/probe_audit/test_review_data.py`` pins
    these descriptions to the canonical semantics in
    ``utils/probe_audit/disagreement.py``.
    """
    return (
        DisagreementCategoryDescription(
            category="stable_agreement",
            short_label="Stable Agreement",
            description=(
                "auditor's pre-view reading matched the probe and the "
                "auditor judged the probe correct post-view, **or** the "
                "audit aggregate was indeterminate and was placed here "
                "by policy."
            ),
        ),
        DisagreementCategoryDescription(
            category="stable_disagreement",
            short_label="Stable Disagreement",
            description=(
                "auditor's pre-view reading disagreed with the probe and "
                "the auditor still judged the probe incorrect post-view. "
                "Strongest evidence of probe error."
            ),
        ),
        DisagreementCategoryDescription(
            category="shifted_against_probe",
            short_label="Shifted Against Probe",
            description=(
                "auditor's pre-view reading matched the probe, but the "
                "auditor judged the probe incorrect post-view. Weaker — "
                "may reflect the probe's own analysis revealing a flaw "
                "on re-read."
            ),
        ),
        DisagreementCategoryDescription(
            category="shifted_to_probe",
            short_label="Shifted Toward Probe",
            description=(
                "auditor's pre-view reading disagreed with the probe, "
                "but the auditor sided with the probe post-view. "
                "Possible anchoring artifact."
            ),
        ),
    )


def _agg_flag(record, flag: Flag) -> bool | None:
    """The record's aggregated probe-correct judgment for ``flag``."""
    return getattr(record, f"aggregated_probe_correct_{flag}")


def per_condition_probe_correctness(
    results: ProbeAuditResults, flag: Flag
) -> list[dict]:
    """Per-condition probe-correctness rate with an anytime-valid 95% one-sided
    lower confidence sequence bound ("alignment is at least ``cs_lower``").

    Uses the beta-binomial mixture martingale inversion from Howard et al.
    (2021) via ``bernoulli_cs_bounds(..., side="lower")``. The probe-auditor
    alignment claim is one-directional, so the full error budget goes on the
    lower bound and there is no upper bound. Unlike fixed-n Wilson intervals,
    the bound maintains nominal coverage at the data-dependent stopping times
    the probe audit actually uses.

    Rows cover ``correct``, ``transparent``, ``opaque`` — one per condition —
    regardless of whether the condition is empty in the sample. Empty rows emit
    ``n=0``, a NaN probe-correctness rate, and ``cs_lower = 0.0`` (the
    maximally-ignorant lower bound) when no observations are available.
    """
    buckets: dict[str, list[bool | None]] = {
        "correct": [],
        "transparent": [],
        "opaque": [],
    }
    for record in results.records:
        cond = condition_from_example_id(record.trial_key[1])
        if cond in buckets:
            buckets[cond].append(_agg_flag(record, flag))

    rows: list[dict] = []
    for cond, values in buckets.items():
        resolved = [v for v in values if v is not None]
        n = len(resolved)
        probe_correct = sum(1 for v in resolved if v is True)
        probe_incorrect = n - probe_correct
        rate = probe_correct / n if n else math.nan
        cs_lower, _ = bernoulli_cs_bounds(hits=probe_correct, valid=n, side="lower")
        rows.append(
            {
                "condition": cond,
                "n": n,
                "probe_correct": probe_correct,
                "probe_incorrect": probe_incorrect,
                "probe_correctness_rate": rate,
                "cs_lower": cs_lower,
            }
        )
    return rows


def _per_value_class_rows(
    records: list,
    flag: Flag,
    probe_flags: dict[tuple[str, str, int, int], dict[str, bool | None]],
) -> list[dict]:
    """Shared shaping for the single-audit and pooled per-value-class views.

    Buckets the given records by ``(condition, probe classification value)``
    and emits six rows (three conditions × ``{True, False}``) carrying the
    within-class alignment count and a one-sided lower CS. Trials the probe
    left indeterminate (``None``) on this flag are dropped — the conditioning
    question ("was the probe right about its positive/negative call?") is
    undefined for them.
    """
    buckets: dict[tuple[str, bool], list[bool | None]] = {
        (cond, value): []
        for cond in ("correct", "transparent", "opaque")
        for value in (True, False)
    }
    for record in records:
        cond = condition_from_example_id(record.trial_key[1])
        if cond not in ("correct", "transparent", "opaque"):
            continue
        probe_value = probe_flags.get(record.trial_key, {}).get(flag)
        if probe_value is None:
            continue
        buckets[(cond, bool(probe_value))].append(_agg_flag(record, flag))

    rows: list[dict] = []
    for (cond, probe_value), values in buckets.items():
        resolved = [v for v in values if v is not None]
        n = len(resolved)
        probe_correct = sum(1 for v in resolved if v is True)
        probe_incorrect = n - probe_correct
        rate = probe_correct / n if n else math.nan
        cs_lower, _ = bernoulli_cs_bounds(hits=probe_correct, valid=n, side="lower")
        rows.append(
            {
                "condition": cond,
                "probe_value": probe_value,
                "n": n,
                "probe_correct": probe_correct,
                "probe_incorrect": probe_incorrect,
                "probe_correctness_rate": rate,
                "cs_lower": cs_lower,
            }
        )
    return rows


def per_condition_per_value_class_probe_correctness(
    results: ProbeAuditResults,
    flag: Flag,
    probe_flags: dict[tuple[str, str, int, int], dict[str, bool | None]],
) -> list[dict]:
    """Per-(condition, probe-classification-value) alignment with a one-sided
    lower CS — the rare/floored-flag view that de-pools base-rate inflation.

    For a flag the coverage-convergent draw floored, the pooled per-condition
    rate mixes the probe's TRUE and FALSE classes at a deliberately enriched
    ratio, so it is NOT a pool rate. Conditioning on the probe's own
    classification value splits that mixture: ``probe_value=True`` rows report
    alignment among trials the probe called positive (where probe errors and
    base-rate inflation live), ``probe_value=False`` rows among the negatives.
    Within each ``(condition, probe_value)`` class the draw is uniform-WOR, so
    the CS is valid; the rare positive classes are small and get censused, so
    at campaign exhaustion the rate is exact. ``probe_flags`` comes from
    :func:`probe_flags_by_trial` over the full probe payload.
    """
    return _per_value_class_rows(list(results.records), flag, probe_flags)


def pooled_per_condition_per_value_class_probe_correctness(
    results_list: list[ProbeAuditResults],
    flag: Flag,
    probe_flags: dict[tuple[str, str, int, int], dict[str, bool | None]],
) -> list[dict]:
    """Pooled per-(condition, probe-value) alignment across audits.

    Row schema matches :func:`per_condition_per_value_class_probe_correctness`.
    The caller MUST have validated ``pool_compatibility(...).errors == ()``
    first (same precondition as the other pooled views).
    """
    return _per_value_class_rows(_pooled_records(results_list), flag, probe_flags)


def disagreement_category_counts(
    results: ProbeAuditResults, flag: Flag
) -> dict[str, int]:
    """Count of each disagreement category across the results."""
    counts = {
        "stable_agreement": 0,
        "stable_disagreement": 0,
        "shifted_against_probe": 0,
        "shifted_to_probe": 0,
    }
    for record in results.records:
        key = getattr(record, f"disagreement_category_{flag}")
        counts[key] = counts.get(key, 0) + 1
    return counts


def disagreement_category_counts_by_condition(
    results: ProbeAuditResults, flag: Flag
) -> list[dict]:
    """Per-(condition, category) disagreement counts as 3 rows.

    Each row pairs a condition (``correct``, ``transparent``, ``opaque``)
    with one int-valued key per :class:`DisagreementCategory` value.
    Three rows always emit, even for empty conditions, so downstream
    renderers can zip these rows with
    :func:`per_condition_probe_correctness` rows by index.

    The downstream agent-facing summary uses this split to surface the
    SM5-relevant cut: of the trials aligned in a given condition, how
    many are Stable Agreement versus Shifted Toward Probe. The flat helper does
    not expose that decomposition.
    """
    buckets: dict[str, dict[str, int]] = {
        "correct": {
            "stable_agreement": 0,
            "stable_disagreement": 0,
            "shifted_against_probe": 0,
            "shifted_to_probe": 0,
        },
        "transparent": {
            "stable_agreement": 0,
            "stable_disagreement": 0,
            "shifted_against_probe": 0,
            "shifted_to_probe": 0,
        },
        "opaque": {
            "stable_agreement": 0,
            "stable_disagreement": 0,
            "shifted_against_probe": 0,
            "shifted_to_probe": 0,
        },
    }
    for record in results.records:
        cond = condition_from_example_id(record.trial_key[1])
        if cond not in buckets:
            continue
        category = getattr(record, f"disagreement_category_{flag}")
        buckets[cond][category] += 1
    return [{"condition": cond, **counts} for cond, counts in buckets.items()]


def rep_split_census(results: ProbeAuditResults) -> dict[str, int]:
    """Count trials whose K reps split on each signal.

    Two keys per flag in ``RATIONALE_ANALYSIS_FLAG_KEYS`` capture the
    auditor side: ``reading_{flag}_split`` for the independent reading
    and ``probe_correct_{flag}_split`` for the probe-correct judgment. A
    "split" is when both ``for`` and ``against`` vote counts are nonzero.
    """
    census: dict[str, int] = {}
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        census[f"reading_{flag}_split"] = 0
        census[f"probe_correct_{flag}_split"] = 0
    for record in results.records:
        for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
            if all(getattr(record, f"independent_reading_votes_{flag}")):
                census[f"reading_{flag}_split"] += 1
            if all(getattr(record, f"probe_correct_votes_{flag}")):
                census[f"probe_correct_{flag}_split"] += 1
    return census


def trials_dataframe(results: ProbeAuditResults) -> list[dict]:
    """One row per trial with flattened key metadata.

    Drives the filterable trials table in the probe-audit review notebook.
    The column set is a stable contract (pinned by tests) so the notebook's
    filter dropdowns and table formatters don't silently desync when
    upstream fields change. Per flag in ``RATIONALE_ANALYSIS_FLAG_KEYS``
    the row carries the aggregated probe-correct judgment, the
    disagreement category, and the probe-correct vote tally.
    """
    rows: list[dict] = []
    for record in results.records:
        config_key, example_id, batch_index, trial = record.trial_key
        row: dict = {
            "config_key": config_key,
            "example_id": example_id,
            "batch_index": batch_index,
            "trial": trial,
            "condition": condition_from_example_id(example_id),
        }
        for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
            votes_for, votes_against = getattr(
                record, f"probe_correct_votes_{flag}"
            )
            row[f"aggregated_probe_correct_{flag}"] = getattr(
                record, f"aggregated_probe_correct_{flag}"
            )
            row[f"disagreement_category_{flag}"] = getattr(
                record, f"disagreement_category_{flag}"
            )
            row[f"{flag}_votes_for"] = votes_for
            row[f"{flag}_votes_against"] = votes_against
        row["k"] = record.k
        rows.append(row)
    return rows


def pooled_trials_dataframe(
    sessions: list[tuple[str, ProbeAuditResults]],
) -> list[dict]:
    """Pooled per-trial rows tagged with their source-audit session ID.

    Wraps ``trials_dataframe`` per audit and prepends a ``session_id``
    column so the notebook's drill-down resolver can dispatch back to
    the source audit's ``ProbeAuditResults``. The compat-check guarantee
    (no overlapping ``trial_keys``) means each row corresponds to exactly
    one audit; this function does not deduplicate.

    The session_id is the snapshot's parent directory name in the
    notebook — the canonical per-audit identifier already used in the
    compat panel's contributions table.
    """
    rows: list[dict] = []
    for session_id, results in sessions:
        for row in trials_dataframe(results):
            rows.append({"session_id": session_id, **row})
    return rows


class PoolCompatibility(BaseModel):
    """Classifies whether a set of ``ProbeAuditResults`` can be pooled.

    The notebook's pooled-CS path calls ``pool_compatibility`` before
    computing any aggregate statistic. If ``errors`` is non-empty,
    pooling is refused — the pooled CS would misrepresent the estimand
    (e.g., mixing different probe snapshots combines populations that
    should remain separate). ``warnings`` surface soft mismatches (K,
    auditor model versions) that are informational but do not block
    pooling.

    The ``shared_*`` fields report the common identity value when every
    input agrees on it; they are ``None`` when inputs disagreed. The
    paired ``*_agrees`` booleans let the notebook render "(mismatch)"
    distinctly from a present-but-shared value. All six identity fields
    (skill fingerprint, probe snapshot SHA, and the byte-aware
    rationale + stimulus source pairs) are required on
    ``ProbeAuditProvenance``, so a ``shared_*`` value of ``None`` paired
    with ``_agrees=True`` cannot arise — it would only appear if the
    inputs disagreed.

    Each source path AND its raw-bytes SHA are checked: same path
    with mutated bytes between two audits is a different audit context
    that must not be pooled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    shared_skill_fingerprint: str | None = None
    skill_fingerprint_agrees: bool = False
    shared_probe_snapshot_sha256: str | None = None
    probe_snapshot_sha256_agrees: bool = False
    shared_rationale_source_path: str | None = None
    rationale_source_path_agrees: bool = False
    shared_rationale_source_sha256: str | None = None
    rationale_source_sha256_agrees: bool = False
    shared_stimulus_source_path: str | None = None
    stimulus_source_path_agrees: bool = False
    shared_stimulus_source_sha256: str | None = None
    stimulus_source_sha256_agrees: bool = False
    shared_batch_allocation: str | None = None
    batch_allocation_agrees: bool = False


def _unique_value(values: list[object]) -> tuple[object, bool]:
    """Return ``(value, is_shared)`` — agreement-or-mismatch summary.

    ``is_shared`` is True iff every entry is equal. The returned value
    is that common value when ``is_shared`` is True; callers use
    ``is_shared`` to decide whether to emit a mismatch error and use
    ``value`` to populate the ``shared_*`` field only on agreement.
    Empty input degrades to ``(None, False)`` so callers can short-
    circuit without a separate length check.
    """
    if not values:
        return (None, False)
    first = values[0]
    is_shared = all(v == first for v in values)
    return (first, is_shared)


def pool_compatibility(
    results_list: list[ProbeAuditResults],
) -> PoolCompatibility:
    """Validate that a list of ``ProbeAuditResults`` can be pooled.

    Hard constraints (each violation → one entry in ``errors``):

    - ``skill_fingerprint`` must match across all inputs — a different
      fingerprint means a different audit protocol, and the estimand
      would mix two populations.
    - ``probe_snapshot_sha256`` must match — different probe bytes
      produce a different measurement population.
    - ``rationale_source_path`` AND ``rationale_source_sha256`` must
      match — the auditors were classifying rationales from a pinned
      source, and two different sources (or the same path with mutated
      bytes between runs) are effectively two different tasks. The
      byte-aware check rejects same-path/mutated-bytes pooling that a
      path-only check would silently accept.
    - ``stimulus_source_path`` AND ``stimulus_source_sha256`` must
      match — the auditor's bundle carries scenario context fields
      derived from this file, so two audits whose auditors saw
      different stimulus bytes saw different ``grader_feedback`` /
      ``ground_truth_description`` values and cannot be pooled.
    - ``trial_keys`` across all inputs must be mutually disjoint — a
      repeated ``trial_key`` would double-count a single trial in the
      pooled ``hits`` / ``valid`` counts. The notebook's inclusion
      table surfaces per-source contributions so a researcher can
      resolve by deselection; this compatibility check is about the
      pooled statistic, which is well-defined only over disjoint trial
      sets.

    Soft constraints (each entry → one entry in ``warnings``):

    - ``K`` differences — the per-trial aggregation uses
      any-affirmative + majority-vote whose noise characteristics shift
      with K. Pooling proceeds but the notebook flags the mismatch.
    - ``auditor_model_versions`` differences — two audits conducted
      under different auditor models are measuring with different
      instruments. Pooling proceeds; the researcher sees the mismatch.

    ``shared_*`` fields are populated iff the corresponding hard
    constraint passes (same value across inputs), so the notebook can
    render a "Shared Provenance" panel without re-deriving the common
    identity.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not results_list:
        return PoolCompatibility(
            errors=("pool_compatibility requires at least one ProbeAuditResults",),
        )

    fingerprints = [r.provenance.skill_fingerprint for r in results_list]
    probe_shas = [r.provenance.probe_snapshot_sha256 for r in results_list]
    rs_paths = [r.provenance.rationale_source_path for r in results_list]
    rs_shas = [r.provenance.rationale_source_sha256 for r in results_list]
    stim_paths = [r.provenance.stimulus_source_path for r in results_list]
    stim_shas = [r.provenance.stimulus_source_sha256 for r in results_list]
    allocations = [r.provenance.batch_allocation for r in results_list]
    ks = [r.provenance.k for r in results_list]
    auditor_versions = [r.provenance.auditor_model_versions for r in results_list]

    shared_fp, fp_agrees = _unique_value(fingerprints)
    shared_probe_sha, probe_agrees = _unique_value(probe_shas)
    shared_rs, rs_agrees = _unique_value(rs_paths)
    shared_rs_sha, rs_sha_agrees = _unique_value(rs_shas)
    shared_stim, stim_agrees = _unique_value(stim_paths)
    shared_stim_sha, stim_sha_agrees = _unique_value(stim_shas)
    shared_alloc, alloc_agrees = _unique_value(allocations)

    if not fp_agrees:
        distinct = sorted({f[:12] for f in fingerprints})
        errors.append(
            f"skill_fingerprint mismatch across {len(results_list)} inputs: "
            f"got {len(distinct)} distinct fingerprints "
            f"({', '.join(f'{d}…' for d in distinct)})"
        )
    if not probe_agrees:
        distinct = sorted({s[:12] for s in probe_shas})
        errors.append(
            f"probe_snapshot_sha256 mismatch across {len(results_list)} inputs: "
            f"got {len(distinct)} distinct probe snapshots "
            f"({', '.join(f'{d}…' for d in distinct)}). Pooling across "
            "different probe bytes would mix populations."
        )
    if not rs_agrees:
        distinct = sorted({str(p) for p in rs_paths})
        errors.append(
            f"rationale_source_path mismatch across {len(results_list)} inputs: "
            f"got {len(distinct)} distinct rationale sources "
            f"({', '.join(distinct)})"
        )
    # SHA check fires INDEPENDENTLY of the path check — two audits with
    # the same rationale_source_path but mutated bytes between runs would
    # pass path-equality but fail SHA-equality. Reject that case.
    if not rs_sha_agrees:
        distinct = sorted({s[:12] for s in rs_shas})
        errors.append(
            f"rationale_source_sha256 mismatch across {len(results_list)} "
            f"inputs: got {len(distinct)} distinct rationale-source byte "
            f"hashes ({', '.join(f'{d}…' for d in distinct)}). Same path "
            "with mutated bytes is a different audit context."
        )
    if not stim_agrees:
        distinct = sorted({str(p) for p in stim_paths})
        errors.append(
            f"stimulus_source_path mismatch across {len(results_list)} "
            f"inputs: got {len(distinct)} distinct stimulus sources "
            f"({', '.join(distinct)}). The auditors saw different "
            "scenario context fields and cannot be pooled."
        )
    if not stim_sha_agrees:
        distinct = sorted({s[:12] for s in stim_shas})
        errors.append(
            f"stimulus_source_sha256 mismatch across {len(results_list)} "
            f"inputs: got {len(distinct)} distinct stimulus-source byte "
            f"hashes ({', '.join(f'{d}…' for d in distinct)}). Same path "
            "with mutated bytes means the auditors saw different "
            "grader_feedback / ground_truth_description values."
        )
    # Allocation-mode mismatch is a hard error: the stratified and
    # coverage-convergent draws are different sampling designs whose pooled
    # statistics are not comparable, and a convergent campaign's convergence
    # guarantee is defined per-mode. Mixing them in one pool is refused.
    if not alloc_agrees:
        distinct = sorted({str(a) for a in allocations})
        errors.append(
            f"batch_allocation mismatch across {len(results_list)} inputs: "
            f"got {len(distinct)} distinct allocation modes "
            f"({', '.join(distinct)}). Stratified and coverage-convergent "
            "draws are different sampling designs and must not be pooled."
        )
    # Trial-key overlap across inputs: the pooled hits/valid sum is only
    # well-defined over a disjoint union. A repeated trial_key means the
    # same observation would enter the count twice.
    seen: dict[tuple[str, str, int, int], int] = {}
    overlaps: list[tuple[str, str, int, int]] = []
    for idx, r in enumerate(results_list):
        for record in r.records:
            tk = record.trial_key
            if tk in seen and seen[tk] != idx:
                overlaps.append(tk)
            else:
                seen[tk] = idx
    if overlaps:
        # De-duplicate the overlap list so the message is compact even
        # when a single trial_key shows up across three+ sessions.
        distinct_overlaps = sorted(set(overlaps))
        head = ", ".join(
            f"({c}|{e}|{b}|{t})" for c, e, b, t in distinct_overlaps[:3]
        )
        suffix = "" if len(distinct_overlaps) <= 3 else f", …{len(distinct_overlaps) - 3} more"
        errors.append(
            f"trial_key overlap across inputs: {len(distinct_overlaps)} "
            f"trial(s) appear in more than one audit ({head}{suffix}). "
            "Pooling would double-count; deselect conflicting rows or "
            "re-run with --exclude-prior-audits to avoid drawing the "
            "same trials twice."
        )

    # Soft constraints.
    if len(set(ks)) > 1:
        warnings.append(
            f"K differs across inputs ({sorted(set(ks))}) — per-trial "
            "aggregation noise characteristics will differ between the "
            "audits. Pooling proceeds but the pooled rate is a mixture "
            "over uneven measurement instruments."
        )
    # Distinct from the K-differs warning: surfaces single-rep audits
    # even when every selected audit shares K=1 (no variation in K).
    # The pooled rate over disjoint trial sets is well-defined, but each
    # contributing K=1 audit carries no within-trial replication evidence
    # — researchers should know that before reading the pooled CS.
    n_singles = sum(
        1 for r in results_list
        if r.provenance.replication_evidence == "none"
    )
    if n_singles > 0:
        warnings.append(
            f"{n_singles} input audit(s) have replication_evidence='none' "
            "(K=1; single rep's judgment is the aggregate). The pooled "
            "rate is well-defined over disjoint trial sets, but each "
            "contributing K=1 audit carries no per-trial replication "
            "evidence."
        )
    distinct_versions = {tuple(v) for v in auditor_versions}
    if len(distinct_versions) > 1:
        warnings.append(
            "auditor_model_versions differs across inputs "
            f"({sorted(distinct_versions)}) — audits ran under different "
            "measurement models. Pooling proceeds but the pooled rate "
            "mixes instruments."
        )

    # Return the common value iff the inputs agreed; ``None`` otherwise.
    # All six identity fields are required on ``ProbeAuditProvenance``,
    # so ``shared_*=None`` paired with ``_agrees=True`` cannot arise —
    # ``None`` paired with ``_agrees=False`` is the sole "mismatch"
    # signal the notebook reads.
    return PoolCompatibility(
        errors=tuple(errors),
        warnings=tuple(warnings),
        shared_skill_fingerprint=shared_fp if fp_agrees else None,
        skill_fingerprint_agrees=fp_agrees,
        shared_probe_snapshot_sha256=shared_probe_sha if probe_agrees else None,
        probe_snapshot_sha256_agrees=probe_agrees,
        shared_rationale_source_path=shared_rs if rs_agrees else None,
        rationale_source_path_agrees=rs_agrees,
        shared_rationale_source_sha256=shared_rs_sha if rs_sha_agrees else None,
        rationale_source_sha256_agrees=rs_sha_agrees,
        shared_stimulus_source_path=shared_stim if stim_agrees else None,
        stimulus_source_path_agrees=stim_agrees,
        shared_stimulus_source_sha256=shared_stim_sha if stim_sha_agrees else None,
        stimulus_source_sha256_agrees=stim_sha_agrees,
        shared_batch_allocation=shared_alloc if alloc_agrees else None,
        batch_allocation_agrees=alloc_agrees,
    )


def _pooled_records(results_list: list[ProbeAuditResults]) -> list:
    """Flatten records across inputs. Assumes compatibility was validated.

    Returned as a plain list (not a tuple) because downstream consumers
    filter / iterate without needing immutability.
    """
    out = []
    for r in results_list:
        out.extend(r.records)
    return out


def pooled_per_condition_probe_correctness(
    results_list: list[ProbeAuditResults], flag: Flag
) -> list[dict]:
    """Pool hits/valid per condition across audits, then invert the one-sided
    martingale once.

    Row schema matches :func:`per_condition_probe_correctness` exactly
    (``condition``, ``n``, ``probe_correct``, ``probe_incorrect``,
    ``probe_correctness_rate``, ``cs_lower``) so the notebook's display code is
    shared between single-audit and pooled views.

    The caller MUST have already run ``pool_compatibility`` and found
    ``errors == ()`` — otherwise the pooled CS would double-count
    overlapping trials or mix populations that should remain separate.
    """
    buckets: dict[str, list[bool | None]] = {
        "correct": [],
        "transparent": [],
        "opaque": [],
    }
    for record in _pooled_records(results_list):
        cond = condition_from_example_id(record.trial_key[1])
        if cond in buckets:
            buckets[cond].append(_agg_flag(record, flag))

    rows: list[dict] = []
    for cond, values in buckets.items():
        resolved = [v for v in values if v is not None]
        n = len(resolved)
        probe_correct = sum(1 for v in resolved if v is True)
        probe_incorrect = n - probe_correct
        rate = probe_correct / n if n else math.nan
        cs_lower, _ = bernoulli_cs_bounds(hits=probe_correct, valid=n, side="lower")
        rows.append(
            {
                "condition": cond,
                "n": n,
                "probe_correct": probe_correct,
                "probe_incorrect": probe_incorrect,
                "probe_correctness_rate": rate,
                "cs_lower": cs_lower,
            }
        )
    return rows


def pooled_disagreement_category_counts(
    results_list: list[ProbeAuditResults], flag: Flag
) -> dict[str, int]:
    """Pooled disagreement-category counts across audits.

    Category set matches :func:`disagreement_category_counts`. The caller
    must have validated compatibility before calling this.
    """
    counts = {
        "stable_agreement": 0,
        "stable_disagreement": 0,
        "shifted_against_probe": 0,
        "shifted_to_probe": 0,
    }
    for record in _pooled_records(results_list):
        key = getattr(record, f"disagreement_category_{flag}")
        counts[key] = counts.get(key, 0) + 1
    return counts


def pooled_disagreement_category_counts_by_condition(
    results_list: list[ProbeAuditResults], flag: Flag
) -> list[dict]:
    """Pooled per-(condition, category) disagreement counts.

    Row schema matches :func:`disagreement_category_counts_by_condition`.
    The caller MUST have validated compatibility via
    :func:`pool_compatibility` before calling — pooling across
    incompatible audits would mix populations.
    """
    buckets: dict[str, dict[str, int]] = {
        "correct": {
            "stable_agreement": 0,
            "stable_disagreement": 0,
            "shifted_against_probe": 0,
            "shifted_to_probe": 0,
        },
        "transparent": {
            "stable_agreement": 0,
            "stable_disagreement": 0,
            "shifted_against_probe": 0,
            "shifted_to_probe": 0,
        },
        "opaque": {
            "stable_agreement": 0,
            "stable_disagreement": 0,
            "shifted_against_probe": 0,
            "shifted_to_probe": 0,
        },
    }
    for record in _pooled_records(results_list):
        cond = condition_from_example_id(record.trial_key[1])
        if cond not in buckets:
            continue
        category = getattr(record, f"disagreement_category_{flag}")
        buckets[cond][category] += 1
    return [{"condition": cond, **counts} for cond, counts in buckets.items()]


def pooled_rep_split_census(
    results_list: list[ProbeAuditResults],
) -> dict[str, int]:
    """Pooled rep-split census across audits.

    Keys match :func:`rep_split_census`. One record contributes to a
    key iff its per-rep votes on that signal were split (both ``for``
    and ``against`` are non-zero). Pooling is a simple sum because
    each trial is counted at most once (compatibility guarantees
    disjoint trial sets).
    """
    census: dict[str, int] = {}
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        census[f"reading_{flag}_split"] = 0
        census[f"probe_correct_{flag}_split"] = 0
    for record in _pooled_records(results_list):
        for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
            if all(getattr(record, f"independent_reading_votes_{flag}")):
                census[f"reading_{flag}_split"] += 1
            if all(getattr(record, f"probe_correct_votes_{flag}")):
                census[f"probe_correct_{flag}_split"] += 1
    return census


def provenance_entries(results: ProbeAuditResults) -> list[tuple[str, str]]:
    """Ordered ``(label, value)`` pairs for the notebook's provenance panel.

    Surfaces every field of ``ProbeAuditProvenance`` untruncated so the
    researcher can copy full 64-hex SHAs, seeds, and args-digest strings
    into a lab journal or audit artifact without re-opening the raw JSON.
    Values are always strings (stringify everything up front) to simplify
    the notebook's rendering — the panel is a two-column markdown table.
    """
    p = results.provenance
    entries: list[tuple[str, str]] = [
        ("Skill fingerprint", p.skill_fingerprint),
        ("Fingerprint method", p.fingerprint_method),
        ("Skill human label", p.skill_human_label if p.skill_human_label else "—"),
        ("Git commit SHA", p.git_commit_sha if p.git_commit_sha else "—"),
        ("Git tree dirty", str(p.git_tree_dirty)),
        ("K", str(p.k)),
        ("Batch allocation", p.batch_allocation),
        (
            "Classified rare flags",
            ", ".join(p.classified_rare_flags)
            if p.classified_rare_flags
            else "—",
        ),
        ("Replication evidence", p.replication_evidence),
        ("Sampling seed", p.sampling_seed),
        ("Rationale source path", p.rationale_source_path),
        ("Rationale source SHA-256", p.rationale_source_sha256),
        ("Stimulus source path", p.stimulus_source_path),
        ("Stimulus source SHA-256", p.stimulus_source_sha256),
        ("Probe snapshot path", p.probe_snapshot_path),
        ("Probe snapshot SHA-256", p.probe_snapshot_sha256),
        ("Auditor model family", p.auditor_model_family),
        ("Auditor model version(s)", ", ".join(p.auditor_model_versions)),
        ("Claude session id", p.claude_session_id if p.claude_session_id else "—"),
        ("Run started", p.run_started_at),
        ("Run completed", p.run_completed_at if p.run_completed_at else "—"),
        ("Args digest", p.args_digest),
        ("Canonical args JSON", p.canonical_args_json),
    ]
    return entries


def probe_flags_by_trial(
    probe_payload: dict,
) -> dict[tuple[str, str, int, int], dict[str, bool | None]]:
    """Return ``trial_key -> {flag: probe_classification}`` from a parsed payload.

    The companion to :func:`utils.probe_audit.aggregation._load_probe_flags_by_trial`,
    but consumes an already-parsed payload dict instead of re-reading the
    snapshot from disk. The notebook data-load cells parse the probe
    snapshot once into ``probe_payload``; this helper exposes a
    ``trial_key``-keyed view of the per-trial flags so consumers do not
    re-parse the payload or duplicate the trial-record loop.

    Currently consumed by ``utils.probe_audit_annotation.loading``'s
    disagreement-queue construction. ``utils.probe_audit.trial_detail``
    still uses an inline single-trial scan (``_probe_record_for``)
    because its access pattern is one trial per drill-down click,
    where building a full lookup dict would be wasted work; that helper
    can adopt this function later if a multi-trial view is added.

    Each of the seven probe-schema flag keys can be ``None`` when fewer
    than ``ceil(K/2)`` of the probe's K reps parsed — surface that
    ``None`` to the caller verbatim rather than coercing to ``False``, so
    the caller can decide whether a missing per-flag value is a hard
    failure (annotation queue) or an acceptable indeterminate.
    """
    result: dict[tuple[str, str, int, int], dict[str, bool | None]] = {}
    for rec in probe_payload.get("trial_records", []):
        key = (
            rec["config_key"],
            rec["example_id"],
            int(rec["batch_index"]),
            int(rec["trial"]),
        )
        result[key] = {
            flag: rec.get(flag) for flag in RATIONALE_ANALYSIS_FLAG_KEYS
        }
    return result
