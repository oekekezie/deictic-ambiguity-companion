"""Disagreement-category classification for audit trial records.

The probe is an LLM rationale-classifier whose own labels have no ground
truth — that is the reason this audit exists. The auditor (K independent
LLM sub-agent reps) re-reads each trial's rationale in two phases: a
**pre-view independent reading** before seeing probe output, and a
**post-view judgment** of whether the probe's labels were correct after
seeing them. The disagreement category encodes how those two readings
line up with the probe across both phases. Throughout this module,
"probe judged correct/incorrect" always means *the auditor's post-view
aggregate judged the probe correct/incorrect* — never a ground truth
claim about the probe's classification.

Four categories, organized as a 2×2 over (pre-view reading == probe?) ×
(post-view: auditor judged probe correct?):

- ``stable_agreement``: pre-view reading matched the probe **and** the
  auditor judged the probe correct post-view. Full consensus.
- ``shifted_against_probe``: pre-view reading matched the probe but the
  auditor judged the probe incorrect post-view. Weaker probe-error
  signal — the probe's own analysis may have surfaced a flaw on
  re-read.
- ``shifted_to_probe``: pre-view reading disagreed with the probe but
  the auditor sided with the probe post-view. Possible anchoring
  artifact — the auditor's pre-view reading was overwritten after
  exposure to the probe.
- ``stable_disagreement``: pre-view reading disagreed with the probe
  **and** the auditor judged the probe incorrect post-view. Strongest
  evidence of probe error.

Indeterminate-fallback policy. When ``aggregated_probe_correct is None``
or ``independent_reading is None`` (insufficient parsed reps), the
function returns ``stable_agreement`` *by policy*, not because actual
agreement is proven. UI surfaces that label this category "Stable Agreement"
must therefore note the fallback explicitly: a "Stable Agreement" row may
include indeterminate aggregates that were placed there because the
audit had nothing decisive to say. When ``probe_value is None`` and the
auditor judged the probe correct, the function still returns
``stable_agreement`` (no realignment can be detected without a probe
value to compare against). When ``probe_value is None`` and the auditor
judged the probe incorrect, the function returns ``shifted_against_probe``
— the auditor flagged the probe wrong even though the probe's own value
is unknown to this classifier.

Reader-facing legend descriptions (kept verbatim here so a doc-sync test
can pin them against ``disagreement_category_descriptions`` in
``review_data.py``):

- Stable Agreement: auditor's pre-view reading matched the probe and the
  auditor judged the probe correct post-view, **or** the audit aggregate
  was indeterminate and was placed here by policy.
- Stable Disagreement: auditor's pre-view reading disagreed with the
  probe and the auditor still judged the probe incorrect post-view.
  Strongest evidence of probe error.
- Shifted Against Probe: auditor's pre-view reading matched the probe, but the
  auditor judged the probe incorrect post-view. Weaker — may reflect
  the probe's own analysis revealing a flaw on re-read.
- Shifted Toward Probe: auditor's pre-view reading disagreed with the
  probe, but the auditor sided with the probe post-view. Possible
  anchoring artifact.
"""

from utils.probe_audit.models import DisagreementCategory


def categorize_disagreement(
    probe_value: bool | None,
    independent_reading: bool | None,
    aggregated_probe_correct: bool | None,
) -> DisagreementCategory:
    """Four-way classification; see module docstring for full semantics.

    ``probe_value`` is frequently ``None`` when only the auditor's
    aggregated judgment is available; in that case the comparison is
    derived directly from ``independent_reading`` and
    ``aggregated_probe_correct``. The indeterminate-fallback policy in
    the module docstring governs every ``None`` branch — read it before
    relying on the return value as evidence of agreement.
    """
    if aggregated_probe_correct is None or independent_reading is None:
        return "stable_agreement"

    if aggregated_probe_correct is True:
        # Auditor judged the probe correct. Without a known probe value
        # we cannot detect a realignment shift, so return stable_agreement
        # by policy. A known probe value that disagreed with the
        # pre-view reading marks a possible anchoring artifact: the
        # auditor disagreed before viewing the probe but sided with it
        # after.
        if probe_value is None or independent_reading == probe_value:
            return "stable_agreement"
        return "shifted_to_probe"
    # Auditor judged the probe incorrect. A pre-view reading that
    # already disagreed with a known probe value is the strongest
    # evidence of probe error; matching pre-view readings indicate the
    # auditor reconsidered after viewing the probe.
    if probe_value is None:
        return "shifted_against_probe"
    return (
        "stable_disagreement"
        if independent_reading != probe_value
        else "shifted_against_probe"
    )
