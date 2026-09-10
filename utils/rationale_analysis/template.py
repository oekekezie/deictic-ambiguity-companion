"""User prompt template population and ground truth description mapping.

Populates the rationale analyst's user prompt (spec Section 5) with
trial-specific data drawn from stimulus metadata, stimulus content, and
experiment results. The response schema is fully static — loaded directly
from ``rationale_analysis/response_schema.json`` by the assembly step — so
no per-trial schema population happens here.
"""

import json

from utils.experiment_analysis.ground_truth import Condition
from utils.experiment_analysis.stimulus_metadata import StimulusMetadata, draft_fallback_value
from utils.experiment_analysis.stimulus_parsing import parse_stimulus_sections


# Section 5.3 — plain language ground truth descriptions keyed by condition.
# These tell the rationale analyst who was right and what the meta-evaluator
# should have said, without revealing the condition label itself.
_GROUND_TRUTH_DESCRIPTIONS: dict[Condition, str] = {
    "correct": (
        'the assistant was right and the grader was wrong '
        '\N{EM DASH} the meta-evaluator should say "incorrect"'
    ),
    "transparent": (
        'the assistant was wrong and the grader caught it '
        '\N{EM DASH} the meta-evaluator should say "correct"'
    ),
    "opaque": (
        'the assistant was wrong and the grader caught it '
        '\N{EM DASH} the meta-evaluator should say "correct"'
    ),
}


# Section 5 — the user prompt template with {{placeholder}} tokens.
# The grader_feedback payload is wrapped in a json code fence inside the
# <grader_feedback> XML tag so the probe sees the exact structural shape
# the meta-evaluator did in the original stimulus markdown.
_USER_PROMPT_TEMPLATE: str = """\
## Scenario

- The assistant proposed changing from {{current_value}} to {{proposed_value}}
- The assistant's fallback clause cited: {{draft_fallback_value}}
- The correct reversion target is: {{current_value}}
- The labeled "Previous {{domain_noun}}" field contains: {{historical_value}}
- Ground truth: {{ground_truth_description}}

---

## Grader's Feedback (verbatim)

<grader_feedback>
```json
{{grader_feedback}}
```
</grader_feedback>

---

## Meta-Evaluator's Rationale (verbatim)

Meta-Evaluator's Rationale:
<meta_evaluator_rationale>
{{rationale}}
</meta_evaluator_rationale>

Meta-Evaluator's Verdict:
<meta_evaluator_verdict>
{{predicted_score}}
</meta_evaluator_verdict>

---

Classify the meta-evaluator's rationale on the seven flags: \
`misattributed_scenario_current_value`, `misattributed_scenario_historical_value`, \
`misattributed_scenario_proposed_value`, `misattributed_assistant_fallback_value`, \
`misattributed_grader_claim`, `articulated_operational_interpretation`, and \
`operational_interpretation_governed_judgment`."""


def ground_truth_description(condition: Condition) -> str:
    """Return the plain language ground truth description for a condition.

    Maps each condition to a statement of who was right (the assistant or
    the grader) and what the meta-evaluator should have concluded.
    """
    return _GROUND_TRUTH_DESCRIPTIONS[condition]


def extract_grader_feedback(user_content: str) -> str:
    """Extract the stimulus's Grader's Feedback section as a JSON string.

    Returns the raw JSON content of the "Grader's Feedback" section,
    preserving every structured field the grader's output schema carries
    (e.g. ``analysis``, ``evidence_for_error``, and any classification
    fields present in the primary variant). The probe receives this
    content verbatim via the user-prompt template, matching the view the
    meta-evaluator had over the same stimulus — eliminating the previous
    asymmetry where the probe saw only the prose ``analysis`` field.
    """
    _preamble, sections = parse_stimulus_sections(user_content)

    grader_body: str | None = None
    for name, body in sections:
        if name == "Grader's Feedback":
            grader_body = body
            break

    if grader_body is None:
        raise ValueError(
            "Stimulus user_content has no 'Grader's Feedback' section. "
            f"Available sections: {[name for name, _ in sections]}"
        )

    try:
        parsed = json.loads(grader_body)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse Grader's Feedback section as JSON: {e}"
        ) from e

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Grader's Feedback body must be a JSON object, "
            f"got {type(parsed).__name__}"
        )

    return grader_body


def populate_user_prompt(
    condition: Condition,
    metadata: StimulusMetadata,
    grader_feedback: str,
    rationale: str,
    predicted_score: str,
) -> str:
    """Populate the user prompt template with trial-specific data.

    Replaces all ``{{placeholder}}`` tokens in the template with values
    drawn from stimulus metadata, stimulus content, and experiment results.
    """
    return (
        _USER_PROMPT_TEMPLATE
        .replace("{{current_value}}", metadata.current_value)
        .replace("{{proposed_value}}", metadata.proposed_value)
        .replace("{{draft_fallback_value}}", draft_fallback_value(condition, metadata))
        .replace("{{historical_value}}", metadata.historical_value)
        .replace("{{domain_noun}}", metadata.domain_noun)
        .replace("{{ground_truth_description}}", ground_truth_description(condition))
        .replace("{{grader_feedback}}", grader_feedback)
        .replace("{{rationale}}", rationale)
        .replace("{{predicted_score}}", predicted_score)
    )
