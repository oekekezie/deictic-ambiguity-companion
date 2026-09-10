"""Shared scenario context extraction for probe-audit.

The probe-audit subagent and the journal template both need the seven
scenario fields the original Kimi K2.6 probe received. This module is the
single shared extraction point for both consumers.

Field sources:
    - current_value / proposed_value / historical_value / domain_noun:
      ``STIMULUS_METADATA_REGISTRY[base_example].<attr>``
    - draft_fallback_value: ``draft_fallback_value(condition, metadata)``
    - grader_feedback:      ``extract_grader_feedback(stimuli[example_id])``
    - ground_truth_description: ``ground_truth_description(condition)``

The fields are for referential disambiguation only — the auditor still
classifies what the meta-evaluator rationale states; the bundle's
scenario context lets the auditor see what "previous" resolves to.
"""

from pydantic import BaseModel, ConfigDict

from utils.experiment_analysis.ground_truth import Condition, parse_example_id
from utils.experiment_analysis.stimulus_metadata import (
    STIMULUS_METADATA_REGISTRY,
    draft_fallback_value,
)
from utils.rationale_analysis.template import (
    extract_grader_feedback,
    ground_truth_description,
)


class ScenarioContext(BaseModel):
    """The seven scenario context fields shared by bundle and journal.

    Frozen + extra=forbid so any drift in field shape is caught at
    construction rather than at downstream rendering.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    current_value: str
    proposed_value: str
    historical_value: str
    domain_noun: str
    draft_fallback_value: str
    grader_feedback: str
    ground_truth_description: str


def condition_from_example_id(example_id: str) -> Condition:
    """Typed-Condition extraction. Raises ``ValueError`` on malformed IDs.

    Replaces the legacy ``_parse_condition`` which silently returned
    ``"unknown"`` for malformed IDs — the new contract fails loud.
    """
    return parse_example_id(example_id)[1]


def build_scenario_context(
    example_id: str,
    stimuli: dict[str, dict[str, str]],
) -> tuple[ScenarioContext, Condition]:
    """Derive the seven scenario fields for ``example_id`` from stimuli + registry.

    Args:
        example_id: full benchmark id, e.g. ``"example_05_incorrect_opaque"``.
        stimuli: mapping produced by ``load_stimuli`` — keys are example_ids,
            values are ``{"system_prompt": str, "user_content": str}``.

    Returns:
        A pair ``(context, condition)`` where the condition is the typed
        Literal extracted alongside the base example.

    Raises:
        ValueError: if ``example_id`` is malformed (via ``parse_example_id``)
            or if the stimulus user_content lacks a parseable Grader's Feedback
            section.
        KeyError: if ``example_id`` is absent from ``stimuli``.
    """
    base_example, condition = parse_example_id(example_id)
    metadata = STIMULUS_METADATA_REGISTRY[base_example]
    user_content = stimuli[example_id]["user_content"]
    context = ScenarioContext(
        current_value=metadata.current_value,
        proposed_value=metadata.proposed_value,
        historical_value=metadata.historical_value,
        domain_noun=metadata.domain_noun,
        draft_fallback_value=draft_fallback_value(condition, metadata),
        grader_feedback=extract_grader_feedback(user_content),
        ground_truth_description=ground_truth_description(condition),
    )
    return context, condition
