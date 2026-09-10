"""Ground truth mapping for the ReFT benchmark.

Parses example IDs into their base example and condition components,
and maps each condition to its deterministic ground truth score.
"""

import re
from typing import Literal

Condition = Literal["correct", "transparent", "opaque"]
ScoreValue = Literal["correct", "incorrect"]

# Dataset filename suffixes → analysis-layer condition labels.
# The "incorrect_" prefix describes the draft; the analysis layer
# strips it because draft-level semantics are a dataset concern.
_CONDITION_SUFFIXES: dict[str, Condition] = {
    "correct": "correct",
    "incorrect_transparent": "transparent",
    "incorrect_opaque": "opaque",
}

_EXAMPLE_ID_PATTERN: re.Pattern[str] = re.compile(
    r"^(example_\d+)_(correct|incorrect_transparent|incorrect_opaque)$"
)

# In the correct-draft condition, the grader is wrong → model should
# output "incorrect". In both incorrect-draft conditions, the grader
# is right → model should output "correct".
_GROUND_TRUTH: dict[Condition, ScoreValue] = {
    "correct": "incorrect",
    "transparent": "correct",
    "opaque": "correct",
}


def parse_example_id(example_id: str) -> tuple[str, Condition]:
    """Decompose an example_id into (base_example, condition).

    >>> parse_example_id("example_04_correct")
    ('example_04', 'correct')
    >>> parse_example_id("example_04_incorrect_opaque")
    ('example_04', 'opaque')
    """
    match = _EXAMPLE_ID_PATTERN.match(example_id)
    if match is None:
        raise ValueError(
            f"example_id {example_id!r} does not match expected pattern "
            f"'example_{{NN}}_{{correct|incorrect_transparent|incorrect_opaque}}'"
        )
    base_example = match.group(1)
    raw_suffix = match.group(2)
    return base_example, _CONDITION_SUFFIXES[raw_suffix]


def ground_truth_for_condition(condition: Condition) -> ScoreValue:
    """Return the deterministic ground truth score for a condition.

    The model acts as a binary classifier: is the grader's assessment
    correct? In the correct-draft condition the grader is wrong, so the
    ground truth is "incorrect". In both incorrect-draft conditions the
    grader is right, so the ground truth is "correct".
    """
    return _GROUND_TRUTH[condition]
