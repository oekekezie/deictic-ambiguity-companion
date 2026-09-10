"""Stimulus metadata for the 10 base examples in the ReFT benchmark.

Loads benchmark design metadata (domain, risk, deictic values) from the
canonical JSON file and exposes it as validated Pydantic models keyed by
base_example identifier.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from utils.experiment_analysis.ground_truth import Condition

# Resolve the JSON file relative to the repository root (two levels up
# from this module: utils/experiment_analysis/ → utils/ → repo root).
_METADATA_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "synthetic_dataset"
    / "stimulus_metadata.json"
)


class StimulusMetadata(BaseModel):
    """Design metadata for a single base example.

    Captures the domain context, risk level, human-readable noun for the
    value under test, and the three deictic values that define each
    scenario's reference-frame trap.
    """

    model_config = ConfigDict(frozen=True)

    domain: str
    risk: str
    domain_noun: str
    current_value: str
    historical_value: str
    proposed_value: str


def _load_registry() -> dict[str, StimulusMetadata]:
    """Read and validate stimulus metadata from the canonical JSON file."""
    raw: dict[str, dict[str, str]] = json.loads(_METADATA_PATH.read_text())
    return {
        key: StimulusMetadata(**entry)
        for key, entry in raw.items()
    }


STIMULUS_METADATA_REGISTRY: dict[str, StimulusMetadata] = _load_registry()


def draft_fallback_value(condition: Condition, metadata: StimulusMetadata) -> str:
    """Return the appropriate fallback value based on the benchmark condition.

    In the correct-draft condition the assistant used the current value,
    so we surface that.  In both incorrect conditions (transparent and
    opaque) the assistant mistakenly used the historical value.
    """
    if condition == "correct":
        return metadata.current_value
    return metadata.historical_value
