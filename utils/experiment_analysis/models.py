"""Data models for the experiment analysis layer.

Defines the two primary data structures (TrialOutcome for arbitrary-dimension
slicing, CellAccumulator for sequential per-cell accumulation) plus their
supporting types. All models are frozen to prevent accidental mutation.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from utils.batch_inference.processed import TokenUsage

# Re-export from ground_truth to keep type aliases co-located with models
from utils.experiment_analysis.ground_truth import Condition, ScoreValue


class AccumulationParams(BaseModel):
    """Bundled parameters for the e-value accumulation fold.

    These come from the experiment plan and are constant across the
    entire pipeline run. Bundled here to avoid threading five separate
    arguments through every function call.
    """

    model_config = ConfigDict(frozen=True)

    p_null: float = 0.5
    alpha: float = 0.05
    futility_bound: float = 0.05
    seed_alternative: float = 0.7
    clamp_epsilon: float = 0.01


class TrialOutcome(BaseModel):
    """One trial with all experimental design coordinates attached.

    The flat, enriched per-trial record. Every trial in the experiment
    becomes one TrialOutcome — the "tidy data" table, groupable by any
    dimension for any slicing question.
    """

    model_config = ConfigDict(frozen=True)

    # Identity (from ProcessedResult)
    custom_id: str
    example_id: str  # e.g. "example_04_incorrect_opaque"
    trial: int

    # Experimental design coordinates (derived from example_id)
    base_example: str  # e.g. "example_04"
    condition: Condition  # "correct" | "transparent" | "opaque"

    # Prediction vs. ground truth
    predicted_score: ScoreValue | None
    ground_truth_score: ScoreValue
    matches_ground_truth: bool | None
    rationale: str | None  # meta-evaluator's stated justification; None on parse failure

    # Configuration identity (derived from ProcessedJob + config registry)
    config_key: str  # e.g. "openai--gpt-5.2--xhigh"
    provider: str
    model_slug: str
    reasoning_effort_level: str

    # Temporal position (per-config, from manifest job_ids list position)
    batch_index: int  # 0-indexed position within config's job sequence

    # Operational (for token usage and thinking rate metrics)
    usage: TokenUsage
    has_reasoning_trace: bool
    parse_error: str | None


class CellKey(BaseModel):
    """Hashable identity for one cell (one stimulus × one configuration)."""

    model_config = ConfigDict(frozen=True)

    example_id: str  # "example_04_incorrect_opaque"
    config_key: str  # "openai--gpt-5.2--xhigh"


class IterationSlice(BaseModel):
    """Audit record for one batch's contribution to a cell's e-value.

    Records the alternative used (determined from prior batches), the
    observed data, and the resulting e-value. Walking a cell's slices
    verifies every step of the sequential validity chain.
    """

    model_config = ConfigDict(frozen=True)

    batch_index: int
    trials: int  # total trials this batch (including parse failures)
    valid: int  # trials with non-None predicted_score
    hits: int  # trials where matches_ground_truth is True
    alternative_used: float  # p_alt from all prior batches' data
    e_value: float  # this batch's e-value (before multiplication)
    log_e_value: float  # log of this batch's e-value; primary quantity


class CellAccumulator(BaseModel):
    """Sequential state for one cell's e-value accumulation.

    Running state per cell across all batches processed so far. ``status``
    is a pure function of the running log e-value, recomputed on every
    fold — it reflects the current evidence and may move in any direction
    as subsequent batches are accumulated (optional continuation).
    ``resolved_at_batch`` records the first threshold crossing as a
    set-once audit record, while counts, slices, and e-values reflect
    all data seen so far.
    """

    model_config = ConfigDict(frozen=True)

    cell_key: CellKey
    condition: Condition
    ground_truth_score: ScoreValue

    # Accumulated totals
    total_trials: int
    valid_trials: int
    hit_count: int
    parse_failures: int

    # Per-batch audit trail
    slices: tuple[IterationSlice, ...]

    # E-value state (log space is primary; raw is derived convenience)
    running_log_e_value: float
    running_e_value: float
    next_alternative: float  # p_alt for the next batch

    status: Literal["active", "rejected", "futile"]
    resolved_at_batch: int | None  # batch_index when threshold first crossed; None if never crossed

    # Mean log e-value across batches; positive ⇒ evidence growing
    empirical_e_power: float


class CellMetrics(BaseModel):
    """Reportable metrics for one cell — what goes into tables and plots."""

    model_config = ConfigDict(frozen=True)

    cell_key: CellKey
    condition: Condition
    accuracy: float
    cs_lower: float  # confidence sequence lower bound (anytime-valid)
    cs_upper: float  # confidence sequence upper bound (anytime-valid)
    parse_failure_rate: float
    e_value: float
    log_e_value: float
    empirical_e_power: float
    status: Literal["active", "rejected", "futile"]
    batches_used: int
    total_trials: int  # total trials accumulated (including parse failures)
    valid_trials: int  # trials with parseable output (denominator for accuracy)
    resolved_at_batch: int | None  # batch where threshold first crossed; None if never crossed
