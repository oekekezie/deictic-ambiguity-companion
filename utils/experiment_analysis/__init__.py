"""Experiment analysis layer for the Reference Frame Tracking Benchmark.

Transforms ProcessedJob artifacts from the batch inference infrastructure
into metrics, e-values, and stopping decisions for sequential experiment
monitoring and post-experiment reporting.
"""

from utils.experiment_analysis.accumulation import (
    accumulate_batch,
    fold_outcomes,
    make_initial_accumulator,
)
from utils.experiment_analysis.comparisons import (
    CONDITION_PAIRS,
    AdjacentPairDegradation,
    ComparisonSlice,
    ConditionPairResult,
    CrossDatasetComparison,
    GlobalNullComparison,
    LevelFold,
    ModelSizeComparison,
    PairwiseComparison,
    SmallerConfigurationFold,
    compute_adjacent_pair_degradation,
    compute_all_comparisons,
    compute_all_cross_dataset_comparisons,
    compute_all_degradation_tests,
    compute_all_global_null_comparisons,
    compute_all_model_size_tests,
    compute_global_null_comparison,
    compute_pairwise_comparison,
)
from utils.experiment_analysis.confidence_sequences import bernoulli_cs_bounds
from utils.experiment_analysis.config_identity import (
    CANONICAL_REASONING_ORDER,
    MODEL_SIZE_PAIRS,
    REASONING_SCALES,
    config_tie_break_key,
    parse_config_key,
    require_compute_position,
    resolve_compute_position,
    resolve_config_key,
)
from utils.experiment_analysis.e_values import (
    bernoulli_lr_log_e_value,
    clamp_alternative,
)
from utils.experiment_analysis.enrichment import enrich_job
from utils.experiment_analysis.ground_truth import (
    Condition,
    ScoreValue,
    ground_truth_for_condition,
    parse_example_id,
)
from utils.experiment_analysis.manifest import ExperimentManifest, ManifestEntry
from utils.experiment_analysis.metrics import (
    ConfigurationSummary,
    UsageCostSummary,
    cell_metrics,
    composite_metrics,
    condition_accuracy,
    configuration_summary,
    thinking_rate,
    usage_cost_summary,
)
from utils.experiment_analysis.models import (
    AccumulationParams,
    CellAccumulator,
    CellKey,
    CellMetrics,
    IterationSlice,
    TrialOutcome,
)
from utils.experiment_analysis.pipeline import (
    CrossDatasetAnalysisResults,
    ExperimentAnalysisResults,
    BatchReport,
    analyze_cross_dataset,
    analyze_experiment,
    find_dataset_jsonl,
    load_experiment,
    process_new_batch,
)
from utils.experiment_analysis.response_parser import (
    ExperimentParsedResponse,
    parse_experiment_response,
)
from utils.experiment_analysis.stimulus_loading import (
    load_stimuli,
    parse_stimuli_jsonl,
)
from utils.experiment_analysis.stimulus_metadata import (
    STIMULUS_METADATA_REGISTRY,
    StimulusMetadata,
    draft_fallback_value,
)
from utils.experiment_analysis.stimulus_parsing import parse_stimulus_sections

__all__ = [
    # Type aliases
    "Condition",
    "ScoreValue",
    # Data models
    "AccumulationParams",
    "AdjacentPairDegradation",
    "CrossDatasetAnalysisResults",
    "CrossDatasetComparison",
    "ModelSizeComparison",
    "ExperimentAnalysisResults",
    "BatchReport",
    "CellAccumulator",
    "CellKey",
    "CellMetrics",
    "ComparisonSlice",
    "ConditionPairResult",
    "ConfigurationSummary",
    "GlobalNullComparison",
    "LevelFold",
    "ExperimentManifest",
    "UsageCostSummary",
    "IterationSlice",
    "ManifestEntry",
    "PairwiseComparison",
    "SmallerConfigurationFold",
    "TrialOutcome",
    # Ground truth
    "ground_truth_for_condition",
    "parse_example_id",
    # Config identity
    "CANONICAL_REASONING_ORDER",
    "MODEL_SIZE_PAIRS",
    "REASONING_SCALES",
    "config_tie_break_key",
    "parse_config_key",
    "require_compute_position",
    "resolve_compute_position",
    "resolve_config_key",
    # E-values
    "bernoulli_cs_bounds",
    "bernoulli_lr_log_e_value",
    "clamp_alternative",
    # Response parsing
    "ExperimentParsedResponse",
    "parse_experiment_response",
    # Enrichment
    "enrich_job",
    # Accumulation
    "accumulate_batch",
    "fold_outcomes",
    "make_initial_accumulator",
    # Metrics
    "cell_metrics",
    "composite_metrics",
    "condition_accuracy",
    "configuration_summary",
    "thinking_rate",
    "usage_cost_summary",
    # Comparisons
    "CONDITION_PAIRS",
    "compute_adjacent_pair_degradation",
    "compute_all_comparisons",
    "compute_all_cross_dataset_comparisons",
    "compute_all_degradation_tests",
    "compute_all_global_null_comparisons",
    "compute_all_model_size_tests",
    "compute_global_null_comparison",
    "compute_pairwise_comparison",
    # Pipeline
    "analyze_cross_dataset",
    "analyze_experiment",
    "find_dataset_jsonl",
    "load_experiment",
    "process_new_batch",
    # Stimulus loading
    "load_stimuli",
    "parse_stimuli_jsonl",
    # Stimulus metadata
    "STIMULUS_METADATA_REGISTRY",
    "StimulusMetadata",
    "draft_fallback_value",
    # Stimulus parsing
    "parse_stimulus_sections",
]
