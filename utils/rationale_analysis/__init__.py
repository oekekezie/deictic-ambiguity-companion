"""Rationale analysis layer for the Reference Frame Tracking Benchmark.

Classifies LLM-generated meta-evaluator rationales on seven binary flags —
five misattribution flags (whether the rationale misreports a fact of the
scenario) and two deictic flags (``articulated_operational_interpretation``
and ``operational_interpretation_governed_judgment``). The flags are
produced per trial via K independent analyst calls with per-flag
aggregation policies.
"""

from utils.rationale_analysis.aggregation import (
    aggregate_repetitions,
    build_rationale_analysis_results,
)
from utils.rationale_analysis.assembly import assemble_rationale_analysis_dataset
from utils.rationale_analysis.custom_id import (
    encode_rationale_analysis_custom_id,
    parse_rationale_analysis_custom_id,
)
from utils.rationale_analysis.eligibility import filter_eligible_trials
from utils.rationale_analysis.loading import load_rationale_analysis_jobs
from utils.rationale_analysis.models import (
    MISATTRIBUTION_FLAG_KEYS,
    RATIONALE_ANALYSIS_FLAG_DISPLAY_NAMES,
    RATIONALE_ANALYSIS_FLAG_KEYS,
    RationaleAnalysisAssemblyMetadata,
    RationaleAnalysisEligibilityResult,
    RationaleAnalysisExcludedTrial,
    RationaleAnalysisFlagResult,
    RationaleAnalysisManifestProvenance,
    RationaleAnalysisParsedResponse,
    RationaleAnalysisProvenance,
    RationaleAnalysisResults,
    RationaleAnalysisTrialRecord,
)
from utils.rationale_analysis.provenance import (
    build_snapshot_provenance,
    resolve_assembly_metadata_path,
)
from utils.rationale_analysis.response_parser import parse_rationale_analysis_response
from utils.rationale_analysis.template import (
    extract_grader_feedback,
    ground_truth_description,
    populate_user_prompt,
)

__all__ = [
    # Flag keys and display names
    "RATIONALE_ANALYSIS_FLAG_KEYS",
    "MISATTRIBUTION_FLAG_KEYS",
    "RATIONALE_ANALYSIS_FLAG_DISPLAY_NAMES",
    # Data models
    "RationaleAnalysisAssemblyMetadata",
    "RationaleAnalysisEligibilityResult",
    "RationaleAnalysisExcludedTrial",
    "RationaleAnalysisFlagResult",
    "RationaleAnalysisManifestProvenance",
    "RationaleAnalysisParsedResponse",
    "RationaleAnalysisProvenance",
    "RationaleAnalysisResults",
    "RationaleAnalysisTrialRecord",
    # Custom ID
    "encode_rationale_analysis_custom_id",
    "parse_rationale_analysis_custom_id",
    # Template
    "extract_grader_feedback",
    "ground_truth_description",
    "populate_user_prompt",
    # Eligibility
    "filter_eligible_trials",
    # Loading
    "load_rationale_analysis_jobs",
    # Response parsing
    "parse_rationale_analysis_response",
    # Aggregation
    "aggregate_repetitions",
    "build_rationale_analysis_results",
    # Provenance
    "build_snapshot_provenance",
    "resolve_assembly_metadata_path",
    # Assembly
    "assemble_rationale_analysis_dataset",
]
