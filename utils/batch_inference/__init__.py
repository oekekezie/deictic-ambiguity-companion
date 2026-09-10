"""Batch inference library for multi-provider LLM evaluation.

Provides configuration models, provider-specific serializers, SDK integration
for job lifecycle management, on-disk storage for job artifacts, and result
extraction / processing for standardized benchmark analysis.
"""

from utils.batch_inference.batch_config import BatchConfig
from utils.batch_inference.dataset import Dataset, parse_jsonl_bytes
from utils.batch_inference.example import Example
from utils.batch_inference.extractors import extract_result
from utils.batch_inference.heal import heal_processed_jobs
from utils.batch_inference.job import JobRecord
from utils.batch_inference.loading import (
    build_record_map,
    find_retry_companions,
    read_healed_job,
    read_processed_job,
)
from utils.batch_inference.llm_configs import (
    AnthropicLLMConfig,
    FireworksLLMConfig,
    GeminiLLMConfig,
    LLMConfig,
    OpenAILLMConfig,
)
from utils.batch_inference.message import Message
from utils.batch_inference.retry import (
    build_retry_dataset,
    filter_to_failed,
    identify_failed_custom_ids,
    load_original_inputs,
    summarize_failures,
)
from utils.batch_inference.processed import (
    ProcessedJob,
    ProcessedResult,
    TokenUsage,
    parse_custom_id,
)
from utils.batch_inference.schema_utils import find_model_family, resolve_schema
from utils.batch_inference.sdk.normalize import normalize_status
from utils.batch_inference.serializers import expand_trials, serialize
from utils.batch_inference.storage import (
    load_index,
    load_job,
    persist_job,
    prune_missing_jobs,
    resolve_job_dir,
    save_results,
    update_job_status,
)
from utils.batch_inference.types import ResponseSchema, Role
from utils.batch_inference.validation import validate_response

__all__ = [
    "AnthropicLLMConfig",
    "BatchConfig",
    "Dataset",
    "Example",
    "FireworksLLMConfig",
    "GeminiLLMConfig",
    "JobRecord",
    "LLMConfig",
    "Message",
    "OpenAILLMConfig",
    "ProcessedJob",
    "ProcessedResult",
    "ResponseSchema",
    "Role",
    "TokenUsage",
    "build_record_map",
    "build_retry_dataset",
    "expand_trials",
    "extract_result",
    "filter_to_failed",
    "find_model_family",
    "find_retry_companions",
    "heal_processed_jobs",
    "identify_failed_custom_ids",
    "load_index",
    "load_job",
    "load_original_inputs",
    "normalize_status",
    "parse_custom_id",
    "parse_jsonl_bytes",
    "persist_job",
    "prune_missing_jobs",
    "read_healed_job",
    "read_processed_job",
    "resolve_job_dir",
    "resolve_schema",
    "save_results",
    "serialize",
    "summarize_failures",
    "update_job_status",
    "validate_response",
]
