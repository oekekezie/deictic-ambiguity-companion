"""JobRecord model for batch inference job metadata."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from utils.batch_inference.llm_configs import LLMConfig


class JobRecord(BaseModel):
    """Immutable metadata for a single batch inference job.

    Stores the complete LLM configuration used for the job, enabling full
    reproducibility. The `provider` and `model` properties derive their
    values from llm_config.

    The `job_dir` field stores the path relative to BATCH_INFERENCE_DIR
    (e.g., "jobs/openai/batch_abc123").

    When multiple providers are submitted together, all resulting JobRecords
    share the same `submission_group` UUID for filtering and bulk actions.

    Frozen to prevent accidental mutation. To produce an updated record,
    use ``record.model_copy(update={...})``.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str
    llm_config: LLMConfig
    status: Literal["submitted", "in_progress", "completed", "failed", "expired", "cancelled"]
    num_requests: int
    num_trials: int
    created_at: str
    completed_at: str | None = None
    provider_job_id: str | None = None
    error_message: str | None = None
    job_dir: str
    submission_group: str | None = None
    retry_of: str | None = None  # job_id of the original job this retries

    @property
    def provider(self) -> str:
        """Derive provider from llm_config."""
        return self.llm_config.provider

    @property
    def model(self) -> str:
        """Derive model from llm_config."""
        return self.llm_config.model
