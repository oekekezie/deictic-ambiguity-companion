"""BatchConfig model for configuring batch inference jobs."""

from pydantic import BaseModel, field_validator

from utils.batch_inference.llm_configs import LLMConfig


class BatchConfig(BaseModel):
    """Top-level configuration for a batch inference job.

    Pairs a provider-specific LLM config with a trial count for
    repeated evaluations. Response schemas are set per-example only.
    """

    llm_config: LLMConfig
    num_trials: int

    @field_validator("num_trials")
    @classmethod
    def num_trials_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_trials must be a positive integer, got {v}")
        return v
