"""SDK integration dispatch for batch inference job lifecycle.

Routes submit, check_status, cancel, and retrieve calls to the appropriate
provider module based on the LLMConfig type in the BatchConfig.
All dispatch functions are async — callers bridge via ``asyncio.run()``.
"""

from pathlib import Path
from typing import Any

from utils.batch_inference.batch_config import BatchConfig
from utils.batch_inference.llm_configs.anthropic import AnthropicLLMConfig
from utils.batch_inference.llm_configs.fireworks import FireworksLLMConfig
from utils.batch_inference.llm_configs.gemini import GeminiLLMConfig
from utils.batch_inference.llm_configs.openai import OpenAILLMConfig
from utils.batch_inference.sdk.anthropic import (
    cancel_anthropic,
    check_status_anthropic,
    retrieve_anthropic,
    submit_anthropic,
)
from utils.batch_inference.sdk.fireworks import (
    cancel_fireworks,
    check_status_fireworks,
    retrieve_fireworks,
    submit_fireworks,
)
from utils.batch_inference.sdk.gemini import (
    cancel_gemini,
    check_status_gemini,
    retrieve_gemini,
    submit_gemini,
)
from utils.batch_inference.sdk.normalize import normalize_status
from utils.batch_inference.sdk.openai import (
    cancel_openai,
    check_status_openai,
    retrieve_openai,
    submit_openai,
)


async def submit(config: BatchConfig, input_path: Path) -> str:
    """Submit a batch job to the appropriate provider.

    Returns the provider's native job identifier. Gemini requires the
    model at submission time; Fireworks requires the full LLM config
    to set inference parameters at the batch job level.
    """
    llm = config.llm_config
    if isinstance(llm, OpenAILLMConfig):
        return await submit_openai(input_path)
    if isinstance(llm, AnthropicLLMConfig):
        return await submit_anthropic(input_path)
    if isinstance(llm, GeminiLLMConfig):
        return await submit_gemini(input_path, model=llm.model)
    if isinstance(llm, FireworksLLMConfig):
        return await submit_fireworks(input_path, llm_config=llm)
    raise TypeError(f"No SDK integration for LLMConfig type: {type(llm).__name__}")


async def check_status(config: BatchConfig, job_id: str) -> dict[str, Any]:
    """Check the status of a batch job. Makes a single API call and returns immediately."""
    llm = config.llm_config
    if isinstance(llm, OpenAILLMConfig):
        return await check_status_openai(job_id)
    if isinstance(llm, AnthropicLLMConfig):
        return await check_status_anthropic(job_id)
    if isinstance(llm, GeminiLLMConfig):
        return await check_status_gemini(job_id)
    if isinstance(llm, FireworksLLMConfig):
        return await check_status_fireworks(job_id)
    raise TypeError(f"No SDK integration for LLMConfig type: {type(llm).__name__}")


async def cancel(config: BatchConfig, job_id: str) -> None:
    """Request cancellation of a batch job with the appropriate provider.

    Cancellation is asynchronous — most providers transition through an
    intermediate state (e.g., "cancelling") before reaching "cancelled".
    Callers should follow up with check_status to get the current state.
    """
    llm = config.llm_config
    if isinstance(llm, OpenAILLMConfig):
        return await cancel_openai(job_id)
    if isinstance(llm, AnthropicLLMConfig):
        return await cancel_anthropic(job_id)
    if isinstance(llm, GeminiLLMConfig):
        return await cancel_gemini(job_id)
    if isinstance(llm, FireworksLLMConfig):
        return await cancel_fireworks(job_id)
    raise TypeError(f"No SDK integration for LLMConfig type: {type(llm).__name__}")


async def retrieve(
    config: BatchConfig,
    job_id: str,
    output_path: Path,
    errors_path: Path | None = None,
) -> None:
    """Retrieve results from a completed batch job.

    Gemini embeds errors inline in results and does not produce a
    separate errors file, so errors_path is ignored for it.
    """
    llm = config.llm_config
    if isinstance(llm, OpenAILLMConfig):
        if errors_path is None:
            raise ValueError("errors_path is required for OpenAI retrieval")
        return await retrieve_openai(job_id, output_path, errors_path)
    if isinstance(llm, AnthropicLLMConfig):
        if errors_path is None:
            raise ValueError("errors_path is required for Anthropic retrieval")
        return await retrieve_anthropic(job_id, output_path, errors_path)
    if isinstance(llm, GeminiLLMConfig):
        return await retrieve_gemini(job_id, output_path)
    if isinstance(llm, FireworksLLMConfig):
        return await retrieve_fireworks(job_id, output_path, errors_path)
    raise TypeError(f"No SDK integration for LLMConfig type: {type(llm).__name__}")


__all__ = [
    "cancel",
    "check_status",
    "normalize_status",
    "retrieve",
    "submit",
]
