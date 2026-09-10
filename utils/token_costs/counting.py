"""Pre-flight input token estimation for batch inference.

Counts input tokens for an example given an LLM configuration,
dispatching to provider-specific tokenization methods. Returns
a worst-case count with all tokens assumed uncached.

Tokenization strategy by provider:
  - OpenAI:                  tiktoken (local, o200k_base encoding)
  - Anthropic:               API endpoint client.messages.count_tokens()
  - Gemini:                  API endpoint client.aio.models.count_tokens()
  - Fireworks GPT-OSS-120B:  tiktoken (local, o200k_base encoding)
  - Fireworks open-weight:   HuggingFace AutoTokenizer with apply_chat_template()

Anthropic and Gemini require their respective API keys.
OpenAI and Fireworks (tiktoken) require no API key.
Fireworks open-weight models download tokenizers on first use.

API-based providers (Anthropic, Gemini) include automatic concurrency
limiting (semaphore-gated) and retry with exponential backoff for
transient HTTP and SDK errors.
"""

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from functools import lru_cache

import anthropic
import httpx
import tiktoken
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
try:
    from transformers import AutoTokenizer
    _transformers_import_error: Exception | None = None
except Exception as exc:
    AutoTokenizer = None
    _transformers_import_error = exc

from utils.batch_inference.example import Example
from utils.batch_inference.llm_configs import LLMConfig
from utils.batch_inference.llm_configs.anthropic import AnthropicLLMConfig
from utils.batch_inference.llm_configs.fireworks import FireworksLLMConfig
from utils.batch_inference.llm_configs.gemini import GeminiLLMConfig
from utils.batch_inference.llm_configs.openai import OpenAILLMConfig
from utils.batch_inference.schema_utils import find_model_family, resolve_schema

_logger = logging.getLogger(__name__)

# HuggingFace model IDs for Fireworks open-weight models.
# Maps model family (from PROVIDER_MODELS display labels) to HF repo.
_FIREWORKS_HF_TOKENIZERS: dict[str, str] = {
    "kimi-k2p5": "moonshotai/Kimi-K2.5",
    "kimi-k2p6": "moonshotai/Kimi-K2.6",
    "glm-4p7": "zai-org/GLM-4.7",
    "glm-5": "zai-org/GLM-5",
}

# Fireworks model families by tokenizer type
_FIREWORKS_TIKTOKEN_FAMILIES: frozenset[str] = frozenset({"gpt-oss-120b"})
_FIREWORKS_HF_FAMILIES: frozenset[str] = frozenset({"kimi-k2p5", "kimi-k2p6", "glm-4p7", "glm-5"})

# Per-message overhead for OpenAI/Fireworks chat format:
# each message incurs ~4 tokens (role delimiter, separators, etc.)
_CHAT_MSG_OVERHEAD: int = 4
# Assistant reply priming adds ~2 tokens
_CHAT_REPLY_OVERHEAD: int = 2


# -- Cached resource loaders --------------------------------------------------


@lru_cache(maxsize=4)
def _get_tiktoken_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Cached tiktoken encoding lookup."""
    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=4)
def _get_hf_tokenizer(model_id: str):
    """Cached HuggingFace tokenizer lookup — Fireworks open-weight models only."""
    if AutoTokenizer is None:
        # Surface the original failure (e.g. CUDA RuntimeError, corrupted OSError)
        # so the user sees the root cause, not just "install transformers"
        msg = (
            "transformers is required for Fireworks open-weight token counting "
            "but failed to import"
        )
        if _transformers_import_error is not None:
            msg += f": {_transformers_import_error}"
        msg += ". Install with: uv add transformers"
        raise ImportError(msg)
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


@lru_cache(maxsize=1)
def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    """Cached async Anthropic client with SDK retries disabled."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required "
            "for Anthropic token counting"
        )
    # Disable SDK-level retries so _retry_transient is the single retry layer.
    return anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)


@lru_cache(maxsize=1)
def _get_gemini_client() -> genai.Client:
    """Cached Gemini client with SDK retries disabled."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required "
            "for Gemini token counting"
        )
    # attempts=1 means one total attempt (zero internal retries).
    retry_options = genai_types.HttpRetryOptions(attempts=1)
    http_options = genai_types.HttpOptions(retry_options=retry_options)
    return genai.Client(api_key=api_key, http_options=http_options)


# -- Concurrency limiting and retry -------------------------------------------

# Semaphores are created lazily because asyncio.Semaphore requires a running
# event loop at construction time. The dict is keyed by provider name.
_semaphores: dict[str, asyncio.Semaphore] = {}

# Per-provider caps on concurrent API calls. Conservative defaults that keep
# connection counts manageable while still providing meaningful parallelism.
_MAX_CONCURRENT_ANTHROPIC: int = 10
_MAX_CONCURRENT_GEMINI: int = 10


def _get_semaphore(provider: str, max_concurrent: int) -> asyncio.Semaphore:
    """Get or create a named semaphore for concurrent API call limiting."""
    if provider not in _semaphores:
        _semaphores[provider] = asyncio.Semaphore(max_concurrent)
    return _semaphores[provider]


# Retry configuration for transient HTTP and SDK errors
_MAX_RETRIES: int = 4
_BASE_DELAY_SECONDS: float = 1.0

# Transport-level errors from httpx (surface through both SDKs)
_RETRYABLE_HTTPX_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.PoolTimeout,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
)

# Anthropic SDK errors that indicate transient issues
_RETRYABLE_ANTHROPIC_ERRORS: tuple[type[Exception], ...] = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

# Gemini SDK errors: ServerError (5xx) is always transient; ClientError (4xx)
# is included to catch 429 rate limits, but filtered by _is_retryable_gemini
# to avoid retrying permanent errors (400, 401, 403, 404, etc.).
_RETRYABLE_GEMINI_ERRORS: tuple[type[Exception], ...] = (
    genai_errors.ServerError,
    genai_errors.ClientError,
)

# HTTP status codes that indicate transient issues worth retrying
# (matches the google-genai SDK's own internal retry set in _api_client.py)
_GEMINI_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429})


def _is_retryable_gemini(exc: Exception) -> bool:
    """Filter Gemini ClientErrors to only retry transient status codes.

    ServerError (5xx) and httpx transport errors are always retryable.
    ClientError (4xx) is only retryable for 408 (timeout) and 429 (rate limit).
    """
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", 0) in _GEMINI_RETRYABLE_STATUS_CODES
    return True


async def _retry_transient[T](
    fn: Callable[[], Awaitable[T]],
    retryable_errors: tuple[type[Exception], ...],
    description: str,
    *,
    is_retryable: Callable[[Exception], bool] | None = None,
    max_retries: int = _MAX_RETRIES,
    base_delay: float = _BASE_DELAY_SECONDS,
) -> T:
    """Retry an async callable with exponential backoff on transient errors.

    Retries on connection-level failures (TLS handshake, server disconnect,
    timeouts) and provider-specific SDK errors (rate limits, server errors).
    Non-retryable exceptions propagate immediately.

    The optional is_retryable predicate provides fine-grained filtering
    within the caught exception types (e.g., inspecting HTTP status codes
    on Gemini ClientError to distinguish 429 from 400).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retryable_errors as exc:
            # Fine-grained filter: re-raise immediately if predicate says
            # this specific exception instance is not worth retrying
            if is_retryable is not None and not is_retryable(exc):
                raise
            last_exc = exc
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            _logger.warning(
                "%s (attempt %d/%d): %s — retrying in %.1fs",
                description, attempt + 1, max_retries + 1,
                type(exc).__name__, delay,
            )
            await asyncio.sleep(delay)
    # Unreachable: the loop either returns or re-raises on the final attempt
    raise RuntimeError(f"Exhausted retries for {description}") from last_exc


# -- Message construction helpers ----------------------------------------------


def _build_chat_messages(
    example: Example,
    system_role: str,
) -> list[dict[str, str]]:
    """Build chat-format messages from an example.

    system_role is the role name for the system prompt: "developer" for
    OpenAI (Responses API), "system" for Fireworks (OpenAI-compatible).
    """
    messages: list[dict[str, str]] = []
    if example.system_prompt is not None:
        messages.append({"role": system_role, "content": example.system_prompt})
    messages.extend(
        {"role": m.role, "content": m.content} for m in example.messages
    )
    return messages


def _schema_json(example: Example) -> str | None:
    """Serialize the response schema to JSON for token overhead estimation."""
    if example.response_schema is None:
        return None
    return json.dumps(resolve_schema(example.response_schema))


# -- Provider-specific counting ------------------------------------------------


def _count_tiktoken(
    messages: list[dict[str, str]],
    schema_json: str | None,
) -> int:
    """Count tokens using tiktoken o200k_base encoding.

    Applies per-message overhead for the chat framing that surrounds
    each message's text content. Schema JSON is counted separately
    as additional input overhead.
    """
    enc = _get_tiktoken_encoding("o200k_base")
    total = 0
    for msg in messages:
        total += _CHAT_MSG_OVERHEAD
        total += len(enc.encode(msg["content"]))
        total += len(enc.encode(msg["role"]))
    # Reply priming tokens
    total += _CHAT_REPLY_OVERHEAD
    # Schema overhead (serialized JSON adds to the system-level context)
    if schema_json is not None:
        total += len(enc.encode(schema_json))
    return total


async def _count_openai(config: OpenAILLMConfig, example: Example) -> int:
    """Count input tokens for OpenAI using tiktoken (local, no API key)."""
    messages = _build_chat_messages(example, system_role="developer")
    return _count_tiktoken(messages, _schema_json(example))


async def _count_anthropic(config: AnthropicLLMConfig, example: Example) -> int:
    """Count input tokens for Anthropic via the count_tokens API endpoint.

    Mirrors the messages.create signature: model, messages, optional
    system prompt, and optional output_config (for response schema).
    Uses a shared client (connection pooling), semaphore-gated concurrency,
    and retry with exponential backoff for transient errors.
    """
    client = _get_anthropic_client()
    sem = _get_semaphore("anthropic", _MAX_CONCURRENT_ANTHROPIC)

    messages = [{"role": m.role, "content": m.content} for m in example.messages]

    kwargs: dict = {
        "model": config.model,
        "messages": messages,
    }
    if example.system_prompt is not None:
        kwargs["system"] = example.system_prompt

    # Include response schema so token count matches the serialized request
    # (serializers/anthropic.py places it in params.output_config.format)
    if example.response_schema is not None:
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": resolve_schema(example.response_schema),
            }
        }

    async def _call() -> int:
        async with sem:
            result = await client.messages.count_tokens(**kwargs)
            return result.input_tokens

    return await _retry_transient(
        _call,
        _RETRYABLE_HTTPX_ERRORS + _RETRYABLE_ANTHROPIC_ERRORS,
        f"anthropic/{config.model}/{example.custom_id}",
    )


async def _count_gemini(config: GeminiLLMConfig, example: Example) -> int:
    """Count input tokens for Gemini via the count_tokens API endpoint.

    The Gemini API's count_tokens does not support system_instruction or
    generation_config in the config parameter (only Vertex AI does — see
    Google bug b/378952792). All countable text is included as content
    entries in a single call, producing an accurate approximation that is
    sufficient for pre-flight cost estimation.
    Uses a shared client (connection pooling), semaphore-gated concurrency,
    and retry with exponential backoff for transient errors.
    """
    client = _get_gemini_client()
    sem = _get_semaphore("gemini", _MAX_CONCURRENT_GEMINI)

    # Gemini API's count_tokens rejects system_instruction and
    # generation_config in the config parameter, so system prompt
    # and response schema are folded into contents.
    contents: list[dict] = []

    if example.system_prompt is not None:
        contents.append(
            {"role": "user", "parts": [{"text": example.system_prompt}]}
        )

    contents.extend(
        {
            "role": "model" if m.role == "assistant" else "user",
            "parts": [{"text": m.content}],
        }
        for m in example.messages
    )

    if example.response_schema is not None:
        schema_str = json.dumps(resolve_schema(example.response_schema))
        contents.append(
            {"role": "user", "parts": [{"text": schema_str}]}
        )

    async def _call() -> int:
        async with sem:
            result = await client.aio.models.count_tokens(
                model=config.model,
                contents=contents,
            )
            return result.total_tokens

    return await _retry_transient(
        _call,
        _RETRYABLE_HTTPX_ERRORS + _RETRYABLE_GEMINI_ERRORS,
        f"gemini/{config.model}/{example.custom_id}",
        is_retryable=_is_retryable_gemini,
    )


async def _count_fireworks(config: FireworksLLMConfig, example: Example) -> int:
    """Count input tokens for Fireworks models.

    GPT-OSS-120B uses tiktoken (same o200k_base encoding as OpenAI).
    Open-weight models (Kimi K2.5, GLM-4.7, GLM-5) use HuggingFace
    AutoTokenizer with apply_chat_template().
    """
    model_id = config.model

    # GPT-OSS-120B uses the same tokenizer as OpenAI models
    family = find_model_family(model_id, _FIREWORKS_TIKTOKEN_FAMILIES)
    if family is not None:
        messages = _build_chat_messages(example, system_role="system")
        return _count_tiktoken(messages, _schema_json(example))

    # Open-weight models use HuggingFace tokenizers
    family = find_model_family(model_id, _FIREWORKS_HF_FAMILIES)
    if family is None:
        raise ValueError(f"No tokenizer mapping for Fireworks model: {model_id!r}")

    hf_model_id = _FIREWORKS_HF_TOKENIZERS[family]
    tokenizer = _get_hf_tokenizer(hf_model_id)
    messages = _build_chat_messages(example, system_role="system")

    # apply_chat_template tokenizes messages with the model's chat format.
    # return_dict=False is required for transformers 5.x, which defaults to
    # returning BatchEncoding (a dict) instead of list[int].
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)

    # Add schema overhead (counted separately since apply_chat_template
    # doesn't include response format instructions)
    schema_str = _schema_json(example)
    if schema_str is not None:
        token_ids_schema = tokenizer.encode(schema_str)
        return len(token_ids) + len(token_ids_schema)

    return len(token_ids)


# -- Fireworks response token counting -----------------------------------------


def count_fireworks_response_tokens(model: str, text: str) -> int:
    """Count tokens in a response text string for a Fireworks model.

    Used to derive reasoning_tokens for Kimi-family reasoning jobs, where
    completion_tokens covers reasoning + response: reasoning is recovered as
    output_tokens - count_fireworks_response_tokens(model, raw_response_text).

    GPT-OSS-120B uses tiktoken (o200k_base); open-weight models (Kimi, GLM)
    use the HuggingFace tokenizer. No chat-template framing is applied — the
    input is a bare response string, not a prompt.
    """
    if find_model_family(model, _FIREWORKS_TIKTOKEN_FAMILIES) is not None:
        return len(_get_tiktoken_encoding("o200k_base").encode(text))

    hf_family = find_model_family(model, _FIREWORKS_HF_FAMILIES)
    if hf_family is None:
        raise ValueError(f"No tokenizer mapping for Fireworks model: {model!r}")

    tokenizer = _get_hf_tokenizer(_FIREWORKS_HF_TOKENIZERS[hf_family])
    return len(tokenizer.encode(text))


# -- Anthropic response token counting -----------------------------------------

# Single-character probe for measuring framing overhead. Anthropic's
# count_tokens API rejects empty content strings (400 Bad Request), so we
# send a minimal non-empty token and subtract 1 from the measured result.
_PROBE_TOKEN: str = "."

# Per-model framing baseline: the token overhead from wrapping text in a
# synthetic user message (role delimiters, reply priming, etc.). Measured
# once per model via count_tokens with a minimal probe token.
_anthropic_framing_baseline: dict[str, int] = {}

# Per-model locks prevent the thundering herd problem: when asyncio.gather
# launches N concurrent coroutines, only the first to acquire the lock makes
# the baseline API call; the rest await the lock and read the cached result.
_anthropic_framing_locks: dict[str, asyncio.Lock] = {}

# Per-model cache of baseline measurement failures (circuit breaker). Once a
# model's baseline call fails with a non-retryable error, subsequent
# coroutines re-raise the cached exception immediately instead of repeating
# the failing API call.
_anthropic_framing_errors: dict[str, Exception] = {}


async def _get_anthropic_framing_baseline(model: str) -> int:
    """Measure and cache the framing overhead for a model's count_tokens call.

    Sends a single-token probe message (_PROBE_TOKEN = 1 token) and
    subtracts 1 to isolate the fixed token overhead (role delimiters,
    message framing, reply priming). Uses a probe token because
    Anthropic's count_tokens API rejects empty content strings with
    400 Bad Request.

    The result is cached per model so subsequent calls are free.

    Uses a per-model asyncio.Lock to prevent redundant API calls when
    many coroutines request the baseline concurrently (e.g., via
    asyncio.gather in _enrich_job). Persistent errors are cached in
    _anthropic_framing_errors to prevent N sequential failures when
    the baseline call is broken (circuit breaker).
    """
    # Fast path: already cached (no lock contention)
    if model in _anthropic_framing_baseline:
        return _anthropic_framing_baseline[model]

    # Circuit breaker: re-raise cached non-retryable errors immediately
    if model in _anthropic_framing_errors:
        raise _anthropic_framing_errors[model]

    # Lazy-create per-model lock (synchronous dict mutation is safe in asyncio)
    if model not in _anthropic_framing_locks:
        _anthropic_framing_locks[model] = asyncio.Lock()

    async with _anthropic_framing_locks[model]:
        # Re-check after acquiring lock: another coroutine may have populated it
        if model in _anthropic_framing_baseline:
            return _anthropic_framing_baseline[model]
        if model in _anthropic_framing_errors:
            raise _anthropic_framing_errors[model]

        client = _get_anthropic_client()
        sem = _get_semaphore("anthropic", _MAX_CONCURRENT_ANTHROPIC)

        async def _call() -> int:
            async with sem:
                result = await client.messages.count_tokens(
                    model=model,
                    messages=[{"role": "user", "content": _PROBE_TOKEN}],
                )
                return result.input_tokens

        try:
            probe_count = await _retry_transient(
                _call,
                _RETRYABLE_HTTPX_ERRORS + _RETRYABLE_ANTHROPIC_ERRORS,
                f"anthropic/{model}/framing_baseline",
            )
        except Exception as exc:
            _anthropic_framing_errors[model] = exc
            raise

        # Subtract the probe token itself to get pure framing overhead
        baseline = probe_count - 1
        _anthropic_framing_baseline[model] = baseline
        _logger.debug("Anthropic framing baseline for %s: %d tokens", model, baseline)
        return baseline


async def count_anthropic_response_tokens(model: str, text: str) -> int:
    """Count tokens in a response text string using Anthropic's count_tokens API.

    Wraps the text in a synthetic user message, calls count_tokens, and
    subtracts the framing overhead (measured once per model via a probe
    token) to isolate the pure text token count. Used to derive
    reasoning_tokens for extended-thinking jobs where
    output_tokens = thinking + response.

    Uses the shared Anthropic client, semaphore-gated concurrency, and
    retry with exponential backoff — same infra as _count_anthropic.
    """
    baseline = await _get_anthropic_framing_baseline(model)

    client = _get_anthropic_client()
    sem = _get_semaphore("anthropic", _MAX_CONCURRENT_ANTHROPIC)

    async def _call() -> int:
        async with sem:
            result = await client.messages.count_tokens(
                model=model,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens

    raw_count = await _retry_transient(
        _call,
        _RETRYABLE_HTTPX_ERRORS + _RETRYABLE_ANTHROPIC_ERRORS,
        f"anthropic/{model}/response_tokens",
    )
    token_count = raw_count - baseline
    if token_count < 0:
        raise ValueError(
            f"count_tokens returned fewer tokens for text than for "
            f"the probe-derived baseline (raw_count={raw_count}, "
            f"baseline={baseline}, model={model!r})"
        )
    return token_count


# -- Public dispatch -----------------------------------------------------------


async def count_input_tokens(config: LLMConfig, example: Example) -> int:
    """Count input tokens for an example given an LLM configuration.

    Dispatches to provider-specific tokenization:
      - OpenAI: tiktoken (local, o200k_base encoding)
      - Anthropic: API endpoint (requires ANTHROPIC_API_KEY)
      - Gemini: API endpoint (requires GOOGLE_API_KEY)
      - Fireworks: tiktoken for GPT-OSS-120B, HuggingFace for open-weight

    API-based providers (Anthropic, Gemini) include built-in concurrency
    limiting and retry with exponential backoff for transient errors.
    Returns a worst-case count with all tokens assumed uncached.
    """
    if isinstance(config, OpenAILLMConfig):
        return await _count_openai(config, example)
    if isinstance(config, AnthropicLLMConfig):
        return await _count_anthropic(config, example)
    if isinstance(config, GeminiLLMConfig):
        return await _count_gemini(config, example)
    if isinstance(config, FireworksLLMConfig):
        return await _count_fireworks(config, example)
    raise ValueError(f"Unsupported config type: {type(config).__name__}")
