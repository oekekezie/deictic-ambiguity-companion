"""Configuration identity — deriving, validating, and positioning config keys.

The config key is the canonical identifier for one model at one reasoning
level, following the filename convention from generate_experiment_configs.py:

    {provider}--{model_slug}--{reasoning_effort_level}

This module provides:
- resolve_config_key: derive a config key from a polymorphic LLMConfig
- parse_config_key: decompose a config key back into its parts
- resolve_compute_position: map a config key to its ordinal position
  within the model's reasoning scale
- config_tie_break_key: deterministic secondary ordering shared by the
  per-configuration tables and the score-ranked figures
- REASONING_SCALES: authoritative ordering of reasoning effort levels per model
- CANONICAL_REASONING_ORDER: global cross-model ordering for presentation
"""

from utils.batch_inference.llm_configs import LLMConfig
from utils.batch_inference.providers import extract_reasoning_effort_level

# Authoritative ordering of reasoning effort levels per model, lowest → highest
# compute. Single source of truth shared by both the experiment config
# generator and the analysis layer.
REASONING_SCALES: dict[tuple[str, str], list[str]] = {
    ("openai", "gpt-5.2"): ["none", "low", "medium", "high", "xhigh"],
    ("openai", "gpt-5-mini"): ["minimal", "low", "medium", "high"],
    ("anthropic", "claude-opus-4-6"): ["low", "medium", "high", "max"],
    ("anthropic", "claude-haiku-4-5"): ["off", "enabled"],
    ("gemini", "gemini-3-flash-preview"): ["minimal", "low", "medium", "high"],
    # EXCLUDED: persistent batch API overload — all submitted jobs expired
    # without results. Retained for infrastructure compatibility.
    ("gemini", "gemini-3.1-pro-preview"): ["low", "medium", "high"],
    # DEPRECATED: discontinued by Google effective March 9, 2026.
    # Retained for existing experiment config compatibility.
    ("gemini", "gemini-3-pro-preview"): ["low", "high"],
}

# Global ordering of reasoning effort levels from lowest to highest compute,
# spanning all model families. This is a design decision about the
# relative positioning of provider-specific levels (e.g., where Haiku's
# "enabled" sits relative to "low"). Must include every level that
# appears in any REASONING_SCALES entry.
CANONICAL_REASONING_ORDER: list[str] = [
    "off", "none", "minimal", "enabled",
    "low", "medium", "high", "max", "xhigh",
]

# Within-provider model size pairs: (smaller_model_slug, larger_model_slug).
# Each slug must appear as the second element of a REASONING_SCALES key.
# Used for the model size comparison e-value tests, which pair every
# reasoning effort level of a provider's larger model with every
# configuration of its smaller model — no configuration is selected.
MODEL_SIZE_PAIRS: dict[str, tuple[str, str]] = {
    "openai": ("gpt-5-mini", "gpt-5.2"),
    "anthropic": ("claude-haiku-4-5", "claude-opus-4-6"),
    "gemini": ("gemini-3-flash-preview", "gemini-3-pro-preview"),
}

# Human-readable display names for model API slugs.
# Consumed by the display_* helpers below to produce publication-quality
# labels from internal config keys.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-5.2": "GPT-5.2",
    "gpt-5-mini": "GPT-5-mini",
}

# Human-readable display names for provider keys.
# Maps the internal provider identifier to its canonical display form.
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "gemini": "Google",
}

# The model that helped author the stimuli. ``paper/draft.md`` states that it
# "is therefore flagged with a dagger", and that promise covers every artifact
# naming a model, so the fact lives here beside the display names rather than
# in a notebook: the figure export path has no access to a notebook constant,
# which is why figures went unmarked while tables carried the dagger.
AUTHORING_MODEL_SLUG: str = "claude-opus-4-6"


def is_authoring_model(model_slug: str) -> bool:
    """Whether this model helped author the stimuli."""
    return model_slug == AUTHORING_MODEL_SLUG


def mark_authoring(label: str, *, is_authoring: bool) -> str:
    """Append the authoring model dagger to a reader-facing label.

    The dagger is closed up against the name, the form paper/CLAUDE.md
    prescribes (Opus 4.6\u2020), so a table cell, a legend entry, and the
    prose read identically. This is the single primitive for the annotation
    across every figure and table. Callers that hold only a slug get
    ``is_authoring`` from ``is_authoring_model``; those fed by the
    visualization data layer pass the flag it already carries.
    """
    return f"{label}\u2020" if is_authoring else label


def display_provider(provider: str) -> str:
    """Return the canonical reader-facing provider label.

    Strict: raises ``ValueError`` on unknown providers so leaks of raw
    identifiers into publication figures fail loudly rather than silently.
    """
    if provider not in PROVIDER_DISPLAY_NAMES:
        raise ValueError(f"Unknown provider for display: {provider!r}")
    return PROVIDER_DISPLAY_NAMES[provider]


def display_model_slug(model_slug: str) -> str:
    """Return the canonical reader-facing model label.

    Strict: raises ``ValueError`` on unknown slugs so leaks of raw
    identifiers into publication figures fail loudly rather than silently.
    """
    if model_slug not in MODEL_DISPLAY_NAMES:
        raise ValueError(f"Unknown model slug for display: {model_slug!r}")
    return MODEL_DISPLAY_NAMES[model_slug]


def display_reasoning_effort_level(level: str) -> str:
    """Return a reader-facing reasoning effort level label.

    Levels print under their provider's own names: OpenAI's "none" and
    Anthropic's "off" are distinct settings and are never merged into a
    shared display category.
    """
    return level


def display_config_key(config_key: str) -> str:
    """Format a config key for reader-facing chart and table labels.

    The authoring model dagger rides the model segment, because the mark
    belongs to the model rather than to the configuration: appended after
    the reasoning effort level it reads as a mark on the level.

    Strict alternative to the permissive ``.get(raw, raw)`` fallback
    pattern: malformed keys and unmapped slugs raise rather than passing
    through the raw identifier. Every call site feeds keys produced by
    ``resolve_config_key`` from the experiment matrix, so every input is
    guaranteed to be mapped in ``MODEL_DISPLAY_NAMES``.
    """
    _provider, model_slug, level = parse_config_key(config_key)
    marked_model = mark_authoring(
        display_model_slug(model_slug), is_authoring=is_authoring_model(model_slug)
    )
    return (
        f"{marked_model} / "
        f"{display_reasoning_effort_level(level)}"
    )


def display_reasoning_transition(
    model_slug: str,
    from_level: str,
    to_level: str,
) -> str:
    """Format a reasoning effort transition label without symbolic arrows.

    Produces labels like ``Claude Opus 4.6: medium to high``. Uses natural
    English ``to`` rather than ``→`` per the paper's terminological
    conventions, and each level prints under its provider's own name.
    """
    from_label = display_reasoning_effort_level(from_level)
    to_label = display_reasoning_effort_level(to_level)
    marked_model = mark_authoring(
        display_model_slug(model_slug), is_authoring=is_authoring_model(model_slug)
    )
    return f"{marked_model}: {from_label} to {to_label}"


def resolve_config_key(llm_config: LLMConfig) -> str:
    """Derive the canonical config key from an LLMConfig.

    The config key combines provider, model identifier, and reasoning
    level into a single string that uniquely identifies the configuration
    within the experiment matrix.

    >>> from utils.batch_inference.llm_configs.openai import OpenAILLMConfig
    >>> cfg = OpenAILLMConfig(model="gpt-5.2", reasoning_effort="xhigh")
    >>> resolve_config_key(cfg)
    'openai--gpt-5.2--xhigh'
    """
    level = extract_reasoning_effort_level(llm_config)
    reasoning_label = level if level is not None else "off"
    return f"{llm_config.provider}--{llm_config.model}--{reasoning_label}"


def parse_config_key(config_key: str) -> tuple[str, str, str]:
    """Decompose a config key into (provider, model_slug, reasoning_effort_level).

    >>> parse_config_key("openai--gpt-5.2--xhigh")
    ('openai', 'gpt-5.2', 'xhigh')
    """
    parts = config_key.split("--")
    if len(parts) != 3:
        raise ValueError(
            f"config_key must have exactly 3 '--'-delimited parts, "
            f"got {len(parts)}: {config_key!r}"
        )
    return parts[0], parts[1], parts[2]


def resolve_compute_position(config_key: str) -> tuple[int, int] | None:
    """Map a config key to its (ordinal, scale_size) within the model's reasoning scale.

    Returns the 0-indexed ordinal position and total number of levels,
    or None if the model or level is not in REASONING_SCALES (e.g., a
    model added after the scales were last updated).

    >>> resolve_compute_position("openai--gpt-5.2--high")
    (3, 5)
    >>> resolve_compute_position("anthropic--claude-haiku-4-5--off")
    (0, 2)
    """
    provider, model_slug, reasoning_effort_level = parse_config_key(config_key)
    scale = REASONING_SCALES.get((provider, model_slug))
    if scale is None or reasoning_effort_level not in scale:
        return None
    return scale.index(reasoning_effort_level), len(scale)


# Sentinel ordinal for configs whose model is not in REASONING_SCALES (e.g.
# ad-hoc additions in test fixtures). Pushes such configs to the end of any
# tie bucket while preserving deterministic ordering by config_key.
_UNKNOWN_COMPUTE_ORDINAL: int = 1_000_000


def config_tie_break_key(config_key: str) -> tuple[str, str, int, str]:
    """Deterministic secondary ordering for configurations.

    Composed after a primary metric key so equal scores resolve the same way
    everywhere: provider, then model, then ascending reasoning effort, with the
    raw key last so the ordering is total. Only the reasoning effort element
    carries published meaning (``REASONING_SCALES`` is the authoritative
    lowest-to-highest sequence); provider, model slug, and config key are
    alphabetical and exist purely to make ties reproducible.

    Used both on its own — to group per-configuration tables by model and read
    them from lowest to highest reasoning effort — and as the tail of the
    balanced accuracy ordering shared by the score-ranked figures.
    """
    provider, model_slug, _ = parse_config_key(config_key)
    position = resolve_compute_position(config_key)
    compute_ordinal = position[0] if position is not None else _UNKNOWN_COMPUTE_ORDINAL
    return (provider, model_slug, compute_ordinal, config_key)


def require_compute_position(config_key: str) -> tuple[int, int]:
    """Map a config key to its (ordinal, scale_size), raising if not found.

    All models in the experiment matrix have entries in REASONING_SCALES.
    A missing entry indicates stale scale definitions or a data integrity
    error — both should surface immediately.
    """
    pos = resolve_compute_position(config_key)
    if pos is None:
        raise ValueError(
            f"No reasoning scale entry for {config_key!r}; "
            f"update REASONING_SCALES in config_identity.py"
        )
    return pos
