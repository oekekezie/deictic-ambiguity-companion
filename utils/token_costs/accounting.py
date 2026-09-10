"""Provider exceptions in how reported usage fields relate to one another.

Two independent relationships govern what a provider's token counts
already include. Each set below names the providers that deviate from the
common convention; every other provider follows it. The two are
orthogonal, and no provider is exceptional on both axes: Anthropic
deviates on the input axis alone, Gemini on the output axis alone.

Both billing (``pricing.py``) and response token derivation
(``calculation.py``) read these sets, so the two cannot disagree about
what a field contains.
"""

INPUT_EXCLUDES_CACHE_PROVIDERS: frozenset[str] = frozenset({"anthropic"})
"""Providers whose ``input_tokens`` excludes the cache fields.

Anthropic reports ``input_tokens`` net of cache reads and writes, so the
full prompt context is ``input_tokens + cache_read + cache_write`` and
``input_tokens`` is already the uncached portion. Every other provider
reports ``input_tokens`` as the total, from which cache reads subtract.
"""

OUTPUT_EXCLUDES_REASONING_PROVIDERS: frozenset[str] = frozenset({"gemini"})
"""Providers whose ``output_tokens`` excludes reasoning tokens.

Gemini reports ``candidatesTokenCount`` as ``output_tokens`` and its
thinking tokens separately, so reasoning must be added back both to bill
it (Google charges thinking at the output rate) and to recover what the
response alone cost. Every other provider folds reasoning into
``output_tokens``, where it is already billed and must be subtracted to
isolate the response.
"""
