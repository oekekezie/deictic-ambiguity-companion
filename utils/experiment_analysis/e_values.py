"""E-value construction for per-cell Bernoulli outcomes.

Provides the log space Bernoulli likelihood ratio used as the per-cell
e-value and the clamping logic that keeps the adaptive plug-in alternative
inside the alternative hypothesis space. Both are folded over batches by
``accumulation.py``.

All functions are pure — no state, no side effects.

The cross-sample comparison families do not use this primitive. They bet on
paired differences between two samples; see ``comparisons.py``.
"""

import math


def bernoulli_lr_log_e_value(
    hits: int,
    valid: int,
    p_null: float,
    p_alt: float,
) -> float:
    """Compute the log e-value for a Bernoulli likelihood ratio test.

    For k hits in n valid trials with null proportion p₀ and
    alternative p₁:

        log E = k × log(p₁/p₀) + (n − k) × log((1−p₁)/(1−p₀))

    Log space computation avoids overflow when evidence is strong
    and underflow when evidence favors the null over many batches.

    The formula is direction-agnostic: it works for p_alt above or
    below p_null. The e-value grows when the observed hit rate is
    closer to p_alt than to p_null.

    Returns 0.0 (neutral evidence) when valid == 0.
    """
    if valid < 0:
        raise ValueError(f"valid must be non-negative, got {valid}")
    if hits < 0 or hits > valid:
        raise ValueError(
            f"hits must be in [0, valid], got hits={hits}, valid={valid}"
        )
    if not (0.0 < p_null < 1.0):
        raise ValueError(f"p_null must be in (0, 1), got {p_null}")
    if not (0.0 < p_alt < 1.0):
        raise ValueError(f"p_alt must be in (0, 1), got {p_alt}")

    if valid == 0:
        return 0.0

    misses = valid - hits
    log_ratio_hit = math.log(p_alt / p_null)
    log_ratio_miss = math.log((1.0 - p_alt) / (1.0 - p_null))
    return hits * log_ratio_hit + misses * log_ratio_miss


def clamp_alternative(
    p_hat: float,
    p_null: float,
    epsilon: float = 0.01,
) -> float:
    """Clamp an estimated accuracy into the alternative hypothesis space.

    Restricts the alternative to ``[p_null + epsilon, 1 − epsilon]``. The
    per-cell test is one-sided above chance — H₀: p ≤ p_null — so the
    alternative must lie strictly above the null. A below-null alternative
    would violate the e-process guarantee against the composite null
    (E_P[E] ≤ 1 for all P ∈ H₀).
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    lower = p_null + epsilon
    upper = 1.0 - epsilon

    if lower > upper:
        raise ValueError(
            f"No valid alternative range: [{lower}, {upper}] is empty. "
            f"Reduce epsilon or adjust p_null."
        )

    return max(lower, min(upper, p_hat))
