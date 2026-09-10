"""Anytime-valid confidence sequences for Bernoulli proportions.

Beta-binomial mixture (test-)martingale construction from Howard, Ramdas,
McAuliffe & Sekhon (2021). For binary data the mixture of likelihood ratios
against a point null is a genuine test martingale (Ville, 1939), so inverting
it yields a confidence sequence with valid coverage at every sample size and
every (possibly data-dependent) stopping time.

The confidence sequence is dual to an anytime test: a value p is excluded from
the sequence exactly when the mixture test of that p has crossed 1/alpha. The
bounds therefore describe accuracy *and* test any null they exclude. (This is a
different test from the per-cell plug-in e-value used for stopping decisions, so
the two need not agree on borderline cells.)

Two-sided (default) mixes the alternative over all of (0, 1). One-sided mixes
only over the alternatives on the relevant side of the null, spending the full
error budget on a single bound -- the right tool when only one direction
matters (e.g. "is the rate at least X?").
"""

import math


def _log_beta(a: float, b: float) -> float:
    """log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, max_iter: int = 400, tiny: float = 1e-300,
            eps: float = 3e-14) -> float:
    """Continued fraction for the incomplete beta function (Lentz's algorithm).

    Standard Numerical-Recipes recurrence; converges fast when
    x < (a + 1) / (a + b + 2). Callers must enforce that branch.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _log_ibeta_tails(c: float, d: float, x: float) -> tuple[float, float]:
    """Return (log I_x(c, d), log(1 - I_x(c, d))) for the regularized
    incomplete beta function I_x, computed in log space.

    I_x(c, d) is the CDF at x of a Beta(c, d) random variable. The branch is
    chosen so the smaller (<= 0.5) tail is computed directly in log space,
    avoiding cancellation; the larger tail is recovered with log1p. This tracks
    tail probabilities far below the smallest positive double without underflow.
    """
    if x <= 0.0:
        return (float("-inf"), 0.0)
    if x >= 1.0:
        return (0.0, float("-inf"))
    log_pref = (
        math.lgamma(c + d) - math.lgamma(c) - math.lgamma(d)
        + c * math.log(x) + d * math.log1p(-x)
    )
    if x < (c + 1.0) / (c + d + 2.0):
        log_lower = log_pref + math.log(_betacf(c, d, x)) - math.log(c)
        if log_lower > 0.0:
            log_lower = 0.0
        log_upper = math.log1p(-math.exp(log_lower)) if log_lower < 0.0 else float("-inf")
        return (log_lower, log_upper)
    else:
        log_upper = log_pref + math.log(_betacf(d, c, 1.0 - x)) - math.log(d)
        if log_upper > 0.0:
            log_upper = 0.0
        log_lower = math.log1p(-math.exp(log_upper)) if log_upper < 0.0 else float("-inf")
        return (log_lower, log_upper)


def _log_mixture_martingale(p: float, hits: int, valid: int, a: float, b: float) -> float:
    """log of the TWO-SIDED mixture test martingale M_n(p).

    M_n(p) = B(k+a, n-k+b) / [B(a, b) * p^k * (1-p)^(n-k)],
    the Beta(a, b)-mixture of binomial likelihood ratios against the null p.
    """
    misses = valid - hits
    return (
        _log_beta(hits + a, misses + b)
        - _log_beta(a, b)
        - hits * math.log(p)
        - misses * math.log(1.0 - p)
    )


def _log_one_sided_martingale(p: float, hits: int, valid: int, a: float, b: float,
                              lower_cs: bool) -> float:
    """log of the ONE-SIDED mixture test martingale.

    Lower CS: mix the alternative only over q > p (the prior Beta(a, b)
    truncated to (p, 1) and renormalized). The renormalization makes this equal
    to the two-sided martingale times the ratio of posterior to prior mass above
    p:

        M_n^+(p) = M_n(p) * P_post(Q > p) / P_prior(Q > p),

    with P_post using Beta(k+a, n-k+b) and P_prior using Beta(a, b). This is a
    test martingale under H0: mean = p (a mixture of likelihood ratios over a
    fixed, data-independent prior), so Ville's inequality applies.

    Upper CS: symmetric, truncating to q < p (uses the lower tails).
    """
    base = _log_mixture_martingale(p, hits, valid, a, b)
    misses = valid - hits
    log_post_lower, log_post_upper = _log_ibeta_tails(hits + a, misses + b, p)
    log_prior_lower, log_prior_upper = _log_ibeta_tails(a, b, p)
    if lower_cs:
        return base + log_post_upper - log_prior_upper
    return base + log_post_lower - log_prior_lower


# --- two-sided bisections (search around p_hat, where M_n is minimized) -------


def _bisect_lower(hits: int, valid: int, a: float, b: float, log_threshold: float,
                  p_hat: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Lower two-sided CS bound: smallest p in (0, p_hat] with M_n(p) < 1/alpha."""
    lo, hi = tol, p_hat
    if _log_mixture_martingale(p_hat, hits, valid, a, b) >= log_threshold:
        return p_hat
    if _log_mixture_martingale(lo, hits, valid, a, b) < log_threshold:
        return 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        if _log_mixture_martingale(mid, hits, valid, a, b) >= log_threshold:
            lo = mid
        else:
            hi = mid
    return hi


def _bisect_upper(hits: int, valid: int, a: float, b: float, log_threshold: float,
                  p_hat: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Upper two-sided CS bound: largest p in [p_hat, 1) with M_n(p) < 1/alpha."""
    lo, hi = p_hat, 1.0 - tol
    if _log_mixture_martingale(p_hat, hits, valid, a, b) >= log_threshold:
        return p_hat
    if _log_mixture_martingale(hi, hits, valid, a, b) < log_threshold:
        return 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        if _log_mixture_martingale(mid, hits, valid, a, b) >= log_threshold:
            hi = mid
        else:
            lo = mid
    return lo


# --- one-sided bisections (single contiguous rejection region) ---------------


def _one_sided_lower_bound(hits: int, valid: int, a: float, b: float,
                           log_threshold: float, tol: float = 1e-12,
                           max_iter: int = 200) -> float:
    """One-sided lower CS bound L, where the CS is (L, 1].

    g(p) = log M_n^+(p) is NOT strictly monotone for concentrated priors (the
    truncation point moves with p), but the super-level set
    {p : g(p) >= threshold} is a single contiguous interval (0, L], so the
    bisection converges to the unique crossing L. Under the weak priors used in
    this codebase (Jeffreys) g is in fact monotone.
    """
    def g(p: float) -> float:
        return _log_one_sided_martingale(p, hits, valid, a, b, True)
    lo, hi = tol, 1.0 - tol
    if g(hi) >= log_threshold:
        return hi
    if g(lo) < log_threshold:
        return 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        if g(mid) >= log_threshold:
            lo = mid
        else:
            hi = mid
    return hi


def _one_sided_upper_bound(hits: int, valid: int, a: float, b: float,
                           log_threshold: float, tol: float = 1e-12,
                           max_iter: int = 200) -> float:
    """One-sided upper CS bound U, where the CS is [0, U).

    Symmetric to ``_one_sided_lower_bound``: g(p) = log M_n^-(p) is non-monotone
    for concentrated priors, but {p : g(p) >= threshold} is the single
    contiguous interval [U, 1), so the bisection finds the unique crossing U.
    """
    def g(p: float) -> float:
        return _log_one_sided_martingale(p, hits, valid, a, b, False)
    lo, hi = tol, 1.0 - tol
    if g(lo) >= log_threshold:
        return lo
    if g(hi) < log_threshold:
        return 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        if g(mid) >= log_threshold:
            hi = mid
        else:
            lo = mid
    return lo


def bernoulli_cs_bounds(
    hits: int,
    valid: int,
    *,
    alpha: float = 0.05,
    v_opt: float = 1.0,
    alpha_opt: float = 0.5,
    side: str = "two-sided",
) -> tuple[float, float]:
    """Anytime-valid confidence sequence (CS) bounds for a Bernoulli proportion.

    Inverts the beta-binomial mixture test martingale of Howard, Ramdas,
    McAuliffe & Sekhon (2021): the CS C_n = {p : M_n(p) < 1/alpha} satisfies
    P(p in C_t for all t >= 1) >= 1 - alpha, so coverage holds at every sample
    size and at any (possibly data-dependent) stopping time. Excluding a value
    from C_n is equivalent to the anytime mixture test of that value rejecting.

    Args:
        hits: Number of successes (0 <= hits <= valid).
        valid: Number of valid trials.
        alpha: Total miscoverage probability (default 0.05 for 95%). For a
            one-sided CS the entire budget is allocated to the single finite
            bound, which is exactly why that bound is tighter than the
            corresponding two-sided bound at the same alpha.
        v_opt: Prior concentration -- the Beta's effective prior sample size
            (a + b), in pseudo-observations. It controls how committed the prior
            is to its center alpha_opt, not a small-n vs. large-n width
            trade-off: a large v_opt makes the CS tight when the truth is near
            alpha_opt and markedly wider when it is not. The default of 1 (with
            alpha_opt = 0.5) is the Jeffreys prior Beta(0.5, 0.5), the standard
            weakly-informative choice that lets the data dominate; v_opt = 2 is
            the uniform Beta(1, 1). Use a strong prior (large v_opt) only when
            you genuinely expect the truth to lie near alpha_opt.
        alpha_opt: Prior mean in (0, 1) -- equivalently, the proportion at which
            the CS is tightest. This choice affects only the WIDTH of the CS,
            never its coverage (any full-support prior is valid).
        side: 'two-sided' (default) returns a two-sided CS [lower, upper].
            'lower' returns a one-sided lower CS (lower, 1.0), mixing the
            alternative only over proportions above the null and spending the
            full alpha on the lower bound. 'upper' returns (0.0, upper)
            symmetrically. One-sided is the right tool when only one direction
            matters -- e.g. "is the rate at least X?".

    Returns:
        (lower, upper). For side='lower' the upper element is 1.0; for
        side='upper' the lower element is 0.0. Returns (0.0, 1.0) when
        valid == 0 (no data).
    """
    if valid < 0:
        raise ValueError(f"valid must be non-negative, got {valid}")
    if hits < 0 or hits > valid:
        raise ValueError(f"hits must be in [0, valid], got hits={hits}, valid={valid}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if v_opt <= 0.0:
        raise ValueError(f"v_opt must be positive, got {v_opt}")
    if not (0.0 < alpha_opt < 1.0):
        raise ValueError(f"alpha_opt must be in (0, 1), got {alpha_opt}")
    if side not in ("two-sided", "lower", "upper"):
        raise ValueError(f"side must be 'two-sided', 'lower', or 'upper', got {side!r}")

    if valid == 0:
        return (0.0, 1.0)

    a = v_opt * alpha_opt
    b = v_opt * (1.0 - alpha_opt)
    log_threshold = math.log(1.0 / alpha)
    p_hat = hits / valid

    if side == "lower":
        if hits == 0:
            return (0.0, 1.0)
        return (_one_sided_lower_bound(hits, valid, a, b, log_threshold), 1.0)

    if side == "upper":
        if hits == valid:
            return (0.0, 1.0)
        return (0.0, _one_sided_upper_bound(hits, valid, a, b, log_threshold))

    # two-sided
    if hits == 0:
        return (0.0, _bisect_upper(hits, valid, a, b, log_threshold, 1e-12))
    if hits == valid:
        return (_bisect_lower(hits, valid, a, b, log_threshold, 1.0 - 1e-12), 1.0)
    lower = _bisect_lower(hits, valid, a, b, log_threshold, p_hat)
    upper = _bisect_upper(hits, valid, a, b, log_threshold, p_hat)
    return (lower, upper)
