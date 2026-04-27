"""High-level convenience entry point: ``estimate_bid``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import config

from bid.distances import check_pm1, pairwise_hamming
from bid.histogram import (
    Histogram,
    empirical_histogram,
    empirical_moments,
    truncate_by_quantiles,
)
from bid.optimize import (
    BIDResult,
    finalize,
    init_state,
    initial_guess,
    minimize_kl,
)

config.update("jax_enable_x64", True)


def estimate_bid(
    X,
    *,
    alphamin: float = 0.0,
    alphamax: float = 1.0,
    delta: float = 5e-4,
    n_steps: int = 1_000_000,
    seed: int = 1,
    L: int | None = None,
    check_format: bool = True,
) -> BIDResult:
    """Estimate the Binary Intrinsic Dimension of a ±1 spin dataset.

    Args:
        X: array-like of shape (N, L) with entries in {-1, +1}.
        alphamin, alphamax: cumulative-probability cutoffs that define the
            distance window used for the fit. ``alphamin=0, alphamax=1``
            uses the full sampled support.
        delta: multiplicative step size for the random proposal in
            ``d0_proposed = d0 * (1 + delta * (U - 1/2))``.
        n_steps: number of optimisation iterations.
        seed: PRNG seed.
        L: number of bits per sample. When supplied, an L-aware grid of
            initial guesses is scanned (Erazo's heuristic); recommended.
        check_format: validate that X is in {-1, +1}. Set False to skip.

    Returns:
        ``BIDResult`` with ``d0`` (the BID), ``d1`` (slope), ``log_kl``,
        the fitted model probability vector, and the acceptance ratio.
    """
    X_j = jnp.asarray(X, dtype=jnp.int32)
    if check_format:
        check_pm1(X_j)
    if L is None:
        L = int(X_j.shape[1])

    D = pairwise_hamming(X_j)
    values, counts = empirical_histogram(D)
    full_mean, _ = empirical_moments(values, counts)
    hist: Histogram = truncate_by_quantiles(values, counts, alphamin, alphamax)

    d0_init, d1_init, _ = initial_guess(hist, full_mean=full_mean, L=L)
    state = init_state(seed=seed, d0=d0_init, d1=d1_init, hist=hist)
    state = minimize_kl(state, hist, delta=delta, n_steps=int(n_steps))
    return finalize(state, hist, n_steps=int(n_steps))
