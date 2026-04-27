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
from bid.shells import (
    ShellFitResult,
    fit_conditional,
    fit_multinomial,
    shell_counts,
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


def estimate_bid_shells(
    X,
    *,
    k1: int | None = None,
    k2: int | None = None,
    d0_init: float | None = None,
    d1_init: float = 0.0,
    L: int | None = None,
    check_format: bool = True,
) -> ShellFitResult:
    """Estimate BID parameters from per-reference shell counts.

    Same generalised-binomial model as :func:`estimate_bid`, but fit by
    maximum likelihood on neighbour-shell statistics rather than by KL on
    the all-pairs distance histogram.

    When ``k1`` and ``k2`` are supplied, fits the I3D-style conditional
    likelihood at radii :math:`(k_1, k_2)` — robust to misspecification of
    :math:`d(r) = d_0 + d_1 r` outside the small-:math:`r` regime where
    :math:`d_0` lives. Otherwise fits the per-reference multinomial.

    Args:
        X: (N, L) array with entries in {-1, +1}.
        k1, k2: integer Hamming radii with 0 <= k1 < k2 <= L. If both
            ``None``, the multinomial fit is used.
        d0_init, d1_init: BFGS initial point. ``None`` for d0 picks ``L``.
        L: bit length. Inferred from ``X`` if not given.
        check_format: validate ±1 entries.

    Returns:
        :class:`ShellFitResult` with fields ``(d0, d1, nll, converged)``.
    """
    X_j = jnp.asarray(X, dtype=jnp.int32)
    if check_format:
        check_pm1(X_j)
    if L is None:
        L = int(X_j.shape[1])

    D = pairwise_hamming(X_j)
    sd = shell_counts(D, L=L)

    if k1 is None and k2 is None:
        return fit_multinomial(sd, d0_init=d0_init, d1_init=d1_init)
    if k1 is None or k2 is None:
        raise ValueError("specify both k1 and k2, or neither")
    return fit_conditional(
        sd, k1=int(k1), k2=int(k2), d0_init=d0_init, d1_init=d1_init
    )
