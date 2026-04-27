"""Empirical Hamming-distance histogram and quantile-based truncation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Histogram(NamedTuple):
    """Truncated empirical distance distribution used as the fit target.

    Fields:
        r:     float distance values (selected slice of the support)
        p_emp: empirical probabilities, renormalised to sum to 1 over `r`
    """

    r: jax.Array
    p_emp: jax.Array


def empirical_histogram(D: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Empirical histogram from a pairwise Hamming distance matrix.

    Each unordered pair (i, j) with i < j contributes one observation. The
    diagonal and lower triangle are discarded.

    Args:
        D: (N, N) integer distance matrix.

    Returns:
        (values, counts) — sorted unique distances actually observed, and
        their multiplicities. Total count is N(N-1)/2.
    """
    n = D.shape[0]
    iu0, iu1 = jnp.triu_indices(n, k=1)
    pair_dists = D[iu0, iu1]
    values, counts = jnp.unique(pair_dists, return_counts=True)
    return values, counts


def truncate_by_quantiles(
    values: jax.Array,
    counts: jax.Array,
    alphamin: float = 0.0,
    alphamax: float = 1.0,
    precision: int = 10,
) -> Histogram:
    """Restrict the histogram to the [alphamin, alphamax] cumulative-probability band.

    Following the BID convention: ``r_min`` is the largest value whose CDF
    is ≤ alphamin, and ``r_max`` is the largest value whose CDF is ≤ alphamax.
    Setting alphamin=0 keeps everything from the smallest sampled distance;
    alphamax=1 keeps everything up to the largest.

    The kept probabilities are renormalised so that they sum to 1.
    """
    probs = counts / jnp.sum(counts)
    probs_r = jnp.round(probs, precision)
    alphamin_r = round(float(alphamin), precision)
    alphamax_r = round(float(alphamax), precision)
    cdf = jnp.cumsum(probs_r)

    below_min = jnp.where(cdf <= alphamin_r)[0]
    idmin = int(below_min[-1]) if below_min.size > 0 else 0

    below_max = jnp.where(cdf <= alphamax_r)[0]
    idmax = int(below_max[-1]) if below_max.size > 0 else 0

    if idmax < idmin:
        raise ValueError(
            f"Empty truncation: alphamin={alphamin} gives idmin={idmin}, "
            f"alphamax={alphamax} gives idmax={idmax}."
        )

    r = values[idmin : idmax + 1].astype(jnp.float64)
    p = probs[idmin : idmax + 1].astype(jnp.float64)
    p = p / jnp.sum(p)
    return Histogram(r=r, p_emp=p)


def empirical_moments(values: jax.Array, counts: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Mean and variance of the (untruncated) empirical distribution."""
    p = counts / jnp.sum(counts)
    mean = jnp.dot(p, values)
    var = jnp.dot(p, values**2) - mean**2
    return mean, var
