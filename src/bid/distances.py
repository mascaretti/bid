"""Pairwise Hamming distances for ±1 binary spin configurations."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def check_pm1(X: jax.Array) -> None:
    """Validate that all entries are in {-1, +1}."""
    vals = jnp.unique(X)
    if vals.shape[0] != 2 or int(vals[0]) != -1 or int(vals[1]) != 1:
        raise ValueError(
            f"spins must be normalised to ±1, got unique values {vals.tolist()}"
        )


@jax.jit
def pairwise_hamming(X: jax.Array) -> jax.Array:
    """Pairwise Hamming distance matrix for ±1 spin samples.

    For ±1 vectors of length L, the Hamming distance equals
    ``(L - <x_i, x_j>) / 2``.

    Args:
        X: array of shape (N, L) with entries in {-1, +1}.

    Returns:
        Integer (N, N) matrix of pairwise Hamming distances. The diagonal
        is exactly 0.
    """
    X = X.astype(jnp.int32)
    L = X.shape[1]
    return ((L - X @ X.T) // 2).astype(jnp.int32)
