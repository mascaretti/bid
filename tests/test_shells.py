"""Tests for the shell-based BID likelihoods."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import pytest

from bid import (
    estimate_bid,
    estimate_bid_shells,
    pairwise_hamming,
    shell_counts,
)


def _random_spins(N: int, L: int, seed: int):
    rng = np.random.RandomState(seed=seed)
    return 2 * rng.randint(0, 2, size=(N, L)) - 1


def test_shell_counts_shape_and_endpoints():
    L, N = 20, 100
    X = _random_spins(N, L, seed=0)
    D = pairwise_hamming(X)
    sd = shell_counts(D, L=L)

    assert sd.n_cum.shape == (N, L + 1)
    assert sd.L == L
    # Each row's final cumulative count = N - 1 (everyone except self).
    np.testing.assert_array_equal(np.asarray(sd.n_cum[:, -1]), N - 1)
    # n_cum is non-decreasing along each row.
    diffs = np.diff(np.asarray(sd.n_cum), axis=1)
    assert (diffs >= 0).all()


def test_multinomial_fit_recovers_dimension_on_random_spins():
    L = 100
    X = _random_spins(N=500, L=L, seed=1)
    res = estimate_bid_shells(X, L=L)
    assert bool(res.converged)
    assert float(res.d0) == pytest.approx(L, abs=1.0)
    assert abs(float(res.d1)) < 0.05


def test_conditional_fit_recovers_dimension_on_random_spins():
    L = 100
    X = _random_spins(N=500, L=L, seed=1)
    # Pick radii bracketing the bulk of the distance distribution
    res = estimate_bid_shells(X, k1=40, k2=60, L=L)
    assert bool(res.converged)
    assert float(res.d0) == pytest.approx(L, abs=0.5)
    assert abs(float(res.d1)) < 0.05


def test_multinomial_agrees_with_bid_kl_fit():
    """Pooled-multinomial MLE and BID's KL fit are mathematically the same
    point estimate (different objective constants, same minimiser)."""
    L = 50
    X = _random_spins(N=300, L=L, seed=2)

    shell = estimate_bid_shells(X, L=L)
    bid = estimate_bid(
        X, alphamin=0.0, alphamax=1.0, delta=5e-4, n_steps=200_000, seed=2, L=L
    )

    # Stochastic optimiser converges noisily; loosen the tolerance accordingly.
    assert float(shell.d0) == pytest.approx(float(bid.d0), abs=0.5)
    assert float(shell.d1) == pytest.approx(float(bid.d1), abs=0.05)


def test_conditional_rejects_invalid_radii():
    L = 20
    X = _random_spins(N=50, L=L, seed=3)
    with pytest.raises(ValueError):
        estimate_bid_shells(X, k1=5, k2=5, L=L)
    with pytest.raises(ValueError):
        estimate_bid_shells(X, k1=10, k2=3, L=L)
