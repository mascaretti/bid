"""Reproduces the BID reference values from DADApy/tests/test_hamming/test0.py.

Random ±1 streams of length L=100 with N=5000 samples, seed=1. The expected
estimates are d0 ≈ L (here ≈99.855), d1 ≈ 0, with log KL ≈ -12.39.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import pytest

from bid import estimate_bid


D0_REF = 99.855
D1_REF = 0.003
LOGKL_REF = -12.39


def test_random_spins_reproduces_dadapy_reference():
    seed = 1
    L = 100
    Ns = 5000

    rng = np.random.RandomState(seed=seed)
    X = 2 * rng.randint(low=0, high=2, size=(Ns, L)) - 1

    result = estimate_bid(
        X,
        alphamin=0.0,
        alphamax=1.0,
        delta=5e-4,
        n_steps=int(1e6),
        seed=seed,
        L=L,
    )

    assert float(result.d0) == pytest.approx(D0_REF, abs=1e-2)
    assert float(result.d1) == pytest.approx(D1_REF, abs=1e-2)
    assert float(result.log_kl) == pytest.approx(LOGKL_REF, abs=1e-1)


def test_pairwise_hamming_matches_naive():
    """Sanity check: the X @ X.T trick agrees with the naive count."""
    from bid import pairwise_hamming

    rng = np.random.RandomState(0)
    X = 2 * rng.randint(0, 2, size=(20, 50)) - 1
    D = np.asarray(pairwise_hamming(X))
    D_naive = np.array([[np.sum(X[i] != X[j]) for j in range(20)] for i in range(20)])
    np.testing.assert_array_equal(D, D_naive)
