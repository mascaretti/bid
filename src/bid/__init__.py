"""Binary Intrinsic Dimension estimator (pure-JAX, functional)."""

from bid.api import estimate_bid
from bid.distances import check_pm1, pairwise_hamming
from bid.histogram import (
    Histogram,
    empirical_histogram,
    empirical_moments,
    truncate_by_quantiles,
)
from bid.model import kl_divergence, p_model
from bid.optimize import (
    BIDResult,
    OptState,
    finalize,
    init_state,
    initial_guess,
    minimize_kl,
    step,
)

__all__ = [
    "estimate_bid",
    "pairwise_hamming",
    "check_pm1",
    "Histogram",
    "empirical_histogram",
    "empirical_moments",
    "truncate_by_quantiles",
    "p_model",
    "kl_divergence",
    "OptState",
    "BIDResult",
    "init_state",
    "step",
    "minimize_kl",
    "initial_guess",
    "finalize",
]
__version__ = "0.1.0"
