"""Binary Intrinsic Dimension estimator (pure-JAX, functional)."""

from bid.api import estimate_bid, estimate_bid_shells
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
from bid.shells import (
    ShellData,
    ShellFitResult,
    cumulative_volume,
    fit_conditional,
    fit_multinomial,
    nll_conditional,
    nll_multinomial,
    shell_counts,
)

__all__ = [
    "estimate_bid",
    "estimate_bid_shells",
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
    "ShellData",
    "ShellFitResult",
    "shell_counts",
    "cumulative_volume",
    "nll_multinomial",
    "nll_conditional",
    "fit_multinomial",
    "fit_conditional",
]
__version__ = "0.1.0"
