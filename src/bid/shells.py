r"""Shell-based likelihoods for the BID model.

Same model as :mod:`bid.model` (generalised binomial parametrised by
``(d0, d1)``), different observational summary: per-reference cumulative
neighbour counts at integer Hamming radii.

Two likelihoods are provided:

* :func:`nll_multinomial` — each reference :math:`s` contributes a multinomial
  draw of its :math:`N-1` neighbours over distances. Pooled across references,
  the MLE coincides with the KL fit on the all-pairs histogram, but the
  per-reference formulation is the statistically honest one (no double-counting
  of correlated pairs).

* :func:`nll_conditional` — I3D-style conditional likelihood at a chosen pair
  of integer radii :math:`(k_1, k_2)`: under the model,
  :math:`n_{k_1}(s) \mid n_{k_2}(s) \sim \mathrm{Binomial}\bigl(n_{k_2}(s),\;
  V(k_1)/V(k_2)\bigr)` where :math:`V(k) = \sum_{r\le k} P(r;d_0,d_1)`.
  Targets the small-:math:`r` regime where :math:`d_0` lives and is robust
  to misspecification of the linear :math:`d(r) = d_0 + d_1 r`.

Both likelihoods are differentiable and minimised via JAX BFGS.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

from bid.model import p_model

_LOG_FLOOR = 1e-300


class ShellData(NamedTuple):
    """Per-reference cumulative shell counts.

    Single-field pytree so the bit length ``L`` is implicit in the static
    array shape (``n_cum.shape[-1] == L + 1``) rather than carried as a
    traced field — keeps the struct safe to pass through ``jit``.

    Fields:
        n_cum: int array of shape ``(N, L+1)``, with
            ``n_cum[s, k] = #{j != s : d_H(s, j) <= k}``.
    """

    n_cum: jax.Array

    @property
    def L(self) -> int:
        return int(self.n_cum.shape[-1]) - 1


class ShellFitResult(NamedTuple):
    d0: jax.Array
    d1: jax.Array
    nll: jax.Array
    converged: jax.Array


def shell_counts(D: jax.Array, L: int) -> ShellData:
    """Cumulative neighbour counts at every integer Hamming radius.

    Args:
        D: (N, N) integer distance matrix (the diagonal must be 0).
        L: bit length of the original samples.
    """
    K = int(L) + 1
    ks = jnp.arange(K)
    sorted_D = jnp.sort(D, axis=1)
    counts = jax.vmap(lambda row: jnp.searchsorted(row, ks, side="right"))(sorted_D)
    return ShellData(n_cum=counts - 1)


def _support_and_pmf(shell_data: ShellData, d0, d1):
    K = shell_data.n_cum.shape[-1]
    support = jnp.arange(K, dtype=jnp.float64)
    return support, p_model(support, d0, d1)


@jax.jit
def cumulative_volume(shell_data: ShellData, d0: jax.Array, d1: jax.Array) -> jax.Array:
    r""":math:`V(k; d_0, d_1) = \sum_{r\le k} P(r; d_0, d_1)` for :math:`k=0,\dots,L`."""
    _, p = _support_and_pmf(shell_data, d0, d1)
    return jnp.cumsum(p)


@jax.jit
def nll_multinomial(
    shell_data: ShellData, d0: jax.Array, d1: jax.Array
) -> jax.Array:
    """Negative log-likelihood (up to a (d0, d1)-independent constant) of the
    per-reference multinomial draw of neighbour distances."""
    n_cum = shell_data.n_cum.astype(jnp.float64)
    zero_col = jnp.zeros((n_cum.shape[0], 1), dtype=n_cum.dtype)
    n_per_shell = jnp.diff(n_cum, axis=1, prepend=zero_col)
    n_total = n_per_shell.sum(axis=0)

    _, p = _support_and_pmf(shell_data, d0, d1)
    log_p = jnp.log(p + _LOG_FLOOR)
    return -jnp.sum(n_total * log_p)


def nll_conditional(
    shell_data: ShellData,
    d0: jax.Array,
    d1: jax.Array,
    k1: int,
    k2: int,
) -> jax.Array:
    """NLL of the I3D-style conditional binomial on the pair :math:`(k_1, k_2)`.

    ``k1`` and ``k2`` are Python ints (static under JIT).
    """
    L = shell_data.L
    if not 0 <= k1 < k2 <= L:
        raise ValueError(
            f"need 0 <= k1 < k2 <= L, got k1={k1}, k2={k2}, L={L}"
        )
    n1 = shell_data.n_cum[:, k1].astype(jnp.float64)
    n2 = shell_data.n_cum[:, k2].astype(jnp.float64)
    V = cumulative_volume(shell_data, d0, d1)
    ratio = V[k1] / V[k2]
    return -jnp.sum(
        n1 * jnp.log(ratio + _LOG_FLOOR)
        + (n2 - n1) * jnp.log1p(-ratio + _LOG_FLOOR)
    )


def select_radii(
    shell_data: ShellData,
    q1: float = 0.25,
    q2: float = 0.75,
) -> tuple[int, int]:
    """Pick conditional-fit radii :math:`(k_1, k_2)` at empirical quantiles
    of the pooled shell distribution.

    Useful when the shell distribution is concentrated and a fixed pair like
    ``(40, 60)`` misses the bulk of the support — common on the ordered side
    of a phase transition. The returned radii satisfy
    ``1 <= k1 < k2 <= L``; degenerate cases (e.g. all mass at ``r=0``) are
    nudged apart so the conditional fit has at least one shell of width.

    Args:
        shell_data: output of :func:`shell_counts`.
        q1, q2: cumulative-probability cutoffs in ``(0, 1)``, ``q1 < q2``.
    """
    if not 0.0 < q1 < q2 < 1.0:
        raise ValueError(f"need 0 < q1 < q2 < 1, got q1={q1}, q2={q2}")

    L = shell_data.L
    cumulative = jnp.asarray(shell_data.n_cum.sum(axis=0), dtype=jnp.float64)
    cdf = cumulative / cumulative[-1]

    k1 = int(jnp.searchsorted(cdf, q1))
    k2 = int(jnp.searchsorted(cdf, q2))
    k1 = max(1, min(k1, L - 1))
    k2 = max(k1 + 1, min(k2, L))
    return k1, k2


def _bfgs_fit(loss_scalar, d0_init: float, d1_init: float) -> ShellFitResult:
    x0 = jnp.array([float(d0_init), float(d1_init)], dtype=jnp.float64)
    res = minimize(lambda x: loss_scalar(x[0], x[1]), x0, method="BFGS")
    return ShellFitResult(
        d0=res.x[0], d1=res.x[1], nll=res.fun, converged=res.success
    )


def fit_multinomial(
    shell_data: ShellData,
    d0_init: float | None = None,
    d1_init: float = 0.0,
) -> ShellFitResult:
    """MLE for (d0, d1) under the per-reference multinomial likelihood."""
    d0_start = float(shell_data.L) if d0_init is None else float(d0_init)
    return _bfgs_fit(
        lambda a, b: nll_multinomial(shell_data, a, b), d0_start, d1_init
    )


def fit_conditional(
    shell_data: ShellData,
    k1: int,
    k2: int,
    d0_init: float | None = None,
    d1_init: float = 0.0,
) -> ShellFitResult:
    """MLE for (d0, d1) under the I3D-style conditional likelihood at (k1, k2)."""
    d0_start = float(shell_data.L) if d0_init is None else float(d0_init)
    return _bfgs_fit(
        lambda a, b: nll_conditional(shell_data, a, b, k1, k2), d0_start, d1_init
    )
