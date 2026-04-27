"""Stochastic-perturbation minimisation of KL(P_emp ‖ P_model)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from bid.histogram import Histogram
from bid.model import kl_divergence, p_model


class OptState(NamedTuple):
    """State of the monotone stochastic-perturbation optimiser.

    Fields:
        key:       JAX PRNG key
        d0, d1:    current accepted parameters
        kl_best:   KL at the current (d0, d1)
        accepted:  number of accepted moves so far
    """

    key: jax.Array
    d0: jax.Array
    d1: jax.Array
    kl_best: jax.Array
    accepted: jax.Array


class BIDResult(NamedTuple):
    """Final BID estimate.

    Fields:
        d0:        intrinsic dimension estimate (intercept of d(r))
        d1:        slope of d(r)
        log_kl:    natural log of the final KL divergence
        p_model:   fitted model probabilities on the truncated support
        acc_ratio: fraction of proposals accepted
    """

    d0: jax.Array
    d1: jax.Array
    log_kl: jax.Array
    p_model: jax.Array
    acc_ratio: jax.Array


def init_state(seed: int, d0: float, d1: float, hist: Histogram) -> OptState:
    """Initialise the optimiser at (d0, d1) with `kl_best` = KL at that point."""
    key = jax.random.PRNGKey(int(seed))
    p = p_model(hist.r, jnp.float64(d0), jnp.float64(d1))
    kl0 = kl_divergence(hist.p_emp, p)
    return OptState(
        key=key,
        d0=jnp.float64(d0),
        d1=jnp.float64(d1),
        kl_best=kl0,
        accepted=jnp.int32(0),
    )


@jax.jit
def step(state: OptState, hist: Histogram, delta: jax.Array) -> OptState:
    """One stochastic-perturbation step. Accept iff KL strictly decreases (≤)."""
    key, sub0 = jax.random.split(state.key)
    u0 = jax.random.uniform(sub0, dtype=jnp.float64)
    d0_prop = state.d0 * (1.0 + delta * (u0 - 0.5))

    key, sub1 = jax.random.split(key)
    u1 = jax.random.uniform(sub1, dtype=jnp.float64)
    d1_prop = state.d1 * (1.0 + delta * (u1 - 0.5))

    p = p_model(hist.r, d0_prop, d1_prop)
    kl = kl_divergence(hist.p_emp, p)
    accept = kl <= state.kl_best

    return OptState(
        key=key,
        d0=jnp.where(accept, d0_prop, state.d0),
        d1=jnp.where(accept, d1_prop, state.d1),
        kl_best=jnp.where(accept, kl, state.kl_best),
        accepted=state.accepted + accept.astype(jnp.int32),
    )


@jax.jit
def minimize_kl(
    state: OptState, hist: Histogram, delta: float, n_steps: int
) -> OptState:
    """Run `n_steps` stochastic perturbation steps. Returns the final state."""
    delta_a = jnp.float64(delta)

    def body(_i, st):
        return step(st, hist, delta_a)

    return lax.fori_loop(0, jnp.int32(n_steps), body, state)


def initial_guess(
    hist: Histogram,
    full_mean: float | None = None,
    L: int | None = None,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
    alpha_step: float = 0.05,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pick an initial (d0, d1) by minimising KL over a small candidate grid.

    Always includes the trivial guess ``(d0=r_max, d1=1)`` (where r_max is
    the largest distance in the truncated support).

    When `L` and `full_mean` are supplied, also scans
    ``d0 = L · α  for α ∈ [alpha_min, alpha_max]``  with
    ``d1 = 2 - d0 / full_mean``  (Erazo's heuristic).

    Args:
        hist:      truncated empirical histogram (the fit target).
        full_mean: mean of the *untruncated* empirical distribution. Only
            used when L is supplied.
        L:         number of bits per sample. Enables the L-aware grid.

    Returns:
        (d0, d1, kl_at_guess)
    """
    d0_list = jnp.array([hist.r[-1]], dtype=jnp.float64)
    d1_list = jnp.array([1.0], dtype=jnp.float64)

    if L is not None and full_mean is not None:
        alphas = jnp.arange(
            alpha_min, alpha_max + 1e-7, alpha_step, dtype=jnp.float64
        )
        d0_grid = jnp.float64(L) * alphas
        d1_grid = 2.0 - d0_grid / jnp.float64(full_mean)
        d0_list = jnp.concatenate([d0_list, d0_grid])
        d1_list = jnp.concatenate([d1_list, d1_grid])

    def eval_kl(d0, d1):
        p = p_model(hist.r, d0, d1)
        return kl_divergence(hist.p_emp, p)

    kls = jax.vmap(eval_kl)(d0_list, d1_list)
    log_kls = jnp.log(kls)
    i = jnp.nanargmin(log_kls)
    return d0_list[i], d1_list[i], kls[i]


def finalize(state: OptState, hist: Histogram, n_steps: int) -> BIDResult:
    """Convert a final OptState into a user-facing BIDResult."""
    p = p_model(hist.r, state.d0, state.d1)
    kl = kl_divergence(hist.p_emp, p)
    return BIDResult(
        d0=state.d0,
        d1=state.d1,
        log_kl=jnp.log(kl),
        p_model=p,
        acc_ratio=state.accepted.astype(jnp.float64) / jnp.float64(n_steps),
    )
