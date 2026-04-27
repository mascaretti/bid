"""Generalised-binomial BID model and KL divergence."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


@jax.jit
def p_model(r: jax.Array, d0: jax.Array, d1: jax.Array) -> jax.Array:
    """Generalised binomial probability mass on the support `r`.

    The continuous-`d` extension of the binomial distribution, with
    ``d(r) = d0 + d1 * r`` playing the role of the number of bits:

        P(r) ∝ Γ(d(r)+1) / [ Γ(r+1) Γ(d(r)-r+1) ] · 2^(-d(r))

    The result is renormalised to sum to 1 over the supplied support.
    """
    d_r = d0 + d1 * r
    log_p = (
        gammaln(d_r + 1.0)
        - gammaln(r + 1.0)
        - gammaln(d_r - r + 1.0)
        - d_r * jnp.log(2.0)
    )
    p = jnp.exp(log_p)
    return p / jnp.sum(p)


@jax.jit
def kl_divergence(p_emp: jax.Array, p_mod: jax.Array) -> jax.Array:
    """KL(p_emp ‖ p_mod). Both inputs must sum to 1."""
    return jnp.sum(p_emp * jnp.log(p_emp / p_mod))
