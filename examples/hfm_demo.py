"""HFM demo: estimate the BID across the critical line of the
Hierarchical Feature Model.

The HFM samples binary strings of length ``n`` as follows: draw a level
``m ∈ {0, ..., n}`` with ``P(m=k) ∝ ξ^k``; if ``m == 0`` the sample is
all-zero, otherwise bit ``m-1`` is set to 1, bits before ``m-1`` are i.i.d.
Bernoulli(1/2), and bits after are 0. The control parameter is
``ξ = 2 exp(-g)`` with critical point ``g_c = log 2`` (i.e. ``ξ_c = 1``).

We sample at several values of ``g`` and compare three intrinsic-dimension
estimators that all share the BID generalised-binomial model:

  1. ``estimate_bid``                     — KL fit on the all-pairs histogram
  2. ``estimate_bid_shells(X)``           — per-reference multinomial MLE
  3. ``estimate_bid_shells(X, k1, k2)``   — I3D-style conditional MLE

Variant (2) is the same MLE as (1) in expectation but uses a different,
statistically honest likelihood; (3) is robust to misspecification of the
linear ``d(r) = d0 + d1 r``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import math

import numpy as np

from bid import (
    estimate_bid,
    estimate_bid_shells,
    pairwise_hamming,
    select_radii,
    shell_counts,
)


def sample_hfm(n: int, xi: float, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorised HFM sampler returning ±1 samples of shape (num_samples, n)."""
    # Level distribution: P(m=k) ∝ xi^k for k in {0, 1, ..., n}.
    log_xi = math.log(max(xi, 1e-300))
    log_w = np.arange(n + 1, dtype=np.float64) * log_xi
    log_w -= log_w.max()
    probs = np.exp(log_w)
    probs /= probs.sum()

    m = rng.choice(n + 1, size=num_samples, p=probs)
    bits = rng.integers(0, 2, size=(num_samples, n), dtype=np.uint8)
    # mask: positions strictly less than m-1 are random, position m-1 is 1, rest is 0.
    pos = np.arange(n)[None, :]
    random_mask = pos < (m[:, None] - 1)
    set_mask = pos == (m[:, None] - 1)
    out = np.where(random_mask, bits, 0)
    out = np.where(set_mask, 1, out)
    return (2 * out.astype(np.int8) - 1).astype(np.int32)


def sweep(
    n: int = 100,
    num_samples: int = 2000,
    g_values: tuple[float, ...] = (0.40, 0.55, math.log(2.0), 0.85, 1.00, 1.20),
    seed: int = 0,
    bid_n_steps: int = 200_000,
    bid_delta: float = 5e-4,
    cond_quantiles: tuple[float, float] = (0.25, 0.75),
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for g in g_values:
        xi = 2.0 * math.exp(-g)
        X = sample_hfm(n, xi, num_samples, rng)
        # Skip degenerate cases (all-zero etc.); BID needs both ±1 present.
        unique = np.unique(X)
        if not (unique.size == 2 and unique[0] == -1 and unique[1] == 1):
            print(f"g={g:.4f} (xi={xi:.4f}): degenerate (unique={unique.tolist()}); skipping.")
            continue

        bid_res = estimate_bid(
            X,
            alphamin=0.0,
            alphamax=1.0,
            delta=bid_delta,
            n_steps=bid_n_steps,
            seed=seed,
            L=n,
        )
        m_res = estimate_bid_shells(X, L=n)
        sd = shell_counts(pairwise_hamming(X), L=n)
        k1, k2 = select_radii(sd, q1=cond_quantiles[0], q2=cond_quantiles[1])
        c_res = estimate_bid_shells(X, k1=k1, k2=k2, L=n)

        rows.append(
            {
                "g": g,
                "xi": xi,
                "bid_d0": float(bid_res.d0),
                "bid_d1": float(bid_res.d1),
                "bid_log_kl": float(bid_res.log_kl),
                "bid_acc_ratio": float(bid_res.acc_ratio),
                "shell_mn_d0": float(m_res.d0),
                "shell_mn_d1": float(m_res.d1),
                "shell_mn_nll": float(m_res.nll),
                "shell_mn_converged": bool(m_res.converged),
                "shell_cd_d0": float(c_res.d0),
                "shell_cd_d1": float(c_res.d1),
                "shell_cd_nll": float(c_res.nll),
                "shell_cd_converged": bool(c_res.converged),
                "shell_cd_k1": int(k1),
                "shell_cd_k2": int(k2),
            }
        )

    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no rows)")
        return
    header = (
        f"{'g':>6} {'xi':>6} | "
        f"{'BID d0':>8} {'d1':>7} {'logKL':>7} {'acc':>5} | "
        f"{'mn d0':>8} {'d1':>7} | "
        f"{'(k1,k2)':>9} {'cond d0':>8} {'d1':>7}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        radii = f"({r['shell_cd_k1']:>2d},{r['shell_cd_k2']:>2d})"
        print(
            f"{r['g']:>6.4f} {r['xi']:>6.4f} | "
            f"{r['bid_d0']:>8.3f} {r['bid_d1']:>+7.4f} "
            f"{r['bid_log_kl']:>7.2f} {r['bid_acc_ratio']:>5.3f} | "
            f"{r['shell_mn_d0']:>8.3f} {r['shell_mn_d1']:>+7.4f} | "
            f"{radii:>9} {r['shell_cd_d0']:>8.3f} {r['shell_cd_d1']:>+7.4f}"
        )


if __name__ == "__main__":
    rows = sweep()
    print_table(rows)
