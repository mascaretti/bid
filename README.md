# bid

Binary Intrinsic Dimension (BID) estimator, extracted from
[DADApy](https://github.com/sissa-data-science/DADApy) and refactored as a
small, pure-JAX, functional library.

The model fits a generalised binomial

```
P(r) = C · 2^(-d(r)) · binom(d(r), r),    d(r) = d0 + d1·r
```

to the empirical distribution of pairwise Hamming distances of binary
samples (±1 spins), by minimising the KL divergence between the empirical
histogram and the model. The fitted `d0` is the Binary Intrinsic Dimension.

Reference: see the BID method in DADApy and the
[tutorial](https://dadapy.readthedocs.io/en/latest/jupyter_example_9.html).

## Install

```bash
pip install -e .
```

The only runtime dependencies are `jax`, `jaxlib`, and `numpy`.

## Usage

```python
import numpy as np
from bid import estimate_bid

# (N, L) array of ±1 spins
rng = np.random.default_rng(0)
X = 2 * rng.integers(0, 2, size=(5000, 100)) - 1

result = estimate_bid(
    X,
    alphamin=0.0,
    alphamax=1.0,
    delta=5e-4,
    n_steps=1_000_000,
    seed=1,
    L=100,
)

print(f"d0     = {result.d0:.3f}")
print(f"d1     = {result.d1:.3f}")
print(f"log KL = {result.log_kl:.3f}")
```

For random ±1 streams of length `L`, the estimator should recover
`d0 ≈ L` and `d1 ≈ 0`.

## Lower-level API

If you need to plug into a custom loop, the building blocks are exposed:

```python
from bid import (
    pairwise_hamming,
    empirical_histogram,
    truncate_by_quantiles,
    log_p_model,
    kl_divergence,
    init_state,
    minimize_kl,
    initial_guess,
)
```

All functions are pure, JIT-compatible, and operate on JAX-friendly pytrees
(`NamedTuple`s for `Histogram`, `OptState`, `BIDResult`).

## Acknowledgments & Citation

This package is a derivative work of [DADApy](https://github.com/sissa-data-science/DADApy)
(`dadapy/hamming.py`), reorganised into a stand-alone, pure-JAX, functional library.
All credit for the BID method and its original implementation belongs to the DADApy
authors and the BID paper's authors.

If you use this package in academic work, please cite the original BID paper:

> Acevedo, S. D., Del Tatto, V., et al. *Binary Intrinsic Dimension* estimator,
> *Nature Communications Physics* (2025).

…and the DADApy library:

```bibtex
@article{dadapy,
    title    = {DADApy: Distance-based analysis of data-manifolds in Python},
    journal  = {Patterns},
    pages    = {100589},
    year     = {2022},
    issn     = {2666-3899},
    doi      = {https://doi.org/10.1016/j.patter.2022.100589},
    author   = {Aldo Glielmo and Iuri Macocco and Diego Doimo and Matteo Carli
                and Claudio Zeni and Romina Wild and Maria d'Errico
                and Alex Rodriguez and Alessandro Laio},
}
```

The original `hamming.py` module in DADApy was authored primarily by
**Santiago Daniel Acevedo** and **Vittorio Del Tatto**.

## License

Apache-2.0, inherited from DADApy. See `LICENSE`.
