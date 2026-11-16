"""
noise.py — Synaptic noise for the per-step hot paths (forgetting, perturbation,
legacy additive noise).

    sample(key, shape, dist) -> array of zero-mean, unit-variance i.i.d. noise

dist "rademacher" (default): ±1 with equal probability, 32 synapses per random
    uint32. Generator cost is per random WORD, so packing makes the noise ~free
    (CPU, 64×274×258: 82 ms Gaussian -> 7 ms incl. the weight arithmetic).
    Valid because every consumer needs only zero mean, unit variance, symmetry
    and independence:
      - forgetting: variance-matched b keeps the first two moments exact and
        symmetry keeps the reflection argument; the stationary law is a sum of
        ~τ_prior/Δ kicks, Gaussian up to excess kurtosis −2(1−a²)/(1+a²) ≈ −2Δ/τ_prior.
      - perturbation: the SPSA perturbation (Spall 1992) — Cov = I (unbiased),
        E[ε⁴] = 1 vs 3 for Gaussian, so a lower-variance credit signal.
dist "gaussian": jax.random.normal — exact Gaussian kicks, one word per synapse.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

DISTS = ("rademacher", "gaussian")


def rademacher(key: jax.Array, shape: tuple) -> jnp.ndarray:
    """±1 i.i.d., packed 32 per uint32 along the last axis."""
    *lead, last = shape
    words = (last + 31) // 32
    bits = jax.random.bits(key, (*lead, words), jnp.uint32)
    s = (bits[..., None] >> jnp.arange(32, dtype=jnp.uint32)) & jnp.uint32(1)
    s = s.reshape(*lead, words * 32)[..., :last]
    return 2.0 * s.astype(jnp.float32) - 1.0


def sample(key: jax.Array, shape: tuple, dist: str = "rademacher") -> jnp.ndarray:
    if dist == "rademacher":
        return rademacher(key, shape)
    if dist == "gaussian":
        return jax.random.normal(key, shape)
    raise ValueError(f"unknown noise dist '{dist}'; have {DISTS}")
