"""Packed Rademacher noise: i.i.d. ±1, and valid for forgetting."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from esp import noise, plasticity
from esp.config import ExperimentConfig, replace


@pytest.mark.parametrize("shape", [(1000, 258), (7, 33), (5, 31), (3, 64)])
def test_rademacher_values_shape_and_moments(shape):
    x = noise.rademacher(jax.random.PRNGKey(0), shape)
    assert x.shape == shape and x.dtype == jnp.float32
    assert set(np.unique(np.asarray(x)).tolist()) <= {-1.0, 1.0}
    if x.size > 100_000:
        assert abs(float(x.mean())) < 0.005


def test_rademacher_each_bit_position_unbiased_and_uncorrelated():
    x = np.asarray(noise.rademacher(jax.random.PRNGKey(1), (20_000, 64)))
    col_means = x.mean(axis=0)                       # each of the 32 bit positions, twice
    assert np.max(np.abs(col_means)) < 0.03          # ~4σ for n=20k
    c = np.corrcoef(x.T)
    off = c[~np.eye(64, dtype=bool)]
    assert np.max(np.abs(off)) < 0.035               # within-word and across-word pairs
    # across time: independent draws per key
    y = np.asarray(noise.rademacher(jax.random.PRNGKey(2), (20_000, 64)))
    assert abs(float(np.mean(x * y))) < 0.005


def test_unknown_dist_rejected():
    with pytest.raises(ValueError):
        noise.sample(jax.random.PRNGKey(0), (4,), "laplace")
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **{"plasticity.noise_dist": "laplace"})


def test_forgetting_with_rademacher_matches_newborn_moments():
    """τ_prior/Δ = 100 (the 1 s floor at Δ = 10 ms): mean/std of the stationary law
    match the half-normal; excess kurtosis of the underlying AR(1) ≈ −2(1−a²)/(1+a²)."""
    S = 0.3
    cfg = replace(ExperimentConfig(), **{"plasticity.rule": "three_factor", "plasticity.forgetting": True,
                                         "plasticity.noise_mode": "matched", "genome.w_init_std": S})
    one = jnp.asarray(1.0)
    rp = plasticity.RuleParams(one, one, one, one, one, jnp.asarray(1000.0))
    w0 = jnp.full((200_000,), 1.5)
    mask = jnp.ones_like(w0, bool)

    def step(w, k):
        return plasticity.forget(w, noise.rademacher(k, w.shape), mask, rp, cfg, 10.0), None

    w, _ = jax.lax.scan(step, w0, jax.random.split(jax.random.PRNGKey(4), 2000))   # 20 τ
    assert abs(float(w.mean()) - S * math.sqrt(2 / math.pi)) < 0.003
    assert abs(float(w.std()) - S * math.sqrt(1 - 2 / math.pi)) < 0.003
