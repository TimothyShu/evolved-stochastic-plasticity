"""Forgetting: reflected AR(1) decay to the newborn distribution |N(0, w_init_std)|."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from esp import plasticity
from esp.config import ExperimentConfig, replace
from esp.genome.direct import N_RULE_PARAMS, connected, random_genome

S = 0.3  # w_init_std


def _cfg(**kw):
    return replace(ExperimentConfig(), **{"plasticity.rule": "three_factor", "plasticity.forgetting": True,
                                          "plasticity.noise_mode": "matched", "genome.w_init_std": S, **kw})


def _rp(tau_prior):
    one = jnp.asarray(1.0)
    return plasticity.RuleParams(one, one, one, one, one, jnp.asarray(float(tau_prior)))


def _simulate(w0, tau, n_steps, cfg, seed=0):
    """Pure forgetting (dw = 0) on a bank of independent plastic synapses."""
    rp, mask = _rp(tau), jnp.ones_like(w0, bool)

    def step(w, k):
        return plasticity.forget(w, jax.random.normal(k, w.shape), mask, rp, cfg, cfg.dt), None

    w, _ = jax.lax.scan(step, w0, jax.random.split(jax.random.PRNGKey(seed), n_steps))
    return w


def test_stationary_law_is_newborn_distribution():
    """From a 'learned' state, unreinforced weights relax to exactly |N(0, S)|."""
    cfg = _cfg()
    tau = 100.0                                   # test only; evolved range starts at 1 s
    w = _simulate(jnp.full((200_000,), 1.5), tau, 2000, cfg)   # 20 τ
    hn_mean, hn_std = S * math.sqrt(2 / math.pi), S * math.sqrt(1 - 2 / math.pi)
    assert abs(float(w.mean()) - hn_mean) < 0.003
    assert abs(float(w.std()) - hn_std) < 0.003
    # compare quantiles with the half-normal (|Z|·S)
    q = np.quantile(np.asarray(w), [0.1, 0.5, 0.9])
    ref = S * np.array([0.12566, 0.67449, 1.64485])
    assert np.allclose(q, ref, atol=0.006)


def test_newborn_distribution_is_invariant():
    """Starting from newborn weights, forgetting leaves the distribution unchanged."""
    cfg = _cfg()
    w0 = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (200_000,))) * S
    w = _simulate(w0, 100.0, 300, cfg)
    assert abs(float(w.mean()) - float(w0.mean())) < 0.003
    assert abs(float(w.std()) - float(w0.std())) < 0.003


def test_learned_deviation_decays_with_tau_prior():
    """Far from 0 the reflection is inactive: E[w_t] = w0·exp(-t/τ)."""
    cfg = _cfg()
    tau, t = 400.0, 200
    w = _simulate(jnp.full((100_000,), 1.5), tau, t, cfg)
    assert abs(float(w.mean()) - 1.5 * math.exp(-t / tau)) < 0.005


def test_fixed_noise_mode_without_noise_is_pure_decay():
    cfg = _cfg(**{"plasticity.noise_mode": "fixed", "plasticity.noise_sigma": 0.0})
    w = _simulate(jnp.full((10,), 1.0), 50.0, 50, cfg)
    assert np.allclose(np.asarray(w), math.exp(-1.0), rtol=1e-5)


def test_firewall_under_forgetting():
    """Non-plastic and off-edge synapses are untouched by decay and noise."""
    from esp.rollout import lifetime
    cfg = replace(_cfg(), **{"genome.n_hidden_max": 16, "genome.init_hidden_active": 8,
                             "task.horizon": 10, "task.substeps": 5, "task.switch_every": 5,
                             "io.n_reward_channels": 2, "genome.tau_prior_range": (50.0, 60.0)})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    half = jax.random.uniform(jax.random.PRNGKey(5), g.plastic.shape) < 0.5
    g = g._replace(plastic=half, rule=jnp.zeros(N_RULE_PARAMS))   # fastest forgetting
    _, _, (ph, c) = lifetime(g, jax.random.PRNGKey(1), cfg)
    frozen = ~ph.plastic_mask
    assert np.array_equal(np.asarray(c.w[frozen]), np.asarray(ph.w0[frozen]))
    assert bool(jnp.all(c.w[~connected(g, cfg)] == 0.0))
    assert bool(jnp.any(c.w[ph.plastic_mask] != ph.w0[ph.plastic_mask]))


def test_matched_requires_forgetting():
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **{"plasticity.noise_mode": "matched"})


def test_tau_prior_decoded_and_floored():
    from esp.genome.direct import decode_rule
    cfg = _cfg()
    lo = decode_rule(jnp.zeros(N_RULE_PARAMS), cfg).tau_prior
    hi = decode_rule(jnp.ones(N_RULE_PARAMS), cfg).tau_prior
    assert math.isclose(float(lo), cfg.genome.tau_prior_range[0], rel_tol=1e-5)
    assert math.isclose(float(hi), cfg.genome.tau_prior_range[1], rel_tol=1e-4)


def test_stationary_law_independent_of_interval():
    """Coarse intervals (one step per env step) keep the exact newborn law."""
    cfg, tau = _cfg(), 100.0
    rp, w0 = _rp(tau), jnp.full((200_000,), 1.5)
    mask = jnp.ones_like(w0, bool)

    def step(w, k):
        return plasticity.forget(w, jax.random.normal(k, w.shape), mask, rp, cfg, 10.0), None

    w, _ = jax.lax.scan(step, w0, jax.random.split(jax.random.PRNGKey(3), 300))   # 30 τ
    assert abs(float(w.mean()) - S * math.sqrt(2 / math.pi)) < 0.003
    assert abs(float(w.std()) - S * math.sqrt(1 - 2 / math.pi)) < 0.003


def test_forgetting_interval_floor_enforced():
    with pytest.raises(ValueError):
        _cfg(**{"genome.tau_prior_range": (50.0, 1e5)})     # < 10 × 10 ms
