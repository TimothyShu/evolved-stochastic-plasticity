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


# ── forget_target: own_init and none ─────────────────────────────────────────

def _run_target(target, w_start, w0, tau, n_steps, sigma, interval=1.0, seed=0):
    cfg = _cfg(**{"plasticity.forget_target": target, "plasticity.noise_mode": "fixed",
                  "plasticity.noise_sigma": sigma})
    rp, mask = _rp(tau), jnp.ones_like(w_start, bool)

    def step(w, k):
        xi = jax.random.normal(k, w.shape)
        return plasticity.forget(w, xi, mask, rp, cfg, interval, w0), None

    w, _ = jax.lax.scan(step, w_start, jax.random.split(jax.random.PRNGKey(seed), n_steps))
    return w


def test_own_init_returns_each_synapse_to_its_own_default():
    """σ = 0: deterministic per-synapse relaxation w0 + (w − w0)·e^(−t/τ)."""
    w0 = jnp.array([0.05, 0.3, 0.8, 1.5])
    w = _run_target("own_init", jnp.full(4, 1.0), w0, 100.0, 200, sigma=0.0)
    expect = w0 + (1.0 - w0) * math.exp(-2.0)
    assert np.allclose(np.asarray(w), np.asarray(expect), rtol=1e-4)


def test_own_init_with_noise_settles_around_own_default():
    w0 = jnp.repeat(jnp.array([0.6, 1.2]), 50_000)
    w = _run_target("own_init", jnp.full_like(w0, 1.9), w0, 100.0, 2000, sigma=0.002)
    lo, hi = np.asarray(w[:50_000]), np.asarray(w[50_000:])
    assert abs(lo.mean() - 0.6) < 0.003 and abs(hi.mean() - 1.2) < 0.003
    sd = 0.002 * math.sqrt(100.0 / 2)                 # OU stationary std σ√(τ/2)
    assert abs(lo.std() - sd) < 0.002


def test_random_walk_is_mean_preserving_away_from_bounds():
    """none: martingale in the interior; variance grows as σ²t; τ_prior ignored."""
    sigma, t = 0.01, 400
    w = _run_target("none", jnp.full((100_000,), 1.0), None, 1e9, t, sigma)
    assert abs(float(w.mean()) - 1.0) < 0.002
    assert abs(float(w.std()) - sigma * math.sqrt(t)) < 0.003
    w_fast = _run_target("none", jnp.full((1000,), 1.0), None, 5.0, t, sigma)   # τ irrelevant
    assert np.array_equal(np.asarray(w_fast), np.asarray(_run_target("none", jnp.full((1000,), 1.0),
                                                                      None, 1e9, t, sigma)))


def test_random_walk_long_run_is_uniform_on_bounds():
    """The documented caveat: bounded symmetric walk -> uniform[0, w_max], mean w_max/2."""
    w = _run_target("none", jnp.full((100_000,), 0.24), None, 1e9, 3000, sigma=0.1)
    w_max = ExperimentConfig().plasticity.w_max
    assert float(w.min()) >= 0.0 and float(w.max()) <= w_max
    assert abs(float(w.mean()) - w_max / 2) < 0.02
    assert abs(float(w.std()) - w_max / math.sqrt(12)) < 0.02


@pytest.mark.parametrize("target,mode,sigma", [("own_init", "matched", 0.0), ("none", "matched", 0.0),
                                               ("none", "fixed", 0.0), ("bogus", "fixed", 0.01)])
def test_invalid_target_configs_rejected(target, mode, sigma):
    with pytest.raises(ValueError):
        _cfg(**{"plasticity.forget_target": target, "plasticity.noise_mode": mode,
                "plasticity.noise_sigma": sigma})


@pytest.mark.parametrize("target", ["own_init", "none"])
def test_firewall_under_each_target(target):
    from esp.rollout import lifetime
    cfg = replace(_cfg(), **{"genome.n_hidden_max": 16, "genome.init_hidden_active": 8,
                             "task.horizon": 10, "task.substeps": 5, "task.switch_every": 5,
                             "io.n_reward_channels": 2, "genome.tau_prior_range": (50.0, 60.0),
                             "plasticity.forget_target": target, "plasticity.noise_mode": "fixed",
                             "plasticity.noise_sigma": 0.05})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    half = jax.random.uniform(jax.random.PRNGKey(5), g.plastic.shape) < 0.5
    g = g._replace(plastic=half, rule=jnp.zeros(N_RULE_PARAMS))
    _, _, (ph, c) = lifetime(g, jax.random.PRNGKey(1), cfg)
    frozen = ~ph.plastic_mask
    assert np.array_equal(np.asarray(c.w[frozen]), np.asarray(ph.w0[frozen]))
    assert bool(jnp.all(c.w[~connected(g, cfg)] == 0.0))
    assert bool(jnp.any(c.w[ph.plastic_mask] != ph.w0[ph.plastic_mask]))
