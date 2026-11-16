"""Weight-perturbation rule: noise kicks + leaky trace, credited by baseline-subtracted reward."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from esp import neuromod, noise, plasticity
from esp.config import ExperimentConfig, replace
from esp.genome.direct import N_RULE_PARAMS, grounding_rows, random_genome

DELTA = 10.0   # env-step interval (ms)


def _cfg(**kw):
    return replace(ExperimentConfig(), **{
        "plasticity.rule": "perturbation", "plasticity.forgetting": True,
        "plasticity.forget_target": "none", "plasticity.noise_mode": "fixed",
        "plasticity.noise_sigma": 0.002, "neuromod.baseline_tau": 200.0, **kw})


def _rp(eta, tau_n):
    one = jnp.asarray(1.0)
    return plasticity.RuleParams(one, one, jnp.asarray(tau_n), jnp.asarray(eta), one, jnp.asarray(1e9))


def _toy(cfg, eta, n_steps, d=400, seed=0, shuffle_reward=False, dist="rademacher"):
    """R = g·w (+ small observation noise). Same order as the rollout:
    reward -> δ -> credit trace -> forget/kick -> trace += kick."""
    kg, kr = jax.random.split(jax.random.PRNGKey(seed))
    g = jax.random.normal(kg, (d,))
    g = g / jnp.linalg.norm(g)
    rp, mask = _rp(eta, 200.0), jnp.ones(d, bool)
    _, b = plasticity.forget_coeffs(rp, cfg, DELTA)
    w0 = jnp.full(d, 1.0)

    def step(carry, k):
        w, n, mod_state = carry
        k_obs, k_xi, k_shuf = jax.random.split(k, 3)
        r = g @ (w - w0) + 0.001 * jax.random.normal(k_obs)
        if shuffle_reward:                       # control: reward uncorrelated with w
            r = 0.001 * jax.random.normal(k_shuf)
        mod_state, m = neuromod.modulate(mod_state, r, cfg)
        w = plasticity.credit_noise(w, n, m, mask, rp, cfg)
        xi = noise.sample(k_xi, (d,), dist)
        w = plasticity.forget(w, xi, mask, rp, cfg, DELTA)
        n = plasticity.update_noise_trace(n, b * xi, rp, DELTA)
        return (w, n, mod_state), None

    (w, _, _), _ = jax.lax.scan(step, (w0, jnp.zeros(d), neuromod.init(cfg)),
                                jax.random.split(kr, n_steps))
    dw = w - w0
    return float(g @ dw), float(g @ dw / (jnp.linalg.norm(dw) + 1e-12))


@pytest.mark.parametrize("dist", ["rademacher", "gaussian"])
def test_climbs_reward_gradient(dist):
    """Gradient-aligned displacement grows far beyond the random-walk scale."""
    cfg, n = _cfg(), 4000
    gain, cos = _toy(cfg, eta=2.0, n_steps=n, dist=dist)
    walk = 0.002 * math.sqrt(DELTA * n)          # std of g·Δw for the noise alone (|g| = 1)
    assert gain > 5 * walk                        # 12 seeds: gain 3.1 ± 0.2 vs walk 0.4 (both dists)
    assert cos > 0.2                              # 12 seeds: 0.26 (both); random direction ~1/√400 = 0.05


def test_no_learning_without_reward_correlation():
    cfg, n = _cfg(), 4000
    gain, cos = _toy(cfg, eta=2.0, n_steps=n, shuffle_reward=True)
    walk = 0.002 * math.sqrt(DELTA * n)
    assert abs(gain) < 4 * walk
    assert abs(cos) < 0.2


def test_eta_zero_is_pure_forgetting_noise():
    cfg, n = _cfg(), 2000
    gain, _ = _toy(cfg, eta=0.0, n_steps=n)
    assert abs(gain) < 4 * 0.002 * math.sqrt(DELTA * n)


def test_trace_decay_floored_at_ten_intervals():
    lam_short = float(plasticity.noise_trace_decay(_rp(0.0, 5.0), DELTA))     # τ_elig < 10Δ
    assert math.isclose(lam_short, math.exp(-0.1), rel_tol=1e-6)
    lam_long = float(plasticity.noise_trace_decay(_rp(0.0, 500.0), DELTA))
    assert math.isclose(lam_long, math.exp(-DELTA / 500.0), rel_tol=1e-6)


@pytest.mark.parametrize("kw", [{"plasticity.forgetting": False, "plasticity.noise_mode": "fixed"},
                                {"plasticity.noise_sigma": 0.0, "plasticity.forget_target": "own_init"}])
def test_requires_noisy_forgetting(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


@pytest.mark.parametrize("target", ["newborn", "own_init", "none"])
def test_lifetime_learns_and_respects_firewall(target):
    from esp.rollout import lifetime
    mode = "matched" if target == "newborn" else "fixed"
    cfg = _cfg(**{"plasticity.forget_target": target, "plasticity.noise_mode": mode,
                  "genome.n_hidden_max": 16, "genome.init_hidden_active": 8,
                  "task.horizon": 20, "task.substeps": 5, "task.switch_every": 5,
                  "io.n_reward_channels": 2, "genome.init_edge_density": 1.0,
                  "genome.tau_prior_range": (50.0, 60.0)})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    g = g._replace(plastic=jnp.ones_like(g.plastic), rule=jnp.full(N_RULE_PARAMS, 1.0))
    _, m, (ph, c) = lifetime(g, jax.random.PRNGKey(3), cfg)
    gr = grounding_rows(cfg)
    assert c.traces.elig.shape == (cfg.n_pre, cfg.n_neurons) and c.traces.x_pre.shape == (0,)
    assert float(m["net_abs_dw"]) > 0
    assert np.array_equal(np.asarray(c.w[gr]), np.asarray(ph.w0[gr]))
    assert bool(jnp.any(c.traces.elig != 0))
