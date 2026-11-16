import jax
import jax.numpy as jnp
import numpy as np

from esp import plasticity
from esp.config import replace
from esp.genome import decode, random_genome
from esp.genome.direct import N_RULE_PARAMS, grounding_rows
from esp.rollout import make_population_eval, rollout


def test_apply_respects_mask(cfg):
    w = jnp.ones((4, 3))
    mask = jnp.zeros((4, 3), bool).at[0, 0].set(True)
    w2 = plasticity.apply(w, jnp.full((4, 3), 0.5), mask, cfg)
    assert float(w2[0, 0]) == 1.5
    assert np.array_equal(np.asarray(w2.at[0, 0].set(1.0)), np.asarray(w))


def test_grounding_rows_never_plastic(cfg):
    cfg = replace(cfg, **{"io.n_reward_channels": 2, "plasticity.rule": "three_factor",
                          "genome.init_edge_density": 1.0})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    ph = decode(g, jax.random.PRNGKey(1), cfg)
    gr = grounding_rows(cfg)
    assert int(gr.sum()) == 2
    assert not bool(jnp.any(ph.plastic_mask[gr]))
    assert bool(jnp.any(ph.w0[gr] != 0))           # wired, but frozen


def test_grounding_weights_bit_identical_after_learning(cfg):
    """Full lifetime with strong noisy learning: grounding weights unchanged."""
    from esp.rollout import lifetime
    cfg = replace(cfg, **{"io.n_reward_channels": 2, "plasticity.rule": "three_factor",
                          "plasticity.noise_sigma": 0.1, "genome.init_edge_density": 1.0})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    # every synapse flagged plastic in the genome — the mask must still hold
    g = g._replace(plastic=jnp.ones_like(g.plastic), rule=jnp.full(N_RULE_PARAMS, 1.0))
    _, m, (ph, c) = jax.jit(lambda k: lifetime(g, k, cfg))(jax.random.PRNGKey(3))
    gr = grounding_rows(cfg)
    assert float(m["net_abs_dw"]) > 0                                      # learning happened
    assert bool(jnp.any(c.w[~gr] != ph.w0[~gr]))
    assert np.array_equal(np.asarray(c.w[gr]), np.asarray(ph.w0[gr]))     # grounding bit-identical


def test_identity_rule_freezes_everything(cfg):
    g = random_genome(jax.random.PRNGKey(0), cfg)
    _, m = rollout(g, jax.random.PRNGKey(1), cfg)
    assert float(m["net_abs_dw"]) == 0.0


def test_three_factor_changes_weights(cfg):
    cfg = replace(cfg, **{"plasticity.rule": "three_factor", "plasticity.noise_sigma": 0.01})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    _, m = rollout(g, jax.random.PRNGKey(1), cfg)
    assert float(m["net_abs_dw"]) > 0.0


def test_rollout_deterministic_given_key(cfg):
    g = random_genome(jax.random.PRNGKey(0), cfg)
    a = rollout(g, jax.random.PRNGKey(5), cfg)
    b = rollout(g, jax.random.PRNGKey(5), cfg)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: bool(jnp.all(x == y)), a, b))


def test_weights_redrawn_per_lifetime(cfg):
    g = random_genome(jax.random.PRNGKey(0), cfg)
    w1 = decode(g, jax.random.PRNGKey(1), cfg).w0
    w2 = decode(g, jax.random.PRNGKey(2), cfg).w0
    assert not bool(jnp.all(w1 == w2))
    cfg_inh = replace(cfg, **{"genome.inherit_weights": True})
    assert bool(jnp.all(decode(g, jax.random.PRNGKey(1), cfg_inh).w0 ==
                        decode(g, jax.random.PRNGKey(2), cfg_inh).w0))


def test_evolved_taus_clipped_to_dt_floor(cfg):
    g = random_genome(jax.random.PRNGKey(0), cfg)._replace(rule=jnp.zeros(N_RULE_PARAMS))
    rp = decode(g, jax.random.PRNGKey(0), cfg).rp
    floor = cfg.genome.tau_floor_factor * cfg.dt
    assert min(float(rp.tau_pre), float(rp.tau_post), float(rp.tau_elig)) >= floor


def test_population_eval_shapes(cfg):
    from esp.evo import init_population
    pop = init_population(jax.random.PRNGKey(0), cfg)
    f, m = make_population_eval(cfg)(pop, jax.random.PRNGKey(1))
    assert f.shape == (cfg.evo.population_size,)
    assert bool(jnp.all((f >= 0) & (f <= 1)))
    assert m["rate_hz"].shape == (cfg.evo.population_size,)


def test_scan_unroll_is_semantically_neutral(cfg):
    cfg = replace(cfg, **{"plasticity.rule": "three_factor", "plasticity.noise_sigma": 0.01})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    a = rollout(g, jax.random.PRNGKey(7), cfg)
    b = rollout(g, jax.random.PRNGKey(7), replace(cfg, scan_unroll=5))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: bool(jnp.allclose(x, y)), a, b))


def test_off_edge_weights_stay_exactly_zero(cfg):
    """No sign/connectivity matrix any more: non-edges must stay 0 via w0 + plastic_mask."""
    from esp.genome.direct import connected
    from esp.rollout import lifetime
    cfg = replace(cfg, **{"plasticity.rule": "three_factor", "plasticity.noise_sigma": 0.1})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    g = g._replace(plastic=jnp.ones_like(g.plastic), rule=jnp.full(N_RULE_PARAMS, 1.0))
    _, _, (ph, c) = lifetime(g, jax.random.PRNGKey(1), cfg)
    off = ~connected(g, cfg)
    assert bool(jnp.all(c.w[off] == 0.0))


# ── pure STDP rule ───────────────────────────────────────────────────────────

def _stdp_setup():
    from esp.config import ExperimentConfig
    cfg = replace(ExperimentConfig(), **{"plasticity.rule": "stdp"})
    one = jnp.asarray(1.0)
    rp = plasticity.RuleParams(jnp.asarray(20.0), jnp.asarray(20.0), one, jnp.asarray(0.01),
                               jnp.asarray(1.0), one)
    return cfg, rp


def _pair(cfg, rp, pre_first: bool, gap: int = 5):
    """One synapse 0→0: pre at t=0 and post at t=gap (or reversed). Returns summed dw."""
    tr = plasticity.Traces(jnp.zeros(1), jnp.zeros(1), jnp.zeros((0, 0)))
    total = 0.0
    for t in range(gap + 1):
        pre = jnp.array([1.0 if t == (0 if pre_first else gap) else 0.0])
        post = jnp.array([1.0 if t == (gap if pre_first else 0) else 0.0])
        tr, dw = plasticity.update(tr, pre, post, jnp.asarray(0.0), jnp.zeros((1, 1)), None, rp, cfg)
        total += float(dw[0, 0])
    return total


def test_stdp_causal_potentiates_anticausal_depresses():
    import math
    cfg, rp = _stdp_setup()
    ltp, ltd = _pair(cfg, rp, True), _pair(cfg, rp, False)
    assert math.isclose(ltp, 0.01 * math.exp(-5 / 20.0), rel_tol=1e-5)    # η·e^(−Δt/τ_pre)
    assert math.isclose(ltd, -0.01 * math.exp(-5 / 20.0), rel_tol=1e-5)   # −η·a₋·e^(−Δt/τ_post)


def test_stdp_ignores_modulator_and_carries_no_eligibility(cfg):
    cfg = replace(cfg, **{"plasticity.rule": "stdp"})
    tr = plasticity.init_traces(cfg)
    assert tr.elig.shape == (0, 0)
    _, rp = _stdp_setup()
    pre = jnp.ones(cfg.n_pre)
    post = jnp.ones(cfg.n_neurons)
    w = jnp.zeros((cfg.n_pre, cfg.n_neurons))
    tr = tr._replace(x_pre=jnp.ones(cfg.n_pre))
    _, dw0 = plasticity.update(tr, pre, post, jnp.asarray(0.0), w, None, rp, cfg)
    _, dw1 = plasticity.update(tr, pre, post, jnp.asarray(5.0), w, None, rp, cfg)
    assert bool(jnp.all(dw0 == dw1)) and bool(jnp.any(dw0 != 0))


def test_stdp_lifetime_learns_and_respects_firewall(cfg):
    from esp.rollout import lifetime
    cfg = replace(cfg, **{"plasticity.rule": "stdp", "io.n_reward_channels": 2,
                          "genome.init_edge_density": 1.0})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    g = g._replace(plastic=jnp.ones_like(g.plastic), rule=jnp.full(N_RULE_PARAMS, 1.0))
    _, m, (ph, c) = lifetime(g, jax.random.PRNGKey(3), cfg)
    gr = grounding_rows(cfg)
    assert float(m["net_abs_dw"]) > 0
    assert np.array_equal(np.asarray(c.w[gr]), np.asarray(ph.w0[gr]))


def test_unknown_rule_rejected():
    import pytest
    from esp.config import ExperimentConfig
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **{"plasticity.rule": "hebb"})
