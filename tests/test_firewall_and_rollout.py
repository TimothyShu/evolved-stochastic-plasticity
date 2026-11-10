import jax
import jax.numpy as jnp
import numpy as np

from esp import plasticity
from esp.config import replace
from esp.genome import decode, random_genome
from esp.genome.direct import grounding_rows
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
    assert bool(jnp.any(ph.sign_conn[gr] != 0))       # wired, but frozen


def test_grounding_weights_bit_identical_after_learning(cfg):
    """Full lifetime with strong noisy learning: grounding weights unchanged."""
    from esp.rollout import lifetime
    cfg = replace(cfg, **{"io.n_reward_channels": 2, "plasticity.rule": "three_factor",
                          "plasticity.noise_sigma": 0.1, "genome.init_edge_density": 1.0})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    # every synapse flagged plastic in the genome — the mask must still hold
    g = g._replace(plastic=jnp.ones_like(g.plastic), rule=jnp.full(5, 1.0))
    _, m, (ph, c) = jax.jit(lambda k: lifetime(g, k, cfg))(jax.random.PRNGKey(3))
    gr = grounding_rows(cfg)
    assert float(m["sum_abs_dw"]) > 0                                      # learning happened
    assert bool(jnp.any(c.w[~gr] != ph.w0[~gr]))
    assert np.array_equal(np.asarray(c.w[gr]), np.asarray(ph.w0[gr]))     # grounding bit-identical


def test_identity_rule_freezes_everything(cfg):
    g = random_genome(jax.random.PRNGKey(0), cfg)
    _, m = rollout(g, jax.random.PRNGKey(1), cfg)
    assert float(m["sum_abs_dw"]) == 0.0


def test_three_factor_changes_weights(cfg):
    cfg = replace(cfg, **{"plasticity.rule": "three_factor", "plasticity.noise_sigma": 0.01})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    _, m = rollout(g, jax.random.PRNGKey(1), cfg)
    assert float(m["sum_abs_dw"]) > 0.0


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
    g = random_genome(jax.random.PRNGKey(0), cfg)._replace(rule=jnp.zeros(5))
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
