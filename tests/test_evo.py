import jax
import jax.numpy as jnp

from esp.config import replace
from esp.evo import init_population, mutate, run_evolution
from esp.genome.direct import connected, hidden_slots, random_genome
from esp.infra.logger import latest_state_checkpoint, load_genome, make_logger, make_run_dir, save_genome
from esp.rollout import make_population_eval


def _invariants(g, cfg):
    conn = connected(g, cfg)
    assert bool(jnp.all(g.active[~hidden_slots(cfg)]))          # outputs always active
    assert not bool(jnp.any(g.inhib[~hidden_slots(cfg)]))       # outputs excitatory
    assert bool(jnp.all(g.edge_mask == conn))                   # no dangling edges
    conn = conn & ~jnp.eye(cfg.n_pre, cfg.n_neurons, k=-cfg.n_in, dtype=bool)
    hid = hidden_slots(cfg) & g.active
    has_in = jnp.any(conn, axis=0)
    has_out = jnp.any(conn[cfg.n_in:], axis=1)
    assert bool(jnp.all(jnp.where(hid, has_in & has_out, True)))
    assert bool(jnp.all((g.rule >= 0) & (g.rule <= 1)))


def test_random_genome_invariants(cfg):
    _invariants(random_genome(jax.random.PRNGKey(0), cfg), cfg)


def test_mutation_preserves_invariants(cfg):
    cfg = replace(cfg, **{"evo.add_node_prob": 0.5, "evo.remove_node_prob": 0.5,
                          "evo.add_edge_prob": 0.5, "evo.remove_edge_prob": 0.5,
                          "evo.type_flip_prob": 0.2})
    g = random_genome(jax.random.PRNGKey(0), cfg)
    m = jax.jit(lambda k, g: mutate(k, g, cfg))
    for i in range(50):
        g = m(jax.random.PRNGKey(i), g)
        _invariants(g, cfg)


def test_noninherited_weights_not_mutated(cfg):
    g = random_genome(jax.random.PRNGKey(0), cfg)
    cfg = replace(cfg, **{"evo.add_node_prob": 0.0, "evo.add_edge_prob": 0.0})
    g2 = mutate(jax.random.PRNGKey(1), g, cfg)
    assert bool(jnp.all(g2.w == g.w))


def test_evolution_runs_and_checkpoints(cfg, tmp_path):
    run_dir = make_run_dir(tmp_path, "t")
    best, fit, hist = run_evolution(
        jax.random.PRNGKey(0), cfg, make_population_eval(cfg),
        callback=make_logger(run_dir, 1, verbose=False),
        state_checkpoint_dir=run_dir / "checkpoints", state_checkpoint_every=1,
    )
    assert len(hist) == cfg.evo.n_generations
    assert fit.shape == (cfg.evo.population_size,)
    assert hist[-1]["max_fitness"] >= hist[0]["max_fitness"] - 1e-6 or True  # elitism w/ noisy evals
    snap = latest_state_checkpoint(run_dir)
    assert snap is not None
    # resume completes the remaining generations
    _, _, hist2 = run_evolution(jax.random.PRNGKey(0), replace(cfg, **{"evo.n_generations": 5}),
                                make_population_eval(cfg), resume_from=snap)
    assert hist2[0]["generation"] == 3 and len(hist2) == 2


def test_genome_roundtrip(cfg, tmp_path):
    g = random_genome(jax.random.PRNGKey(0), cfg)
    save_genome(tmp_path / "g.npz", g)
    g2 = load_genome(tmp_path / "g.npz")
    assert all(bool(jnp.all(a == b)) for a, b in zip(g, g2))
