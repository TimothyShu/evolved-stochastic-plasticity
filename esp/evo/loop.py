"""
evo/loop.py — Evolutionary loop, ported from ctrnn_evo.evolution.

Touches only genomes and scalar fitnesses (plan §3.7). The rollout is injected
as `evaluate(pop, key) -> (fitness_raw [P], metrics)`, so evo never sees inside
a lifetime.

Kept from ctrnn_evo: tournament selection, elitism (1), percentage-fitness
penalties (penalty = frac·f_raw·C/C0), penalty warm-up ramp, cyclic penalty
loosening, mutation-sigma warm-up, resumable state snapshots, early-stop helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from ..config import ExperimentConfig
from ..genome.direct import Genome, connected, decode_rule, grounding_rows, random_genome
from .mutation import mutate


# ── population ───────────────────────────────────────────────────────────────

def init_population(key, cfg: ExperimentConfig) -> Genome:
    keys = jax.random.split(key, cfg.evo.population_size)
    return jax.vmap(lambda k: random_genome(k, cfg))(keys)


# ── costs (ported from ctrnn_evo.cost) ──────────────────────────────────────

def edge_count_cost(g: Genome, cfg: ExperimentConfig) -> jnp.ndarray:
    return jnp.sum(connected(g, cfg)).astype(jnp.float32)


def dist_cost(g: Genome, cfg: ExperimentConfig) -> jnp.ndarray:
    """Wire length over neuron→neuron edges (inputs have no position)."""
    d = jnp.linalg.norm(g.position[:, None, :] - g.position[None, :, :], axis=-1)
    return jnp.sum(d * connected(g, cfg)[cfg.n_in:])


def _warmup_ramp(gen, ec):
    return min(gen / ec.penalty_warmup_gens, 1.0) if ec.penalty_warmup_gens > 0 else 1.0


def _cycle_ramp(gen, ec):
    if ec.penalty_cycle_gens > 0 and ec.penalty_cycle_free_gens > 0 and gen >= ec.penalty_warmup_gens:
        pos = (gen - ec.penalty_warmup_gens) % ec.penalty_cycle_gens
        if pos >= ec.penalty_cycle_gens - ec.penalty_cycle_free_gens:
            return 0.0
    return 1.0


def mutation_scale(gen, ec):
    if ec.mutation_warmup_scale > 1.0 and ec.penalty_warmup_gens > 0:
        r = min(gen / ec.penalty_warmup_gens, 1.0)
        return ec.mutation_warmup_scale * (1.0 - r) + r
    return 1.0


def adjusted_fitness(f_raw, pop: Genome, cfg: ExperimentConfig, gen: int):
    ec = cfg.evo
    ramp = _warmup_ramp(gen, ec) * _cycle_ramp(gen, ec)
    if ramp == 0.0 or (ec.edge_frac == 0.0 and ec.dist_frac == 0.0):
        return f_raw
    c_edge = jax.vmap(lambda g: edge_count_cost(g, cfg))(pop)
    c_dist = jax.vmap(lambda g: dist_cost(g, cfg))(pop)
    pen = ec.edge_frac * c_edge / ec.C0_edge + ec.dist_frac * c_dist / ec.C0_wiring
    return f_raw * (1.0 - ramp * pen)


# ── selection / reproduction ─────────────────────────────────────────────────

def tournament_select(key, fitness, k):
    cand = jax.random.choice(key, fitness.shape[0], (k,), replace=False)
    return cand[jnp.argmax(fitness[cand])]


def make_evolve_step(cfg: ExperimentConfig):
    P, T = cfg.evo.population_size, cfg.evo.tournament_size

    @jax.jit
    def evolve_step(key, pop: Genome, fitness, sigma_scale):
        k_sel, k_mut = jax.random.split(key)
        parents = jax.vmap(lambda kk: tournament_select(kk, fitness, T))(jax.random.split(k_sel, P))
        off = jax.tree_util.tree_map(lambda x: x[parents], pop)
        off = jax.vmap(lambda kk, g: mutate(kk, g, cfg, sigma_scale))(jax.random.split(k_mut, P), off)
        best = jnp.argmax(fitness)
        return jax.tree_util.tree_map(lambda o, x: o.at[0].set(x[best]), off, pop)   # elitism

    return evolve_step


def rule_stats(pop: Genome, fitness, cfg: ExperimentConfig) -> dict:
    """Evolved plasticity meta-parameters, in physical units: population mean and
    the best genome's values. plastic_frac = flagged-plastic share of connected,
    non-grounding synapses (the genome's flags — reported even under the identity
    rule, where decode zeroes the mask, so drift in the flags stays visible)."""
    rp = jax.vmap(lambda r: decode_rule(r, cfg))(pop.rule)

    def frac(g):
        eligible = connected(g, cfg) & ~grounding_rows(cfg)[:, None]
        return jnp.sum(g.plastic & eligible) / jnp.maximum(jnp.sum(eligible), 1)

    pf = jax.vmap(frac)(pop)
    b = int(jnp.argmax(fitness))
    out = {}
    for name, v in [("eta", rp.eta), ("a_minus", rp.a_minus), ("tau_elig", rp.tau_elig),
                    ("tau_pre", rp.tau_pre), ("tau_post", rp.tau_post), ("plastic_frac", pf)]:
        out[f"mean_{name}"] = float(jnp.mean(v))
        out[f"best_{name}"] = float(v[b])
    # η × plastic share: one number for "how much can this population learn"
    out["mean_eff_eta"] = float(jnp.mean(rp.eta * pf))
    return out


def collect_stats(gen, fitness, f_raw, metrics, pop: Genome, cfg) -> dict:
    stats = {
        "generation": gen,
        "max_fitness": float(jnp.max(fitness)),
        "mean_fitness": float(jnp.mean(fitness)),
        "max_f_raw": float(jnp.max(f_raw)),
        "mean_n_active": float(jnp.mean(jnp.sum(pop.active, axis=-1))),
        "mean_edges": float(jnp.mean(jax.vmap(lambda g: edge_count_cost(g, cfg))(pop))),
    }
    for k, v in metrics.items():
        stats[f"mean_{k}"] = float(jnp.mean(v))
    stats.update(rule_stats(pop, fitness, cfg))
    return stats


# ── driver ───────────────────────────────────────────────────────────────────

def run_evolution(
    key,
    cfg: ExperimentConfig,
    evaluate: Callable,
    callback: Callable | None = None,
    early_stop_fn: Callable | None = None,
    resume_from: str | Path | None = None,
    state_checkpoint_dir: str | Path | None = None,
    state_checkpoint_every: int = 50,
):
    from ..infra.logger import load_training_state, save_training_state

    evolve_step = make_evolve_step(cfg)

    if resume_from is not None:
        pop, fitness, f_raw, key, start_gen = load_training_state(resume_from, cfg)
        metrics = {}
    else:
        start_gen = 0
        key, k_init, k_eval = jax.random.split(key, 3)
        pop = init_population(k_init, cfg)
        f_raw, metrics = evaluate(pop, k_eval)
        fitness = adjusted_fitness(f_raw, pop, cfg, 0)

    if state_checkpoint_dir is not None:
        Path(state_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    history = []
    for gen in range(start_gen, cfg.evo.n_generations):
        stats = collect_stats(gen, fitness, f_raw, metrics, pop, cfg)
        history.append(stats)
        if callback is not None:
            b = int(jnp.argmax(fitness))
            callback(stats, jax.tree_util.tree_map(lambda x: x[b], pop))
        if early_stop_fn is not None and early_stop_fn(stats):
            break

        key, k_step, k_eval = jax.random.split(key, 3)
        pop = evolve_step(k_step, pop, fitness, mutation_scale(gen, cfg.evo))
        f_raw, metrics = evaluate(pop, k_eval)
        fitness = adjusted_fitness(f_raw, pop, cfg, gen + 1)

        nxt = gen + 1
        if state_checkpoint_dir is not None and nxt % state_checkpoint_every == 0:
            save_training_state(Path(state_checkpoint_dir) / f"state_gen_{nxt:06d}.npz",
                                pop, fitness, f_raw, key, nxt)

    b = int(jnp.argmax(fitness))
    return jax.tree_util.tree_map(lambda x: x[b], pop), fitness, history


def fitness_threshold(min_fitness: float):
    return lambda s: s["max_fitness"] >= min_fitness


def convergence_stop(window: int = 50, tol: float = 1e-3):
    recent: list[float] = []

    def _check(s):
        recent.append(s["max_fitness"])
        return len(recent) >= window and recent[-1] - recent[-window] < tol

    return _check
