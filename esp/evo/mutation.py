"""
evo/mutation.py — Mutation operators, ported from ctrnn_evo.mutation and
adapted to the esp genome (pre axis = inputs ‖ neurons; E/I types; plasticity
flags; normalised rule genes). All operators are pure and vmap-able.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import EvoConfig, ExperimentConfig
from ..genome.direct import Genome, connected, hidden_slots, pre_active, prune_isolated


def _random_eligible(key, eligible):
    """Random True index of a 1-D mask → (idx, any). Ported verbatim in spirit."""
    scores = jnp.where(eligible, jax.random.uniform(key, eligible.shape), -1.0)
    return jnp.argmax(scores), jnp.any(eligible)


def _scaled(sigma, scale):
    return sigma * scale


# ── parameter mutations ──────────────────────────────────────────────────────

def perturb_rule(key, g: Genome, sigma: float) -> Genome:
    return g._replace(rule=jnp.clip(g.rule + sigma * jax.random.normal(key, g.rule.shape), 0.0, 1.0))


def perturb_weights(key, g: Genome, sigma: float, w_max: float) -> Genome:
    w = jnp.clip(g.w + sigma * jax.random.normal(key, g.w.shape), 0.0, w_max)
    return g._replace(w=w)


def perturb_position(key, g: Genome, sigma: float) -> Genome:
    return g._replace(position=jnp.clip(g.position + sigma * jax.random.normal(key, g.position.shape), 0.0, 1.0))


def type_flip(key, g: Genome, cfg: ExperimentConfig, p: float) -> Genome:
    flip = (jax.random.uniform(key, g.inhib.shape) < p) & hidden_slots(cfg)
    return g._replace(inhib=g.inhib ^ flip)


def plastic_flip(key, g: Genome, p: float) -> Genome:
    flip = jax.random.uniform(key, g.plastic.shape) < p
    return g._replace(plastic=g.plastic ^ flip)


# ── structural mutations ─────────────────────────────────────────────────────

def add_node(key, g: Genome, cfg: ExperimentConfig) -> Genome:
    """Activate the first free hidden slot with one in-edge and one out-edge."""
    k_t, k_p, k_src, k_dst, k_wi, k_wo = jax.random.split(key, 6)
    free = hidden_slots(cfg) & ~g.active
    slot, any_free = jnp.argmax(free), jnp.any(free)

    src, any_src = _random_eligible(k_src, pre_active(g, cfg))   # pre row
    dst, any_dst = _random_eligible(k_dst, g.active)             # post col
    row = cfg.n_in + slot
    wi = jnp.abs(jax.random.normal(k_wi)) * cfg.genome.w_init_std + 0.01
    wo = jnp.abs(jax.random.normal(k_wo)) * cfg.genome.w_init_std + 0.01

    new = g._replace(
        active=g.active.at[slot].set(True),
        inhib=g.inhib.at[slot].set(jax.random.uniform(k_t) < cfg.genome.frac_inhibitory),
        position=g.position.at[slot].set(jax.random.uniform(k_p, (2,))),
        edge_mask=g.edge_mask.at[src, slot].set(True).at[row, dst].set(True),
        w=g.w.at[src, slot].set(wi).at[row, dst].set(wo),
    )
    ok = any_free & any_src & any_dst
    return jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, g)


def remove_node(key, g: Genome, cfg: ExperimentConfig) -> Genome:
    slot, ok = _random_eligible(key, hidden_slots(cfg) & g.active)
    row = cfg.n_in + slot
    new = g._replace(
        active=g.active.at[slot].set(False),
        edge_mask=g.edge_mask.at[:, slot].set(False).at[row, :].set(False),
    )
    return jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, g)


def add_edge(key, g: Genome, cfg: ExperimentConfig) -> Genome:
    k_s, k_w = jax.random.split(key)
    eligible = pre_active(g, cfg)[:, None] & g.active[None, :] & ~g.edge_mask
    flat, ok = _random_eligible(k_s, eligible.reshape(-1))
    w_new = jnp.abs(jax.random.normal(k_w)) * cfg.genome.w_init_std + 0.01
    em = g.edge_mask.reshape(-1).at[flat].set(True).reshape(g.edge_mask.shape)
    w = g.w.reshape(-1).at[flat].set(w_new).reshape(g.w.shape)
    return g._replace(edge_mask=jnp.where(ok, em, g.edge_mask), w=jnp.where(ok, w, g.w))


def remove_edge(key, g: Genome, cfg: ExperimentConfig) -> Genome:
    """Remove an edge unless it would strand a hidden endpoint (ported rule)."""
    conn = connected(g, cfg)
    hidden = hidden_slots(cfg)
    hidden_pre = jnp.concatenate([jnp.zeros(cfg.n_in, bool), hidden])
    out_deg = jnp.concatenate([jnp.zeros(cfg.n_in), jnp.sum(conn[cfg.n_in:], axis=1)])  # by pre row
    in_deg = jnp.sum(conn, axis=0)                                                      # by post col
    strand_src = hidden_pre[:, None] & (out_deg[:, None] <= 1)
    strand_dst = hidden[None, :] & (in_deg[None, :] <= 1)
    eligible = conn & ~strand_src & ~strand_dst
    flat, ok = _random_eligible(key, eligible.reshape(-1))
    em = g.edge_mask.reshape(-1).at[flat].set(False).reshape(g.edge_mask.shape)
    return g._replace(edge_mask=jnp.where(ok, em, g.edge_mask))


# ── combined operator ────────────────────────────────────────────────────────

def mutate(key, g: Genome, cfg: ExperimentConfig, sigma_scale: float = 1.0) -> Genome:
    ec: EvoConfig = cfg.evo
    k = jax.random.split(key, 14)

    g = perturb_rule(k[0], g, ec.rule_sigma * sigma_scale)
    if cfg.genome.inherit_weights:           # non-heritable weights are never mutated
        g = perturb_weights(k[1], g, ec.weight_sigma * sigma_scale, cfg.plasticity.w_max)
    g = perturb_position(k[2], g, ec.position_sigma * sigma_scale)
    g = type_flip(k[3], g, cfg, ec.type_flip_prob)
    g = plastic_flip(k[4], g, ec.plastic_flip_prob)

    ops = [
        (ec.add_node_prob, add_node), (ec.remove_node_prob, remove_node),
        (ec.add_edge_prob, add_edge), (ec.remove_edge_prob, remove_edge),
    ]
    for i, (p, op) in enumerate(ops):
        do = jax.random.uniform(k[5 + i]) < p
        g = jax.lax.cond(do, lambda g_, kk=k[9 + i], op=op: op(kk, g_, cfg), lambda g_: g_, g)

    return prune_isolated(g, cfg)
