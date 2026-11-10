"""
genome/direct.py — Direct (padded, fixed-capacity) encoding.

    random_genome(key, cfg) -> Genome
    decode(genome, key, cfg) -> Phenotype

Layout
------
Neurons: N = n_hidden_max + n_out slots; outputs are the last n_out (always
active, excitatory). Weight-like arrays are [n_pre, N] with the presynaptic axis
ordered (input channels ‖ neurons): row i < n_in is input channel i, row n_in+j
is neuron j.

Heritability (validity control, plan §6 Stage 2)
------------------------------------------------
Heritable: topology (active, edge_mask), E/I types (⇒ synaptic signs, Dale's
law), positions, per-synapse plasticity flags, rule meta-parameters.
Weights: heritable only if cfg.genome.inherit_weights. Otherwise `decode` draws
magnitudes from a fixed, non-heritable |N(0, w_init_std)| every lifetime and the
`w` field is ignored (and not mutated), so no weight information can be
smuggled through the init distribution. Signs remain inherited via E/I type —
structural information, declared as such.

Firewall (plan §4)
------------------
plastic_mask = genome.plastic ∧ connected ∧ ¬grounding. Grounding rows (the
reward/interoceptive input channels) are excluded by construction, so no genome
and no rule can make them plastic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..config import ExperimentConfig
from ..plasticity import RuleParams

N_RULE_PARAMS = 5   # tau_pre, tau_post, tau_elig, eta, a_minus — normalised to [0, 1]


class Genome(NamedTuple):
    active: jnp.ndarray     # [N] bool
    inhib: jnp.ndarray      # [N] bool — Dale's law type (True = inhibitory)
    position: jnp.ndarray   # [N, 2] in [0, 1]² (wiring cost; spatial decoder later)
    edge_mask: jnp.ndarray  # [n_pre, N] bool
    plastic: jnp.ndarray    # [n_pre, N] bool — heritable plasticity flag
    w: jnp.ndarray          # [n_pre, N] magnitudes (used iff inherit_weights)
    rule: jnp.ndarray       # [N_RULE_PARAMS] in [0, 1]


class Phenotype(NamedTuple):
    w0: jnp.ndarray            # [n_pre, N] initial magnitudes for this lifetime
    sign_conn: jnp.ndarray     # [n_pre, N] float: Dale sign × connectivity (0 = no synapse)
    plastic_mask: jnp.ndarray  # [n_pre, N] bool — the firewall
    active: jnp.ndarray        # [N] bool
    rp: RuleParams


# ── index helpers ────────────────────────────────────────────────────────────

def hidden_slots(cfg: ExperimentConfig) -> jnp.ndarray:
    return jnp.arange(cfg.n_neurons) < cfg.genome.n_hidden_max


def output_slots(cfg: ExperimentConfig) -> jnp.ndarray:
    return ~hidden_slots(cfg)


def grounding_rows(cfg: ExperimentConfig) -> jnp.ndarray:
    """[n_pre] bool: reward/interoceptive input channels (never plastic)."""
    start = cfg.n_obs * cfg.io.in_pop
    idx = jnp.arange(cfg.n_pre)
    return (idx >= start) & (idx < cfg.n_in)


def pre_active(genome: Genome, cfg: ExperimentConfig) -> jnp.ndarray:
    return jnp.concatenate([jnp.ones(cfg.n_in, bool), genome.active])


def connected(genome: Genome, cfg: ExperimentConfig) -> jnp.ndarray:
    """[n_pre, N] bool: edges whose both endpoints are active."""
    return genome.edge_mask & pre_active(genome, cfg)[:, None] & genome.active[None, :]


# ── structural invariant (ported from ctrnn_evo.prune_isolated) ─────────────

def prune_isolated(genome: Genome, cfg: ExperimentConfig) -> Genome:
    """Deactivate hidden neurons lacking an in-edge or out-edge (self-loops don't
    count) from another active node, iterated to a fixed point — removing one
    neuron can orphan its neighbours (ctrnn_evo's single pass missed this
    cascade). Then clear dead edges."""
    hidden = hidden_slots(cfg)
    not_self = ~jnp.eye(cfg.n_pre, cfg.n_neurons, k=-cfg.n_in, dtype=bool)

    def prune_once(active):
        g = genome._replace(active=active)
        conn = connected(g, cfg) & not_self
        has_in = jnp.any(conn, axis=0)
        has_out = jnp.any(conn[cfg.n_in:, :], axis=1)
        return active & ((has_in & has_out) | ~hidden)

    def cond(s):
        prev, cur = s
        return jnp.any(prev != cur)

    def body(s):
        _, cur = s
        return cur, prune_once(cur)

    _, active = jax.lax.while_loop(cond, body, (genome.active, prune_once(genome.active)))
    g = genome._replace(active=active)
    return g._replace(edge_mask=connected(g, cfg))


def random_genome(key: jax.Array, cfg: ExperimentConfig) -> Genome:
    gc = cfg.genome
    N, n_pre = cfg.n_neurons, cfg.n_pre
    k = jax.random.split(key, 7)

    perm = jax.random.permutation(k[0], gc.n_hidden_max)
    hidden_on = jnp.zeros(gc.n_hidden_max, bool).at[perm[: gc.init_hidden_active]].set(True)
    active = jnp.concatenate([hidden_on, jnp.ones(gc.n_out, bool)])

    inhib = (jax.random.uniform(k[1], (N,)) < gc.frac_inhibitory) & hidden_slots(cfg)
    position = jax.random.uniform(k[2], (N, 2))
    w = jnp.abs(jax.random.normal(k[3], (n_pre, N))) * gc.w_init_std
    edge_mask = jax.random.uniform(k[4], (n_pre, N)) < gc.init_edge_density
    plastic = jnp.ones((n_pre, N), bool)
    rule = jax.random.uniform(k[5], (N_RULE_PARAMS,))

    g = Genome(active, inhib, position, edge_mask, plastic, w, rule)
    return prune_isolated(g._replace(edge_mask=connected(g, cfg)), cfg)


# ── decode ───────────────────────────────────────────────────────────────────

def _log_interp(u, lo, hi):
    return jnp.exp(jnp.log(lo) + u * (jnp.log(hi) - jnp.log(lo)))


def decode_rule(rule: jnp.ndarray, cfg: ExperimentConfig) -> RuleParams:
    """Map normalised genes to physical units; evolved taus clipped to ≥ floor·dt."""
    gc = cfg.genome
    floor = gc.tau_floor_factor * cfg.dt
    u = jnp.clip(rule, 0.0, 1.0)
    tau_pre = jnp.maximum(_log_interp(u[0], *gc.tau_trace_range), floor)
    tau_post = jnp.maximum(_log_interp(u[1], *gc.tau_trace_range), floor)
    tau_elig = jnp.maximum(_log_interp(u[2], *gc.tau_elig_range), floor)
    eta = gc.eta_range[0] + u[3] * (gc.eta_range[1] - gc.eta_range[0])
    a_minus = 2.0 * u[4]
    return RuleParams(tau_pre, tau_post, tau_elig, eta, a_minus)


def decode(genome: Genome, key: jax.Array, cfg: ExperimentConfig) -> Phenotype:
    gc = cfg.genome
    conn = connected(genome, cfg)
    if gc.inherit_weights:
        w0 = genome.w
    else:
        w0 = jnp.abs(jax.random.normal(key, genome.w.shape)) * gc.w_init_std
    w0 = jnp.clip(w0, 0.0, cfg.plasticity.w_max) * conn

    sign_pre = jnp.concatenate([jnp.ones(cfg.n_in), jnp.where(genome.inhib, -1.0, 1.0)])
    sign_conn = sign_pre[:, None] * conn

    plastic_mask = genome.plastic & conn & ~grounding_rows(cfg)[:, None]
    if cfg.plasticity.rule == "identity":
        plastic_mask = jnp.zeros_like(plastic_mask)

    return Phenotype(w0, sign_conn, plastic_mask, genome.active, decode_rule(genome.rule, cfg))
