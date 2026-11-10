"""
rollout.py — The one composition seam (plan §2).

    rollout(genome, key, cfg) -> (fitness, metrics)

One lifetime: decode → reset → scan over env steps, each an inner scan over
`substeps` neural steps of dt:  encode → ALIF → plasticity → readout.

Data partition = the firewall (plan §4):
  closure (evolved-frozen within a lifetime): phenotype.sign_conn,
      phenotype.plastic_mask, phenotype.rp, phenotype.active
  carry   (plastic):  w, traces, neuron state, modulator state, env state, key
Weights are carry; nothing the network does can reach the closure.

Pure and jit/vmap-able. `make_population_eval` vmaps over (population ×
n_evals lifetimes); each lifetime gets fresh weights when weights are
non-heritable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import io, neuromod, plasticity, tasks
from .config import ExperimentConfig
from .genome import Genome, decode
from .substrate import alif


class Carry(NamedTuple):
    neuron: alif.ALIFState
    w: jnp.ndarray
    traces: plasticity.Traces
    mod_state: jnp.ndarray
    m: jnp.ndarray            # current third factor (held across substeps)
    env_state: tuple
    obs: jnp.ndarray
    reward: jnp.ndarray       # last reward (feeds grounding channels)
    readout: jnp.ndarray      # [n_out] filtered output spikes
    key: jax.Array
    # accumulators
    ret: jnp.ndarray
    n_spikes: jnp.ndarray
    n_silent: jnp.ndarray
    abs_dw: jnp.ndarray


def rollout(genome: Genome, key: jax.Array, cfg: ExperimentConfig):
    fitness, metrics, _ = lifetime(genome, key, cfg)
    return fitness, metrics


def lifetime(genome: Genome, key: jax.Array, cfg: ExperimentConfig):
    """rollout plus (phenotype, final carry) — for analysis and firewall tests."""
    task = tasks.get(cfg.task.name)
    learns = cfg.plasticity.rule != "identity" or cfg.plasticity.noise_sigma > 0
    noisy = cfg.plasticity.noise_sigma > 0
    n_out = cfg.genome.n_out

    k_dec, k_env, k_run = jax.random.split(key, 3)
    ph = decode(genome, k_dec, cfg)
    w_eff_static = ph.sign_conn * ph.w0          # used only when nothing learns

    def substep(c: Carry, _):
        k_next, k_enc, k_noise = jax.random.split(c.key, 3)
        x = io.encode(k_enc, c.obs, c.reward, cfg)
        pre = jnp.concatenate([x, c.neuron.z])
        w_eff = ph.sign_conn * c.w if learns else w_eff_static
        neuron, z = alif.step(c.neuron, pre, w_eff, ph.active, cfg.alif, cfg.dt)

        if learns:
            noise = jax.random.normal(k_noise, c.w.shape) if noisy else None
            traces, dw = plasticity.update(c.traces, pre, z, c.m, c.w, noise, ph.rp, cfg)
            w = plasticity.apply(c.w, dw, ph.plastic_mask, cfg)
            abs_dw = c.abs_dw + jnp.sum(jnp.abs(w - c.w))
        else:
            traces, w, abs_dw = c.traces, c.w, c.abs_dw

        readout = io.update_readout(c.readout, z[-n_out:], cfg)
        c = c._replace(neuron=neuron, w=w, traces=traces, readout=readout, key=k_next,
                       n_spikes=c.n_spikes + jnp.sum(z), abs_dw=abs_dw)
        return c, None

    def env_step(c: Carry, _):
        c, _ = jax.lax.scan(substep, c, None, length=cfg.task.substeps)
        action = io.decode(c.readout, cfg)
        env_state, obs, reward, done = task.step(c.env_state, action, cfg)
        mod_state, m = neuromod.modulate(c.mod_state, reward, cfg)
        c = c._replace(env_state=env_state, obs=obs, reward=reward, mod_state=mod_state, m=m,
                       ret=c.ret + reward, n_silent=c.n_silent + (action < 0))
        return c, None

    env_state, obs = task.reset(k_env, cfg)
    zero = jnp.zeros((), jnp.float32)
    c0 = Carry(
        neuron=alif.init_state(cfg.n_neurons), w=ph.w0, traces=plasticity.init_traces(cfg),
        mod_state=neuromod.init(cfg), m=zero, env_state=env_state, obs=obs, reward=zero,
        readout=jnp.zeros(n_out, jnp.float32), key=k_run,
        ret=zero, n_spikes=zero, n_silent=jnp.zeros((), jnp.int32), abs_dw=zero,
    )
    c, _ = jax.lax.scan(env_step, c0, None, length=cfg.task.horizon)

    fitness = c.ret / task.max_return(cfg)
    T_s = cfg.lifetime_steps * cfg.dt * 1e-3
    n_active = jnp.maximum(jnp.sum(ph.active), 1)
    metrics = {
        "return": c.ret,
        "rate_hz": c.n_spikes / n_active / T_s,
        "frac_silent": c.n_silent / cfg.task.horizon,
        "sum_abs_dw": c.abs_dw,
    }
    return fitness, metrics, (ph, c)


def make_population_eval(cfg: ExperimentConfig):
    """jit(pop_genomes, key) -> (mean fitness [P], mean metrics {k: [P]}) over n_evals lifetimes."""
    P, E = cfg.evo.population_size, cfg.evo.n_evals

    def per_genome(g, keys):
        f, m = jax.vmap(lambda k: rollout(g, k, cfg))(keys)
        return jnp.mean(f), jax.tree_util.tree_map(jnp.mean, m)

    @jax.jit
    def evaluate(pop: Genome, key: jax.Array):
        keys = jax.random.split(key, P * E).reshape(P, E, -1)
        return jax.vmap(per_genome)(pop, keys)

    return evaluate
