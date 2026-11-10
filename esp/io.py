"""
io.py — encode(obs) -> input spikes; decode(output traces) -> action.

Encoder "poisson": each obs dimension in [0, 1] drives `in_pop` channels that
spike independently with p = r_max·obs·dt (Bernoulli approximation of Poisson,
valid while r_max·dt ≪ 1; asserted below). Reward/grounding channels are
appended after the obs channels and rate-code max(r, 0) the same way.

Decoder "argmax": output neurons are low-pass filtered (τ_readout); the action is
the argmax trace, or -1 (no action) when no output has spiked recently.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .config import ExperimentConfig


def encode(key: jax.Array, obs: jnp.ndarray, reward: jnp.ndarray, cfg: ExperimentConfig) -> jnp.ndarray:
    p_max = cfg.io.r_max * cfg.dt * 1e-3
    if p_max > 0.2:
        raise ValueError(f"r_max·dt = {p_max:.3f}: Bernoulli approx to Poisson invalid; lower r_max")
    rates = jnp.repeat(jnp.clip(obs, 0.0, 1.0), cfg.io.in_pop)
    if cfg.io.n_reward_channels:
        r = jnp.clip(reward, 0.0, 1.0)
        rates = jnp.concatenate([rates, jnp.full(cfg.io.n_reward_channels, r)])
    return (jax.random.uniform(key, rates.shape) < rates * p_max).astype(jnp.float32)


def readout_decay(cfg: ExperimentConfig) -> float:
    return math.exp(-cfg.dt / cfg.io.tau_readout)


def update_readout(trace: jnp.ndarray, out_spikes: jnp.ndarray, cfg: ExperimentConfig) -> jnp.ndarray:
    d = readout_decay(cfg)
    return d * trace + (1.0 - d) * out_spikes


def decode(trace: jnp.ndarray, cfg: ExperimentConfig) -> jnp.ndarray:
    """Discrete action in {-1, 0, ..., n_out-1}; -1 = silent outputs."""
    silent = jnp.max(trace) <= 1e-6
    return jnp.where(silent, -1, jnp.argmax(trace)).astype(jnp.int32)
