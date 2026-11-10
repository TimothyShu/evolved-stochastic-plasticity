"""
neuromod.py — Produces the third factor (modulator).

    init(cfg) -> mod_state
    modulate(mod_state, reward, cfg) -> (mod_state, m)

kind "external" (Stages 1–2): m is the task reward, optionally baseline-
subtracted (δ = r − r̄, r̄ a running mean with time constant baseline_tau, in ms
of lifetime). The modulator is computed once per env step and held across that
step's neural substeps.

Stage 3 adds kind "critic" (a learned readout producing self-δ). Its grounding
wiring is frozen by the plastic_mask produced in genome.decode — not here.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from .config import ExperimentConfig


def init(cfg: ExperimentConfig) -> jnp.ndarray:
    return jnp.zeros((), jnp.float32)   # running reward baseline r̄


def modulate(state: jnp.ndarray, reward: jnp.ndarray, cfg: ExperimentConfig):
    nm = cfg.neuromod
    if nm.kind != "external":
        raise ValueError(f"neuromod kind '{nm.kind}' not implemented before Stage 3")
    if nm.baseline_tau > 0:
        env_dt = cfg.dt * cfg.task.substeps
        d = math.exp(-env_dt / nm.baseline_tau)
        m = reward - state
        state = d * state + (1.0 - d) * reward
        return state, m
    return state, reward
