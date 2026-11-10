"""
substrate/alif.py — Adaptive leaky integrate-and-fire (ALIF), hard spikes.

    step(state, pre_spikes, w_eff, cfg, dt) -> (state, spikes)

Dynamics per step (exact exponential decay, so unconditionally stable; the
dt <= tau/10 bound is an accuracy constraint on spike timing):

    v  <- α·v + W_effᵀ·pre            α = exp(-dt/τ_m)
    A   = v_th + β·a                   adaptive threshold
    z   = [v >= A] ∧ [ref == 0]        hard Heaviside — no surrogate
    v  <- v − z·A                      soft reset (LSNN convention)
    a  <- ρ·a + (1−ρ)·z                ρ = exp(-dt/τ_a)
    ref<- z ? n_ref : max(ref−1, 0)    n_ref = ceil(t_ref/dt)

Synaptic input is a delta jump of W per presynaptic spike (units of v_th), so
the PSP amplitude per spike is dt-invariant; input firing rates are specified in
Hz by io, so total drive is dt-invariant as well.

`a` is the slow variable that bridges task delays (τ_a ≫ τ_m).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

from ..config import ALIFConfig


class ALIFState(NamedTuple):
    v: jnp.ndarray      # [N] membrane potential
    a: jnp.ndarray      # [N] adaptation variable
    ref: jnp.ndarray    # [N] int32 refractory counter (steps remaining)
    z: jnp.ndarray      # [N] spikes emitted on the previous step (float 0/1)


def init_state(n: int) -> ALIFState:
    zeros = jnp.zeros(n, jnp.float32)
    return ALIFState(v=zeros, a=zeros, ref=jnp.zeros(n, jnp.int32), z=zeros)


def n_ref_steps(cfg: ALIFConfig, dt: float) -> int:
    return int(math.ceil(cfg.t_ref / dt))


def step(
    state: ALIFState,
    pre: jnp.ndarray,       # [n_pre] presynaptic spikes (inputs ++ previous z)
    w_eff: jnp.ndarray,     # [n_pre, N] signed effective weights
    active: jnp.ndarray,    # [N] bool — inactive slots never spike
    cfg: ALIFConfig,
    dt: float,
) -> tuple[ALIFState, jnp.ndarray]:
    alpha = math.exp(-dt / cfg.tau_m)
    rho = math.exp(-dt / cfg.tau_a)

    v = alpha * state.v + pre @ w_eff
    thr = cfg.v_th + cfg.beta * state.a
    z = ((v >= thr) & (state.ref == 0) & active).astype(jnp.float32)
    v = v - z * thr
    a = rho * state.a + (1.0 - rho) * z
    ref = jnp.where(z > 0, n_ref_steps(cfg, dt), jnp.maximum(state.ref - 1, 0))
    return ALIFState(v=v, a=a, ref=ref.astype(jnp.int32), z=z), z
