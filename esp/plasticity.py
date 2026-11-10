"""
plasticity.py — Online weight updates inside the lifetime scan.

    init_traces(cfg) -> traces
    update(traces, pre, post, modulator, w, noise, rp, cfg) -> (traces, dw)
    apply(w, dw, plastic_mask, cfg) -> w

The rule is a deterministic function of its inputs; stochasticity enters only
through the `noise` array, sampled by the rollout scan from the carried key.
`noise=None` (when cfg.plasticity.noise_sigma == 0) skips the N² draw entirely.

The firewall (plan §4) lives in `apply`: dw is multiplied by `plastic_mask`
before it touches w, so frozen entries (grounding pathway, non-edges) are
bit-identical across the whole lifetime regardless of what the rule computes.

Weights are non-negative magnitudes; Dale's sign is applied downstream by the
phenotype, so LTP on an inhibitory synapse strengthens inhibition.

Rules
-----
identity      dw = 0 (Stage 0).
three_factor  Spike-native pair-based STDP feeding an eligibility trace, gated
              by the third factor m (Stage 1):
                  x_pre  <- d_pre·x_pre  + pre        (count traces)
                  x_post <- d_post·x_post + post
                  stdp    = x_pre ⊗ post − a_minus · pre ⊗ x_post
                  e      <- d_e·e + stdp
                  dw      = η·m·e  (+ σ·√dt·ξ,  ξ ~ N(0, 1))
              Noise std scales as √dt so σ is a per-√ms diffusion coefficient
              and the weight random walk is dt-invariant.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

from .config import ExperimentConfig


class RuleParams(NamedTuple):
    """Decoded (physical-unit) rule meta-parameters; evolved, frozen per lifetime."""
    tau_pre: jnp.ndarray
    tau_post: jnp.ndarray
    tau_elig: jnp.ndarray
    eta: jnp.ndarray
    a_minus: jnp.ndarray


class Traces(NamedTuple):
    x_pre: jnp.ndarray    # [n_pre]
    x_post: jnp.ndarray   # [N]
    elig: jnp.ndarray     # [n_pre, N]  (shape (0, 0) for identity: nothing carried)


def init_traces(cfg: ExperimentConfig) -> Traces:
    if cfg.plasticity.rule == "identity":
        z = jnp.zeros((0,), jnp.float32)
        return Traces(z, z, jnp.zeros((0, 0), jnp.float32))
    return Traces(
        x_pre=jnp.zeros(cfg.n_pre, jnp.float32),
        x_post=jnp.zeros(cfg.n_neurons, jnp.float32),
        elig=jnp.zeros((cfg.n_pre, cfg.n_neurons), jnp.float32),
    )


def update(
    traces: Traces,
    pre: jnp.ndarray,           # [n_pre]
    post: jnp.ndarray,          # [N]
    modulator: jnp.ndarray,     # scalar third factor
    w: jnp.ndarray,             # [n_pre, N] (unused by current rules; weight-dependent rules)
    noise: jnp.ndarray | None,  # [n_pre, N] standard normal, or None
    rp: RuleParams,
    cfg: ExperimentConfig,
) -> tuple[Traces, jnp.ndarray]:
    rule = cfg.plasticity.rule
    if rule == "identity":
        dw = jnp.zeros_like(w)
    elif rule == "three_factor":
        dt = cfg.dt
        d_pre, d_post, d_e = (jnp.exp(-dt / t) for t in (rp.tau_pre, rp.tau_post, rp.tau_elig))
        stdp = jnp.outer(traces.x_pre, post) - rp.a_minus * jnp.outer(pre, traces.x_post)
        traces = Traces(
            x_pre=d_pre * traces.x_pre + pre,
            x_post=d_post * traces.x_post + post,
            elig=d_e * traces.elig + stdp,
        )
        dw = rp.eta * modulator * traces.elig
    else:
        raise ValueError(f"unknown plasticity rule '{rule}'")

    if noise is not None:
        dw = dw + cfg.plasticity.noise_sigma * math.sqrt(cfg.dt) * noise
    return traces, dw


def apply(w: jnp.ndarray, dw: jnp.ndarray, plastic_mask: jnp.ndarray, cfg: ExperimentConfig) -> jnp.ndarray:
    """Firewall: only plastic_mask entries can change; result clipped to [0, w_max]."""
    w_new = jnp.clip(w + dw, 0.0, cfg.plasticity.w_max)
    return jnp.where(plastic_mask, w_new, w)
