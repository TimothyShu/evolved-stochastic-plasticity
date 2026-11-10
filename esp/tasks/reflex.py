"""
tasks/reflex.py — Stage 0 loop exerciser.

A one-hot cue (n_cues options) is shown; the agent must emit the matching
action. The cue is resampled every `switch_every` env steps; the first
`grace_steps` after each switch are unscored, so pure latency isn't punished.
Reward is 1 for a correct action on a scored step, else 0.

Solvable by topology alone (cue_k inputs → output_k), so it tests the loop,
not learning. gymnax-style signature; the key lives in the env state.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..config import ExperimentConfig


class ReflexState(NamedTuple):
    t: jnp.ndarray
    cue: jnp.ndarray
    key: jax.Array


def _obs(cue, cfg):
    return jax.nn.one_hot(cue, cfg.task.n_cues)


def reset(key: jax.Array, cfg: ExperimentConfig):
    k_cue, k_state = jax.random.split(key)
    cue = jax.random.randint(k_cue, (), 0, cfg.task.n_cues)
    s = ReflexState(t=jnp.zeros((), jnp.int32), cue=cue, key=k_state)
    return s, _obs(cue, cfg)


def step(state: ReflexState, action: jnp.ndarray, cfg: ExperimentConfig):
    tc = cfg.task
    phase = state.t % tc.switch_every
    scored = phase >= tc.grace_steps
    reward = jnp.where(scored & (action == state.cue), 1.0, 0.0)

    t = state.t + 1
    key, k_cue = jax.random.split(state.key)
    new_cue = jax.random.randint(k_cue, (), 0, tc.n_cues)
    cue = jnp.where(t % tc.switch_every == 0, new_cue, state.cue)
    done = t >= tc.horizon
    return ReflexState(t, cue, key), _obs(cue, cfg), reward, done


def max_return(cfg: ExperimentConfig) -> float:
    tc = cfg.task
    return float(sum(1 for t in range(tc.horizon) if t % tc.switch_every >= tc.grace_steps))
