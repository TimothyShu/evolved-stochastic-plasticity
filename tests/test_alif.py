import math

import jax.numpy as jnp

from esp.config import ALIFConfig
from esp.substrate import alif

CFG = ALIFConfig()
DT = 1.0


def run(drive, n_steps, cfg=CFG, n=1):
    """Constant presynaptic drive via a single always-on input of weight `drive`."""
    s = alif.init_state(n)
    pre = jnp.ones(1)
    w = jnp.full((1, n), drive)
    zs, states = [], []
    for _ in range(n_steps):
        s, z = alif.step(s, pre, w, jnp.ones(n, bool), cfg, DT)
        zs.append(float(z[0]))
        states.append(s)
    return zs, states


def test_silent_without_input():
    zs, _ = run(0.0, 100)
    assert sum(zs) == 0


def test_strong_input_spikes():
    zs, _ = run(5.0, 50)
    assert sum(zs) > 0


def test_refractory_enforced():
    n_ref = alif.n_ref_steps(CFG, DT)
    zs, _ = run(100.0, 200)             # saturating drive
    idx = [i for i, z in enumerate(zs) if z]
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert min(gaps) == n_ref + 1        # fire, n_ref blocked steps, fire


def test_adaptation_raises_threshold_and_slows_rate():
    # Near-threshold drive (v_ss ≈ 2): rate is set by integration, not refractoriness.
    zs, states = run(0.1, 3000)
    zb, _ = run(0.1, 3000, ALIFConfig(beta=0.0))
    assert float(states[-1].a[0]) > 0
    assert sum(zs[:200]) == sum(zb[:200])      # identical before a accumulates
    assert sum(zs[-1000:]) < sum(zb[-1000:])   # spike-frequency adaptation at steady state


def test_inactive_never_spikes():
    s = alif.init_state(2)
    s, z = alif.step(s, jnp.ones(1), jnp.full((1, 2), 100.0), jnp.array([True, False]), CFG, DT)
    assert z.tolist() == [1.0, 0.0]


def test_membrane_decay_is_exact_exponential():
    s = alif.init_state(1)._replace(v=jnp.array([0.5]))
    s, _ = alif.step(s, jnp.zeros(1), jnp.zeros((1, 1)), jnp.ones(1, bool), CFG, DT)
    assert math.isclose(float(s.v[0]), 0.5 * math.exp(-DT / CFG.tau_m), rel_tol=1e-6)
