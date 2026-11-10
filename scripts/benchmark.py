#!/usr/bin/env python3
"""
benchmark.py — Stage 0 throughput gate (plan §6).

Measures genome·lifetime-timesteps/sec (one "lifetime timestep" = one neural dt
step of one genome's lifetime) for the full rollout — ALIF scan + online weight
carry — swept over network size, population and plasticity rule. Compile time is
excluded; each point is timed over --repeats evaluations after a warm-up call.

    python scripts/benchmark.py --hidden 64,256,512 --pop 256,1024 \
        --rules identity,three_factor --out bench.csv

Why a sweep, not one number: with plastic weights every genome carries
[n_pre, N] weights (+ an eligibility trace of the same shape) through the scan,
so each step streams ~P·N² floats. Expect the regime to flip from compute- to
memory-bandwidth-bound as N grows; the crossover decides Stage 2 sizing.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esp.config import load_config, replace  # noqa: E402
from esp.evo import init_population  # noqa: E402
from esp.rollout import make_population_eval  # noqa: E402


def ints(s):
    return [int(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/stage0_reflex.yaml"))
    p.add_argument("--hidden", type=ints, default=[64, 256])
    p.add_argument("--pop", type=ints, default=[256, 1024])
    p.add_argument("--rules", default="identity,three_factor")
    p.add_argument("--noise", type=float, default=0.0, help="plasticity.noise_sigma (adds N² RNG per step)")
    p.add_argument("--n-evals", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="bench.csv")
    a = p.parse_args()

    base = load_config(a.config)
    print(f"device: {jax.devices()[0]}  dt={base.dt} ms  lifetime={base.lifetime_steps} steps")
    rows = []
    for rule in a.rules.split(","):
        for h in a.hidden:
            for P in a.pop:
                cfg = replace(base, **{
                    "genome.n_hidden_max": h, "genome.init_hidden_active": h // 2,
                    "evo.population_size": P, "evo.n_evals": a.n_evals,
                    "plasticity.rule": rule, "plasticity.noise_sigma": a.noise,
                })
                pop = init_population(jax.random.PRNGKey(0), cfg)
                ev = make_population_eval(cfg)
                try:
                    t0 = time.perf_counter()
                    jax.block_until_ready(ev(pop, jax.random.PRNGKey(1)))
                    compile_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for r in range(a.repeats):
                        jax.block_until_ready(ev(pop, jax.random.PRNGKey(2 + r)))
                    run_s = (time.perf_counter() - t0) / a.repeats
                except Exception as e:  # e.g. OOM at large P·N²
                    print(f"{rule:13s} N={cfg.n_neurons:4d} P={P:5d}  FAILED: {type(e).__name__}")
                    continue
                gts = P * a.n_evals * cfg.lifetime_steps / run_s
                row = {"rule": rule, "n_neurons": cfg.n_neurons, "n_pre": cfg.n_pre, "pop": P,
                       "n_evals": a.n_evals, "lifetime_steps": cfg.lifetime_steps,
                       "compile_s": round(compile_s, 2), "eval_s": round(run_s, 4),
                       "genome_steps_per_s": round(gts)}
                rows.append(row)
                print(f"{rule:13s} N={cfg.n_neurons:4d} P={P:5d}  {run_s:8.3f} s/eval  "
                      f"{gts:12.3e} genome·steps/s  (compile {compile_s:.1f}s)", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
