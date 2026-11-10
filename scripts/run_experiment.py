#!/usr/bin/env python3
"""
run_experiment.py — Run evolution for a YAML experiment config.

    python scripts/run_experiment.py --config configs/stage0_reflex.yaml \
        --n-replicates 3 --set evo.population_size=512 --set plasticity.rule=three_factor

Forge-queue compatible: every param maps to a flag (FORGE_PARAM_* → --flag);
the queue forces --output-dir. Writes per-replicate run dirs plus results.json
(consumed by scripts/forge_emit.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esp.config import load_config, replace  # noqa: E402
from esp.evo import run_evolution  # noqa: E402
from esp.evo.loop import convergence_stop  # noqa: E402
from esp.infra.logger import latest_state_checkpoint, make_logger, make_run_dir, save_config  # noqa: E402
from esp.infra.notify import ntfy  # noqa: E402
from esp.rollout import make_population_eval  # noqa: E402

SHORTCUTS = {  # flag -> dotted config key
    "pop_size": "evo.population_size",
    "n_generations": "evo.n_generations",
    "n_evals": "evo.n_evals",
    "rule": "plasticity.rule",
    "horizon": "task.horizon",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n-replicates", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted override, YAML-parsed value (repeatable)")
    for flag in SHORTCUTS:
        p.add_argument("--" + flag.replace("_", "-"), default=None)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--early-stop-window", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="resume each replicate from its latest state")
    p.add_argument("--smoke-test", action="store_true", help="tiny run: 1 rep × 3 gens × pop 16")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    for flag, key in SHORTCUTS.items():
        v = getattr(a, flag)
        if v is not None:
            overrides[key] = yaml.safe_load(v)
    if a.smoke_test:
        overrides.update({"evo.population_size": 16, "evo.n_generations": 3, "evo.n_evals": 1})
        a.n_replicates = 1

    cfg = load_config(a.config, overrides)
    if a.seed is not None:
        cfg = replace(cfg, seed=a.seed)

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluate = make_population_eval(cfg)
    results = []

    for rep in range(a.n_replicates):
        run_dir = make_run_dir(out, f"{cfg.name}_rep{rep:02d}")
        save_config(run_dir, cfg)
        key = jax.random.PRNGKey(cfg.seed * 1000 + rep)
        resume = latest_state_checkpoint(run_dir) if a.resume else None
        t0 = time.time()
        best, fitness, hist = run_evolution(
            key, cfg, evaluate,
            callback=make_logger(run_dir, a.checkpoint_every, verbose=not a.quiet),
            early_stop_fn=convergence_stop(a.early_stop_window) if a.early_stop_window else None,
            resume_from=resume,
            state_checkpoint_dir=run_dir / "checkpoints",
            state_checkpoint_every=a.checkpoint_every,
        )
        elapsed = time.time() - t0
        r = {
            "condition": cfg.plasticity.rule, "replicate": rep, "run_dir": str(run_dir),
            "final_max_fitness": float(fitness.max()), "generations_run": len(hist),
            "best_n_active": int(best.active.sum()), "elapsed_s": round(elapsed, 1),
        }
        results.append(r)
        print(json.dumps(r))
        ntfy(f"{cfg.name} rep {rep}: max fitness {r['final_max_fitness']:.3f} in {elapsed/60:.1f} min")

    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
