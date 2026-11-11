"""
infra/logger.py — Run directories, history, genome archives, resumable state.
Ported from ctrnn_evo.logger; genome (de)serialisation is now generic over the
Genome NamedTuple's fields, so new genome fields need no logger changes.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ..config import ExperimentConfig, from_dict, to_dict
from ..genome.direct import Genome

_POP = "pop_"


def make_run_dir(base: str | Path = "runs", run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(base) / run_id
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    return d


def save_config(run_dir: Path, cfg: ExperimentConfig) -> None:
    (Path(run_dir) / "config.yaml").write_text(yaml.safe_dump(_listify(to_dict(cfg)), sort_keys=False))


def load_run_config(run_dir: Path) -> ExperimentConfig:
    return from_dict(yaml.safe_load((Path(run_dir) / "config.yaml").read_text()))


def _listify(x):
    if isinstance(x, dict):
        return {k: _listify(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return list(x)
    return x


def save_genome(path: str | Path, g: Genome) -> None:
    np.savez(path, **{f: np.asarray(getattr(g, f)) for f in Genome._fields})


def load_genome(path: str | Path) -> Genome:
    with np.load(path) as z:
        return Genome(*(jnp.asarray(z[f]) for f in Genome._fields))


def append_history(run_dir: Path, stats: dict) -> None:
    with open(Path(run_dir) / "history.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")


def load_history(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "history.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def make_logger(run_dir: Path, checkpoint_every: int = 50, verbose: bool = True) -> Callable:
    run_dir = Path(run_dir)
    t0 = time.time()

    def callback(stats: dict, best: Genome) -> None:
        stats = {**stats, "elapsed_s": round(time.time() - t0, 2)}
        append_history(run_dir, stats)
        g = stats["generation"]
        if checkpoint_every and g % checkpoint_every == 0:
            save_genome(run_dir / "checkpoints" / f"best_gen_{g:06d}.npz", best)
        save_genome(run_dir / "best_genome.npz", best)
        if verbose:
            print(f"gen {g:5d}  max {stats['max_fitness']:.4f}  mean {stats['mean_fitness']:.4f}  "
                  f"rate {stats.get('mean_rate_hz', float('nan')):6.1f} Hz  "
                  f"active {stats['mean_n_active']:.1f}  "
                  f"eta {stats.get('mean_eta', float('nan')):.4f}  "
                  f"a- {stats.get('mean_a_minus', float('nan')):.2f}  "
                  f"plastic {stats.get('mean_plastic_frac', float('nan')):.2f}  "
                  f"({stats['elapsed_s']:.0f}s)", flush=True)

    return callback


def save_training_state(path, pop: Genome, fitness, f_raw, key, generation: int) -> None:
    arrs = {_POP + f: np.asarray(getattr(pop, f)) for f in Genome._fields}
    np.savez(path, fitness=np.asarray(fitness), f_raw=np.asarray(f_raw),
             rng_key=np.asarray(jax.random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key),
             generation=generation, **arrs)


def load_training_state(path, cfg: ExperimentConfig):
    with np.load(path) as z:
        pop = Genome(*(jnp.asarray(z[_POP + f]) for f in Genome._fields))
        return (pop, jnp.asarray(z["fitness"]), jnp.asarray(z["f_raw"]),
                jnp.asarray(z["rng_key"]), int(z["generation"]))


def latest_state_checkpoint(run_dir) -> Path | None:
    snaps = sorted((Path(run_dir) / "checkpoints").glob("state_gen_*.npz"))
    return snaps[-1] if snaps else None
