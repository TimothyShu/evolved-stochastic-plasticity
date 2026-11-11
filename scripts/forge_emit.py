#!/usr/bin/env python3
"""forge_emit.py — esp → forge-queue result-contract adapter.

Writes into <run_dir> (== FORGE_RUN_DIR), per forge-queue contract/README.md §6:
  records.csv    one row per replicate; column 0 a string id
  metrics.json   flat {ident: scalar}: per-condition mean/std of final fitness
  series.csv     WIDE: column 0 = generation (numeric, monotonic); one column per
                 <condition>_<key>, plus <condition>_<key>_std companions (bands)
  best_network_repNN.npz  viewer-format graph per replicate (square matrices)
  manifest.json  tags the viewer files as `network`

Viewer graph: nodes = input channels ‖ neurons (n_pre nodes). Weights are
non-heritable by default, so `weight_matrix` is the 0/1 edge mask unless the run
inherited weights — the genome's `w` field is meaningless otherwise.
Usage: forge_emit.py <run_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import yaml

SERIES_KEYS = ["max_fitness", "mean_fitness", "mean_rate_hz", "mean_n_active",
               "mean_eta", "mean_a_minus", "mean_tau_elig", "mean_plastic_frac", "mean_eff_eta"]


def write_records(run_dir: Path, results: list) -> None:
    cols = ["run_id", "condition", "replicate", "final_max_fitness", "best_n_active",
            "generations_run", "elapsed_s"]
    with (run_dir / "records.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            rid = f"{r['condition']}_rep{int(r['replicate']):02d}"
            w.writerow([rid] + [r.get(c, "") for c in cols[1:]])


def write_metrics(run_dir: Path, results: list) -> None:
    m = {"n_runs": len(results)}
    for cond in sorted({r["condition"] for r in results}):
        fits = [float(r["final_max_fitness"]) for r in results if r["condition"] == cond]
        m[f"{cond}_mean_fitness"] = round(mean(fits), 4)
        m[f"{cond}_std_fitness"] = round(pstdev(fits), 4) if len(fits) > 1 else 0.0
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))


def write_series(run_dir: Path, results: list) -> None:
    conds = sorted({r["condition"] for r in results})
    agg: dict = {}   # cond -> gen -> key -> [values across reps]
    for r in results:
        hp = Path(r["run_dir"]) / "history.jsonl"
        if not hp.exists():
            continue
        for line in hp.read_text().splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            for k in SERIES_KEYS:
                if k in s:
                    agg.setdefault(r["condition"], {}).setdefault(s["generation"], {}).setdefault(k, []).append(s[k])
    gens = sorted({g for c in agg.values() for g in c})
    header = ["generation"]
    for c in conds:
        for k in SERIES_KEYS:
            header += [f"{c}_{k}", f"{c}_{k}_std"]
    with (run_dir / "series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for g in gens:
            row = [g]
            for c in conds:
                for k in SERIES_KEYS:
                    v = agg.get(c, {}).get(g, {}).get(k)
                    row += [round(mean(v), 4), round(pstdev(v), 4)] if v else ["", ""]
            w.writerow(row)


def write_network(run_dir: Path, r: dict) -> Path | None:
    rd = Path(r["run_dir"])
    gp, cp = rd / "best_genome.npz", rd / "config.yaml"
    if not (gp.exists() and cp.exists()):
        return None
    cfg = yaml.safe_load(cp.read_text())
    with np.load(gp) as z:
        edge, inhib, active, w = z["edge_mask"], z["inhib"], z["active"], z["w"]
    n_pre, N = edge.shape
    n_in = n_pre - N
    inherit = bool(cfg.get("genome", {}).get("inherit_weights", False))
    sq_edge = np.zeros((n_pre, n_pre), bool)
    sq_edge[:, n_in:] = edge
    sq_w = np.zeros((n_pre, n_pre), np.float32)
    sq_w[:, n_in:] = np.where(edge, w if inherit else 1.0, 0.0)
    ntype = np.concatenate([np.zeros(n_in, np.int32), inhib.astype(np.int32)])
    out = run_dir / f"best_network_rep{int(r['replicate']):02d}.npz"
    np.savez(out, weight_matrix=sq_w, edge_mask=sq_edge, neuron_type=ntype,
             active_mask=np.concatenate([np.ones(n_in, bool), active]),
             n_in=np.int32(n_in), n_out=np.int32(cfg.get("genome", {}).get("n_out", 2)),
             neuron_type_names=np.array(["excitatory", "inhibitory"]))
    return out


def main() -> int:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    p = run_dir / "results.json"
    results = json.loads(p.read_text()) if p.exists() else []
    if not results:
        print("[forge_emit] no results.json — nothing to emit")
        return 0
    write_records(run_dir, results)
    write_metrics(run_dir, results)
    write_series(run_dir, results)
    nets = [n for n in (write_network(run_dir, r) for r in results) if n is not None]
    # forge-queue reads artifacts as a dict {filename: {"type": ...}} (forge_json.finalize)
    artifacts = {n.name: {"type": "network"} for n in nets}
    (run_dir / "manifest.json").write_text(json.dumps({"artifacts": artifacts}, indent=2))
    print("[forge_emit] wrote records.csv / metrics.json / series.csv / manifest.json"
          f" + {len(nets)} network file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
