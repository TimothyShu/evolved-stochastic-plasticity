#!/usr/bin/env python3
"""forge_emit.py — esp → forge-queue result-contract adapter (ported from ctrnn_evo).

Reads results.json + per-run history.jsonl and writes records.csv,
metrics.json, series.csv, manifest.json into the run dir. Keeps
run_experiment.py forge-agnostic. Usage: forge_emit.py <run_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

SERIES_KEYS = ["max_fitness", "mean_fitness", "mean_rate_hz", "mean_n_active"]


def main() -> int:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    p = run_dir / "results.json"
    results = json.loads(p.read_text()) if p.exists() else []
    if not results:
        print("[forge_emit] no results.json — nothing to emit")
        return 0

    cols = ["condition", "replicate", "final_max_fitness", "best_n_active", "generations_run", "elapsed_s"]
    with (run_dir / "records.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "") for c in cols])

    metrics = {"n_runs": len(results)}
    for cond in sorted({r["condition"] for r in results}):
        fits = [float(r["final_max_fitness"]) for r in results if r["condition"] == cond]
        metrics[f"{cond}_mean_fitness"] = round(mean(fits), 4)
        metrics[f"{cond}_std_fitness"] = round(pstdev(fits), 4) if len(fits) > 1 else 0.0
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    agg: dict = {}
    for r in results:
        hp = Path(r["run_dir"]) / "history.jsonl"
        if not hp.exists():
            continue
        for line in hp.read_text().splitlines():
            s = json.loads(line)
            for k in SERIES_KEYS:
                if k in s:
                    agg.setdefault((r["condition"], s["generation"], k), []).append(s[k])
    with (run_dir / "series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "generation", "key", "mean", "std", "n"])
        for (cond, gen, k), v in sorted(agg.items()):
            w.writerow([cond, gen, k, mean(v), pstdev(v) if len(v) > 1 else 0.0, len(v)])

    artifacts = [{"path": str(Path(r["run_dir"]) / "best_genome.npz"), "kind": "network",
                  "condition": r["condition"], "replicate": r["replicate"]} for r in results]
    (run_dir / "manifest.json").write_text(json.dumps({"artifacts": artifacts}, indent=2))
    print("[forge_emit] wrote records.csv / metrics.json / series.csv / manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
