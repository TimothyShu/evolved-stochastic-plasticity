#!/usr/bin/env bash
# run.sh — forge-queue entrypoint for esp.
#
# The watcher has already pinned a GPU (CUDA_VISIBLE_DEVICES), activated the
# venv, and cd'd here. Contract: forge-queue backend/docs/conventions.md.
#
# Job-spec params:
#   mode:   evolve (default) | benchmark
#   config: base YAML (default configs/stage0_reflex.yaml) — passed as --config,
#           not flattened: esp configs are nested.
#   set:    space-separated dotted overrides, e.g.
#           "plasticity.rule=three_factor evo.population_size=512"
#           (dots can't survive FORGE_PARAM_* env names, hence this param)
#   <flag>: any other param becomes --<flag> <value> for the chosen script;
#           booleans: true -> bare flag, false -> omitted.
set -euo pipefail

cd "${FORGE_PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

git pull --ff-only 2>&1 || echo "[run.sh] WARN: git pull failed, using current code"
echo "[run.sh] commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) gpu=${FORGE_GPU:-?}"

MODE="${FORGE_PARAM_MODE:-evolve}"
case "$MODE" in
    evolve)    SCRIPT=scripts/run_experiment.py ;;
    benchmark) SCRIPT=scripts/benchmark.py ;;
    *) echo "[run.sh] unknown mode '$MODE' (evolve|benchmark)"; exit 2 ;;
esac

readarray -t ARGS < <(MODE="$MODE" python3 - <<'PY'
import os, shlex

params = {k[len("FORGE_PARAM_"):].lower().replace("_", "-"): v
          for k, v in os.environ.items() if k.startswith("FORGE_PARAM_")}
params.pop("mode", None)
mode = os.environ["MODE"]

# Flags each script accepts; the dashboard submits every form field, so params
# belonging to the other mode are dropped here rather than crashing argparse.
ACCEPTS = {
    "evolve": {"n-replicates", "seed", "pop-size", "n-generations", "n-evals", "rule",
               "horizon", "checkpoint-every", "early-stop-window", "resume", "smoke-test", "quiet"},
    "benchmark": {"hidden", "pop", "rules", "unroll", "noise", "n-evals", "repeats", "allow-cpu"},
}[mode]

out = ["--config", params.pop("config", "") or "configs/stage0_reflex.yaml"]
sets = shlex.split(params.pop("set", ""))
if mode == "evolve":
    for kv in sets:
        out += ["--set", kv]
elif sets:
    import sys; print(f"# WARN: 'set' ignored in benchmark mode: {sets}", file=sys.stderr)
for k in [k for k in params if k not in ACCEPTS]:
    import sys; print(f"# dropping param '{k}' (not used by {mode})", file=sys.stderr)
    params.pop(k)
for k, v in params.items():
    s = str(v).strip().lower()
    if s in ("true", "false"):
        if s == "true":
            out.append(f"--{k}")
    elif v not in (None, ""):
        out += [f"--{k}", str(v)]
print("\n".join(out))
PY
)

if [ -n "${FORGE_RUN_DIR:-}" ]; then
    if [ "$MODE" = evolve ]; then
        ARGS+=(--output-dir "$FORGE_RUN_DIR")
    else
        ARGS+=(--out "$FORGE_RUN_DIR/bench.csv")
    fi
fi

echo "[run.sh] $SCRIPT ${ARGS[*]}"
set +e
python3 -u "$SCRIPT" "${ARGS[@]}"
rc=$?
set -e

if [ "$MODE" = evolve ] && [ -n "${FORGE_RUN_DIR:-}" ]; then
    python3 scripts/forge_emit.py "$FORGE_RUN_DIR" || echo "[run.sh] WARN: forge_emit failed"
fi

exit "$rc"
