# esp — evolved-stochastic-plasticity

Evolve the **structure** of a spiking (ALIF) network — topology, E/I types,
plasticity flags, rule meta-parameters — and learn the **weights online within
each lifetime** via a noise-driven three-factor rule. No gradients through the
network. Build contract: `project_plan.md`.

Everything composes through one pure, `jit`/`vmap`-able function:

```python
fitness, metrics = esp.rollout(genome, key, cfg)
```

## Layout

| module | role (plan §3) |
|---|---|
| `esp/genome/` | `decode(genome, key, cfg) -> Phenotype`; direct encoding (indirect at Stage 2) |
| `esp/substrate/alif.py` | ALIF `step`, hard spikes, exact exponential integration |
| `esp/plasticity.py` | `update(traces, pre, post, m, w, noise, rp, cfg)`, `apply` = the firewall |
| `esp/neuromod.py` | third factor; external δ (Stages 1–2) |
| `esp/io.py` | Poisson population encoder, filtered-trace argmax decoder |
| `esp/tasks/` | gymnax-style tasks; `reflex` (Stage 0) |
| `esp/rollout.py` | the lifetime scan; carry vs closure partition |
| `esp/evo/` | selection, elitism, mutation, percentage penalties (ported from ctrnn_evo) |
| `esp/infra/` | logging, checkpoints/resume, ntfy |

## Invariants enforced in code

- **dt fidelity** — `dt <= tau/10` for every fixed time constant and `t_ref >= dt`,
  asserted at config load; evolved taus clipped to `>= 10·dt` in `decode`.
- **Firewall** — `plastic_mask = genome.plastic ∧ connected ∧ ¬grounding`; `apply`
  only touches masked entries. Grounding rows are bit-identical over a lifetime (tested).
- **Non-heritable weights** (`genome.inherit_weights: false`) — redrawn from a fixed
  distribution every lifetime and never mutated; signs inherited via E/I type.
- **Stochasticity enters only via the `noise` argument** to the rule, drawn from the
  carried key; `noise_sigma: 0` skips the draw.

## Use

```bash
pip install -e ".[dev]"            # or .[cuda] on forge
pytest
python scripts/run_experiment.py --config configs/stage0_reflex.yaml --smoke-test
python scripts/benchmark.py --hidden 64,256,512 --pop 256,1024,4096 --out bench.csv   # Stage 0 gate
```

Set `ESP_NTFY_TOPIC` for push notifications. `forge.project.yaml` +
`scripts/forge_emit.py` plug into forge-queue as ctrnn_evo did.
