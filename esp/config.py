"""
config.py — Experiment configuration.

One frozen (hashable) dataclass tree, loaded from YAML. An experiment is a
selection of one implementation per module (plan §3) plus hyperparameters.

Units: time in ms, rates in Hz. Membrane potential is dimensionless with the
baseline threshold v_th = 1.

The dt fidelity rule (plan §5) is enforced in ExperimentConfig.__post_init__:
    dt <= tau / 10   for every time constant tau in the system
    t_ref >= dt      (refractory is a duration, not a time constant)
Evolved time constants cannot be checked here; genome.decode clips them to
>= 10·dt instead (see GenomeConfig.tau_floor_factor).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ALIFConfig:
    tau_m: float = 20.0     # membrane time constant (ms)
    tau_a: float = 700.0    # threshold-adaptation time constant (ms); plan range 200–2000
    t_ref: float = 2.0      # absolute refractory period (ms) — a duration
    v_th: float = 1.0       # baseline threshold (dimensionless)
    beta: float = 1.8       # adaptation strength (LSNN convention); tune at Stage 1


@dataclass(frozen=True)
class IOConfig:
    encoder: str = "poisson"    # obs in [0,1] -> Bernoulli spikes, p = r_max·obs·dt
    decoder: str = "argmax"     # filtered output traces -> discrete action
    in_pop: int = 8             # input channels per observation dimension (population code)
    r_max: float = 100.0        # max input rate (Hz)
    tau_readout: float = 20.0   # output trace filter (ms)
    n_reward_channels: int = 0  # grounding inputs (reward/interoceptive); frozen by plastic_mask


@dataclass(frozen=True)
class PlasticityConfig:
    rule: str = "identity"      # identity | three_factor
    w_max: float = 2.0          # weight-magnitude ceiling (units of v_th jump per spike)
    noise_sigma: float = 0.0    # std of additive weight noise per ms (0 => no noise drawn)


@dataclass(frozen=True)
class NeuromodConfig:
    kind: str = "external"      # external scalar δ (Stages 1–2)
    baseline_tau: float = 0.0   # >0: δ = r − running mean (ms); 0: δ = r


@dataclass(frozen=True)
class GenomeConfig:
    decoder: str = "direct"
    n_hidden_max: int = 64          # hidden slots (padded)
    n_out: int = 2                  # output neurons (last slots)
    init_hidden_active: int = 32
    init_edge_density: float = 0.15
    inherit_weights: bool = False   # False: weights redrawn every lifetime (non-heritable dist)
    w_init_std: float = 0.3         # fixed, non-heritable init distribution |N(0, std)|
    frac_inhibitory: float = 0.2    # E/I split at init (Dale's law; types are heritable)
    # Evolved plasticity time constants are clipped to >= tau_floor_factor·dt
    tau_floor_factor: float = 10.0
    tau_trace_range: tuple = (10.0, 200.0)   # pre/post trace taus (ms)
    tau_elig_range: tuple = (20.0, 2000.0)   # eligibility trace tau (ms)
    eta_range: tuple = (0.0, 0.05)           # learning rate


@dataclass(frozen=True)
class TaskConfig:
    name: str = "reflex"
    horizon: int = 50           # env steps per lifetime
    substeps: int = 10          # neural steps (of dt) per env step
    n_cues: int = 2
    switch_every: int = 10      # reflex: cue changes every N env steps
    grace_steps: int = 2        # reflex: env steps after a switch that aren't scored


@dataclass(frozen=True)
class EvoConfig:
    population_size: int = 256
    tournament_size: int = 4
    n_evals: int = 4                # lifetimes per fitness estimate
    n_generations: int = 200
    # Mutation intensities (ported from ctrnn_evo MutationRates)
    weight_sigma: float = 0.1       # only used when genome.inherit_weights
    rule_sigma: float = 0.1         # in log/normalised rule-param space
    position_sigma: float = 0.05
    type_flip_prob: float = 0.02
    plastic_flip_prob: float = 0.01
    add_node_prob: float = 0.05
    remove_node_prob: float = 0.05
    add_edge_prob: float = 0.1
    remove_edge_prob: float = 0.1
    # Percentage-fitness penalties (ported): penalty = frac·f_raw·C/C0
    edge_frac: float = 0.0
    dist_frac: float = 0.0
    C0_edge: float = 150.0
    C0_wiring: float = 75.0
    penalty_warmup_gens: int = 0
    penalty_cycle_gens: int = 0
    penalty_cycle_free_gens: int = 0
    mutation_warmup_scale: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "stage0_reflex"
    seed: int = 0
    dt: float = 1.0             # ms — fidelity constraint, NOT a throughput knob
    scan_unroll: int = 1        # inner-scan unroll: pure performance knob, no semantic effect
    alif: ALIFConfig = field(default_factory=ALIFConfig)
    io: IOConfig = field(default_factory=IOConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    neuromod: NeuromodConfig = field(default_factory=NeuromodConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    evo: EvoConfig = field(default_factory=EvoConfig)

    def __post_init__(self):
        check_dt(self)

    # ── derived sizes ────────────────────────────────────────────────────────
    @property
    def n_neurons(self) -> int:
        return self.genome.n_hidden_max + self.genome.n_out

    @property
    def n_obs(self) -> int:
        return self.task.n_cues

    @property
    def n_in(self) -> int:
        """Input channels: population-coded obs + grounding (reward) channels."""
        return self.n_obs * self.io.in_pop + self.io.n_reward_channels

    @property
    def n_pre(self) -> int:
        """Presynaptic axis of the weight matrix: input channels then neurons."""
        return self.n_in + self.n_neurons

    @property
    def lifetime_steps(self) -> int:
        """Neural timesteps per lifetime — the scan length (dominant cost)."""
        return self.task.horizon * self.task.substeps


def check_dt(cfg: ExperimentConfig) -> None:
    """Plan §5: dt <= tau/10 for every fixed time constant; t_ref >= dt."""
    fixed_taus = {
        "alif.tau_m": cfg.alif.tau_m,
        "alif.tau_a": cfg.alif.tau_a,
        "io.tau_readout": cfg.io.tau_readout,
        "genome.tau_trace_range[0]": cfg.genome.tau_trace_range[0],
        "genome.tau_elig_range[0]": cfg.genome.tau_elig_range[0],
    }
    if cfg.neuromod.baseline_tau > 0:
        fixed_taus["neuromod.baseline_tau"] = cfg.neuromod.baseline_tau
    bad = {k: v for k, v in fixed_taus.items() if cfg.dt > v / 10.0}
    if bad:
        raise ValueError(
            f"dt={cfg.dt} ms violates dt <= tau/10 for {bad}. "
            "dt is a fidelity constraint; do not coarsen it for throughput."
        )
    if cfg.alif.t_ref < cfg.dt:
        raise ValueError(f"t_ref={cfg.alif.t_ref} ms < dt={cfg.dt} ms: refractory unresolvable")
    if cfg.genome.tau_floor_factor < 10.0:
        raise ValueError("tau_floor_factor < 10 would let evolved taus break the dt bound")


# ── YAML I/O ─────────────────────────────────────────────────────────────────

def _build(cls, d: dict[str, Any]):
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for k, v in d.items():
        if k not in fields:
            raise KeyError(f"Unknown config key '{k}' for {cls.__name__}")
        ftype = fields[k].type
        sub = _SUBCONFIGS.get(k) if cls is ExperimentConfig else None
        if sub is not None:
            kwargs[k] = _build(sub, v or {})
        elif isinstance(v, list):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


_SUBCONFIGS = {
    "alif": ALIFConfig, "io": IOConfig, "plasticity": PlasticityConfig,
    "neuromod": NeuromodConfig, "genome": GenomeConfig, "task": TaskConfig,
    "evo": EvoConfig,
}


def from_dict(d: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, d)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load YAML; `overrides` uses dotted keys, e.g. {"evo.population_size": 512}."""
    d = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, v in (overrides or {}).items():
        node = d
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    return from_dict(d)


def to_dict(cfg: ExperimentConfig) -> dict:
    return dataclasses.asdict(cfg)


def replace(cfg: ExperimentConfig, **dotted) -> ExperimentConfig:
    """replace(cfg, **{"evo.population_size": 64}) — dotted-key functional update."""
    d = to_dict(cfg)
    for k, v in dotted.items():
        node = d
        *parents, leaf = k.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = v
    return from_dict(d)
