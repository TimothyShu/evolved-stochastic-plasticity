"""esp — evolved-stochastic-plasticity.

Evolve SNN structure; learn weights online within each lifetime via a
noise-driven three-factor rule. Everything composes through

    rollout(genome, key, cfg) -> (fitness, metrics)
"""

from .config import ExperimentConfig, load_config
from .rollout import rollout, make_population_eval

__version__ = "0.0.1"
__all__ = ["ExperimentConfig", "load_config", "rollout", "make_population_eval"]
