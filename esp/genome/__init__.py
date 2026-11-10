"""genome — `decode(genome, key, cfg) -> Phenotype`. Direct encoding now;
the indirect/spatial decoder arrives at Stage 2 behind the same signature."""

from .direct import Genome, Phenotype, decode, random_genome, prune_isolated

__all__ = ["Genome", "Phenotype", "decode", "random_genome", "prune_isolated"]
