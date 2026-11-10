import pytest

from esp.config import ExperimentConfig, replace


@pytest.fixture
def cfg():
    """Small, fast config."""
    return replace(ExperimentConfig(), **{
        "genome.n_hidden_max": 16, "genome.init_hidden_active": 8,
        "task.horizon": 10, "task.substeps": 5, "task.switch_every": 5,
        "evo.population_size": 8, "evo.n_evals": 2, "evo.n_generations": 3,
    })
