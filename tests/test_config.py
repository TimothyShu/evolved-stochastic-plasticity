import pytest

from esp.config import ExperimentConfig, load_config, replace


def test_defaults_pass_dt_bound():
    ExperimentConfig()


@pytest.mark.parametrize("key,val", [
    ("dt", 2.5),                       # > tau_m/10
    ("alif.t_ref", 0.5),               # refractory shorter than dt
    ("genome.tau_trace_range", (5.0, 100.0)),
    ("genome.tau_floor_factor", 5.0),
])
def test_dt_violations_rejected(key, val):
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **{key: val})


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("evo:\n  populaton_size: 3\n")
    with pytest.raises(KeyError):
        load_config(p)


def test_yaml_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("task:\n  horizon: 7\n")
    c = load_config(p, {"evo.population_size": 11})
    assert c.task.horizon == 7 and c.evo.population_size == 11
    assert c.lifetime_steps == 7 * c.task.substeps
