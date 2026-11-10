"""tasks — gymnax-style: reset(key, cfg) -> (state, obs);
step(state, action, cfg) -> (state, obs, reward, done); max_return(cfg)."""

from types import ModuleType

from . import reflex

_REGISTRY: dict[str, ModuleType] = {"reflex": reflex}


def get(name: str) -> ModuleType:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown task '{name}'; have {sorted(_REGISTRY)}") from None
