from typing import Callable, Any


REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str):
    def decorator(func):
        if name in REGISTRY:
            raise ValueError(f"Name {name} is already used.")
        REGISTRY[name] = func
        return func
    return decorator


def get_available_answer_extractor_function():
    return ', '.join(REGISTRY.keys())


def get_field_extractor_function(name: str):
    try:
        extractor = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Extractor {name} not registry."
            f"Evailable: {get_available_answer_extractor_function()}"
        )
    return extractor
