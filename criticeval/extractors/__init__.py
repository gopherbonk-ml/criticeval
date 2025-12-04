from .registry import get_answer_extractor_function
from . import extractors as _default_extractors  # noqa: F401 - imports register built-in extractors


__all__ = ["get_answer_extractor_function"]
