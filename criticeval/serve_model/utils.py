def import_backend_module(name: str):
    """Dynamically import and return a backend module by name."""
    if name == "vllm":
        import importlib
        return importlib.import_module("vllm")
    if name == "vllm_npu":
        import importlib
        return importlib.import_module("vllm_ascend")
    raise ValueError(f"Unknown backend module: {name}")


def detect_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
            return "npu"
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def empty_cashe():
    try:
        import torch

        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except Exception:
        pass
