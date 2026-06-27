"""PyTorch device selection for training and inference."""

import torch


def resolve_device(device: str | None = None) -> torch.device:
    """
    Resolve a torch device from an optional override.

    ``None`` or ``"auto"`` picks CUDA when available, else CPU.
    Accepts ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"mps"``.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            _enable_cudnn()
            return torch.device("cuda")
        return torch.device("cpu")

    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device}' but CUDA is not available. "
                "Install a CUDA-enabled PyTorch build and ensure NVIDIA drivers are loaded."
            )
        _enable_cudnn()
    elif dev.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError(f"Requested device '{device}' but MPS is not available.")
    return dev


def device_label(device: torch.device) -> str:
    """Human-readable device description for logging."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return str(device)


def supports_amp(device: torch.device) -> bool:
    """Whether autocast mixed precision is supported on this device."""
    return device.type == "cuda"


def _enable_cudnn() -> None:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
