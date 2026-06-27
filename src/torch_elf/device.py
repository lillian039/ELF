from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypedDict

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: "torch.device"
    backend: str
    description: str
    supports_amp: bool
    amp_dtype: Optional["torch.dtype"]


class AutocastKwargs(TypedDict, total=False):
    enabled: bool
    device_type: str
    dtype: "torch.dtype"


def _require_torch():
    import torch

    return torch


def detect_device(preferred: str = "auto") -> DeviceInfo:
    torch = _require_torch()
    pref = (preferred or "auto").lower()

    def cpu() -> DeviceInfo:
        return DeviceInfo(torch.device("cpu"), "cpu", "CPU", False, None)

    def cuda_like() -> DeviceInfo:
        name = torch.cuda.get_device_name(0)
        hip = getattr(torch.version, "hip", None)
        backend = "rocm" if hip else "cuda"
        return DeviceInfo(torch.device("cuda"), backend, f"{backend.upper()}:{name}", True, torch.float16)

    def xpu() -> DeviceInfo:
        return DeviceInfo(torch.device("xpu"), "xpu", f"XPU:{torch.xpu.get_device_name(0)}", True, getattr(torch, "float16", None))

    def mps() -> DeviceInfo:
        return DeviceInfo(torch.device("mps"), "mps", "Apple Metal Performance Shaders", False, None)

    available = {
        "cuda": torch.cuda.is_available() and getattr(torch.version, "hip", None) is None,
        "rocm": torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None,
        "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu": True,
    }

    if pref != "auto":
        if pref in {"cuda", "rocm"} and (available["cuda"] or available["rocm"]):
            return cuda_like()
        if pref == "xpu" and available["xpu"]:
            return xpu()
        if pref == "mps" and available["mps"]:
            return mps()
        if pref == "cpu":
            return cpu()
        raise RuntimeError(f"Requested device '{preferred}' is not available.")

    if available["cuda"] or available["rocm"]:
        return cuda_like()
    if available["xpu"]:
        return xpu()
    if available["mps"]:
        return mps()
    return cpu()


def format_device_info(info: DeviceInfo) -> str:
    torch = _require_torch()
    parts = [
        f"torch={getattr(torch, '__version__', 'unknown')}",
        f"backend={info.backend}",
        f"device={info.device}",
        f"description={info.description}",
    ]
    cuda = getattr(torch.version, "cuda", None)
    hip = getattr(torch.version, "hip", None)
    if cuda:
        parts.append(f"cuda_runtime={cuda}")
    if hip:
        parts.append(f"hip_runtime={hip}")
    if info.supports_amp and info.amp_dtype is not None:
        parts.append(f"amp_dtype={info.amp_dtype}")
    return " | ".join(parts)


def get_autocast_kwargs(info: DeviceInfo) -> AutocastKwargs:
    if not info.supports_amp or info.amp_dtype is None:
        return {"enabled": False, "device_type": info.device.type}
    return {"enabled": True, "device_type": info.device.type, "dtype": info.amp_dtype}
