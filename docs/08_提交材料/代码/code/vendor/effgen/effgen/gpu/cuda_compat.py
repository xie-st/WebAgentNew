"""CUDA / driver compatibility detection for effGen.

This is the single source of truth for the question "can effGen actually use the
NVIDIA GPUs on this machine?" — and, just as importantly, for explaining *why not*
when the answer is no.

The motivating failure: a fresh install can pull a PyTorch wheel built for a newer
CUDA runtime (e.g. ``torch ...+cu130``) than the host driver supports (e.g. a
CUDA 12.4 driver). PyTorch then reports ``torch.cuda.is_available() == False`` and
quietly runs everything on the CPU, even though ``nvidia-smi`` shows several idle
GPUs. The user sees only a slow run, with no hint that the install picked an
incompatible torch build.

This module:
  * detects the number of *physical* NVIDIA GPUs (independent of whether torch can
    use them), via torch's device count, then NVML, then ``nvidia-smi``;
  * reads the host driver's maximum supported CUDA version and the CUDA version
    torch was built against;
  * decides whether the GPUs are *usable* (the one definition every other module
    should defer to); and
  * emits exactly one clear, actionable warning per process when physical GPUs
    exist but torch cannot use them.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import warnings
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Set EFFGEN_NO_GPU_WARN=1 to silence the one-time CUDA mismatch warning (e.g. when
# you have deliberately chosen a CPU-only run on a GPU box).
_SILENCE_ENV = "EFFGEN_NO_GPU_WARN"

_warn_lock = threading.Lock()
_warned = False

# Documentation pointer included in the mismatch warning.
_DOCS_HINT = (
    "See the CUDA install matrix in docs/installation.md "
    "(install a torch build matching your driver, e.g. "
    "pip install torch --index-url https://download.pytorch.org/whl/cu124)."
)


def _torch_module():
    """Return the imported ``torch`` module, or ``None`` if torch is unavailable."""
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # pragma: no cover - torch is a base dep but stay defensive
        return None


def torch_cuda_version() -> str | None:
    """CUDA version PyTorch was built against (e.g. ``"12.4"``), or None."""
    torch = _torch_module()
    if torch is None:
        return None
    return getattr(torch.version, "cuda", None)


def cuda_usable() -> bool:
    """The single source of truth for "are the GPUs usable by torch right now?".

    Every other module (``gpu.utils``, ``hardware.platform``) should defer to this
    so they cannot disagree.
    """
    torch = _torch_module()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except Exception:
        return False


def _nvml_info() -> tuple[int, str | None]:
    """Return ``(physical_gpu_count, driver_cuda_version)`` via NVML.

    ``driver_cuda_version`` is the maximum CUDA runtime the installed driver
    supports (e.g. ``"12.4"``). Returns ``(0, None)`` if NVML is unavailable.
    """
    try:
        import pynvml  # noqa: PLC0415
    except Exception:
        return 0, None

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        count = int(pynvml.nvmlDeviceGetCount())
        driver_cuda = None
        try:
            raw = int(pynvml.nvmlSystemGetCudaDriverVersion_v2())
            # NVML encodes the version as (major * 1000 + minor * 10).
            driver_cuda = f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:
            driver_cuda = None
        return count, driver_cuda
    except Exception:
        return 0, None
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _nvidia_smi_info() -> tuple[int, str | None]:
    """Fallback detection via the ``nvidia-smi`` binary.

    Returns ``(physical_gpu_count, driver_cuda_version)``; ``(0, None)`` if the
    binary is missing or fails.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return 0, None
    count = 0
    driver_cuda: str | None = None
    try:
        listing = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if listing.returncode == 0:
            count = len([ln for ln in listing.stdout.splitlines() if ln.strip()])
    except Exception:
        return 0, None
    try:
        header = subprocess.run([smi], capture_output=True, text=True, timeout=5)
        match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", header.stdout)
        if match:
            driver_cuda = match.group(1)
    except Exception:
        driver_cuda = None
    return count, driver_cuda


def physical_gpu_count() -> int:
    """Best-effort count of physical NVIDIA GPUs, regardless of torch usability.

    ``torch.cuda.device_count()`` reports the physical count even when
    ``torch.cuda.is_available()`` is False (the exact mismatch case), so it is
    tried first; NVML and ``nvidia-smi`` are fallbacks for CPU-only torch builds.
    """
    torch = _torch_module()
    if torch is not None:
        try:
            count = int(torch.cuda.device_count())
            if count > 0:
                return count
        except Exception:
            pass
    count, _ = _nvml_info()
    if count > 0:
        return count
    count, _ = _nvidia_smi_info()
    return count


def driver_cuda_version() -> str | None:
    """Maximum CUDA runtime version supported by the installed driver (e.g. ``"12.4"``)."""
    _, driver_cuda = _nvml_info()
    if driver_cuda:
        return driver_cuda
    _, driver_cuda = _nvidia_smi_info()
    return driver_cuda


def _nvml_utilization() -> dict[int, float]:
    """Map physical GPU index → GPU utilization percent (0–100) via NVML.

    Returns ``{}`` if NVML is unavailable. Indices are *physical* (NVML's own
    ordering), which matches ``nvidia-smi`` but not necessarily torch's device
    ordering when ``CUDA_VISIBLE_DEVICES`` remaps it — callers should only trust
    the mapping when that variable is unset.
    """
    try:
        import pynvml  # noqa: PLC0415
    except Exception:
        return {}
    out: dict[int, float] = {}
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        for i in range(int(pynvml.nvmlDeviceGetCount())):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                out[i] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                continue
    except Exception:
        return out
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    return out


def _nvidia_smi_utilization() -> dict[int, float]:
    """Fallback: physical GPU index → utilization percent via ``nvidia-smi``."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    out: dict[int, float] = {}
    try:
        res = subprocess.run(
            [smi, "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode != 0:
            return {}
        for line in res.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    out[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue
    except Exception:
        return {}
    return out


@dataclass
class GpuMemoryStatus:
    """Physical memory + utilization snapshot for one GPU.

    ``used_bytes``/``free_bytes`` reflect the CUDA driver's view across *all*
    processes (via ``torch.cuda.mem_get_info``), so this is the real "is this GPU
    free?" picture — not just the current process's reservations.
    """

    index: int
    name: str
    total_bytes: int
    free_bytes: int
    used_bytes: int
    utilization_pct: float | None = None


def per_gpu_status() -> list[GpuMemoryStatus]:
    """Per-GPU *physical* memory and utilization for the devices torch can see.

    Uses ``torch.cuda.mem_get_info`` for true driver-level used/free (counting
    every process on the card), and NVML/``nvidia-smi`` for utilization. Returns
    an empty list if torch/CUDA is unavailable. Utilization is reported only when
    it can be attributed to the right device — when ``CUDA_VISIBLE_DEVICES`` remaps
    ordering, the per-device utilization is left as ``None`` rather than risk
    mislabelling it.
    """
    torch = _torch_module()
    if torch is None:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        num = int(torch.cuda.device_count())
    except Exception:
        return []

    # Utilization is keyed by physical index; only trust it when the visible-device
    # ordering is the identity mapping (no CUDA_VISIBLE_DEVICES remap).
    util_map: dict[int, float] = {}
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        util_map = _nvml_utilization() or _nvidia_smi_utilization()

    out: list[GpuMemoryStatus] = []
    for i in range(num):
        try:
            props = torch.cuda.get_device_properties(i)
            name = props.name
            total = int(props.total_memory)
        except Exception:
            name, total = f"GPU {i}", 0
        try:
            free, total_drv = torch.cuda.mem_get_info(i)
            free = int(free)
            total = int(total_drv) or total
        except Exception:
            # Older torch without a live CUDA context: degrade to per-process view.
            try:
                free = max(total - int(torch.cuda.memory_reserved(i)), 0)
            except Exception:
                free = 0
        used = max(total - free, 0)
        out.append(GpuMemoryStatus(
            index=i, name=name, total_bytes=total,
            free_bytes=free, used_bytes=used,
            utilization_pct=util_map.get(i),
        ))
    return out


@dataclass(frozen=True)
class CudaStatus:
    """A consistent snapshot of the CUDA usability picture."""

    usable: bool
    physical_gpus: int
    torch_cuda: str | None
    driver_cuda: str | None
    mismatch: bool
    message: str | None
    torch_installed: bool = True

    @property
    def is_mismatch(self) -> bool:
        """True when the installed torch CUDA build does not match the driver."""
        return self.mismatch


def get_cuda_status() -> CudaStatus:
    """Compute the current CUDA usability picture.

    A *mismatch* is the specific, actionable failure: the machine has physical
    NVIDIA GPUs but torch cannot use them (almost always a torch-CUDA / driver
    version skew). A plain CPU-only machine is *not* a mismatch.
    """
    torch_installed = _torch_module() is not None
    usable = cuda_usable()
    physical = physical_gpu_count()
    torch_cuda = torch_cuda_version()
    driver_cuda = driver_cuda_version()

    mismatch = (not usable) and physical > 0
    message: str | None = None
    if mismatch:
        gpu_word = "GPU" if physical == 1 else "GPUs"
        if not torch_installed:
            # No torch at all is a different problem from a torch built for the
            # wrong CUDA line, and it has a different fix.
            message = (
                f"{physical} NVIDIA {gpu_word} detected, but PyTorch is not installed, "
                f"so no local model can run here. Install it with: pip install torch "
                f"(see docs/installation.md for the build matching your driver). "
                f"Cloud models need no local engine."
            )
        else:
            built = f"CUDA {torch_cuda}" if torch_cuda else "no CUDA (CPU-only build)"
            supports = f"CUDA {driver_cuda}" if driver_cuda else "an unknown CUDA version"
            message = (
                f"{physical} NVIDIA {gpu_word} detected, but torch.cuda is unavailable, "
                f"so models will run on CPU (much slower). The installed PyTorch is built "
                f"for {built} while this host's driver supports up to {supports}. "
                f"{_DOCS_HINT}"
            )
    return CudaStatus(
        usable=usable,
        physical_gpus=physical,
        torch_cuda=torch_cuda,
        driver_cuda=driver_cuda,
        mismatch=mismatch,
        message=message,
        torch_installed=torch_installed,
    )


def warn_cuda_mismatch_once(*, force: bool = False) -> bool:
    """Emit one process-wide warning if physical GPUs exist but torch can't use them.

    Returns True if a mismatch was detected (whether or not a warning was emitted;
    the warning is suppressed after the first time and when ``EFFGEN_NO_GPU_WARN``
    is set). ``force=True`` re-checks but still honours the once-per-process guard.
    """
    global _warned

    status = get_cuda_status()
    if not status.mismatch:
        return False

    if os.environ.get(_SILENCE_ENV, "").strip() not in ("", "0", "false", "False"):
        # User opted out of the warning; still report the mismatch to callers.
        return True

    with _warn_lock:
        if _warned and not force:
            return True
        _warned = True

    assert status.message is not None
    logger.warning(status.message)
    warnings.warn(status.message, RuntimeWarning, stacklevel=2)
    return True


def _reset_warned_for_tests() -> None:
    """Reset the once-per-process warning guard (test helper only)."""
    global _warned
    with _warn_lock:
        _warned = False
