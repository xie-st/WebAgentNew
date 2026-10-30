"""Hardware platform detection: Apple Silicon, CUDA, MLX availability."""

from effgen.hardware.platform import (
    HardwarePlatform,
    get_best_local_backend,
    get_unified_memory_gb,
    is_apple_silicon,
    is_cuda_available,
    is_mlx_available,
    is_mlx_vlm_available,
)

__all__ = [
    "is_apple_silicon",
    "is_cuda_available",
    "is_mlx_available",
    "is_mlx_vlm_available",
    "get_unified_memory_gb",
    "get_best_local_backend",
    "HardwarePlatform",
]
