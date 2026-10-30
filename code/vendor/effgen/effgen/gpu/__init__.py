"""GPU management: allocation strategies, real-time monitoring and utility helpers.

Includes:
- GPU allocation with multiple strategies
- Real-time GPU monitoring with alerts
- Utility functions for GPU operations and memory estimation

Components:
    - GPUAllocator: Smart GPU allocation and resource management
    - GPUMonitor: Real-time monitoring with threshold alerts
    - Utility functions: Memory estimation, device selection, etc.

Author: effGen Team
License: Apache-2.0
"""

from effgen.gpu import utils as gpu_utils
from effgen.gpu.allocator import (
    Allocation,
    AllocationRequest,
    AllocationStrategy,
    GPUAllocator,
    GPUInfo,
    ParallelismType,
)
from effgen.gpu.cuda_compat import (
    CudaStatus,
    cuda_usable,
    driver_cuda_version,
    get_cuda_status,
    physical_gpu_count,
    torch_cuda_version,
    warn_cuda_mismatch_once,
)
from effgen.gpu.monitor import (
    Alert,
    AlertLevel,
    GPUMetrics,
    GPUMonitor,
    MetricType,
    MonitorConfig,
)

__all__ = [
    # Allocator classes
    "GPUAllocator",
    "AllocationStrategy",
    "ParallelismType",
    "GPUInfo",
    "AllocationRequest",
    "Allocation",

    # Monitor classes
    "GPUMonitor",
    "MonitorConfig",
    "GPUMetrics",
    "Alert",
    "AlertLevel",
    "MetricType",

    # Utilities module
    "gpu_utils",

    # CUDA / driver compatibility
    "CudaStatus",
    "cuda_usable",
    "get_cuda_status",
    "physical_gpu_count",
    "driver_cuda_version",
    "torch_cuda_version",
    "warn_cuda_mismatch_once",
]
