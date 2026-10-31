"""Regression tests for GPU allocator correctness.

Covers three previously-shipped defects:
  * ``GPUAllocator.reset()`` deadlocked after an allocation (non-reentrant lock
    held by ``reset`` then re-acquired by ``deallocate``).
  * Memory accounting ignored other processes (used per-process
    ``memory_reserved`` instead of the driver's real free memory), so the
    allocator over-committed GPUs already busy with other tenants.
  * Device discovery/refresh mutated the process-wide CUDA device via
    ``torch.cuda.set_device``.

The lock/overlay tests run without a GPU by injecting synthetic device/allocation
state; the GPU-marked tests validate the real-hardware behavior.
"""

from __future__ import annotations

import threading

import pytest

from effgen.gpu.allocator import (
    Allocation,
    AllocationRequest,
    AllocationStrategy,
    GPUAllocator,
    GPUInfo,
    ParallelismType,
)

GB = 1024**3


def _make_allocator_with_devices(devices: dict[int, GPUInfo]) -> GPUAllocator:
    """Build an allocator and replace its discovered devices with synthetic ones.

    Avoids any real CUDA dependency so the lock/overlay logic can be tested on a
    CPU-only box.
    """
    alloc = GPUAllocator(enable_monitoring=False)
    alloc._devices = devices
    alloc._allocations = {}
    return alloc


def _gpu_info(device_id: int, total_gb: float, available_gb: float) -> GPUInfo:
    return GPUInfo(
        device_id=device_id,
        name=f"FakeGPU{device_id}",
        total_memory=int(total_gb * GB),
        available_memory=int(available_gb * GB),
        utilization=0.0,
    )


# --------------------------------------------------------------------------- #
# #5 — reset() deadlock                                                        #
# --------------------------------------------------------------------------- #


def test_reset_after_allocation_does_not_deadlock():
    """reset() must return promptly even with active allocations (no self-deadlock)."""
    alloc = _make_allocator_with_devices({0: _gpu_info(0, 40, 40)})
    # Inject an active allocation directly (works without a real GPU).
    alloc._allocations["agent_1"] = Allocation(
        requester_id="agent_1",
        device_ids=[0],
        memory_allocated={0: 1 * GB},
        parallelism_type=ParallelismType.NONE,
    )

    done = threading.Event()

    def _call_reset():
        alloc.reset()
        done.set()

    t = threading.Thread(target=_call_reset, daemon=True)
    t.start()
    t.join(timeout=10.0)

    assert done.is_set(), "reset() deadlocked (did not return within 10s)"
    assert alloc.list_allocations() == {}, "reset() should clear all allocations"


def test_reset_is_reentrant_safe_no_allocations():
    alloc = _make_allocator_with_devices({0: _gpu_info(0, 40, 40)})
    done = threading.Event()

    def _call_reset():
        alloc.reset()
        done.set()

    t = threading.Thread(target=_call_reset, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert done.is_set()


# --------------------------------------------------------------------------- #
# #16 — memory accounting respects other processes (overlay model)            #
# --------------------------------------------------------------------------- #


def test_selection_respects_external_usage():
    """A GPU that is nearly full (per the driver's free memory) must not be selected."""
    # GPU 0: 40GB total but only 2GB really free (busy with another process).
    # GPU 1: 40GB total, 38GB free.
    alloc = _make_allocator_with_devices(
        {0: _gpu_info(0, 40, 2), 1: _gpu_info(1, 40, 38)}
    )
    # Disable refresh so our synthetic available_memory survives.
    alloc._refresh_device_info = lambda: None  # type: ignore[method-assign]

    req = AllocationRequest(
        requester_id="agent_1",
        memory_required=10 * GB,
        num_gpus=1,
        strategy=AllocationStrategy.GREEDY,
    )
    result = alloc.allocate(req)
    assert result is not None
    assert result.device_ids == [1], "should skip the externally-busy GPU 0"


def test_no_allocation_when_all_gpus_externally_busy():
    alloc = _make_allocator_with_devices(
        {0: _gpu_info(0, 40, 1), 1: _gpu_info(1, 40, 1)}
    )
    alloc._refresh_device_info = lambda: None  # type: ignore[method-assign]

    req = AllocationRequest(
        requester_id="agent_1", memory_required=8 * GB, num_gpus=1
    )
    assert alloc.allocate(req) is None


def test_available_memory_overlay_subtracts_internal_reservation():
    alloc = _make_allocator_with_devices({0: _gpu_info(0, 40, 30)})
    alloc._refresh_device_info = lambda: None  # type: ignore[method-assign]
    alloc._allocations["agent_1"] = Allocation(
        requester_id="agent_1",
        device_ids=[0],
        memory_allocated={0: 10 * GB},
        parallelism_type=ParallelismType.NONE,
    )
    # 30GB driver-free minus 10GB effGen reservation = 20GB.
    assert alloc.get_available_memory(0) == 20 * GB


# --------------------------------------------------------------------------- #
# #17 — no global CUDA device mutation                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.gpu
def test_discovery_does_not_mutate_global_device():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 usable CUDA GPUs")

    torch.cuda.set_device(0)
    before = torch.cuda.current_device()
    alloc = GPUAllocator(enable_monitoring=False)
    alloc.get_device_info()  # triggers refresh
    alloc.get_available_memory(1)
    after = torch.cuda.current_device()
    assert before == after == 0, f"current device mutated: {before} -> {after}"


@pytest.mark.gpu
def test_available_memory_matches_driver_free():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a usable CUDA GPU")

    alloc = GPUAllocator(enable_monitoring=False)
    info = alloc.get_device_info()
    for did, gi in info.items():
        driver_free, _total = torch.cuda.mem_get_info(did)
        # Allocator's available should track the driver's real free memory
        # (respecting other processes), within a small slack for timing.
        assert abs(gi.available_memory - driver_free) < 2 * GB


@pytest.mark.gpu
def test_real_allocate_then_reset_no_hang():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a usable CUDA GPU")

    alloc = GPUAllocator(enable_monitoring=False)
    req = AllocationRequest(requester_id="agent_1", memory_required=1 * GB, num_gpus=1)
    result = alloc.allocate(req)
    assert result is not None

    done = threading.Event()

    def _call_reset():
        alloc.reset()
        done.set()

    t = threading.Thread(target=_call_reset, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert done.is_set(), "real allocate->reset deadlocked"
    assert alloc.list_allocations() == {}
