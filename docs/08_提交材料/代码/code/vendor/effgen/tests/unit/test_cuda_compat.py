"""Tests for CUDA / driver compatibility detection (effgen.gpu.cuda_compat).

These exercise the real hardware on whatever machine runs the suite (no faked
torch / model calls):
  * the usability invariants that keep gpu.utils, hardware.platform and
    torch.cuda from disagreeing, and
  * the one-warning-per-process mismatch guard and its env opt-out.

The mismatch-warning mechanism is exercised against a synthetic CudaStatus so the
behaviour is deterministic regardless of whether the test host has a torch-CUDA /
driver mismatch.
"""

from __future__ import annotations

import warnings

import pytest

from effgen.gpu import cuda_compat as cc


@pytest.fixture(autouse=True)
def _reset_warn_guard():
    cc._reset_warned_for_tests()
    yield
    cc._reset_warned_for_tests()


def _make_status(*, usable, physical, torch_cuda="13.0", driver_cuda="12.4"):
    """Build a CudaStatus the way get_cuda_status() would, for the warn mechanism."""
    mismatch = (not usable) and physical > 0
    message = None
    if mismatch:
        message = (
            f"{physical} NVIDIA GPUs detected, but torch.cuda is unavailable... "
            f"built for CUDA {torch_cuda} while driver supports CUDA {driver_cuda}."
        )
    return cc.CudaStatus(
        usable=usable,
        physical_gpus=physical,
        torch_cuda=torch_cuda,
        driver_cuda=driver_cuda,
        mismatch=mismatch,
        message=message,
    )


class TestRealHardwareInvariants:
    """Run against the actual host; assert internal consistency only."""

    def test_status_is_self_consistent(self):
        st = cc.get_cuda_status()
        assert isinstance(st.physical_gpus, int)
        assert st.physical_gpus >= 0
        # mismatch is exactly "physical GPUs exist but torch can't use them".
        assert st.mismatch == ((not st.usable) and st.physical_gpus > 0)
        # a mismatch always carries an explanatory message; a non-mismatch never does.
        assert (st.message is not None) == st.mismatch

    def test_single_source_of_truth_agrees(self):
        from effgen.gpu.utils import is_gpu_available
        from effgen.hardware.platform import is_cuda_available

        is_cuda_available.cache_clear()
        usable = cc.cuda_usable()
        assert is_gpu_available() == usable
        assert is_cuda_available() == usable

    def test_usable_implies_no_mismatch(self):
        st = cc.get_cuda_status()
        if st.usable:
            assert not st.mismatch
            assert st.physical_gpus >= 1

    def test_physical_count_at_least_usable_count(self):
        # If torch can use GPUs, they are by definition physically present.
        st = cc.get_cuda_status()
        if st.usable:
            assert st.physical_gpus > 0


class TestPerGpuStatus:
    """`per_gpu_status` reports *physical* (driver) memory, not per-process."""

    def test_returns_consistent_physical_snapshot(self):
        gpus = cc.per_gpu_status()
        assert isinstance(gpus, list)
        if not cc.cuda_usable():
            # No usable CUDA → empty list, never a crash.
            assert gpus == []
            return
        assert len(gpus) >= 1
        for g in gpus:
            assert isinstance(g, cc.GpuMemoryStatus)
            assert g.total_bytes > 0
            # used + free should reconcile to total (driver view).
            assert g.used_bytes + g.free_bytes == g.total_bytes
            assert 0 <= g.used_bytes <= g.total_bytes
            assert g.utilization_pct is None or 0.0 <= g.utilization_pct <= 100.0

    def test_used_reflects_driver_not_current_process(self):
        # A fresh process reserves ~0 of its own memory, but the driver view
        # (mem_get_info) must report whatever is physically allocated on the card.
        # We can't force another tenant, but we can assert we are NOT reading the
        # current process's reservation (which would be ~0 used on every card).
        if not cc.cuda_usable():
            pytest.skip("no usable CUDA on this host")
        import torch
        gpus = cc.per_gpu_status()
        for g in gpus:
            free, total = torch.cuda.mem_get_info(g.index)
            assert g.free_bytes == int(free)
            assert g.used_bytes == int(total) - int(free)


class TestWarnOnceMechanism:
    def test_warns_exactly_once_on_mismatch(self, monkeypatch):
        monkeypatch.setattr(
            cc, "get_cuda_status", lambda: _make_status(usable=False, physical=8)
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert cc.warn_cuda_mismatch_once() is True
            assert cc.warn_cuda_mismatch_once() is True  # still a mismatch...
        # ...but only one warning is ever emitted per process.
        mismatch_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(mismatch_warns) == 1
        assert "NVIDIA GPUs detected" in str(mismatch_warns[0].message)

    def test_no_warning_when_usable(self, monkeypatch):
        monkeypatch.setattr(
            cc, "get_cuda_status", lambda: _make_status(usable=True, physical=8)
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert cc.warn_cuda_mismatch_once() is False
        assert len(w) == 0

    def test_no_warning_on_pure_cpu_host(self, monkeypatch):
        # No physical GPUs => CPU-only, not a mismatch, no warning.
        monkeypatch.setattr(
            cc, "get_cuda_status", lambda: _make_status(usable=False, physical=0)
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert cc.warn_cuda_mismatch_once() is False
        assert len(w) == 0

    def test_env_opt_out_suppresses_warning(self, monkeypatch):
        monkeypatch.setenv(cc._SILENCE_ENV, "1")
        monkeypatch.setattr(
            cc, "get_cuda_status", lambda: _make_status(usable=False, physical=8)
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Still reports the mismatch to callers, but emits no warning.
            assert cc.warn_cuda_mismatch_once() is True
        assert len(w) == 0

    def test_env_opt_out_falsey_values_do_not_suppress(self, monkeypatch):
        monkeypatch.setenv(cc._SILENCE_ENV, "0")
        monkeypatch.setattr(
            cc, "get_cuda_status", lambda: _make_status(usable=False, physical=8)
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cc.warn_cuda_mismatch_once()
        assert len(w) == 1


class TestVersionHelpers:
    def test_torch_cuda_version_type(self):
        v = cc.torch_cuda_version()
        assert v is None or isinstance(v, str)

    def test_driver_cuda_version_type(self):
        v = cc.driver_cuda_version()
        assert v is None or isinstance(v, str)
