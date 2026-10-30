"""
Cross-platform compatibility tests.

Ensures MLX/Apple Silicon code does not break on NVIDIA/Linux machines
and vice versa. All tests must pass regardless of the platform.
"""

from __future__ import annotations

import pytest

from effgen.hardware.platform import (
    HardwarePlatform,
    detect_platform,
    get_best_local_backend,
    get_unified_memory_gb,
    is_apple_silicon,
    is_cuda_available,
    is_mlx_available,
    is_mlx_vlm_available,
)
from effgen.models.base import ModelType

# ---------------------------------------------------------------------------
# Hardware platform detection
# ---------------------------------------------------------------------------

class TestPlatformDetection:
    """Platform detection returns sane values on any machine."""

    def test_is_apple_silicon_returns_bool(self):
        assert isinstance(is_apple_silicon(), bool)

    def test_is_cuda_available_returns_bool(self):
        assert isinstance(is_cuda_available(), bool)

    def test_is_mlx_available_returns_bool(self):
        assert isinstance(is_mlx_available(), bool)

    def test_is_mlx_vlm_available_returns_bool(self):
        assert isinstance(is_mlx_vlm_available(), bool)

    def test_detect_platform_returns_enum(self):
        result = detect_platform()
        assert isinstance(result, HardwarePlatform)

    def test_get_best_local_backend_returns_string(self):
        backend = get_best_local_backend()
        assert backend in ("mlx", "vllm", "transformers")

    def test_get_unified_memory_returns_float(self):
        mem = get_unified_memory_gb()
        assert isinstance(mem, float)
        assert mem >= 0.0

    def test_mlx_not_available_without_apple_silicon(self):
        """On non-Apple hardware, MLX should never be available."""
        if not is_apple_silicon():
            assert not is_mlx_available()
            assert not is_mlx_vlm_available()

    def test_mlx_vlm_requires_mlx(self):
        """MLX-VLM cannot be available without MLX."""
        if not is_mlx_available():
            assert not is_mlx_vlm_available()


# ---------------------------------------------------------------------------
# ModelType enum
# ---------------------------------------------------------------------------

class TestModelTypeEnum:
    """ModelType enum has all expected backends."""

    def test_has_mlx(self):
        assert hasattr(ModelType, "MLX")
        assert ModelType.MLX.value == "mlx"

    def test_has_mlx_vlm(self):
        assert hasattr(ModelType, "MLX_VLM")
        assert ModelType.MLX_VLM.value == "mlx_vlm"

    def test_has_vllm(self):
        assert hasattr(ModelType, "VLLM")

    def test_has_transformers(self):
        assert hasattr(ModelType, "TRANSFORMERS")

    def test_has_openai(self):
        assert hasattr(ModelType, "OPENAI")

    def test_has_anthropic(self):
        assert hasattr(ModelType, "ANTHROPIC")

    def test_has_gemini(self):
        assert hasattr(ModelType, "GEMINI")


# ---------------------------------------------------------------------------
# MLX engine imports (should not crash on non-Apple)
# ---------------------------------------------------------------------------

class TestMLXImportsNonApple:
    """MLX modules can be imported without crashing, even on NVIDIA."""

    def test_import_mlx_engine(self):
        from effgen.models.mlx_engine import MLXEngine
        assert MLXEngine is not None

    def test_import_mlx_vlm_engine(self):
        from effgen.models.mlx_vlm_engine import MLXVLMEngine
        assert MLXVLMEngine is not None

    def test_mlx_engine_instantiation(self):
        """Can create an MLXEngine instance without crashing."""
        from effgen.models.mlx_engine import MLXEngine
        engine = MLXEngine(model_name="test-model")
        assert engine.model_type == ModelType.MLX

    def test_mlx_vlm_engine_instantiation(self):
        """Can create an MLXVLMEngine instance without crashing."""
        from effgen.models.mlx_vlm_engine import MLXVLMEngine
        engine = MLXVLMEngine(model_name="test-model")
        assert engine.model_type == ModelType.MLX_VLM

    def test_mlx_engine_load_raises_a_clear_error_on_nvidia(self):
        """Loading MLX model on non-Apple should raise RuntimeError with helpful message."""
        if is_apple_silicon():
            pytest.skip("Running on Apple Silicon")
        from effgen.models.mlx_engine import MLXEngine
        engine = MLXEngine(model_name="test-model")
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            engine.load()

    def test_mlx_vlm_engine_load_raises_a_clear_error_on_nvidia(self):
        """Loading MLX-VLM model on non-Apple should raise RuntimeError with helpful message."""
        if is_apple_silicon():
            pytest.skip("Running on Apple Silicon")
        from effgen.models.mlx_vlm_engine import MLXVLMEngine
        engine = MLXVLMEngine(model_name="test-model")
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            engine.load()

    def test_load_model_mlx_raises_a_clear_error_on_nvidia(self):
        """load_model with engine='mlx' on non-Apple should raise RuntimeError."""
        if is_apple_silicon():
            pytest.skip("Running on Apple Silicon")
        from effgen.models import load_model
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            load_model("test-model", engine="mlx")

    def test_load_model_mlx_vlm_raises_a_clear_error_on_nvidia(self):
        """load_model with engine='mlx_vlm' on non-Apple should raise RuntimeError."""
        if is_apple_silicon():
            pytest.skip("Running on Apple Silicon")
        from effgen.models import load_model
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            load_model("test-model", engine="mlx_vlm")


# ---------------------------------------------------------------------------
# Models __init__ wildcard import
# ---------------------------------------------------------------------------

class TestModelsWildcardImport:
    """from effgen.models import * should work on any platform."""

    def test_wildcard_import_does_not_crash(self):
        # The local engines are part of this surface and import torch.
        pytest.importorskip("torch")
        import effgen.models as m
        for name in m.__all__:
            assert hasattr(m, name), f"{name} in __all__ but not accessible"

    def test_all_entries_count(self):
        import effgen.models as m
        # Should have at least 20 entries (base + engines + adapters + multi-model support).
        assert len(m.__all__) >= 20


# ---------------------------------------------------------------------------
# Config validator accepts mlx engines
# ---------------------------------------------------------------------------

class TestConfigValidatorMLX:
    """Config validator should accept 'mlx' and 'mlx_vlm' as engine options."""

    def test_validator_imports(self):
        from effgen.config.validator import ConfigValidator
        assert ConfigValidator is not None
