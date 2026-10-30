"""Device placement and CUDA-fault recovery for the Transformers engine.

Resolves the device map, probes it before sampling, and retries a pinned
placement after a device-side assert. Mixed into
:class:`~effgen.models.transformers_engine.TransformersEngine`; not usable on
its own.
"""

from __future__ import annotations

import logging

try:
    import torch
except ImportError as exc:  # pragma: no cover - only on an install without torch
    from effgen.models._adapter_utils import missing_torch_error

    raise missing_torch_error("transformers") from exc

from ._vram import free_vram_gb

logger = logging.getLogger("effgen.models.transformers_engine")


class TransformersPlacementMixin:
    """Device-map resolution and CUDA-fault recovery."""

    @staticmethod
    def _is_cuda_device_side_assert(exc: BaseException) -> bool:
        """True when CUDA sampling/forward failed with a device-side assert."""
        msg = str(exc).lower()
        return (
            "device-side assert" in msg
            or "cudaerrorassert" in msg
            or "probability tensor contains" in msg
        )

    def _effective_device_map(self) -> str | dict[str, int]:
        """Device map for from_pretrained; pin to GPU 0 only after a CUDA assert."""
        if self.device == "cuda" and self._pin_device_map_for_cuda:
            return {"": 0}
        return self.device_map

    def _last_logits_look_valid(self, logits: torch.Tensor) -> bool:
        """Heuristic: invalid logits from device_map='auto' precede CUDA sampling asserts."""
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return False
        return float(logits.max()) > 10.0

    def _probe_auto_device_map_logits(self) -> bool:
        """
        Return True if a short forward pass produces plausible logits.

        device_map='auto' can shard across GPUs and yield broken logits; sampling
        then triggers a CUDA device-side assert. Probing avoids poisoning CUDA.
        """
        if self.model is None or self.tokenizer is None:
            return True

        probe_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        probe_inputs = self.tokenizer(
            probe_prompt, return_tensors="pt", truncation=True, max_length=64
        )
        if self.device != "cpu":
            probe_inputs = {
                k: v.to(self.model.device) for k, v in probe_inputs.items()
            }

        with torch.no_grad():
            probe_out = self.model(
                input_ids=probe_inputs["input_ids"],
                attention_mask=probe_inputs.get("attention_mask"),
            )
        last_logits = probe_out.logits[:, -1, :]
        return self._last_logits_look_valid(last_logits)

    def _apply_cuda_device_map_pin_fallback(self, *, reason: str) -> None:
        """Reload weights on GPU 0 (device_map={'': 0})."""
        if self._pin_device_map_for_cuda:
            return

        self._pin_device_map_for_cuda = True
        logger.warning("%s Reloading on GPU 0 (device_map={'': 0}).", reason)
        self._drop_model_weights()
        try:
            self._load_model_weights()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reload model after pinning to GPU 0: {exc}"
            ) from exc
        if self.model is None:
            raise RuntimeError("Model reload after GPU pin produced no model.")
        if getattr(self, "_metadata", None):
            self._metadata["num_parameters"] = self.model.num_parameters()

    def _ensure_device_map_viable_before_sampling(self) -> None:
        """
        Pin to GPU 0 when device_map='auto' would cause a CUDA device-side assert.

        A CUDA assert poisons the GPU context, so recovery must happen before
        sampling (detected via a forward-pass probe), not after.
        """
        if (
            self._pin_device_map_for_cuda
            or self._device_map_probe_passed
            or self.device != "cuda"
            or self.device_map != "auto"
        ):
            return

        try:
            logits_ok = self._probe_auto_device_map_logits()
        except Exception as exc:
            if self._is_cuda_device_side_assert(exc):
                logits_ok = False
            else:
                logger.warning("device_map probe failed (%s); continuing.", exc)
                return

        if logits_ok:
            # These weights are good; do not spend a forward pass on the next
            # call asking the same question again.
            self._device_map_probe_passed = True
            return

        self._apply_cuda_device_map_pin_fallback(
            reason=(
                "device_map='auto' produced invalid logits on this system "
                "(would cause a CUDA device-side assert during sampling)."
            ),
        )

    def _maybe_retry_after_cuda_assert(self, exc: BaseException, operation):
        """
        Retry generation after detecting a CUDA device-side assert.

        If the GPU context is already poisoned, reload is not possible in-process;
        callers must restart the Python process.
        """
        if self._retrying_after_cuda_assert or not self._is_cuda_device_side_assert(exc):
            raise exc

        if self._pin_device_map_for_cuda:
            raise RuntimeError(
                "CUDA device-side assert during generation. The GPU context is no "
                "longer usable in this process — restart Python, or load with "
                "device_map={'': 0} / CUDA_VISIBLE_DEVICES=0."
            ) from exc

        self._retrying_after_cuda_assert = True
        try:
            self._apply_cuda_device_map_pin_fallback(
                reason=(
                    f"CUDA device-side assert during generation with "
                    f"device_map={self.device_map!r}."
                ),
            )
            return operation()
        except Exception as retry_exc:
            if self._is_cuda_device_side_assert(retry_exc) or self.model is None:
                raise RuntimeError(
                    "CUDA device-side assert during generation. Restart the Python "
                    "process, or set CUDA_VISIBLE_DEVICES=0 before loading."
                ) from exc
            raise
        finally:
            self._retrying_after_cuda_assert = False

    def _resolve_placement(self) -> str:
        """Return where the model parameters actually reside after loading.

        Returns 'cuda' if every parameter is on a GPU, 'cpu' if every parameter
        is on CPU (or disk), or 'mixed' if the model is split across GPU and
        CPU/disk. accelerate records the per-module placement in
        ``model.hf_device_map`` when ``device_map`` dispatch is used; otherwise
        the placement is read from the parameters directly.
        """
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            on_gpu = on_host = False
            for dev in device_map.values():
                if isinstance(dev, int):
                    on_gpu = True
                    continue
                text = str(dev).lower()
                if text.startswith("cuda") or text.isdigit():
                    on_gpu = True
                else:  # "cpu", "disk", "meta"
                    on_host = True
            if on_gpu and on_host:
                return "mixed"
            return "cuda" if on_gpu else "cpu"
        try:
            param = next(self.model.parameters())
        except StopIteration:
            return str(self.device)
        return "cuda" if param.is_cuda else "cpu"

    @staticmethod
    def _free_vram_gb() -> float:
        """Total free VRAM (GB) across the visible CUDA devices, or 0.0 if none.

        Returns:
            Free memory in gibibytes, summed over the visible devices.
        """
        return free_vram_gb()
