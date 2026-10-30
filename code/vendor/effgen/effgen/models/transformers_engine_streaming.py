"""Token-by-token streaming for the Transformers engine.

Runs generation on a worker thread behind a text-iterator streamer and yields
decoded chunks. Mixed into
:class:`~effgen.models.transformers_engine.TransformersEngine`; not usable on
its own.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from effgen.models._adapter_utils import (
    apply_call_overrides,
    not_loaded_error,
    provider_runtime_error,
)
from effgen.models.base import GenerationConfig
from effgen.models.transformers_engine_support import _reraise_if_classified

logger = logging.getLogger("effgen.models.transformers_engine")


class TransformersStreamingMixin:
    """Token-by-token streaming generation."""

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise not_loaded_error("transformers", self.model_name, "generate_stream")

        self.validate_prompt(prompt)

        config = apply_call_overrides(config, kwargs)
        generation_config, stop_sequences = self._create_generation_config(config)

        try:
            # Apply the chat template (same as generate()) so instruct/chat
            # models receive their expected role-tagged format. Without this,
            # streaming fed the raw prompt and the model rambled / never emitted
            # its stop token.
            tools_for_template = kwargs.pop("tools", None)
            formatted_prompt = self._apply_chat_template(prompt, tools_for_template)
            self._ensure_device_map_viable_before_sampling()

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_length
            )

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Use TextIteratorStreamer for streaming
            from threading import Event, Thread

            from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
                timeout=30.0,  # prevent indefinite block if generation thread dies
                clean_up_tokenization_spaces=False,  # non-destructive for BPE; see generate()
            )

            stop_event = Event()

            class _StopOnEvent(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs) -> bool:
                    return stop_event.is_set()

            stopping_criteria = kwargs.pop("stopping_criteria", None)
            if stopping_criteria is None:
                stopping_criteria = StoppingCriteriaList([_StopOnEvent()])
            elif isinstance(stopping_criteria, StoppingCriteriaList):
                stopping_criteria.append(_StopOnEvent())
            else:
                try:
                    criteria_items = list(stopping_criteria)
                except TypeError:
                    criteria_items = [stopping_criteria]
                stopping_criteria = StoppingCriteriaList([*criteria_items, _StopOnEvent()])

            # Note: stop_sequences not fully supported in streaming mode
            # They would need to be checked in the consumer of the stream

            self._seed_sampling(config, kwargs)

            # Fold any per-call generation kwargs into the config (Transformers
            # 5.x deprecates passing them alongside a generation_config); forward
            # only leftover non-generation kwargs.
            generation_config, extra_kwargs = self._fold_into_generation_config(
                generation_config, self._sanitize_generation_kwargs(kwargs)
            )

            # Generate in a separate thread
            generation_kwargs = {
                **inputs,
                "generation_config": generation_config,
                "stopping_criteria": stopping_criteria,
                "streamer": streamer,
                **extra_kwargs
            }

            gen_exception: list[BaseException] = []

            def _generate_with_error_capture() -> None:
                try:
                    self.model.generate(**generation_kwargs)
                except BaseException as exc:
                    gen_exception.append(exc)
                    # Unblock the streamer queue so yield-from terminates
                    streamer.end()

            thread = Thread(target=_generate_with_error_capture, daemon=False)
            thread.start()

            try:
                # Yield tokens as they're generated
                yield from streamer
            finally:
                stop_event.set()
                try:
                    streamer.end()
                except Exception:
                    logger.debug("Failed to end streamer", exc_info=True)
                thread.join(timeout=30.0)

            if thread.is_alive():
                raise RuntimeError("Streaming generation thread did not exit cleanly")
            if gen_exception:
                exc = gen_exception[0]
                if self._is_cuda_device_side_assert(exc) and not self._retrying_after_cuda_assert:
                    self._retrying_after_cuda_assert = True
                    try:
                        self._apply_cuda_device_map_pin_fallback(
                            reason=(
                                f"CUDA device-side assert during streaming with "
                                f"device_map={self.device_map!r}."
                            ),
                        )
                        yield from self.generate_stream(prompt, config, **kwargs)
                        return
                    except Exception as retry_exc:
                        if self._is_cuda_device_side_assert(retry_exc) or self.model is None:
                            raise RuntimeError(
                                "CUDA device-side assert during streaming. Restart the "
                                "Python process, or set CUDA_VISIBLE_DEVICES=0 before loading."
                            ) from exc
                        raise
                    finally:
                        self._retrying_after_cuda_assert = False
                raise exc

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            _reraise_if_classified(e)
            raise provider_runtime_error(
                "transformers", self.model_name, "generate_stream", e,
                message="Streaming generation failed",
            ) from e
