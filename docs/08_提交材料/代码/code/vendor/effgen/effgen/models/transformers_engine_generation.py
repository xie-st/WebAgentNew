"""Blocking and batched text generation for the Transformers engine.

Holds the single-prompt and batched generation paths and the token counter.
Mixed into :class:`~effgen.models.transformers_engine.TransformersEngine`; not
usable on its own.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - only on an install without torch
    from effgen.models._adapter_utils import missing_torch_error

    raise missing_torch_error("transformers") from exc

from effgen.models._adapter_utils import (
    apply_call_overrides,
    normalize_finish_reason,
    not_loaded_error,
    provider_runtime_error,
)
from effgen.models.base import (
    GenerationConfig,
    GenerationResult,
    TokenCount,
)
from effgen.models.transformers_engine_support import (
    _reraise_if_classified,
    _strip_special_tokens_keep_tool_calls,
)

logger = logging.getLogger("effgen.models.transformers_engine")


class TransformersGenerationMixin:
    """Blocking single-prompt generation, batched generation and token counting."""

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise not_loaded_error("transformers", self.model_name, "generate")

        self.validate_prompt(prompt)

        config = apply_call_overrides(config, kwargs)
        generation_config, stop_sequences = self._create_generation_config(config)

        # Serialize fast-tokenizer + generate so concurrent local calls (e.g.
        # batch at concurrency>1) never trip the tokenizer's "Already borrowed".
        self._tokenizer_lock.acquire()
        try:
            # Extract tool definitions before sanitizing kwargs — these are
            # passed to the chat template, not to HF generate()
            tools_for_template = kwargs.pop("tools", None)

            self._seed_sampling(config, kwargs)

            # Sanitize per-call kwargs (OpenAI-style → HuggingFace) and fold them
            # into the GenerationConfig. Passing generation params alongside a
            # generation_config is deprecated in Transformers 5.x, so we merge
            # them in and forward only the leftover non-generation kwargs.
            sanitized_kwargs = self._sanitize_generation_kwargs(kwargs)
            generation_config, extra_kwargs = self._fold_into_generation_config(
                generation_config, sanitized_kwargs
            )

            # Apply chat template if available for better model compatibility
            # Many modern models like Qwen expect a specific format
            formatted_prompt = self._apply_chat_template(prompt, tools_for_template)

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_length
            )

            self._ensure_device_map_viable_before_sampling()

            # Generate with sanitized kwargs. Wrapped so a CUDA device-side
            # assert can be retried once on a model pinned to GPU 0.
            def _run_generate():
                if self.model is None:
                    raise RuntimeError("Model is not loaded.")
                gen_inputs = inputs
                if self.device != "cpu":
                    gen_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    return self.model.generate(
                        **gen_inputs,
                        generation_config=generation_config,
                        **extra_kwargs,
                    )

            try:
                outputs = _run_generate()
            except Exception as e:
                outputs = self._maybe_retry_after_cuda_assert(e, _run_generate)

            # Decode output
            # When native tool calling is active, preserve tool-call tokens
            # like <tool_call>, </tool_call>, [TOOL_CALLS] etc. but strip
            # chat-template end markers like <|im_end|>, </s>, <|eot_id|>.
            # clean_up_tokenization_spaces=False: the cleanup step is destructive
            # for BPE/SentencePiece tokenizers (it strips spaces before
            # punctuation) and Transformers warns + ignores it for them anyway.
            # Passing False explicitly preserves spacing and silences the warning.
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            if tools_for_template:
                # Preserve native tool-call delimiters for the parser, then
                # strip every other special token so chat-template turn/end
                # markers never leak into the answer (see helper docstring).
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                generated_text = _strip_special_tokens_keep_tool_calls(
                    generated_text, self.tokenizer,
                )
            else:
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            # Apply stop sequences post-generation
            # This is more reliable than trying to use them during generation
            finish_reason = "stop"
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        # Find first occurrence of any stop sequence
                        stop_index = generated_text.find(stop_seq)
                        if stop_index != -1:
                            generated_text = generated_text[:stop_index]
                            finish_reason = "stop_sequence"
                            logger.debug(f"Stopped generation at stop sequence: '{stop_seq}'")
                            break

            # Calculate tokens
            prompt_tokens = inputs["input_ids"].shape[1]
            completion_tokens = len(generated_ids)

            # HuggingFace `generate()` doesn't report whether decoding stopped
            # at EOS or was cut off at the token budget. Infer the budget case:
            # no stop-sequence match, the last token isn't an EOS id, and the
            # model produced the full requested budget.
            if finish_reason == "stop":
                eos_ids = generation_config.eos_token_id
                if eos_ids is None:
                    eos_ids = ()
                elif isinstance(eos_ids, int):
                    eos_ids = (eos_ids,)
                last_token = generated_ids[-1].item() if completion_tokens else None
                max_new = generation_config.max_new_tokens
                if (
                    max_new is not None
                    and completion_tokens >= max_new
                    and last_token not in eos_ids
                ):
                    finish_reason = "length"

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=normalize_finish_reason(finish_reason),
                model_name=self.model_name,
                metadata={
                    "raw_finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "stop_sequences_applied": stop_sequences if stop_sequences else [],
                    "device": self.device,
                }
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            _reraise_if_classified(e)
            raise provider_runtime_error(
                "transformers", self.model_name, "generate", e,
                message="Generation failed",
            ) from e
        finally:
            self._tokenizer_lock.release()

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        **kwargs: Any
    ) -> list[GenerationResult]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If any prompt exceeds context length
        """
        if not self._is_loaded:
            raise not_loaded_error("transformers", self.model_name, "generate_batch")

        # Validate all prompts
        for prompt in prompts:
            self.validate_prompt(prompt)

        generation_config, stop_sequences = self._create_generation_config(config)

        # Serialize fast-tokenizer + generate (see generate(): thread-safety).
        self._tokenizer_lock.acquire()
        try:
            self._ensure_device_map_viable_before_sampling()

            # Apply the chat template to each prompt, the way generate() does.
            # An instruct model that never sees its role tags continues the text
            # instead of answering it.
            tools_for_template = kwargs.pop("tools", None)
            formatted_prompts = [
                self._apply_chat_template(prompt, tools_for_template)
                for prompt in prompts
            ]

            # Tokenize all inputs. Decoder-only batching pads on the LEFT:
            # generation continues from the last position of each row, and the
            # decode below slices every row at the shared padded width. Padding
            # on the right would have the model continue from pad tokens and cut
            # the slice in the wrong place.
            previous_padding_side = getattr(self.tokenizer, "padding_side", None)
            if previous_padding_side is not None:
                self.tokenizer.padding_side = "left"
            try:
                inputs = self.tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._context_length
                )
            finally:
                if previous_padding_side is not None:
                    self.tokenizer.padding_side = previous_padding_side

            # Fold per-call generation kwargs into the config (Transformers 5.x
            # deprecates passing them alongside a generation_config); forward only
            # leftover non-generation kwargs.
            generation_config, extra_kwargs = self._fold_into_generation_config(
                generation_config, self._sanitize_generation_kwargs(kwargs)
            )

            # Generate. Wrapped so a CUDA device-side assert can be retried once
            # on a model pinned to GPU 0.
            def _run_batch_generate():
                if self.model is None:
                    raise RuntimeError("Model is not loaded.")
                gen_inputs = inputs
                if self.device != "cpu":
                    gen_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    return self.model.generate(
                        **gen_inputs,
                        generation_config=generation_config,
                        **extra_kwargs,
                    )

            try:
                outputs = _run_batch_generate()
            except Exception as e:
                outputs = self._maybe_retry_after_cuda_assert(e, _run_batch_generate)

            # Decode outputs
            results = []
            for i, output in enumerate(outputs):
                # Get only the generated part (exclude input)
                # Rows share a padded width, which is where the generated
                # tokens begin; the prompt's own length is what it actually
                # tokenized to, so padding is not reported as prompt tokens.
                padded_width = inputs["input_ids"][i].shape[0]
                mask = inputs.get("attention_mask")
                prompt_length = (
                    int(mask[i].sum()) if mask is not None else padded_width
                )
                generated_ids = output[padded_width:]

                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                results.append(GenerationResult(
                    text=generated_text,
                    tokens_used=len(generated_ids),
                    finish_reason="stop",
                    model_name=self.model_name,
                    metadata={
                        "prompt_tokens": prompt_length,
                        "completion_tokens": len(generated_ids),
                        "total_tokens": prompt_length + len(generated_ids),
                        "device": self.device,
                    }
                ))

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            _reraise_if_classified(e)
            raise provider_runtime_error(
                "transformers", self.model_name, "generate_batch", e,
                message="Batch generation failed",
            ) from e
        finally:
            self._tokenizer_lock.release()

    def count_tokens(self, text: str) -> TokenCount:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._is_loaded or self.tokenizer is None:
            raise not_loaded_error("transformers", self.model_name, "count_tokens")

        # Serialize tokenizer access (see generate(): "Already borrowed").
        with self._tokenizer_lock:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return TokenCount(count=len(tokens), model_name=self.model_name)
            except Exception as e:
                logger.error(f"Token counting failed: {e}")
                raise provider_runtime_error(
                    "transformers", self.model_name, "count_tokens", e,
                    message="Token counting failed",
                ) from e
