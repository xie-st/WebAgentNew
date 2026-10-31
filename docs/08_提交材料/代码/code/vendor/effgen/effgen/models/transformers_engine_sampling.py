"""Generation-config assembly and chat-template rendering for the Transformers engine.

Builds the HuggingFace generation config for a call, normalizes per-call
sampling kwargs, seeds sampling, and renders the chat template. Mixed into
:class:`~effgen.models.transformers_engine.TransformersEngine`; not usable on
its own.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import torch  # noqa: F401  probes the local-inference extra
except ImportError as exc:  # pragma: no cover - only on an install without torch
    from effgen.models._adapter_utils import missing_torch_error

    raise missing_torch_error("transformers") from exc

from transformers import (
    GenerationConfig as HFGenerationConfig,
)
from transformers import (
    set_seed as _hf_set_seed,
)

from effgen.models.base import GenerationConfig

logger = logging.getLogger("effgen.models.transformers_engine")


class TransformersSamplingMixin:
    """Generation-config assembly, sampling kwargs and chat-template rendering."""

    def _eos_token_ids(self) -> int | list[int] | None:
        """Return every token id that should end generation for this model.

        A model's own ``generation_config`` may declare several terminators
        while ``tokenizer.eos_token_id`` holds only one. Llama 3.x is the case
        that matters: the tokenizer reports ``<|eot_id|>`` (end of turn), but a
        tool call ends with ``<|eom_id|>`` (end of message). Passing only the
        tokenizer's id leaves ``<|eom_id|>`` a normal token, so after emitting a
        tool call the model keeps going and writes the assistant turn that
        should have followed the tool's result — inventing an observation it
        never received. Merging both sources stops generation where the model
        intends to stop, and leaves single-terminator models unchanged.
        """
        ids: list[int] = []

        def _add(value: Any) -> None:
            candidates = value if isinstance(value, list | tuple) else [value]
            for candidate in candidates:
                # bool is an int subclass and is never a token id.
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    if candidate not in ids:
                        ids.append(candidate)

        _add(getattr(getattr(self.model, "generation_config", None), "eos_token_id", None))
        _add(getattr(self.tokenizer, "eos_token_id", None))

        if not ids:
            return None
        return ids[0] if len(ids) == 1 else ids

    def _create_generation_config(
        self,
        config: GenerationConfig | None = None
    ) -> tuple[HFGenerationConfig, list[str]]:
        """
        Create HuggingFace GenerationConfig from our GenerationConfig.

        Args:
            config: Our generation configuration

        Returns:
            Tuple of (HuggingFace GenerationConfig object, stop_sequences list)

        Notes:
            HuggingFace Transformers doesn't support stop sequences natively like OpenAI,
            so we return them separately for post-generation processing.
        """
        if config is None:
            config = GenerationConfig()

        eos_token_id = self._eos_token_ids()

        # Normalize deterministic generation. Transformers 5.x rejects
        # temperature<=0 ("has to be a strictly positive float") and warns when
        # sampling params (temperature/top_p/top_k) are set while do_sample is
        # False. Treat temperature<=0 as greedy decoding: set do_sample=False and
        # omit the sampling params entirely so the same effGen config works
        # identically across Transformers versions and other backends.
        do_sample = config.temperature is not None and config.temperature > 0
        if do_sample:
            hf_config = HFGenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_tokens or 512,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
        else:
            # Greedy decoding. Set the sampling params to their no-op defaults
            # (temperature=1.0, top_p=1.0, top_k=50) explicitly: leaving them unset
            # lets Transformers merge the model's own generation_config.json
            # sampling values (e.g. Qwen's temperature=0.7) into the config, which
            # then triggers a "generation flags not valid for do_sample=False"
            # warning on every greedy call. Explicit no-op values suppress it
            # without affecting greedy output.
            hf_config = HFGenerationConfig(
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                max_new_tokens=config.max_tokens or 512,
                repetition_penalty=config.repetition_penalty,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        # Return stop sequences separately for post-processing
        # NOTE: We DON'T set them as eos_token_id because that would stop generation
        # at the first token match, not the full sequence match
        return hf_config, config.stop_sequences if config.stop_sequences else []

    # HuggingFace GenerationConfig fields effGen forwards from per-call kwargs.
    _HF_GEN_PARAMS = (
        "temperature", "top_p", "top_k", "repetition_penalty",
        "num_beams", "do_sample", "pad_token_id", "eos_token_id",
        "max_new_tokens",
    )

    def _seed_sampling(self, config: GenerationConfig | None, kwargs: dict[str, Any]) -> None:
        """Seed the sampling RNGs so a fixed seed reproduces the text on-device.

        The seed comes from a per-call ``seed=`` kwarg when given (removed from
        ``kwargs`` so it is not forwarded on as an unknown generation parameter),
        otherwise from ``config.seed`` — matching the precedence every other
        per-call generation override follows. ``set_seed()`` covers torch (CPU +
        CUDA), numpy and random; it only affects sampling, so greedy decoding
        (``temperature<=0``) is unchanged either way.
        """
        seed = kwargs.pop("seed", None)
        if seed is None:
            seed = getattr(config, "seed", None)
        if seed is not None:
            _hf_set_seed(seed)

    def _sanitize_generation_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate per-call OpenAI-style generation kwargs to HuggingFace names.

        ``max_tokens`` becomes ``max_new_tokens``; recognised HuggingFace
        generation params pass through; a non-positive ``temperature`` collapses
        to greedy decoding; a positive ``temperature`` enables sampling. Unknown
        keys are logged and skipped so a stray kwarg never crashes generation.

        These values are folded INTO an existing ``GenerationConfig`` (see
        :meth:`_fold_into_generation_config`), so a per-call override must fully
        supersede the config's sampling fields — not merely add ``do_sample`` —
        or a config that defaulted to sampling (``temperature=0.7``) would keep
        ``temperature``/``top_p`` set alongside ``do_sample=False`` and trip the
        Transformers "generation flags are not valid" warning.
        """
        sanitized: dict[str, Any] = {}
        for key, value in kwargs.items():
            try:
                if key == "max_tokens":
                    sanitized["max_new_tokens"] = value
                    logger.debug("Converted max_tokens=%s to max_new_tokens", value)
                elif key in self._HF_GEN_PARAMS:
                    sanitized[key] = value
                else:
                    logger.warning("Skipping unknown generation parameter: %s=%s", key, value)
            except Exception as e:
                logger.error("Error processing generation parameter %s: %s", key, e)
                continue

        if "temperature" in sanitized:
            temp = sanitized["temperature"]
            if temp is None or temp <= 0:
                # Greedy decoding. Overwrite the sampling fields with their no-op
                # defaults (not just do_sample=False) so they override whatever
                # the base config carried — mirroring the greedy branch of
                # _create_generation_config and keeping the call warning-free.
                sanitized["temperature"] = 1.0
                sanitized["top_p"] = 1.0
                sanitized["top_k"] = 50
                sanitized["do_sample"] = False
            else:
                # A positive per-call temperature is an explicit request to
                # sample; enable it so the override isn't silently ignored when
                # the base config was greedy.
                sanitized.setdefault("do_sample", True)
        return sanitized

    def _fold_into_generation_config(
        self, generation_config: HFGenerationConfig, params: dict[str, Any]
    ) -> tuple[HFGenerationConfig, dict[str, Any]]:
        """Merge generation *params* INTO *generation_config*; return leftover kwargs.

        Passing a ``generation_config`` together with generation-related keyword
        arguments is deprecated in Transformers 5.x and prints a warning on every
        call. Folding the recognised parameters into the config object — and
        forwarding only genuinely non-generation kwargs (e.g. ``streamer``,
        ``stopping_criteria``) separately — keeps each call quiet with identical
        decoding behaviour (per-call values still override the config). Returns
        the mutated config and the kwargs it did not consume.
        """
        if not params:
            return generation_config, {}
        unused = generation_config.update(**params)
        return generation_config, unused or {}

    def _apply_chat_template(self, prompt: str, tools_for_template: Any = None) -> str:
        """Wrap *prompt* with the tokenizer's chat template when one exists.

        Instruct/chat models (Qwen, Llama-3, …) expect their role-tagged format;
        feeding the raw text makes them ramble and skip the stop token. Both the
        batched ``generate`` and ``generate_stream`` paths use this so streamed
        and non-streamed answers are formatted identically.
        """
        if not (hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            # Caller-supplied template arguments, e.g. enable_thinking=False on a
            # Qwen3 template. A template that does not declare the name ignores
            # it, so this is safe to pass to any model.
            template_kwargs.update(getattr(self, "chat_template_kwargs", None) or {})
            if tools_for_template:
                template_kwargs["tools"] = tools_for_template
                logger.debug(
                    f"Passing {len(tools_for_template)} tool definitions "
                    "to chat template for native function calling"
                )
            formatted = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            logger.debug("Applied chat template to prompt")
            return formatted
        except TypeError as e:
            if tools_for_template:
                logger.debug(
                    f"Chat template does not accept tools param: {e}, "
                    "falling back to plain template"
                )
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            logger.warning(f"Failed to apply chat template: {e}")
            return prompt
        except Exception as e:
            logger.warning(f"Failed to apply chat template, using raw prompt: {e}")
            return prompt
