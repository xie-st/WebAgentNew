"""
Replicate model registry for effGen.

Full model catalog fetched from the Replicate API on 2026-04-29 and
reconciled against it on 2026-08-07.
Includes all language models from the official Replicate language-models
collection plus additional text-generation models.

Replicate billing model:
  - Charged per second of GPU compute (predict_time from metrics).
  - Prices vary by hardware (T4, L40S, A100, H100, etc.).
  - Typical rates:
      - T4 GPU:    ~$0.000225/second
      - L40S GPU:  ~$0.000975/second
      - A100 GPU:  ~$0.001400/second
      - H100 GPU:  ~$0.001525/second
  - Some models (via official/hosted deployments) have per-token pricing.
  - cost_per_second_usd is set to the typical hardware cost; actual billing
    depends on which hardware Replicate assigns at run time.

REGISTRY_FETCH_DATE: str = "2026-08-07"
Use refresh_models() to fetch the live catalog and detect drift vs this snapshot.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_FETCH_DATE = "2026-08-07"

# ---------------------------------------------------------------------------
# Main registry — keyed by "owner/name" (canonical Replicate model ID)
# ---------------------------------------------------------------------------
REPLICATE_MODELS: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Meta Llama 3 family  (most popular models on Replicate)
    # -------------------------------------------------------------------------
    "meta/meta-llama-3-8b-instruct": {
        "display_name": "Llama 3 8B Instruct",
        "family": "llama3",
        "organization": "Meta",
        "context": 8_192,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "cost_per_second_usd": 0.000225,
        "run_count": 408_247_347,
        "version": "5a6809ca6288247d06daf6365557e5e429063f32a21146b2a807c682652136b8",
    },
    "meta/meta-llama-3-70b-instruct": {
        "display_name": "Llama 3 70B Instruct",
        "family": "llama3",
        "organization": "Meta",
        "context": 8_192,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "cost_per_second_usd": 0.000975,
        "run_count": 167_971_116,
        "version": "fbfb20b472b2f3bdfb1a7e9d09d2f8cd374d7aef3bd61c08be5fe79fc38d8a43",
    },
    "meta/meta-llama-3-8b": {
        "display_name": "Llama 3 8B (Base)",
        "family": "llama3",
        "organization": "Meta",
        "context": 8_192,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.000225,
        "run_count": 51_310_430,
        "version": "9a9e68fc8695f5847d42e8bc339e2d1de77bc69049f25cb0d9b3c29a08fcdc39",
    },
    "meta/meta-llama-3-70b": {
        "display_name": "Llama 3 70B (Base)",
        "family": "llama3",
        "organization": "Meta",
        "context": 8_192,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.000975,
        "run_count": 866_849,
        "version": "83c5bdea9941e83e8a83b1f04ec01285c6b2b05e4e5e4eda3e9fd41bf940b4e8",
    },
    # -------------------------------------------------------------------------
    # IBM Granite — supports native tool-calling via messages API
    # -------------------------------------------------------------------------
    "ibm-granite/granite-3.3-8b-instruct": {
        "display_name": "Granite 3.3 8B Instruct",
        "family": "granite",
        "organization": "IBM",
        "context": 128_000,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system_prompt",
        "prompt_template": None,
        "cost_per_second_usd": 0.000225,
        "run_count": 1_693_126,
        "version": "618ecbe80773609e96ea19d8c96e708f6f2b368bb89be8fad509983194466bf8",
    },
    # -------------------------------------------------------------------------
    # DeepSeek family
    # -------------------------------------------------------------------------
    "deepseek-ai/deepseek-r1": {
        "display_name": "DeepSeek R1",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 64_000,
        "max_output": 16_384,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.001400,
        "run_count": 2_224_948,
        "version": "f5c66b7abc414e3a8882d58cc88f2ae41d39edce3a1c4e3d28ea5f15ff4e5b79",
    },
    "deepseek-ai/deepseek-v3": {
        "display_name": "DeepSeek V3",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 64_000,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.001400,
        "run_count": 5_106_887,
        "version": "b65d70e71acdaa2c8e93dac5ea1b07de83e42d1d3a5bf1ce0ad1aad2b0d66b9e",
    },
    "deepseek-ai/deepseek-v3.1": {
        "display_name": "DeepSeek V3.1",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 64_000,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.001400,
        "run_count": 460_188,
        "version": "f257f380598d0760a0ea8c1b1d02d8bce4c80b9f3a3a08d34ceac7e77a81df5a",
    },
    # -------------------------------------------------------------------------
    # Qwen family
    # -------------------------------------------------------------------------
    "qwen/qwen3-235b-a22b-instruct-2507": {
        "display_name": "Qwen3 235B Instruct",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 16_384,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.001525,
        "run_count": 814_009,
        "version": "a96f2c4d13a1fa84ebe2bf2e3fcff1c5ca8a4b5426d9571cde35e1e8e8975cbf",
    },
    # -------------------------------------------------------------------------
    # Moonshot Kimi
    # -------------------------------------------------------------------------
    "moonshotai/kimi-k2.5": {
        "display_name": "Kimi K2.5",
        "family": "kimi",
        "organization": "Moonshot AI",
        "context": 131_072,
        "max_output": 16_384,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.001525,
        "run_count": 36_417,
        "version": "b4d8427a98a2de295fa0454e7d1c5d84e6d20f96e3e0f2fdf95db15ecc2d4f07",
    },
    # -------------------------------------------------------------------------
    # Anthropic models on Replicate (uses Anthropic's API under the hood)
    # -------------------------------------------------------------------------
    "anthropic/claude-3.7-sonnet": {
        "display_name": "Claude 3.7 Sonnet",
        "family": "claude",
        "organization": "Anthropic",
        "context": 200_000,
        "max_output": 64_000,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 3.0,
        "pricing_per_1m_output": 15.0,
        "run_count": 4_080_811,
        "version": "81a891bd00c339f3d80e5c647a7aae02b5a48e3bdbbea79c38e3e2e9cbfe9979",
    },
    "anthropic/claude-3.5-haiku": {
        "display_name": "Claude 3.5 Haiku",
        "family": "claude",
        "organization": "Anthropic",
        "context": 200_000,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 0.8,
        "pricing_per_1m_output": 4.0,
        "run_count": 3_046_120,
        "version": "291420afd8302405c1e67d9e9b90ae9e06a2f4e31cebeaeccc06caad1cc5607d",
    },
    "anthropic/claude-4.5-sonnet": {
        "display_name": "Claude 4.5 Sonnet",
        "family": "claude",
        "organization": "Anthropic",
        "context": 200_000,
        "max_output": 64_000,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 3.0,
        "pricing_per_1m_output": 15.0,
        "run_count": 1_142_543,
        "version": "459655107e29a6835a41e177b83dd2b39e68d27b4b1a28e5b4eff52ef2649d5a",
    },
    "anthropic/claude-4.5-haiku": {
        "display_name": "Claude 4.5 Haiku",
        "family": "claude",
        "organization": "Anthropic",
        "context": 200_000,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 0.8,
        "pricing_per_1m_output": 4.0,
        "run_count": 650_701,
        "version": "1ad171f62532e20962b5a4f7e83f9e649b80b0e3e6f9c4e79a6a4ea97dae46b5",
    },
    "anthropic/claude-opus-4.6": {
        "display_name": "Claude Opus 4.6",
        "family": "claude",
        "organization": "Anthropic",
        "context": 200_000,
        "max_output": 128_000,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 15.0,
        "pricing_per_1m_output": 75.0,
        "run_count": 149_270,
        "version": "5a2f72c8a00f561db0e4c85c04ef3c9b9f12da0ecd20a7e12bc9d52c2a6c3af4",
    },
    "anthropic/claude-4-sonnet": {
        "display_name": "Claude Sonnet 4",
        "family": "claude",
        "organization": "Anthropic",
        "context": 200_000,
        "max_output": 64_000,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 3.0,
        "pricing_per_1m_output": 15.0,
        "run_count": 2_900_189,
        "version": "3380fe4ca9cac053296b74d6c2bab3b1e1e3f07b94a88c59f9e7d01fc03a97c7",
    },
    # -------------------------------------------------------------------------
    # OpenAI models on Replicate
    # -------------------------------------------------------------------------
    "openai/gpt-4o": {
        "display_name": "GPT-4o",
        "family": "gpt4",
        "organization": "OpenAI",
        "context": 128_000,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 2.5,
        "pricing_per_1m_output": 10.0,
        "run_count": 650_739,
        "version": "42f4eb858641c70ce2d50e9264b2c76f6f82671e5de54c7a58c9ccdc5e3e7e22",
    },
    "openai/gpt-4o-mini": {
        "display_name": "GPT-4o Mini",
        "family": "gpt4",
        "organization": "OpenAI",
        "context": 128_000,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 0.15,
        "pricing_per_1m_output": 0.60,
        "run_count": 37_618_445,
        "version": "86d7f12d34e3f9b68e07df3875c1dd7e13bd49dbb5c8a4e00f77fce64e43f1c4",
    },
    "openai/gpt-4.1": {
        "display_name": "GPT-4.1",
        "family": "gpt4",
        "organization": "OpenAI",
        "context": 1_000_000,
        "max_output": 32_768,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 2.0,
        "pricing_per_1m_output": 8.0,
        "run_count": 317_250,
        "version": "f7e65222875892b7bb22f1f14e6e7d32bfbdfa9c46e0b8a6a6d08e8f36d69d39",
    },
    "openai/gpt-4.1-mini": {
        "display_name": "GPT-4.1 Mini",
        "family": "gpt4",
        "organization": "OpenAI",
        "context": 1_000_000,
        "max_output": 32_768,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 0.40,
        "pricing_per_1m_output": 1.60,
        "run_count": 2_130_578,
        "version": "aca77ca43d7155cf79e7b4f853a74617f4f68b02ed30a3a3c90b7d38af0b8cd6",
    },
    "openai/gpt-4.1-nano": {
        "display_name": "GPT-4.1 Nano",
        "family": "gpt4",
        "organization": "OpenAI",
        "context": 1_000_000,
        "max_output": 32_768,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.40,
        "run_count": 1_675_667,
        "version": "756e9851b24d755b5f9c47d3a25cc4d9b49b89ade93bc3562f73d4dc7e9e3d93",
    },
    "openai/o1": {
        "display_name": "OpenAI o1",
        "family": "o1",
        "organization": "OpenAI",
        "context": 200_000,
        "max_output": 100_000,
        "supports_native_tools": True,
        "supports_streaming": False,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 15.0,
        "pricing_per_1m_output": 60.0,
        "run_count": 18_503,
        "version": "729043ff117dccc6b9d42f6a32e0db3f27c4040fee6e7e3c1abb41e3b49d9f89",
    },
    "openai/o1-mini": {
        "display_name": "OpenAI o1-mini",
        "family": "o1",
        "organization": "OpenAI",
        "context": 128_000,
        "max_output": 65_536,
        "supports_native_tools": True,
        "supports_streaming": False,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 3.0,
        "pricing_per_1m_output": 12.0,
        "run_count": 3_319,
        "version": "afb63ac5dc420133de5f4a0be1a5dcd9deb33e2b57b3b82f50eb3ee7c5b3b0e7",
    },
    "openai/o4-mini": {
        "display_name": "OpenAI o4-mini",
        "family": "o4",
        "organization": "OpenAI",
        "context": 200_000,
        "max_output": 100_000,
        "supports_native_tools": True,
        "supports_streaming": False,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 1.10,
        "pricing_per_1m_output": 4.40,
        "run_count": 448_546,
        "version": "04ce1dc5eea6a7dd6b4ab5df960bfc8e30dcd6a11bdcf8ef2b92b61bf85f5e3e",
    },
    "openai/gpt-5": {
        "display_name": "GPT-5",
        "family": "gpt5",
        "organization": "OpenAI",
        "context": 1_000_000,
        "max_output": 32_768,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 2.0,
        "pricing_per_1m_output": 8.0,
        "run_count": 1_814_525,
        "version": "feacd077889bbeea68d6aeef0b1a13c19d58ec00a80254c4e7eedfe9dfbfe84b",
    },
    "openai/gpt-5-mini": {
        "display_name": "GPT-5 Mini",
        "family": "gpt5",
        "organization": "OpenAI",
        "context": 1_000_000,
        "max_output": 32_768,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 0.40,
        "pricing_per_1m_output": 1.60,
        "run_count": 1_859_387,
        "version": "8fd5dbfbc0f88570e3f4e2cb3f8ee35f4f68be78b5b6caa75e92a2f7ab2c3a30",
    },
    "openai/gpt-5-nano": {
        "display_name": "GPT-5 Nano",
        "family": "gpt5",
        "organization": "OpenAI",
        "context": 1_000_000,
        "max_output": 32_768,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 0.05,
        "pricing_per_1m_output": 0.20,
        "run_count": 10_861_416,
        "version": "034fc01c1d162ba0b4f24e00e8da8b0e7e04b05e8b6b3e19a9765e7e4a8baf54",
    },
    "openai/gpt-5.2": {
        "display_name": "GPT-5.2",
        "family": "gpt5",
        "organization": "OpenAI",
        "context": 1_000_000,
        "max_output": 32_768,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 2.0,
        "pricing_per_1m_output": 8.0,
        "run_count": 533_640,
        "version": "e805d0794e42cc94c2a4f72e90b2a9069ad5f8d573b26e3c0e3fb7649f4c7a02",
    },
    "openai/gpt-oss-120b": {
        "display_name": "GPT-OSS 120B",
        "family": "gpt-oss",
        "organization": "OpenAI",
        "context": 128_000,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.001400,
        "run_count": 182_436,
        "version": "25230470188a9477c41547aeebe1e69e2aeabfec0cd42e4b3be3e8dda1e0bea3",
    },
    "openai/gpt-oss-20b": {
        "display_name": "GPT-OSS 20B",
        "family": "gpt-oss",
        "organization": "OpenAI",
        "context": 128_000,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.000975,
        "run_count": 363_056,
        "version": "5cfc571b53ad7aa2c5d5f58bbdce0e5e5a697cfde15b9e5b7d4baf3a7f7a0c4d",
    },
    # -------------------------------------------------------------------------
    # Google / Gemini models on Replicate
    # -------------------------------------------------------------------------
    "google/gemini-2.5-flash": {
        "display_name": "Gemini 2.5 Flash",
        "family": "gemini",
        "organization": "Google",
        "context": 1_000_000,
        "max_output": 65_536,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 0.15,
        "pricing_per_1m_output": 0.60,
        "run_count": 4_924_565,
        "version": "6585308f2652e91cec7e6e8e06a32f37b1e22e43fd67d0a3024af35a08f4e0bc",
    },
    "google/gemini-3-pro": {
        "display_name": "Gemini 3 Pro",
        "family": "gemini",
        "organization": "Google",
        "context": 1_000_000,
        "max_output": 65_536,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 1.25,
        "pricing_per_1m_output": 5.0,
        "run_count": 1_126_387,
        "version": "6c727b6aa9d5663b5d3c17e5cdcd51c12fd4cac7f07462d4a84c5bc3ea8e1c5f",
    },
    "google/gemini-3.1-pro": {
        "display_name": "Gemini 3.1 Pro",
        "family": "gemini",
        "organization": "Google",
        "context": 1_000_000,
        "max_output": 65_536,
        "supports_native_tools": True,
        "supports_streaming": True,
        "input_schema": "messages",
        "system_prompt_key": "system",
        "prompt_template": None,
        "cost_per_second_usd": 0.0,
        "pricing_per_1m_input": 1.25,
        "pricing_per_1m_output": 5.0,
        "run_count": 463_672,
        "version": "68423ca56f33c1a0cebbb03c7e0a9c2b2d6d8c6d8eb6b5be7e3cf8e1b5a7a2c9",
    },
    # -------------------------------------------------------------------------
    # Google DeepMind  —  Gemma
    # -------------------------------------------------------------------------
    "google-deepmind/gemma-2b-it": {
        "display_name": "Gemma 2B IT",
        "family": "gemma",
        "organization": "Google DeepMind",
        "context": 8_192,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.000225,
        "run_count": 134_714,
        "version": "dff94eaf770e1fc884627c9c4ddb24a6a7cc24a61c2d5e5b6be8a1e3d98d5d0b",
    },
    # -------------------------------------------------------------------------
    # Replicate community models
    # -------------------------------------------------------------------------
    "replicate/flan-t5-xl": {
        "display_name": "Flan-T5 XL",
        "family": "flan-t5",
        "organization": "Google",
        "context": 2_048,
        "max_output": 2_048,
        "supports_native_tools": False,
        "supports_streaming": False,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.000225,
        "run_count": 151_473,
        "version": "eec2f71c986dfa3bd46e4b4adc25f93d7f83af82c06f90d02a6a3ac7f0abcea7",
    },
    "stability-ai/stablelm-tuned-alpha-7b": {
        "display_name": "StableLM Tuned Alpha 7B",
        "family": "stablelm",
        "organization": "Stability AI",
        "context": 4_096,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.000225,
        "run_count": 140_611,
        "version": "943c4afb4d0273cf47cc7ffbd2c4e7fb9e7ee0c5d07e6cb9f11d44dddea1ee40",
    },
    "replit/replit-code-v1-3b": {
        "display_name": "Replit Code v1 3B",
        "family": "replit-code",
        "organization": "Replit",
        "context": 2_048,
        "max_output": 2_048,
        "supports_native_tools": False,
        "supports_streaming": True,
        "input_schema": "prompt_only",
        "system_prompt_key": None,
        "prompt_template": None,
        "cost_per_second_usd": 0.000225,
        "run_count": 1_867,
        "version": None,
    },
}

# ---------------------------------------------------------------------------
# Default model (cheapest + most popular text model with streaming)
# ---------------------------------------------------------------------------
REPLICATE_DEFAULT_MODEL = "meta/meta-llama-3-8b-instruct"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return all registered model IDs."""
    return list(REPLICATE_MODELS.keys())


def get_model_info(model_id: str) -> dict[str, Any] | None:
    """Return registry entry for ``model_id``, or None if not found."""
    return REPLICATE_MODELS.get(model_id)


def models_by_family(family: str) -> list[str]:
    """Return all model IDs that belong to the given family."""
    return [mid for mid, info in REPLICATE_MODELS.items() if info.get("family") == family]


def tool_capable_models() -> list[str]:
    """Return model IDs that support native tool calling."""
    return [mid for mid, info in REPLICATE_MODELS.items() if info.get("supports_native_tools")]


def streaming_models() -> list[str]:
    """Return model IDs that support streaming."""
    return [mid for mid, info in REPLICATE_MODELS.items() if info.get("supports_streaming")]


def refresh_models(api_token: str | None = None) -> dict[str, Any]:
    """Fetch the live Replicate catalog and report drift against the bundled registry.

    This function makes live HTTP requests to the Replicate API.  It does NOT
    modify the in-memory ``REPLICATE_MODELS`` dict — it only reports drift so
    users can update the registry if needed.

    Args:
        api_token: Replicate API token.  Falls back to ``REPLICATE_API_TOKEN``
            environment variable.

    Returns:
        dict with keys:
        - ``"added"``: list of model IDs in live catalog but not in registry.
        - ``"removed"``: list of model IDs in registry but not in live catalog.
        - ``"registry_count"``: number of models in the bundled registry.
        - ``"live_count"``: number of models returned by the live catalog.
        - ``"fetch_date"``: ISO date of the live fetch.
        - ``"registry_fetch_date"``: the date the bundled registry was fetched.
    """
    import datetime

    try:
        import httpx
    except ImportError:
        raise RuntimeError(
            "httpx is required for refresh_models(). "
            "Install with: pip install httpx"
        )

    token = api_token or os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise ValueError(
            "REPLICATE_API_TOKEN not set. Pass api_token= or set the env var."
        )

    headers = {"Authorization": f"Bearer {token}"}

    # Paginate through the language-models collection
    live_ids: set[str] = set()
    url = "https://api.replicate.com/v1/collections/language-models"
    while url:
        resp = httpx.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        models_data = data.get("models", {})
        if isinstance(models_data, dict):
            results = models_data.get("results", [])
            next_url: str | None = models_data.get("next")
        else:
            results = models_data if isinstance(models_data, list) else []
            next_url = None

        for m in results:
            owner = m.get("owner", "")
            name = m.get("name", "")
            if owner and name:
                live_ids.add(f"{owner}/{name}")

        url = next_url  # type: ignore[assignment]

    bundled_ids = set(REPLICATE_MODELS.keys())
    added = sorted(live_ids - bundled_ids)
    removed = sorted(bundled_ids - live_ids)

    today = datetime.date.today().isoformat()
    report = {
        "added": added,
        "removed": removed,
        "registry_count": len(bundled_ids),
        "live_count": len(live_ids),
        "fetch_date": today,
        "registry_fetch_date": REGISTRY_FETCH_DATE,
    }

    if added:
        logger.warning(
            "Replicate registry drift: %d new model(s) not in the bundled registry "
            "(registry_date=%s). Call refresh_models() for details or update "
            "effgen/models/replicate_models.py: %s",
            len(added),
            REGISTRY_FETCH_DATE,
            added,
        )
    if removed:
        logger.warning(
            "Replicate registry drift: %d model(s) in the bundled registry may have "
            "been removed from the live catalog: %s",
            len(removed),
            removed,
        )

    return report


def print_registry_summary() -> None:
    """Print a human-readable summary of the bundled registry."""
    total = len(REPLICATE_MODELS)
    streaming = len(streaming_models())
    tool_capable = len(tool_capable_models())
    families = sorted({info.get("family", "?") for info in REPLICATE_MODELS.values()})

    print(f"Replicate model registry (fetched {REGISTRY_FETCH_DATE})")
    print(f"  Total models   : {total}")
    print(f"  Streaming      : {streaming}")
    print(f"  Tool-capable   : {tool_capable}")
    print(f"  Families       : {', '.join(families)}")
    print()
    print(f"{'Model ID':<55} {'Family':<12} {'Context':>9} {'Native Tools'}")
    print("-" * 100)
    for mid, info in sorted(REPLICATE_MODELS.items()):
        ctx = f"{info.get('context', 0) // 1000}K"
        nt = "yes" if info.get("supports_native_tools") else "no"
        print(f"  {mid:<53} {info.get('family', '?'):<12} {ctx:>9}   {nt}")
