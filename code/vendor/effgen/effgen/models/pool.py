"""
Model Pool for effGen framework.

Manages multiple loaded models in memory with:
- LRU eviction when GPU memory is tight
- Pre-warming: load models before they are needed
- Hot-swap: switch models without restarting agents
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from effgen.models.base import BaseModel
from effgen.models.capabilities import estimate_capability
from effgen.models.model_loader import ModelLoader

logger = logging.getLogger(__name__)


def _load_torch():
    """Import torch lazily so importing the pool (and `effgen.models`) never pulls
    torch. Returns the module, or ``None`` if torch is unavailable."""
    try:
        import torch
        return torch
    except ImportError:
        return None


@dataclass
class PoolEntry:
    """Tracks a model managed by the pool."""
    model: BaseModel
    model_name: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    gpu_memory_gb: float = 0.0

    def touch(self) -> None:
        """Mark as recently used."""
        self.last_used = time.time()
        self.use_count += 1


@dataclass
class PoolConfig:
    """Configuration for the ModelPool.

    Attributes:
        max_loaded_models: Hard cap on concurrently loaded models.
        gpu_memory_limit_gb: Max total GPU memory the pool may consume.
            None means no explicit limit (rely on max_loaded_models).
        eviction_headroom_gb: Extra GB to free when evicting.
        auto_evict: Whether to automatically evict LRU models when limits
            are exceeded.
    """
    max_loaded_models: int = 4
    gpu_memory_limit_gb: float | None = None
    eviction_headroom_gb: float = 2.0
    auto_evict: bool = True


class ModelPool:
    """Manages a pool of loaded models with LRU eviction.

    Thread-safe: all mutations go through ``_lock``.

    Example::

        pool = ModelPool()
        model = pool.get_or_load("Qwen/Qwen2.5-1.5B-Instruct")
        result = model.generate("Hello")
        pool.release("Qwen/Qwen2.5-1.5B-Instruct")
    """

    def __init__(
        self,
        config: PoolConfig | None = None,
        loader: ModelLoader | None = None,
    ) -> None:
        self.config = config or PoolConfig()
        self._loader = loader or ModelLoader()
        self._lock = threading.Lock()
        # OrderedDict preserves insertion order; we move-to-end on access.
        self._entries: OrderedDict[str, PoolEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_load(
        self,
        model_name: str,
        engine: str | None = None,
        engine_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Return a loaded model, loading it first if necessary.

        If the pool is at capacity, the least-recently-used model is
        evicted before loading the new one.

        Args:
            model_name: Model identifier.
            engine: Optional engine override ("vllm", "transformers").
            engine_config: Optional engine-specific config.
            **kwargs: Extra args forwarded to ModelLoader.

        Returns:
            The loaded BaseModel instance.
        """
        with self._lock:
            if model_name in self._entries:
                entry = self._entries[model_name]
                entry.touch()
                # Move to end (most recently used)
                self._entries.move_to_end(model_name)
                logger.debug("Pool hit: '%s'", model_name)
                return entry.model

        # Load outside the lock (loading is slow)
        self._maybe_evict_for(model_name)

        loader = ModelLoader(force_engine=engine) if engine else self._loader
        model = loader.load_model(model_name, engine_config, **kwargs)

        cap = estimate_capability(model_name)

        with self._lock:
            # Double-check (another thread may have loaded it)
            if model_name in self._entries:
                # Unload the one we just loaded; use the existing entry
                model.unload()
                entry = self._entries[model_name]
                entry.touch()
                self._entries.move_to_end(model_name)
                return entry.model

            entry = PoolEntry(
                model=model,
                model_name=model_name,
                gpu_memory_gb=cap.gpu_memory_gb,
            )
            self._entries[model_name] = entry

        logger.info("Pool loaded '%s' (pool size: %d)", model_name, len(self._entries))
        return model

    def add(self, model: BaseModel) -> None:
        """Add an already-loaded model to the pool.

        Args:
            model: A loaded BaseModel instance.
        """
        name = getattr(model, "model_name", str(model))
        cap = estimate_capability(name)
        with self._lock:
            if name in self._entries:
                logger.debug("Pool already has '%s', updating entry", name)
                self._entries[name].model = model
                self._entries[name].touch()
                self._entries.move_to_end(name)
                return
            entry = PoolEntry(
                model=model,
                model_name=name,
                gpu_memory_gb=cap.gpu_memory_gb,
            )
            self._entries[name] = entry
        logger.info("Pool added '%s' (pool size: %d)", name, len(self._entries))

    def get(self, model_name: str) -> BaseModel | None:
        """Get a model from the pool without loading.

        Returns None if the model is not in the pool.
        """
        with self._lock:
            entry = self._entries.get(model_name)
            if entry is not None:
                entry.touch()
                self._entries.move_to_end(model_name)
                return entry.model
        return None

    def release(self, model_name: str) -> bool:
        """Unload and remove a model from the pool.

        Returns True if the model was found and removed.
        """
        with self._lock:
            entry = self._entries.pop(model_name, None)
        if entry is None:
            logger.warning("Pool release: '%s' not found", model_name)
            return False
        try:
            entry.model.unload()
        except Exception as e:
            logger.error("Error unloading '%s': %s", model_name, e)
        logger.info("Pool released '%s' (pool size: %d)", model_name, len(self._entries))
        return True

    def release_all(self) -> None:
        """Unload and remove all models from the pool."""
        with self._lock:
            entries = list(self._entries.items())
            self._entries.clear()
        for name, entry in entries:
            try:
                entry.model.unload()
            except Exception as e:
                logger.error("Error unloading '%s': %s", name, e)
        logger.info("Pool released all models")

    def prewarm(
        self,
        model_names: list[str],
        engine: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Pre-load models into the pool in advance.

        Args:
            model_names: List of model identifiers to pre-load.
            engine: Optional engine override.
            **kwargs: Extra args forwarded to loader.

        Returns:
            List of model names that were successfully loaded.
        """
        loaded = []
        for name in model_names:
            try:
                self.get_or_load(name, engine=engine, **kwargs)
                loaded.append(name)
            except Exception as e:
                logger.error("Prewarm failed for '%s': %s", name, e)
        return loaded

    def hot_swap(
        self,
        old_model_name: str,
        new_model_name: str,
        engine: str | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Replace one model with another without downtime.

        Loads the new model first, then unloads the old one.

        Args:
            old_model_name: Model to replace.
            new_model_name: New model to load.
            engine: Optional engine override.
            **kwargs: Extra args forwarded to loader.

        Returns:
            The newly loaded BaseModel.
        """
        # Load new model first
        new_model = self.get_or_load(new_model_name, engine=engine, **kwargs)
        # Then release old
        self.release(old_model_name)
        logger.info("Hot-swapped '%s' -> '%s'", old_model_name, new_model_name)
        return new_model

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> list[dict[str, Any]]:
        """Return status info for all pooled models."""
        with self._lock:
            result = []
            for name, entry in self._entries.items():
                result.append({
                    "model_name": name,
                    "loaded": entry.model.is_loaded() if hasattr(entry.model, "is_loaded") else True,
                    "use_count": entry.use_count,
                    "last_used": entry.last_used,
                    "loaded_at": entry.loaded_at,
                    "gpu_memory_gb": entry.gpu_memory_gb,
                })
            return result

    def loaded_model_names(self) -> list[str]:
        """Return names of all loaded models in LRU order (oldest first)."""
        with self._lock:
            return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, model_name: str) -> bool:
        return model_name in self._entries

    def __repr__(self) -> str:
        return f"ModelPool(loaded={len(self._entries)}, max={self.config.max_loaded_models})"

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _maybe_evict_for(self, incoming_name: str) -> None:
        """Evict LRU models if the pool is at capacity or GPU memory is tight."""
        if not self.config.auto_evict:
            return

        with self._lock:
            # 1. Count-based eviction
            while len(self._entries) >= self.config.max_loaded_models:
                self._evict_lru_locked()

            # 2. GPU memory-based eviction
            if self.config.gpu_memory_limit_gb is not None:
                total_used = sum(e.gpu_memory_gb for e in self._entries.values())
                incoming_cap = estimate_capability(incoming_name)
                needed = total_used + incoming_cap.gpu_memory_gb

                while (
                    needed > self.config.gpu_memory_limit_gb
                    and self._entries
                ):
                    freed = self._evict_lru_locked()
                    needed -= freed

    def _evict_lru_locked(self) -> float:
        """Evict the least-recently-used entry. Must hold ``_lock``.

        Returns the estimated GPU memory freed in GB.
        """
        if not self._entries:
            return 0.0
        # OrderedDict: first item is LRU
        name, entry = next(iter(self._entries.items()))
        del self._entries[name]
        freed = entry.gpu_memory_gb
        logger.info("Pool evicted LRU model '%s' (freed ~%.1f GB)", name, freed)
        # Unload outside lock would be cleaner but we hold lock here; unload
        # is typically fast (del tensors).
        try:
            entry.model.unload()
        except Exception as e:
            logger.error("Error unloading evicted model '%s': %s", name, e)
        return freed

    def _get_gpu_memory_free_gb(self) -> float:
        """Return total free GPU memory in GB across all visible devices."""
        torch = _load_torch()
        if torch is None or not torch.cuda.is_available():
            return float("inf")
        free_total = 0.0
        for i in range(torch.cuda.device_count()):
            free, _total = torch.cuda.mem_get_info(i)
            free_total += free / (1024 ** 3)
        return free_total
