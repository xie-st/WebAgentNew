"""
Tool registry for the effGen framework.

This module provides a central registry for tool registration, discovery,
lazy loading, and dependency management.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import logging
import os
import threading
import weakref
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..utils.async_bridge import run_coroutine_sync
from .base_tool import BaseTool, ToolCategory, ToolMetadata

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_discovery():
    """Silence library log noise while merely *listing* tools.

    Discovery imports every built-in module and instantiates each tool class
    once to read its metadata. A few tools log at construction time (e.g. a
    missing optional key or an unavailable backend). Listing the catalog should
    never behave like activating a tool, so we pin the package logger to ERROR
    for the duration of discovery. Genuine errors still surface; the per-tool
    informational chatter does not. The previous level is always restored.
    """
    pkg_logger = logging.getLogger("effgen")
    previous = pkg_logger.level
    pkg_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        pkg_logger.setLevel(previous)


class ToolDependencyError(Exception):
    """Raised when tool dependencies cannot be satisfied."""
    pass


class ToolRegistrationError(Exception):
    """Raised when tool registration fails."""
    pass


class ToolRegistry:
    """
    Central registry for managing tools in the effGen framework.

    Features:
    - Tool registration with metadata validation
    - Lazy loading of tools
    - Dependency management and resolution
    - Tool discovery and filtering
    - Version management
    - Plugin system for external tools

    Example:
        registry = ToolRegistry()
        registry.register_tool(MyTool)          # a BaseTool subclass
        registry.register_tool(my_function_tool)  # or a BaseTool instance (e.g. from @tool)
        tool = await registry.get_tool("my_tool")
        result = await tool.execute(param="value")
    """

    def __init__(self) -> None:
        """Initialize the tool registry."""
        # A registered entry is either a class (constructed fresh on first
        # lookup) or a zero-arg factory closing over a pre-built instance
        # (registered directly, e.g. a ``@tool``-decorated function).
        self._tools: dict[str, type[BaseTool] | Callable[[], BaseTool]] = {}
        self._instances: dict[str, BaseTool] = {}
        self._metadata_cache: dict[str, ToolMetadata] = {}
        self._dependencies: dict[str, set[str]] = defaultdict(set)
        self._categories: dict[ToolCategory, set[str]] = defaultdict(set)
        self._plugins: dict[str, Path] = {}
        self._initialized_tools: set[str] = set()
        # Serializes the async loader. One lock per event loop: agents commonly
        # run on a thread each, ``get_tool_sync`` drives the loader on the
        # calling thread's own loop, and an ``asyncio.Lock`` awaited from two
        # loops raises rather than waiting. Keyed on the loop itself and held
        # weakly, so an entry lasts exactly as long as the loop it belongs to —
        # ``get_tool_sync`` runs each call on a loop of its own, and a map keyed
        # on the loop's id would both grow without bound and hand a new loop the
        # lock of a dead one that happened to be allocated at the same address.
        self._loop_locks: "weakref.WeakKeyDictionary[Any, asyncio.Lock]" = (
            weakref.WeakKeyDictionary()
        )
        # The loader lock for callers with no running loop at all, which have no
        # loop object to key on.
        self._loopless_lock = asyncio.Lock()
        # Tools whose ``initialize()`` is running right now, and the event each
        # waiter blocks on. Initialization must happen once however many threads
        # ask at once: a tool that opens a connection or starts a worker on
        # ``initialize()`` would otherwise start one per caller.
        self._initializing: dict[str, threading.Event] = {}
        # Guards every shared mapping below against concurrent mutation from
        # threads. Re-entrant because discovery registers tools, which takes it
        # again on the same thread.
        self._mutex = threading.RLock()
        # Whether the built-in catalog has been auto-discovered yet. Discovery is
        # deferred until first access so importing the package stays cheap, but a
        # programmatic caller never sees a surprisingly empty registry.
        self._builtins_discovered = False
        # Whether installed entry-point plugins have been auto-discovered yet.
        self._plugins_discovered = False

    @property
    def _lock(self) -> asyncio.Lock:
        """The loader lock for the event loop currently running.

        Kept per loop so a registry shared by agents on several threads is
        serialized within each loop without an ``asyncio.Lock`` ever being
        awaited from a loop other than the one that first waited on it. Threads
        contending across loops are serialized by :attr:`_mutex` instead.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._loopless_lock
        with self._mutex:
            lock = self._loop_locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                self._loop_locks[loop] = lock
            return lock

    def _ensure_builtins(self) -> None:
        """Lazily discover built-in tools (and installed plugins) on first access.

        Programmatic users expect ``get_registry().list_tools()`` to be
        populated without first calling :meth:`discover_builtin_tools`. We run
        discovery once, the first time the registry is queried, and remember it
        so repeated lookups stay cheap and quiet (no re-import, no log spam).
        Installed ``effgen.plugins`` entry-point packages are folded in at the
        same time so a published plugin's tools show up everywhere built-in
        tools do (``effgen tools list``, ``effgen run -t <tool>``, the API).

        Held under the registry mutex so a thread arriving while another is
        mid-discovery waits for the catalog rather than being handed the empty
        registry discovery has not filled yet.
        """
        with self._mutex:
            if self._builtins_discovered:
                self._ensure_plugins()
                return
            # Mark first to avoid re-entry if discovery itself queries the
            # registry; the mutex is re-entrant, so that path does not deadlock.
            self._builtins_discovered = True
            self.discover_builtin_tools()
            self._ensure_plugins()

    def _ensure_plugins(self) -> None:
        """Lazily load installed entry-point (and user-dir) plugins once.

        A plugin published with an ``effgen.plugins`` entry point — exactly what
        ``effgen create-plugin`` scaffolds — is auto-discovered here so its tools
        are usable without any manual ``register()`` call. Set
        ``EFFGEN_DISABLE_PLUGINS=1`` to skip this (e.g. for a hermetic run).
        Discovery is best-effort: a broken plugin is logged and skipped, never
        fatal.
        """
        with self._mutex:
            if self._plugins_discovered:
                return
            self._plugins_discovered = True
            if os.environ.get("EFFGEN_DISABLE_PLUGINS"):
                return
            self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Load every installed plugin package; a broken one is skipped."""
        try:
            # Local import: plugin.py imports from this module, so importing it
            # at module load time would create a circular import.
            from .plugin import PluginManager
            loaded = PluginManager(self).discover_all()
            if loaded:
                logger.debug("Auto-discovered plugins: %s", ", ".join(loaded))
        except Exception:  # noqa: BLE001 — discovery must never break the registry
            logger.debug("Plugin auto-discovery failed", exc_info=True)

    def register_tool(
        self,
        tool_class: type[BaseTool] | BaseTool,
        override: bool = False
    ) -> None:
        """
        Register a tool with the registry.

        Accepts either a ``BaseTool`` subclass (the registry constructs and owns
        each instance) or a ready-made ``BaseTool`` instance — including the
        ``FunctionTool`` produced by ``@tool`` / ``Tool.from_function``, which
        returns instances rather than classes. A registered instance is reused
        for every lookup instead of being reconstructed.

        Args:
            tool_class: A ``BaseTool`` subclass, or a ``BaseTool`` instance.
            override: Whether to override existing tool with same name

        Raises:
            ToolRegistrationError: If registration fails
        """
        if isinstance(tool_class, BaseTool):
            instance = tool_class
            metadata = instance.metadata
            name = metadata.name
            constructor: Callable[[], BaseTool] = lambda: instance  # noqa: E731
            dependencies = instance.dependencies
        elif inspect.isclass(tool_class) and issubclass(tool_class, BaseTool):
            # Create temporary instance to get metadata
            try:
                temp_instance = tool_class()
                metadata = temp_instance.metadata
                name = metadata.name
            except Exception as e:
                raise ToolRegistrationError(
                    f"Failed to instantiate tool {tool_class.__name__}: {e}. "
                    "A tool class must be constructible with no arguments; "
                    "give its __init__ defaults, or register an instance you "
                    "built yourself."
                )
            constructor = tool_class
            dependencies = temp_instance.dependencies
        elif inspect.isclass(tool_class):
            raise ToolRegistrationError(
                f"Tool class {tool_class.__name__} must inherit from BaseTool. "
                "Subclass effgen.tools.BaseTool, or wrap the callable with "
                "effgen.tool() or Tool.from_function()."
            )
        else:
            raise ToolRegistrationError(
                f"Expected a BaseTool subclass or instance, got {type(tool_class)!r}. "
                "A plain function must be wrapped first, e.g. "
                "register_tool(effgen.tool(fn)) or register_tool(Tool.from_function(fn))."
            )

        with self._mutex:
            # Check for name collision
            if name in self._tools and not override:
                # Silently skip re-registration instead of throwing error/warning
                logger.debug(f"Tool '{name}' already registered, skipping")
                return

            # Register the tool
            self._tools[name] = constructor
            self._metadata_cache[name] = metadata
            self._categories[metadata.category].add(name)

            # Store dependencies
            if dependencies:
                self._dependencies[name] = set(dependencies)

        logger.debug(f"Registered tool: {name} (v{metadata.version})")

    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool from the registry.

        Args:
            name: Name of the tool to unregister

        Raises:
            KeyError: If tool is not registered
        """
        with self._mutex:
            if name not in self._tools:
                raise KeyError(f"Tool '{name}' is not registered")
            instance = self._instances.pop(name, None)

        # Clean up instance if exists. Run cleanup to completion regardless of
        # whether an event loop is already running — never leave the coroutine
        # un-awaited (the old asyncio.create_task path raised "no running event
        # loop" when called from synchronous code). Awaited outside the registry
        # mutex so a slow teardown does not hold every other thread's lookups.
        if instance is not None:
            try:
                run_coroutine_sync(instance.cleanup())
            except Exception:
                logger.debug("Cleanup for tool '%s' failed during unregister", name, exc_info=True)

        with self._mutex:
            # Remove from all tracking structures
            metadata = self._metadata_cache.get(name)
            if metadata is not None:
                self._categories[metadata.category].discard(name)
            self._tools.pop(name, None)
            self._metadata_cache.pop(name, None)
            self._dependencies.pop(name, None)
            self._initialized_tools.discard(name)

        logger.debug(f"Unregistered tool: {name}")

    async def get_tool(self, name: str, initialize: bool = True) -> BaseTool:
        """
        Get a tool instance by name.

        Implements lazy loading - tools are only instantiated when first requested.

        Args:
            name: Name of the tool to get
            initialize: Whether to initialize the tool if not already initialized

        Returns:
            BaseTool: The tool instance

        Raises:
            KeyError: If tool is not registered
            ToolDependencyError: If tool dependencies cannot be satisfied
        """
        self._ensure_builtins()
        with self._mutex:
            if name not in self._tools:
                raise KeyError(f"Tool '{name}' is not registered")
            existing = self._instances.get(name)
            ready = name in self._initialized_tools
            has_dependencies = name in self._dependencies

        if existing is not None and (ready or not initialize):
            return existing

        # Dependencies are loaded through this same method, so they are resolved
        # before the loader lock is taken rather than under it — the lock is not
        # re-entrant, and a tool declaring a dependency would otherwise wait on
        # itself forever.
        if has_dependencies and existing is None:
            await self._resolve_dependencies(name)

        async with self._lock:
            with self._mutex:
                tool = self._instances.get(name)
                if tool is None:
                    tool = self._tools[name]()
                    self._instances[name] = tool
                    logger.debug(f"Loaded tool: {name}")
                ready = name in self._initialized_tools

            if initialize and not ready:
                await self._initialize_once(name, tool)
            return tool

    async def _initialize_once(self, name: str, tool: BaseTool) -> None:
        """Run ``tool.initialize()`` exactly once, however many callers arrive.

        The loader lock is per event loop, so two threads driving their own
        loops both reach this point. Whichever gets there first claims the tool
        and initializes it; the others wait for that to finish and return. A
        tool that opens a connection, starts a worker process or registers a
        handler on ``initialize()`` therefore does so once per registry rather
        than once per caller.
        """
        while True:
            with self._mutex:
                if name in self._initialized_tools:
                    return
                pending = self._initializing.get(name)
                if pending is None:
                    pending = threading.Event()
                    self._initializing[name] = pending
                    mine = True
                else:
                    mine = False
            if mine:
                try:
                    await tool.initialize()
                    with self._mutex:
                        self._initialized_tools.add(name)
                finally:
                    with self._mutex:
                        self._initializing.pop(name, None)
                    pending.set()
                return
            # Someone else is initializing this tool. Wait for them without
            # blocking this event loop, then re-check: their attempt may have
            # failed, in which case this caller takes its own turn.
            while not pending.is_set():
                await asyncio.sleep(0.001)

    def get_tool_sync(self, name: str, initialize: bool = True) -> BaseTool:
        """Synchronous accessor for a tool instance.

        ``get_tool`` is a coroutine; calling it without ``await`` silently
        yields a coroutine object (the source of ``'coroutine' object has no
        attribute 'name'`` errors). This convenience wrapper drives the async
        loader to completion whether or not an event loop is already running,
        so synchronous callers (CLI, scripts, notebooks) get the real instance.

        Args:
            name: Name of the tool to get.
            initialize: Whether to initialize the tool if not already initialized.

        Returns:
            BaseTool: The tool instance.
        """
        return run_coroutine_sync(self.get_tool(name, initialize=initialize))

    async def _resolve_dependencies(self, tool_name: str) -> None:
        """
        Resolve and load tool dependencies.

        Args:
            tool_name: Name of the tool whose dependencies to resolve

        Raises:
            ToolDependencyError: If dependencies cannot be satisfied
        """
        # Read the graph once, under the mutex, so the checks below run against
        # one consistent view rather than one another thread is editing.
        with self._mutex:
            dependencies = set(self._dependencies.get(tool_name, set()))
            if not dependencies:
                return
            registered = set(self._tools)
            graph = {name: set(deps) for name, deps in self._dependencies.items()}
            loaded = set(self._instances)

        # Check all dependencies are registered
        missing = dependencies - registered
        if missing:
            raise ToolDependencyError(
                f"Tool '{tool_name}' has missing dependencies: {missing}. "
                "Register those tools first, or drop them from this tool's "
                "dependencies."
            )

        # Detect circular dependencies
        visited = set()
        path = []

        def check_circular(name: str) -> None:
            if name in path:
                cycle = " -> ".join(path + [name])
                raise ToolDependencyError(
                    f"Circular dependency detected: {cycle}. Break the cycle — "
                    "one of these tools must not depend on the others."
                )
            if name in visited:
                return

            path.append(name)
            visited.add(name)

            for dep in graph.get(name, ()):
                check_circular(dep)

            path.pop()

        check_circular(tool_name)

        # Load dependencies
        for dep_name in dependencies:
            if dep_name not in loaded:
                await self.get_tool(dep_name, initialize=True)

    def list_tools(
        self,
        category: ToolCategory | None = None,
        tags: list[str] | None = None,
        name_filter: str | None = None
    ) -> list[str]:
        """
        List available tools with optional filtering.

        Args:
            category: Filter by tool category
            tags: Filter by tags (returns tools with ANY of these tags)
            name_filter: Filter by name substring (case-insensitive)

        Returns:
            List[str]: List of tool names matching the filters
        """
        self._ensure_builtins()
        with self._mutex:
            tools = set(self._tools)
            by_category = set(self._categories.get(category, set())) if category else None
            metadata_by_name = dict(self._metadata_cache) if tags else {}

        # Filter by category
        if by_category is not None:
            tools &= by_category

        # Filter by tags
        if tags:
            matching = set()
            for name in tools:
                metadata = metadata_by_name.get(name)
                if metadata is not None and any(tag in metadata.tags for tag in tags):
                    matching.add(name)
            tools = matching

        # Filter by name
        if name_filter:
            filter_lower = name_filter.lower()
            tools = {name for name in tools if filter_lower in name.lower()}

        return sorted(tools)

    def get_metadata(self, name: str) -> ToolMetadata:
        """
        Get metadata for a registered tool.

        Args:
            name: Name of the tool

        Returns:
            ToolMetadata: Tool metadata

        Raises:
            KeyError: If tool is not registered
        """
        self._ensure_builtins()
        with self._mutex:
            metadata = self._metadata_cache.get(name)
        if metadata is None:
            raise KeyError(f"Tool '{name}' is not registered")
        return metadata

    def get_all_metadata(self) -> dict[str, ToolMetadata]:
        """
        Get metadata for all registered tools.

        Returns:
            Dict[str, ToolMetadata]: Mapping of tool names to metadata
        """
        self._ensure_builtins()
        with self._mutex:
            return self._metadata_cache.copy()

    def get_tools_by_category(self, category: ToolCategory) -> list[str]:
        """
        Get all tools in a specific category.

        Args:
            category: The tool category

        Returns:
            List[str]: List of tool names in the category
        """
        self._ensure_builtins()
        with self._mutex:
            return sorted(self._categories.get(category, set()))

    def is_registered(self, name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            name: Name of the tool

        Returns:
            bool: True if registered, False otherwise
        """
        self._ensure_builtins()
        with self._mutex:
            return name in self._tools

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """
        Get the dependency graph for all tools.

        Returns:
            Dict[str, List[str]]: Mapping of tool names to their dependencies
        """
        with self._mutex:
            return {name: sorted(deps) for name, deps in self._dependencies.items()}

    async def initialize_all(self) -> None:
        """Initialize all registered tools."""
        with self._mutex:
            names = list(self._tools)
            pending = [n for n in names if n not in self._initialized_tools]
        for name in pending:
            await self.get_tool(name, initialize=True)

    async def cleanup_all(self) -> None:
        """Clean up all initialized tools."""
        with self._mutex:
            cleanup_tasks = [
                tool.cleanup()
                for name, tool in self._instances.items()
                if name in self._initialized_tools
            ]
            self._instances.clear()
            self._initialized_tools.clear()

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    def register_plugin(self, plugin_path: Path) -> None:
        """
        Register a plugin directory for external tools.

        Args:
            plugin_path: Path to the plugin directory

        Raises:
            ToolRegistrationError: If plugin loading fails
        """
        if not plugin_path.is_dir():
            raise ToolRegistrationError(
                f"Plugin path {plugin_path} is not a directory. Pass the "
                "package directory that holds the plugin's __init__.py."
            )

        # Try to import the plugin module
        try:
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_path.name}",
                plugin_path / "__init__.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all BaseTool subclasses in the module
                for _name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseTool) and obj is not BaseTool:
                        self.register_tool(obj)

                self._plugins[plugin_path.name] = plugin_path
                logger.debug(f"Registered plugin: {plugin_path.name}")

        except Exception as e:
            raise ToolRegistrationError(
                f"Failed to load plugin {plugin_path}: {e}. Import the "
                "plugin's __init__.py directly to see the underlying error."
            )

    def discover_builtin_tools(self) -> None:
        """
        Automatically discover and register built-in tools.

        Scans the effgen.tools.builtin package for tool classes.
        """
        # Calling this directly also satisfies the lazy-discovery guard so a
        # later read access does not re-scan the package.
        self._builtins_discovered = True
        try:
            from . import builtin
            builtin_path = Path(builtin.__file__).parent

            # Import all Python files in builtin directory. Listing the catalog
            # must stay quiet — a tool that logs while constructing its metadata
            # probe should not spam the user.
            with _quiet_discovery():
                for file_path in builtin_path.glob("*.py"):
                    if file_path.name.startswith("_"):
                        continue

                    module_name = f"effgen.tools.builtin.{file_path.stem}"
                    try:
                        module = importlib.import_module(module_name)

                        # Find all BaseTool subclasses
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, BaseTool) and
                                obj is not BaseTool and
                                obj.__module__ == module_name):
                                try:
                                    self.register_tool(obj)
                                except ToolRegistrationError as e:
                                    # Already handled — just debug log
                                    logger.debug(f"Skipping {name}: {e}")

                    except ImportError as e:
                        logger.warning(f"Failed to import {module_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover builtin tools: {e}")

    def export_schemas(self, format: str = "json") -> dict[str, Any]:
        """
        Export all tool schemas in the specified format.

        Args:
            format: Output format ("json" or "openai")

        Returns:
            Dict[str, Any]: Tool schemas
        """
        self._ensure_builtins()
        schemas = {}

        with self._mutex:
            cached = list(self._metadata_cache.items())
        for name, metadata in cached:
            if format == "openai":
                schemas[name] = metadata.to_json_schema()
            else:
                schemas[name] = metadata.to_dict()

        return schemas

    def __len__(self) -> int:
        """Return number of registered tools."""
        self._ensure_builtins()
        with self._mutex:
            return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        self._ensure_builtins()
        with self._mutex:
            return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry(tools={len(self._tools)}, initialized={len(self._initialized_tools)})>"


# Global tool registry instance
_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    Returns:
        ToolRegistry: The global registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global tool registry (mainly for testing).

    Cleans up all initialized tool instances *synchronously*, working whether or
    not an event loop is already running in the calling thread. Prefer
    :func:`async_reset_registry` from async code to avoid blocking the loop on a
    worker thread.
    """
    global _global_registry
    if _global_registry is not None:
        try:
            run_coroutine_sync(_global_registry.cleanup_all())
        except Exception:
            logger.debug("Registry cleanup_all failed during reset", exc_info=True)
    _global_registry = None


async def async_reset_registry() -> None:
    """Async variant of :func:`reset_registry`.

    Awaits tool cleanup directly on the current event loop, then clears the
    global registry. Use this from coroutines/async test fixtures.
    """
    global _global_registry
    if _global_registry is not None:
        await _global_registry.cleanup_all()
    _global_registry = None
