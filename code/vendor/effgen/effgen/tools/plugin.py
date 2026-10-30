"""
effGen Plugin System — Load external tool plugins via entry points and directories.

Plugin discovery sources (in order of precedence):
1. ``effgen.plugins`` entry-point group (installed packages)
2. ``~/.effgen/plugins/`` directory (user-local plugins)
3. ``EFFGEN_PLUGINS_DIR`` environment variable (custom directory)

Writing a plugin
----------------
1. Create a class that extends ``ToolPlugin``.
2. Set ``name``, ``version``, and ``tools`` (list of BaseTool subclasses).
3. Publish as a package with an entry point::

       [project.entry-points."effgen.plugins"]
       my_tools = "my_package.plugin:MyPlugin"
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path

from effgen.tools.base_tool import BaseTool
from effgen.tools.registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)


class ToolPlugin:
    """Base class for external tool plugins.

    Subclass this and populate ``name``, ``version``, and ``tools`` with
    your custom ``BaseTool`` subclasses — or ready-made ``BaseTool``
    instances, such as a function decorated with ``@tool``.

    Example::

        class MyPlugin(ToolPlugin):
            name = "my_tools"
            version = "1.0.0"
            tools = [MyCustomTool, AnotherTool, my_decorated_tool]
    """

    _DEFAULT_VERSION = "0.0.0"

    name: str = ""
    version: str = _DEFAULT_VERSION
    description: str = ""
    tools: list[type[BaseTool] | BaseTool] = []

    def __init__(
        self,
        name: str | None = None,
        version: str | None = None,
        description: str | None = None,
        tools: list[type[BaseTool] | BaseTool] | None = None,
    ) -> None:
        # Constructor args take priority, then class-level attrs, then defaults
        if name is not None:
            self.name = name
        elif self.name == "" and type(self).name != "":
            self.name = type(self).name
        if version is not None:
            self.version = version
        elif self.version == self._DEFAULT_VERSION and type(self).version != self._DEFAULT_VERSION:
            self.version = type(self).version
        if description is not None:
            self.description = description
        elif self.description == "" and type(self).description != "":
            self.description = type(self).description
        if tools is not None:
            self.tools = tools
        elif not self.tools and type(self).tools:
            self.tools = list(type(self).tools)

    def register(self, registry: ToolRegistry) -> list[str]:
        """Register all tools from this plugin into *registry*.

        Returns:
            List of registered tool names.
        """
        registered: list[str] = []
        for tool_obj in self.tools:
            try:
                registry.register_tool(tool_obj)
                if isinstance(tool_obj, BaseTool):
                    tool_name = tool_obj.metadata.name
                else:
                    tool_name = tool_obj().metadata.name if hasattr(tool_obj, 'metadata') else tool_obj.__name__
                registered.append(tool_name)
                logger.info(
                    "Plugin '%s' registered tool '%s'", self.name, tool_name
                )
            except Exception as exc:
                logger.warning(
                    "Plugin '%s' failed to register tool %s: %s",
                    self.name, getattr(tool_obj, "__name__", type(tool_obj).__name__), exc,
                )
        return registered


class PluginManager:
    """Discovers and loads tool plugins from multiple sources."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_registry()
        self._loaded: dict[str, ToolPlugin] = {}

    @property
    def loaded_plugins(self) -> dict[str, ToolPlugin]:
        """Return mapping of loaded plugin name → ToolPlugin."""
        return dict(self._loaded)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_all(self) -> list[str]:
        """Run all discovery methods and return names of loaded plugins."""
        loaded: list[str] = []
        loaded.extend(self.discover_entry_points())
        loaded.extend(self.discover_user_dir())
        loaded.extend(self.discover_env_dir())
        return loaded

    def discover_entry_points(self) -> list[str]:
        """Load plugins from the ``effgen.plugins`` entry-point group."""
        loaded: list[str] = []
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="effgen.plugins")
        except Exception:
            logger.debug("No entry points found for effgen.plugins")
            return loaded

        for ep in eps:
            try:
                plugin_cls = ep.load()
                plugin = plugin_cls() if isinstance(plugin_cls, type) else plugin_cls
                if not isinstance(plugin, ToolPlugin):
                    logger.warning("Entry point '%s' is not a ToolPlugin — skipping.", ep.name)
                    continue
                plugin.register(self.registry)
                self._loaded[plugin.name] = plugin
                loaded.append(plugin.name)
                logger.info("Loaded plugin '%s' v%s from entry point.", plugin.name, plugin.version)
            except Exception as exc:
                logger.warning("Failed to load entry-point plugin '%s': %s", ep.name, exc)
        return loaded

    def discover_user_dir(self) -> list[str]:
        """Load plugins from ``~/.effgen/plugins/``."""
        user_dir = Path.home() / ".effgen" / "plugins"
        return self._discover_directory(user_dir)

    def discover_env_dir(self) -> list[str]:
        """Load plugins from ``EFFGEN_PLUGINS_DIR`` environment variable."""
        env_path = os.environ.get("EFFGEN_PLUGINS_DIR")
        if not env_path:
            return []
        return self._discover_directory(Path(env_path))

    def _discover_directory(self, directory: Path) -> list[str]:
        """Scan *directory* for Python files containing ToolPlugin subclasses."""
        loaded: list[str] = []
        if not directory.is_dir():
            return loaded

        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                module = self._load_module_from_file(py_file)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, ToolPlugin)
                        and attr is not ToolPlugin
                    ):
                        plugin = attr()
                        plugin.register(self.registry)
                        self._loaded[plugin.name] = plugin
                        loaded.append(plugin.name)
                        logger.info(
                            "Loaded plugin '%s' from %s", plugin.name, py_file
                        )
            except Exception as exc:
                logger.warning("Failed to load plugin from %s: %s", py_file, exc)
        return loaded

    @staticmethod
    def _load_module_from_file(path: Path):
        """Import a Python file as a module."""
        module_name = f"effgen_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_plugin(self, plugin: ToolPlugin) -> list[str]:
        """Manually load and register a single plugin instance."""
        registered = plugin.register(self.registry)
        self._loaded[plugin.name] = plugin
        return registered


def discover_plugins(registry: ToolRegistry | None = None) -> list[str]:
    """Convenience function: discover and load all available plugins."""
    mgr = PluginManager(registry)
    return mgr.discover_all()
