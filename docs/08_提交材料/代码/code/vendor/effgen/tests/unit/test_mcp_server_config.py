"""``EffGenMCPServer(config=EffGenMCPServerConfig(...))`` must actually work.

``EffGenMCPServerConfig`` is exported public surface whose fields mirror the
server's constructor kwargs, so ``config=`` is the natural call a contributor
tries first. The constructor must accept it (mapping its fields), not just the
individual keyword arguments.
"""

from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp")

from effgen.tools.protocols.mcp_official import (  # noqa: E402
    EffGenMCPServer,
    EffGenMCPServerConfig,
)
from effgen.tools.registry import ToolRegistry  # noqa: E402


def test_server_accepts_config_object():
    cfg = EffGenMCPServerConfig(
        name="temp-tools",
        version="2.0.0",
        allowed_tools=["calculator"],
    )
    server = EffGenMCPServer(config=cfg)
    assert server.name == "temp-tools"
    assert server.version == "2.0.0"
    assert server.allowed_tools == ["calculator"]


def test_server_direct_kwargs_still_work():
    """Backward compatible: the pre-existing direct-kwargs form is unaffected."""
    server = EffGenMCPServer(name="direct", version="3.0.0")
    assert server.name == "direct"
    assert server.version == "3.0.0"


def test_server_config_drives_real_tool_exposure():
    registry = ToolRegistry()
    cfg = EffGenMCPServerConfig(
        name="temp-tools",
        tools_registry=registry,
        allowed_tools=["calculator"],
    )
    server = EffGenMCPServer(config=cfg)
    asyncio.run(server.initialize())
    try:
        assert server._exposed_tools == ["calculator"]
    finally:
        asyncio.run(server.cleanup())


def _register_probe_tool(name: str):
    from effgen.tools.base_tool import (
        BaseTool,
        ParameterSpec,
        ParameterType,
        ToolCategory,
        ToolMetadata,
    )

    class _Probe(BaseTool):
        def __init__(self):
            super().__init__(metadata=ToolMetadata(
                name=name,
                description="probe tool for registry-default tests",
                category=ToolCategory.COMPUTATION,
                parameters=[ParameterSpec(name="x", type=ParameterType.STRING,
                                          description="x", required=False)],
            ))

        async def _execute(self, **kwargs):  # pragma: no cover - not called
            return {}

    return _Probe()


def test_server_with_no_registry_arg_defaults_to_shared_registry():
    """EffGenMCPServer() with no tools_registry= exposes a tool registered on
    the shared get_registry() singleton elsewhere in the same process — the
    sequence the report calls out (register_tool() then EffGenMCPServer())."""
    from effgen.tools.registry import get_registry

    name = "shared_registry_default_tool"
    get_registry().register_tool(_register_probe_tool(name))
    try:
        server = EffGenMCPServer()  # no tools_registry= given
        assert server.tools_registry is get_registry()
        asyncio.run(server.initialize())
        try:
            assert name in server._exposed_tools
        finally:
            asyncio.run(server.cleanup())
    finally:
        get_registry().unregister_tool(name)


def test_explicit_isolated_registry_still_supported():
    """Passing tools_registry=ToolRegistry() explicitly (an isolated registry,
    not the shared one) is still honored — the fix only changes the *default*
    used when no registry is given."""
    from effgen.tools.registry import get_registry

    name = "only_on_shared_registry"
    get_registry().register_tool(_register_probe_tool(name))
    try:
        isolated = ToolRegistry()
        server = EffGenMCPServer(tools_registry=isolated)
        asyncio.run(server.initialize())
        try:
            assert name not in server._exposed_tools
        finally:
            asyncio.run(server.cleanup())
    finally:
        get_registry().unregister_tool(name)
