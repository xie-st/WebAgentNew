"""Unit tests for protocol implementations."""

import pytest

from effgen.tools.protocols.acp.protocol import (
    ACPProtocolHandler,
    AgentManifest,
    CapabilityDefinition,
    SchemaDefinition,
    TaskStatus,
)


class TestACPProtocol:
    """Tests for ACP Protocol Handler."""

    @pytest.fixture
    def manifest(self):
        return AgentManifest(
            agentId="test-agent",
            name="Test Agent",
            version="1.0.0",
            description="A test agent",
            capabilities=[
                CapabilityDefinition(
                    name="calculate",
                    description="Perform calculations",
                    inputSchema=SchemaDefinition(
                        type="object",
                        properties={"expression": {"type": "string"}},
                        required=["expression"],
                    ),
                )
            ],
        )

    @pytest.fixture
    def handler(self, manifest):
        return ACPProtocolHandler(manifest)

    def test_create_handler(self, handler):
        assert handler is not None

    def test_manifest_properties(self, manifest):
        assert manifest.agentId == "test-agent"
        assert manifest.name == "Test Agent"
        assert len(manifest.capabilities) == 1

    def test_create_request(self, handler):
        request = handler.create_request("calculate", {"expression": "2+2"})
        assert request is not None
        assert request.capability == "calculate"

    def test_validate_valid_request(self, handler):
        request = handler.create_request("calculate", {"expression": "2+2"})
        is_valid, error = handler.validate_request(request)
        assert is_valid is True

    def test_validate_missing_required_field(self, handler):
        request = handler.create_request("calculate", {})
        is_valid, error = handler.validate_request(request)
        assert is_valid is False
        assert "expression" in str(error).lower() or error is not None

    def test_task_status_values(self):
        assert TaskStatus.PENDING is not None
        assert TaskStatus.RUNNING is not None
        assert TaskStatus.COMPLETED is not None
        assert TaskStatus.FAILED is not None


class TestMCPServerToolSelection:
    """The MCP server must hide unsafe tools by default (fail-closed)."""

    def _server(self, **kwargs):
        from effgen.tools.protocols.mcp_official.server import EffGenMCPServer

        server = EffGenMCPServer(**kwargs)
        server.tools_registry.discover_builtin_tools()
        return server

    def test_unsafe_tools_hidden_by_default(self):
        from effgen.tools.protocols.mcp_official.server import UNSAFE_TOOLS

        server = self._server()
        selected = set(server._select_tools())
        assert "calculator" in selected
        assert "datetime" in selected
        # No code execution / shell / filesystem tools leak by default.
        assert not (selected & UNSAFE_TOOLS), selected & UNSAFE_TOOLS
        assert "bash" not in selected
        assert "python_repl" not in selected

    def test_expose_unsafe_opt_in(self):
        server = self._server(expose_unsafe_tools=True)
        selected = set(server._select_tools())
        assert "bash" in selected

    def test_allowlist_is_explicit_opt_in(self):
        server = self._server(allowed_tools=["calculator", "bash"])
        selected = set(server._select_tools())
        assert selected == {"calculator", "bash"}

    def test_blocklist_removes_even_safe_tools(self):
        server = self._server(blocked_tools=["calculator"])
        selected = set(server._select_tools())
        assert "calculator" not in selected
        assert "datetime" in selected

    def test_build_signature_has_real_parameters(self):
        server = self._server()
        meta = server.tools_registry.get_metadata("calculator")
        sig = server._build_signature(meta)
        params = sig.parameters
        # ctx is injected for FastMCP; the tool's own params are present.
        assert "ctx" in params
        assert "expression" in params
        # required param has no default; optional ones do.
        assert params["expression"].default is __import__("inspect").Parameter.empty

    def test_run_cli_bad_mode_returns_error_code(self):
        """The launch entry dispatches modes and reports a usage error cleanly
        (the package `__main__` and the module guard share this)."""
        from effgen.tools.protocols.mcp_official.server import run_cli

        assert run_cli(["definitely-not-a-mode"]) == 1

    def test_package_has_main_module(self):
        """`python -m effgen.tools.protocols.mcp_official` works (and is
        warning-free) because the package ships a __main__ that reuses run_cli."""
        import importlib.util

        spec = importlib.util.find_spec("effgen.tools.protocols.mcp_official.__main__")
        assert spec is not None
        from effgen.tools.protocols.mcp_official import __main__ as pkg_main
        from effgen.tools.protocols.mcp_official.server import run_cli
        assert pkg_main.run_cli is run_cli


class TestMCPOfficialBridgeTypes:
    """The MCP->effGen bridge must map JSON types to real ParameterType members."""

    def test_json_type_map_members_exist(self):
        from effgen.tools.base_tool import ParameterType
        from effgen.tools.protocols.mcp_official.client import _MCPOfficialToolBridge

        for ptype in _MCPOfficialToolBridge._JSON_TYPE_MAP.values():
            assert isinstance(ptype, ParameterType)


class TestACPSecureDefaults:
    """ACP server must default to loopback and no wildcard CORS."""

    def test_default_host_is_loopback(self):
        from effgen.tools.protocols.acp.server import ACPServerConfig

        assert ACPServerConfig().host == "127.0.0.1"

    def test_default_cors_is_empty(self):
        from effgen.tools.protocols.acp.server import ACPServerConfig

        assert ACPServerConfig().cors_origins == []


class TestACPServerHandleRequest:
    """In-process ACP request/response incl. the auth-reject path."""

    async def _echo_handler(self, input_data, context):
        return {"echo": input_data.get("text", "")}

    async def test_sync_capability_executes(self):
        from effgen.tools.protocols.acp import ACPRequest, ACPServer

        server = ACPServer(agent_id="a", name="A", version="1.0.0", description="d")
        server.register_capability(
            name="echo", description="echo",
            input_schema={"type": "object"}, handler=self._echo_handler,
        )
        resp = await server.handle_request(ACPRequest(capability="echo", input={"text": "hi"}))
        assert resp.status.value == "completed"
        assert resp.output == {"echo": "hi"}

    async def test_missing_token_rejected(self):
        from effgen.tools.protocols.acp import ACPRequest, ACPServer, ACPServerConfig

        server = ACPServer(agent_id="a", name="A", version="1.0.0", description="d",
                           config=ACPServerConfig(require_auth=True))
        server.register_capability(
            name="echo", description="echo",
            input_schema={"type": "object"}, handler=self._echo_handler,
        )
        resp = await server.handle_request(ACPRequest(capability="echo", input={}))
        assert resp.status.value == "failed"
        assert resp.error is not None and resp.error.code == "UNAUTHORIZED"

    async def test_valid_token_accepted(self):
        from effgen.tools.protocols.acp import ACPRequest, ACPServer, ACPServerConfig
        from effgen.tools.protocols.acp.protocol import CapabilityToken

        server = ACPServer(agent_id="a", name="A", version="1.0.0", description="d",
                           config=ACPServerConfig(require_auth=True))
        server.register_capability(
            name="echo", description="echo",
            input_schema={"type": "object"}, handler=self._echo_handler,
        )
        server.add_capability_token(
            CapabilityToken(tokenId="tk", agentId="a", capabilities=["echo"], expires=None)
        )
        resp = await server.handle_request(
            ACPRequest(capability="echo", input={"text": "yo"}), token_id="tk"
        )
        assert resp.status.value == "completed"


class TestA2AAgentCard:
    """A2A agent-card construction, validation, and JSON round-trip."""

    def _card(self):
        from effgen.tools.protocols.a2a import (
            AgentCard,
            AuthScheme,
            Capability,
            CapabilityType,
            EndpointConfig,
        )
        return AgentCard(
            name="math", description="adds", version="1.0.0",
            capabilities=[Capability(
                name="add", type=CapabilityType.TASK_EXECUTION,
                description="add", inputSchema={"type": "object"},
            )],
            endpoint=EndpointConfig(url="http://127.0.0.1:9", protocol="http"),
            authSchemes=[AuthScheme.BEARER],
        )

    def test_card_validates(self):
        ok, err = self._card().validate()
        assert ok, err

    def test_card_json_round_trip(self):
        from effgen.tools.protocols.a2a import AgentCard

        card = self._card()
        restored = AgentCard.from_json(card.to_json())
        assert restored.name == card.name
        assert len(restored.capabilities) == 1

    async def test_bearer_auth_handler_sets_header(self):
        from effgen.tools.protocols.a2a import BearerAuthHandler

        headers = await BearerAuthHandler("tok").apply_auth({})
        assert headers["Authorization"] == "Bearer tok"
