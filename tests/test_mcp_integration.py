# tests/test_mcp_integration.py
"""
Tests for MCP (Model Context Protocol) tool integration.

Unit tests use mocks for the MCPManager.
Integration tests spawn a real MCP server process via stdio.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Unit tests — MCPTool adapter
# ---------------------------------------------------------------------------


class TestMCPToolAdapter:
    """Test MCPTool without needing real MCP connections."""

    def _make_tool(
        self,
        name: str = "test.echo",
        mcp_tool_name: str = "echo",
        server_id: str = "test",
        schema: Dict[str, Any] | None = None,
        description: str = "Echo back",
    ) -> Any:
        from app.runtime.tools.adapters.mcp import MCPTool

        manager = MagicMock()
        manager.call_tool_sync.return_value = {"result": "hello"}

        if schema is None:
            schema = {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to echo"},
                },
                "required": ["message"],
            }

        return (
            MCPTool(
                name=name,
                mcp_tool_name=mcp_tool_name,
                server_id=server_id,
                schema=schema,
                description=description,
                manager=manager,
            ),
            manager,
        )

    def test_execute_calls_manager(self):
        tool, manager = self._make_tool()
        result = tool.execute({"message": "hi", "unrelated": "data"}, {})

        manager.call_tool_sync.assert_called_once_with(
            server_id="test",
            tool_name="echo",
            arguments={"message": "hi"},
            timeout=30,
        )
        assert result == {"result": "hello"}

    def test_extract_arguments_filters_by_schema(self):
        tool, _ = self._make_tool(
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            }
        )
        args = tool._extract_arguments({"a": 1, "b": 2, "thread_id": "t1", "extra": "noise"})
        assert args == {"a": 1, "b": 2}

    def test_extract_arguments_no_schema_passes_all(self):
        tool, _ = self._make_tool(schema={})
        args = tool._extract_arguments({"x": 1, "y": None, "z": "val"})
        # None values should be excluded
        assert args == {"x": 1, "z": "val"}

    def test_describe_includes_parameters(self):
        tool, _ = self._make_tool()
        desc = tool.describe()

        assert desc["name"] == "test.echo"
        assert desc["type"] == "mcp"
        assert desc["server"] == "test"
        assert "message" in desc["description"]
        assert "(required)" in desc["description"]
        assert "input_schema" in desc

    def test_describe_no_params(self):
        tool, _ = self._make_tool(schema={}, description="Simple tool")
        desc = tool.describe()
        assert desc["description"] == "Simple tool"
        assert "Parameters" not in desc["description"]

    def test_execute_handles_timeout(self):
        tool, manager = self._make_tool()
        manager.call_tool_sync.side_effect = TimeoutError("timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            tool.execute({"message": "hi"}, {})

    def test_execute_handles_connection_error(self):
        tool, manager = self._make_tool()
        manager.call_tool_sync.side_effect = ConnectionError("server down")

        with pytest.raises(RuntimeError, match="lost connection"):
            tool.execute({"message": "hi"}, {})

    def test_implements_itool(self):
        from app.runtime.tools.interface import ITool

        tool, _ = self._make_tool()
        assert isinstance(tool, ITool)

    def test_callable_interface(self):
        """MCPTool should be directly callable (ITool.__call__)."""
        tool, manager = self._make_tool()
        result = tool({"message": "hi"}, {})
        assert result == {"result": "hello"}


# ---------------------------------------------------------------------------
# Unit tests — ToolRegistry MCP integration
# ---------------------------------------------------------------------------


class TestRegistryMCPLoading:
    """Test that ToolRegistry.load_mcp_servers works correctly."""

    def test_load_mcp_servers_no_configs(self):
        from app.runtime.tools.registry import ToolRegistry

        registry = ToolRegistry()
        # Should not raise
        registry.load_mcp_servers([])
        registry.load_mcp_servers(None)

    def test_load_mcp_servers_all_disabled(self):
        from app.runtime.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.load_mcp_servers(
            [
                {"id": "s1", "enabled": False, "transport": "stdio", "command": "x"},
                {"id": "s2", "_disabled": True, "transport": "stdio", "command": "x"},
            ]
        )
        # No tools should be registered
        assert registry.all_names() == []

    @patch("app.runtime.tools.registry.MCPManager", create=True)
    def test_load_mcp_servers_registers_tools(self, mock_manager_cls):
        """Verify that load_mcp_servers calls MCPManager and registers tools."""
        from app.runtime.tools.adapters.mcp import MCPTool
        from app.runtime.tools.registry import ToolRegistry

        # Set up mock
        mock_manager = MagicMock()
        mock_manager.is_connected.return_value = False

        mock_tool = MagicMock(spec=MCPTool)
        mock_tool.name = "srv.echo"
        mock_tool.describe.return_value = {"name": "srv.echo", "type": "mcp"}
        # MCPTool inherits from ITool, so isinstance check needs to pass
        mock_tool.__class__ = MCPTool

        mock_manager.get_tools.return_value = {"srv.echo": mock_tool}

        with patch(
            "app.runtime.tools.mcp_manager.MCPManager.get_instance",
            return_value=mock_manager,
        ):
            registry = ToolRegistry()
            registry.load_mcp_servers(
                [
                    {"id": "srv", "transport": "stdio", "command": "python", "args": ["-m", "x"]},
                ]
            )

        # The tool should be in the registry
        assert "srv.echo" in registry.all_names()

    def test_load_mcp_servers_graceful_without_package(self):
        """If mcp package is not importable, should log warning and continue."""
        from app.runtime.tools.registry import ToolRegistry

        registry = ToolRegistry()

        with patch.dict("sys.modules", {"app.runtime.tools.mcp_manager": None}):
            with patch("builtins.__import__", side_effect=ImportError("no mcp")):
                # Should not raise
                registry.load_mcp_servers(
                    [
                        {"id": "s1", "transport": "stdio", "command": "x"},
                    ]
                )

    def test_shutdown_calls_manager(self):
        from app.runtime.tools.registry import ToolRegistry

        registry = ToolRegistry()
        mock_manager = MagicMock()
        registry._mcp_manager = mock_manager

        registry.shutdown()
        mock_manager.shutdown.assert_called_once()
        assert registry._mcp_manager is None


# ---------------------------------------------------------------------------
# Unit tests — build_registry with MCP
# ---------------------------------------------------------------------------


class TestBuildRegistryMCP:
    """Test that build_registry passes mcp_servers correctly."""

    def test_build_registry_without_mcp(self):
        from app.runtime.tools import build_registry

        registry = build_registry()
        # Should have default stubs
        assert "verify_identity" in registry.all_names()

    def test_build_registry_with_mcp_param(self):
        from app.runtime.tools import build_registry

        with patch.object(
            __import__("app.runtime.tools.registry", fromlist=["ToolRegistry"]).ToolRegistry,
            "load_mcp_servers",
        ) as mock_load:
            build_registry(mcp_servers=[{"id": "test"}])
            mock_load.assert_called_once_with([{"id": "test"}])


# ---------------------------------------------------------------------------
# Unit tests — MCPManager._parse_result
# ---------------------------------------------------------------------------


class TestMCPManagerParseResult:
    """Test result parsing without needing real connections."""

    @pytest.fixture
    def manager(self):
        """Create an MCPManager instance (requires mcp package)."""
        try:
            from app.runtime.tools.mcp_manager import MCPManager, _MCP_AVAILABLE

            if not _MCP_AVAILABLE:
                pytest.skip("mcp package not installed")

            mgr = MCPManager()
            yield mgr
            mgr.shutdown()
        except ImportError:
            pytest.skip("mcp package not installed")

    def test_parse_json_text_content(self, manager):
        """TextContent with JSON should be parsed to dict."""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = None

        text_content = MagicMock()
        text_content.text = '{"status": "ok", "count": 42}'
        mock_result.content = [text_content]

        parsed = manager._parse_result(mock_result)
        assert parsed == {"status": "ok", "count": 42}

    def test_parse_plain_text_content(self, manager):
        """TextContent with non-JSON should be stored as 'result'."""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = None

        text_content = MagicMock()
        text_content.text = "Hello world"
        mock_result.content = [text_content]

        parsed = manager._parse_result(mock_result)
        assert parsed == {"result": "Hello world"}

    def test_parse_structured_content(self, manager):
        """structuredContent should be preferred over text content."""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = {"key": "value"}
        mock_result.content = []

        parsed = manager._parse_result(mock_result)
        assert parsed == {"key": "value"}

    def test_parse_error_result(self, manager):
        """Error results should raise RuntimeError."""
        mock_result = MagicMock()
        mock_result.isError = True

        text_content = MagicMock()
        text_content.text = "Something went wrong"
        mock_result.content = [text_content]

        with pytest.raises(RuntimeError, match="MCP tool returned error"):
            manager._parse_result(mock_result)

    def test_parse_empty_content(self, manager):
        """Empty content should return status: completed."""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = None
        mock_result.content = []

        parsed = manager._parse_result(mock_result)
        assert parsed == {"status": "completed"}


# ---------------------------------------------------------------------------
# Integration tests — Real MCP server (requires `mcp` package installed)
# ---------------------------------------------------------------------------


MOCK_SERVER_PATH = str(Path(__file__).parent / "fixtures" / "mock_mcp_server.py")


@pytest.mark.skipif(
    not Path(MOCK_SERVER_PATH).exists(),
    reason="Mock MCP server fixture not found",
)
class TestMCPIntegration:
    """
    Integration tests that spawn a real MCP server process.
    These require the `mcp` package to be installed.
    """

    @pytest.fixture
    def manager(self):
        """Get a fresh MCPManager (resets singleton)."""
        try:
            from app.runtime.tools.mcp_manager import MCPManager, _MCP_AVAILABLE

            if not _MCP_AVAILABLE:
                pytest.skip("mcp package not installed")

            MCPManager.reset_instance()
            mgr = MCPManager()
            yield mgr
            mgr.shutdown()
            MCPManager._instance = None
        except ImportError:
            pytest.skip("mcp package not installed")

    def test_connect_and_discover_tools(self, manager):
        """Connect to mock server and verify tools are discovered."""
        discovery = manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                    "tool_prefix": True,
                }
            ]
        )

        assert "test" in discovery
        tool_names = discovery["test"]
        assert "echo" in tool_names
        assert "add" in tool_names
        assert "lookup_customer" in tool_names

    def test_get_tools_returns_mcp_tools(self, manager):
        """Verify get_tools() returns MCPTool instances."""
        from app.runtime.tools.adapters.mcp import MCPTool
        from app.runtime.tools.interface import ITool

        manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                    "tool_prefix": True,
                }
            ]
        )

        tools = manager.get_tools()
        assert len(tools) >= 3

        for name, tool in tools.items():
            assert isinstance(tool, MCPTool)
            assert isinstance(tool, ITool)
            assert name.startswith("test.")

    def test_call_echo_tool(self, manager):
        """Call the echo tool and verify the result."""
        manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                    "tool_prefix": False,
                }
            ]
        )

        tools = manager.get_tools()
        echo_tool = tools["echo"]
        result = echo_tool.execute({"message": "hello world"}, {})
        assert "hello world" in str(result.get("result", ""))

    def test_call_add_tool(self, manager):
        """Call the add tool and verify the result."""
        manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                    "tool_prefix": False,
                }
            ]
        )

        tools = manager.get_tools()
        add_tool = tools["add"]
        result = add_tool.execute({"a": 7, "b": 3}, {})
        # The mock server returns str(a+b), parsed as text content
        assert "10" in str(result.get("result", ""))

    def test_call_lookup_customer(self, manager):
        """Call lookup_customer and verify JSON result is parsed correctly."""
        manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                    "tool_prefix": False,
                }
            ]
        )

        tools = manager.get_tools()
        tool = tools["lookup_customer"]
        result = tool.execute({"customer_id": "C-42"}, {})

        assert result.get("customer_id") == "C-42"
        assert result.get("account_status") == "active"
        assert result.get("customer_found") is True

    def test_tool_describe_has_schema(self, manager):
        """Verify describe() returns schema information."""
        manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                    "tool_prefix": True,
                }
            ]
        )

        tools = manager.get_tools()
        desc = tools["test.echo"].describe()

        assert desc["type"] == "mcp"
        assert desc["name"] == "test.echo"
        assert "input_schema" in desc
        assert "message" in desc["description"]

    def test_tool_prefix_disabled(self, manager):
        """When tool_prefix=False, tool names should not be prefixed."""
        manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                    "tool_prefix": False,
                }
            ]
        )

        tools = manager.get_tools()
        assert "echo" in tools
        assert "test.echo" not in tools

    def test_registry_integration(self, manager):
        """Verify MCP tools integrate into ToolRegistry."""
        from app.runtime.tools.registry import ToolRegistry

        registry = ToolRegistry()

        # Register MCP tools
        manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                    "tool_prefix": True,
                }
            ]
        )
        mcp_tools = manager.get_tools()
        for tool_name, tool in mcp_tools.items():
            registry.register(tool_name, tool)

        # Verify they show up in registry
        names = registry.all_names()
        assert "test.echo" in names

        # Verify describe_all includes them
        descriptions = registry.describe_all()
        mcp_descs = [d for d in descriptions if d.get("type") == "mcp"]
        assert len(mcp_descs) >= 3

    def test_failed_server_does_not_crash(self, manager):
        """A bad server config should log warning but not raise."""
        discovery = manager.connect_servers(
            [
                {
                    "id": "bad",
                    "transport": "stdio",
                    "command": "nonexistent_command_xyz",
                    "args": [],
                }
            ]
        )
        # Should have no tools from the failed server
        assert "bad" not in discovery

    def test_shutdown_is_clean(self, manager):
        """Verify shutdown doesn't raise."""
        manager.connect_servers(
            [
                {
                    "id": "test",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [MOCK_SERVER_PATH],
                }
            ]
        )

        assert manager.is_connected()
        manager.shutdown()
        # After shutdown, connections should be cleared
        assert not manager.is_connected()
