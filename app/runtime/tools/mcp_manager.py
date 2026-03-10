# app/runtime/tools/mcp_manager.py
"""
MCPManager — manages MCP server connections, tool discovery, and lifecycle.

Provides a sync-to-async bridge so the synchronous ITool.execute() interface
can call MCP servers (which use an async SDK) without changing the engine.

Usage:
    manager = MCPManager.get_instance()
    manager.connect_servers([
        {"id": "my_tools", "transport": "stdio",
         "command": "python", "args": ["-m", "my_server"]},
    ])
    tools = manager.get_tools()  # Dict[str, MCPTool]

    # tools are ITool instances ready to register in ToolRegistry
    for name, tool in tools.items():
        registry.register(name, tool)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MCP SDK imports — guarded so the module can be imported even when
# the `mcp` package is not installed.
# ---------------------------------------------------------------------------
_MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    _MCP_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConnection:
    """Represents a single connected MCP server."""

    server_id: str
    config: Dict[str, Any]
    session: Any  # mcp.ClientSession
    tools: Dict[str, Any] = field(default_factory=dict)  # MCP tool name → Tool schema
    # Context manager references for cleanup
    _transport_cm: Any = None
    _session_cm: Any = None
    _lock: Any = None  # asyncio.Lock for serialising calls


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class MCPManager:
    """
    Manages MCP server connections, tool discovery, and lifecycle.

    Singleton-per-application. Maintains a dedicated asyncio event loop in a
    background daemon thread for all MCP async operations.
    """

    _instance: Optional["MCPManager"] = None
    _class_lock = threading.Lock()

    # ---- Singleton --------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "MCPManager":
        """Thread-safe singleton access."""
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = MCPManager()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Tear down the singleton (for testing)."""
        with cls._class_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None

    # ---- Init -------------------------------------------------------------

    def __init__(self) -> None:
        if not _MCP_AVAILABLE:
            raise ImportError("MCP package not installed. Install with: pip install 'mcp>=1.26.0'")
        self._connections: Dict[str, MCPServerConnection] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-event-loop")
        self._thread.start()

    def _run_loop(self) -> None:
        """Background thread: run the asyncio event loop forever."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro: Any, timeout: float = 60) -> Any:
        """Submit a coroutine to the background loop and block for result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ---- Public API -------------------------------------------------------

    def is_connected(self) -> bool:
        return len(self._connections) > 0

    def connect_servers(self, server_configs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Connect to all enabled MCP servers and discover their tools.

        Args:
            server_configs: List of server config dicts from tools_config.json.

        Returns:
            {server_id: [tool_names]} for logging/diagnostics.
        """
        enabled = [c for c in server_configs if c.get("enabled", True) and not c.get("_disabled")]
        if not enabled:
            _log.info("[MCP] No enabled MCP servers configured.")
            return {}

        discovery: Dict[str, List[str]] = {}
        for config in enabled:
            server_id = config.get("id", "unnamed")
            try:
                conn = self._run_async(
                    self._connect_server(config),
                    timeout=config.get("timeout", 30) + 10,
                )
                if conn is not None:
                    discovery[server_id] = list(conn.tools.keys())
                    _log.info(
                        "[MCP] Connected '%s': %d tools discovered — %s",
                        server_id,
                        len(conn.tools),
                        list(conn.tools.keys()),
                    )
            except Exception as exc:
                _log.warning("[MCP] Failed to connect '%s': %s", server_id, exc)

        return discovery

    def get_tools(self) -> Dict[str, Any]:
        """
        Return all discovered MCP tools as MCPTool adapter instances.

        Keys are the registered tool names (prefixed with server_id if
        tool_prefix is enabled).
        """
        from app.runtime.tools.adapters.mcp import MCPTool

        tools: Dict[str, Any] = {}
        for conn in self._connections.values():
            use_prefix = conn.config.get("tool_prefix", True)
            timeout = conn.config.get("timeout", 30)

            for mcp_tool_name, mcp_tool in conn.tools.items():
                registered_name = (
                    f"{conn.server_id}.{mcp_tool_name}" if use_prefix else mcp_tool_name
                )

                # Extract schema and description from the MCP Tool object
                schema = {}
                description = ""
                if hasattr(mcp_tool, "inputSchema"):
                    schema = mcp_tool.inputSchema if isinstance(mcp_tool.inputSchema, dict) else {}
                if hasattr(mcp_tool, "description"):
                    description = mcp_tool.description or ""

                tools[registered_name] = MCPTool(
                    name=registered_name,
                    mcp_tool_name=mcp_tool_name,
                    server_id=conn.server_id,
                    schema=schema,
                    description=description,
                    manager=self,
                    timeout=timeout,
                )

        return tools

    def call_tool_sync(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 30,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper: call an MCP tool and return the result dict.

        Submits the async call to the background event loop and blocks.
        """
        return self._run_async(
            self._call_tool(server_id, tool_name, arguments),
            timeout=timeout + 5,
        )

    def shutdown(self) -> None:
        """Gracefully close all MCP server connections and stop the event loop."""
        if not self._loop.is_running():
            return

        try:
            self._run_async(self._shutdown_async(), timeout=15)
        except Exception as exc:
            _log.warning("[MCP] Error during shutdown: %s", exc)

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._connections.clear()
        _log.info("[MCP] All servers disconnected.")

    # ---- Async internals --------------------------------------------------

    async def _connect_server(self, config: Dict[str, Any]) -> Optional[MCPServerConnection]:
        """Connect to a single MCP server and discover its tools."""
        server_id = config.get("id", "unnamed")
        transport = config.get("transport", "stdio").lower()

        try:
            if transport == "stdio":
                transport_cm, streams = await self._enter_stdio(config)
            elif transport in ("streamable_http", "sse", "http"):
                transport_cm, streams = await self._enter_http(config)
            else:
                _log.warning("[MCP] Unknown transport '%s' for server '%s'", transport, server_id)
                return None

            read_stream, write_stream = streams

            # Create and initialise the client session.
            # IMPORTANT: must enter ClientSession as context manager to start
            # its background read task — otherwise initialize() hangs.
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()

            # Discover tools
            tools_result = await session.list_tools()
            discovered_tools = {tool.name: tool for tool in tools_result.tools}

            conn = MCPServerConnection(
                server_id=server_id,
                config=config,
                session=session,
                tools=discovered_tools,
                _transport_cm=transport_cm,
                _session_cm=session,  # session entered as CM, needs __aexit__
                _lock=asyncio.Lock(),
            )
            self._connections[server_id] = conn
            return conn

        except Exception as exc:
            _log.warning("[MCP] Connection failed for '%s': %s", server_id, exc)
            return None

    async def _enter_stdio(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Manually enter the stdio_client context manager for a long-lived connection."""
        env = config.get("env")
        if env:
            # Expand ${ENV_VAR} tokens in env values
            env = {k: os.path.expandvars(str(v)) for k, v in env.items()}

        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=env,
        )

        cm = stdio_client(server_params)
        streams = await cm.__aenter__()
        return cm, streams

    async def _enter_http(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Manually enter the streamable_http_client context manager."""
        url = config["url"]
        headers = config.get("headers", {})
        # Expand ${ENV_VAR} tokens in header values
        headers = {k: os.path.expandvars(str(v)) for k, v in headers.items()}

        cm = streamable_http_client(url=url, headers=headers)
        streams = await cm.__aenter__()
        # streamable_http_client yields (read, write, session_id) — we only need read/write
        read_stream = streams[0]
        write_stream = streams[1]
        return cm, (read_stream, write_stream)

    async def _call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call a tool on a specific MCP server."""
        conn = self._connections.get(server_id)
        if conn is None:
            raise ConnectionError(f"MCP server '{server_id}' not connected.")

        async with conn._lock:
            result = await conn.session.call_tool(tool_name, arguments)

        return self._parse_result(result)

    def _parse_result(self, result: Any) -> Dict[str, Any]:
        """Normalize an MCP CallToolResult into a plain dict for slot updates."""
        # Check for isError flag
        if hasattr(result, "isError") and result.isError:
            error_text = ""
            if hasattr(result, "content"):
                for item in result.content:
                    if hasattr(item, "text"):
                        error_text += item.text
            raise RuntimeError(f"MCP tool returned error: {error_text or 'unknown error'}")

        # Prefer structuredContent if available (already dict-like)
        if hasattr(result, "structuredContent") and result.structuredContent:
            sc = dict(result.structuredContent)
            # If structuredContent has a single "result" key with a JSON string,
            # try to parse it (common pattern with FastMCP tools returning str)
            if len(sc) == 1 and "result" in sc and isinstance(sc["result"], str):
                try:
                    parsed = json.loads(sc["result"])
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            return sc

        # Fall back to parsing text content
        combined: Dict[str, Any] = {}
        if hasattr(result, "content"):
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    text = content_item.text
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            combined.update(parsed)
                        else:
                            combined["result"] = parsed
                    except (json.JSONDecodeError, TypeError):
                        if "result" in combined:
                            # Multiple text items — append
                            combined["result"] = str(combined["result"]) + "\n" + text
                        else:
                            combined["result"] = text

        return combined if combined else {"status": "completed"}

    async def _shutdown_async(self) -> None:
        """Close all sessions and transports."""
        for server_id, conn in self._connections.items():
            try:
                # Close the session context manager first
                if conn._session_cm is not None:
                    await conn._session_cm.__aexit__(None, None, None)
            except Exception as exc:
                _log.debug("[MCP] Session cleanup error for '%s': %s", server_id, exc)
            try:
                # Then close the transport context manager
                if conn._transport_cm is not None:
                    await conn._transport_cm.__aexit__(None, None, None)
            except Exception as exc:
                _log.debug("[MCP] Transport cleanup error for '%s': %s", server_id, exc)
