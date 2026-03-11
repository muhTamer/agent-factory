# app/runtime/tools/adapters/mcp.py
"""
MCPTool — adapts a single MCP tool to the ITool interface.

Each instance wraps one tool from one MCP server.  The MCPManager handles
the actual async call; this class bridges to the synchronous ITool.execute()
contract used by the ReAct engine.

Usage:
    # Normally created by MCPManager.get_tools(), not directly:
    tool = MCPTool(
        name="crm.lookup_customer",
        mcp_tool_name="lookup_customer",
        server_id="crm",
        schema={"type": "object", "properties": {"customer_id": {"type": "string"}}},
        description="Look up a customer by ID",
        manager=mcp_manager,
    )
    result = tool.execute({"customer_id": "C-123"}, {})
"""
from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from app.runtime.tools.interface import ITool

if TYPE_CHECKING:
    from app.runtime.tools.mcp_manager import MCPManager


class MCPTool(ITool):
    """Adapts a single MCP tool to the ITool interface."""

    def __init__(
        self,
        name: str,
        mcp_tool_name: str,
        server_id: str,
        schema: Dict[str, Any],
        description: str,
        manager: "MCPManager",
        timeout: int = 30,
    ) -> None:
        self.name = name
        self.mcp_tool_name = mcp_tool_name
        self.server_id = server_id
        self.schema = schema or {}
        self._description = description or f"MCP tool '{mcp_tool_name}'"
        self._manager = manager
        self.timeout = timeout

    # ------------------------------------------------------------------
    # ITool
    # ------------------------------------------------------------------

    def execute(self, slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the MCP tool via the manager.

        Extracts arguments from slots based on the tool's input schema,
        calls the MCP server, and returns the result as slot updates.
        """
        arguments = self._extract_arguments(slots)

        try:
            return self._manager.call_tool_sync(
                server_id=self.server_id,
                tool_name=self.mcp_tool_name,
                arguments=arguments,
                timeout=self.timeout,
            )
        except TimeoutError as exc:
            raise RuntimeError(f"MCPTool '{self.name}' timed out after {self.timeout}s") from exc
        except ConnectionError as exc:
            raise RuntimeError(
                f"MCPTool '{self.name}' lost connection to server " f"'{self.server_id}': {exc}"
            ) from exc

    def describe(self) -> Dict[str, Any]:
        """
        Rich description including argument schema for LLM prompt building.

        The DomainAgentEngine._build_react_prompt() uses the 'description'
        field in tool_lines.  We append parameter info so the LLM knows what
        arguments to pass when calling this tool.
        """
        full_description = self._description

        # Build human-readable parameter summary
        params_desc = self._format_parameters()
        if params_desc:
            full_description += f" | Parameters: {params_desc}"

        return {
            "name": self.name,
            "type": "mcp",
            "description": full_description,
            "server": self.server_id,
            "input_schema": self.schema,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_arguments(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract tool arguments from slots based on the tool's JSON Schema.

        If the schema defines properties, only pass those keys from slots.
        This prevents leaking unrelated slot data to the MCP tool.
        """
        properties = self.schema.get("properties", {})
        if not properties:
            # No schema defined — pass all non-None slots
            return {k: v for k, v in slots.items() if v is not None}

        args: Dict[str, Any] = {}
        for prop_name in properties:
            if prop_name in slots and slots[prop_name] is not None:
                args[prop_name] = slots[prop_name]
        return args

    def _format_parameters(self) -> str:
        """Build a human-readable parameter summary from the JSON Schema."""
        properties = self.schema.get("properties", {})
        if not properties:
            return ""

        required = set(self.schema.get("required", []))
        parts: list[str] = []

        for pname, pschema in properties.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            req_marker = " (required)" if pname in required else ""
            entry = f"{pname}: {ptype}{req_marker}"
            if pdesc:
                entry += f" - {pdesc}"
            parts.append(entry)

        return "; ".join(parts)
