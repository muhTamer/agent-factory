# app/runtime/tools/registry.py
"""
ToolRegistry — resolves tool names to ITool implementations at runtime.

Usage:

    from app.runtime.tools.registry import ToolRegistry
    from app.runtime.tools.adapters.stub import StubTool
    from app.runtime.tools.adapters.http import HttpTool
    from app.runtime.tools.adapters.sql import SqlTool

    registry = ToolRegistry()

    # Register stubs (dev / demo)
    registry.register("initiate_refund", StubTool("initiate_refund", my_callable))

    # Register real customer tools from config
    registry.load_config([
        {"name": "initiate_refund", "type": "http",
         "url": "https://erp.customer.com/api/refunds", "method": "POST"},
        {"name": "lookup_customer", "type": "sql",
         "dsn": "sqlite:///data/customers.db",
         "query": "SELECT account_status, kyc_status FROM customers WHERE id = :customer_id"},
    ])

    # Pass to engine — no engine changes required
    engine = GenericWorkflowEngine(..., tools=registry.as_callable_dict())
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from app.runtime.tools.interface import ITool


class ToolRegistry:
    """
    Central registry mapping tool names → ITool implementations.

    Designed so that:
    - Customer A can swap in their SQL/HTTP tools without touching agent code.
    - The workflow engine receives a plain {name: callable} dict (no API change).
    - Tools can be registered programmatically or loaded from a config list.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ITool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, name: str, tool: ITool) -> None:
        """Register (or replace) a tool by name."""
        if not isinstance(tool, ITool):
            raise TypeError(
                f"Tool '{name}' must implement ITool. Got {type(tool).__name__}. "
                "Wrap plain callables with StubTool."
            )
        self._tools[name] = tool

    def get(self, name: str) -> Optional[ITool]:
        return self._tools.get(name)

    def all_names(self) -> List[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Config-driven loading
    # ------------------------------------------------------------------
    def load_config(self, config: List[Dict[str, Any]]) -> None:
        """
        Load tools from a list of config dicts.  Each entry must have at
        minimum:  { "name": str, "type": "stub"|"http"|"sql" }

        Supported types and their required keys:
            stub:
                No additional keys — uses the StubTool already registered
                under that name, or a no-op if not present. To override a
                stub name use: { "name": "...", "type": "stub" }

            http:
                url     : str   — endpoint URL (required)
                method  : str   — HTTP method (default "POST")
                headers : dict  — static headers; ${ENV_VAR} is expanded
                timeout : int   — seconds (default 10)
                slot_map: dict  — {response_key: slot_key} mapping (optional)

            sql:
                dsn     : str  — SQLite path ("sqlite:///db.sqlite3") or
                                 any SQLAlchemy-compatible DSN (requires
                                 sqlalchemy installed)
                query   : str  — parameterised query using :slot_name syntax
                slot_map: dict — {column_name: slot_key} (optional)
        """
        from app.runtime.tools.adapters.http import HttpTool
        from app.runtime.tools.adapters.sql import SqlTool
        from app.runtime.tools.adapters.stub import StubTool

        for entry in config:
            # Skip comment-only entries and explicitly disabled examples
            if entry.get("_disabled") or not entry.get("name"):
                continue

            name = entry.get("name")
            kind = (entry.get("type") or "").lower()

            if kind == "stub":
                # Keep the existing stub if already registered; otherwise no-op stub
                if name not in self._tools:
                    self._tools[name] = StubTool(name, lambda s, c: {})

            elif kind == "http":
                self._tools[name] = HttpTool(
                    name=name,
                    url=entry["url"],
                    method=entry.get("method", "POST"),
                    headers=entry.get("headers", {}),
                    timeout=int(entry.get("timeout", 10)),
                    slot_map=entry.get("slot_map", {}),
                )

            elif kind == "sql":
                self._tools[name] = SqlTool(
                    name=name,
                    dsn=entry["dsn"],
                    query=entry["query"],
                    slot_map=entry.get("slot_map", {}),
                )

            else:
                raise ValueError(
                    f"Unknown tool type '{kind}' for tool '{name}'. " "Expected: stub | http | sql"
                )

    # ------------------------------------------------------------------
    # Engine bridge
    # ------------------------------------------------------------------
    def as_callable_dict(self) -> Dict[str, Callable]:
        """
        Return a {name: callable} dict compatible with GenericWorkflowEngine.

        Because ITool implements __call__, each ITool instance IS a callable
        that satisfies the engine contract:  tool(slots, context) -> dict
        """
        return {name: tool for name, tool in self._tools.items()}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def describe_all(self) -> List[Dict[str, Any]]:
        """Return describe() for every registered tool (useful for docs/UI)."""
        return [tool.describe() for tool in self._tools.values()]

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolRegistry(tools={list(self._tools.keys())})"
