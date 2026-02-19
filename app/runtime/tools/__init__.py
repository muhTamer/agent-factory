# app/runtime/tools/__init__.py
"""
Pluggable tool layer for Agent Factory workflow engines.

Quick-start
-----------
Default registry (demo stubs — works out of the box):

    from app.runtime.tools import DEFAULT_REGISTRY
    engine = GenericWorkflowEngine(..., tools=DEFAULT_REGISTRY.as_callable_dict())

Customer tool override:

    from app.runtime.tools import build_registry

    registry = build_registry([
        {"name": "initiate_refund", "type": "http",
         "url": "https://erp.customer.com/api/refunds"},
        {"name": "lookup_customer",  "type": "sql",
         "dsn": "sqlite:///data/customers.db",
         "query": "SELECT account_status, kyc_status FROM customers WHERE id = :customer_id"},
        # Any tool not listed here falls back to the built-in stub
        {"name": "verify_identity",  "type": "stub"},
    ])
    engine = GenericWorkflowEngine(..., tools=registry.as_callable_dict())
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.runtime.tools.adapters.stub import StubTool
from app.runtime.tools.interface import ITool
from app.runtime.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Build the default registry from existing stub callables
# ---------------------------------------------------------------------------
def _build_default_registry() -> ToolRegistry:
    from app.runtime.tools.stub_tools import STUB_TOOLS

    registry = ToolRegistry()
    for name, fn in STUB_TOOLS.items():
        registry.register(name, StubTool(name, fn))
    return registry


DEFAULT_REGISTRY: ToolRegistry = _build_default_registry()


# ---------------------------------------------------------------------------
# Public factory: merge customer overrides on top of stubs
# ---------------------------------------------------------------------------
def build_registry(
    config: Optional[List[Dict[str, Any]]] = None,
) -> ToolRegistry:
    """
    Build a ToolRegistry starting from the default stubs, then apply
    any customer-provided overrides from `config`.

    Args:
        config: List of tool config dicts (see ToolRegistry.load_config).
                Pass None or [] to get a registry that is identical to
                DEFAULT_REGISTRY (all stubs).

    Returns:
        A new ToolRegistry instance with customer overrides applied.
    """
    from app.runtime.tools.stub_tools import STUB_TOOLS

    registry = ToolRegistry()
    # Seed with all stubs as fallback
    for name, fn in STUB_TOOLS.items():
        registry.register(name, StubTool(name, fn))

    # Apply customer overrides (may replace individual stubs)
    if config:
        registry.load_config(config)

    return registry


__all__ = [
    "ITool",
    "ToolRegistry",
    "StubTool",
    "DEFAULT_REGISTRY",
    "build_registry",
]
