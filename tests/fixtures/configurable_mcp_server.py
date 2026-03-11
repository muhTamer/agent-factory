# tests/fixtures/configurable_mcp_server.py
"""
Config-driven MCP server — customers define tools and responses in a JSON file.

Usage:
    python tests/fixtures/configurable_mcp_server.py                          # uses default config
    python tests/fixtures/configurable_mcp_server.py my_tools.json            # custom config
    python tests/fixtures/configurable_mcp_server.py --config my_tools.json   # explicit flag

Config file format (see mcp_tools_config.json for full example):

    {
      "server_name": "my-server",
      "tools": [
        {
          "name": "lookup_customer",
          "description": "Look up a customer by ID",
          "parameters": {
            "customer_id": {"type": "string", "required": true}
          },
          "response": {
            "customer_id": "{{customer_id}}",
            "account_status": "active",
            "customer_found": true
          },
          "scenarios": [
            {
              "when": {"customer_id": {"starts_with": "FAIL"}},
              "response": {"customer_found": false, "message": "Not found"}
            }
          ]
        }
      ]
    }

Template variables:
    {{param_name}}          → replaced with the input parameter value
    {{param_name:default}}  → replaced with value, or "default" if not provided

Scenario conditions (evaluated in order, first match wins):
    "starts_with": "PREFIX"        → string prefix match (case-insensitive)
    "equals": "VALUE"              → exact match
    "greater_than": 5000           → numeric comparison
    "less_than": 100               → numeric comparison
    "in": ["a", "b", "c"]         → membership check
    "contains": "substring"        → substring match (case-insensitive)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Default config path (next to this file)
DEFAULT_CONFIG = Path(__file__).parent / "mcp_tools_config.json"


def _resolve_template(value: Any, params: Dict[str, Any]) -> Any:
    """Resolve {{param}} templates in a value."""
    if isinstance(value, str):

        def replacer(match: re.Match) -> str:
            key = match.group(1).strip()
            if ":" in key:
                key, default = key.split(":", 1)
                key = key.strip()
                default = default.strip()
            else:
                default = key  # echo the key name if not provided
            val = params.get(key, default)
            return str(val)

        return re.sub(r"\{\{(.+?)\}\}", replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_template(v, params) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_template(v, params) for v in value]
    return value


def _check_condition(condition: Dict[str, Any], params: Dict[str, Any]) -> bool:
    """Check if a scenario condition matches the given parameters."""
    for param_name, checks in condition.items():
        param_value = params.get(param_name)
        if param_value is None:
            return False

        if not isinstance(checks, dict):
            # Simple equality: {"customer_id": "C001"}
            if str(param_value).lower() != str(checks).lower():
                return False
            continue

        for op, expected in checks.items():
            pv_str = str(param_value).upper()

            if op == "starts_with":
                if not pv_str.startswith(str(expected).upper()):
                    return False
            elif op == "equals":
                if pv_str != str(expected).upper():
                    return False
            elif op == "contains":
                if str(expected).upper() not in pv_str:
                    return False
            elif op == "greater_than":
                try:
                    if float(param_value) <= float(expected):
                        return False
                except (ValueError, TypeError):
                    return False
            elif op == "less_than":
                try:
                    if float(param_value) >= float(expected):
                        return False
                except (ValueError, TypeError):
                    return False
            elif op == "in":
                if str(param_value).lower() not in [str(v).lower() for v in expected]:
                    return False
            else:
                return False
    return True


def _find_matching_scenario(
    scenarios: List[Dict[str, Any]], params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Find the first matching scenario, or None."""
    for scenario in scenarios:
        condition = scenario.get("when", {})
        if _check_condition(condition, params):
            return scenario
    return None


def _build_tool_handler(tool_name: str, config_path: Path):
    """Create a tool handler that re-reads config on every call (hot-reload)."""

    def handler(**kwargs) -> str:
        # Re-read config from disk so edits take effect without restart
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tool_def = next(
            (t for t in config.get("tools", []) if t["name"] == tool_name),
            None,
        )
        if tool_def is None:
            return json.dumps({"error": f"Tool '{tool_name}' not found in config"})

        default_response = tool_def.get("response", {"status": "ok"})
        scenarios = tool_def.get("scenarios", [])

        # Check scenarios first (first match wins)
        matched = _find_matching_scenario(scenarios, kwargs)
        if matched:
            # Merge: start with default response, override with scenario response
            response = {**default_response, **matched["response"]}
        else:
            response = dict(default_response)

        # Resolve templates
        resolved = _resolve_template(response, kwargs)
        return json.dumps(resolved)

    return handler


def _build_param_annotations(params_def: Dict[str, Any]) -> Dict[str, Any]:
    """Build Python type annotations from parameter definitions."""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    annotations = {}
    for name, spec in params_def.items():
        ptype = spec.get("type", "string") if isinstance(spec, dict) else "string"
        annotations[name] = type_map.get(ptype, str)
    return annotations


def build_server(config_path: Path) -> FastMCP:
    """Build a FastMCP server from a JSON config file."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    server_name = config.get("server_name", "configurable-server")
    server = FastMCP(server_name)

    for tool_def in config.get("tools", []):
        name = tool_def["name"]
        description = tool_def.get("description", f"Tool: {name}")
        params_def = tool_def.get("parameters", {})

        # Build the handler (re-reads config on each call for hot-reload)
        handler = _build_tool_handler(name, config_path)

        # Build parameter info for the function signature
        annotations = _build_param_annotations(params_def)
        required = {
            k for k, v in params_def.items() if isinstance(v, dict) and v.get("required", False)
        }
        defaults = {}
        for pname, pspec in params_def.items():
            if isinstance(pspec, dict) and "default" in pspec:
                defaults[pname] = pspec["default"]
            elif pname not in required:
                # Provide sensible defaults for optional params
                ptype = pspec.get("type", "string") if isinstance(pspec, dict) else "string"
                defaults[pname] = {"string": "", "integer": 0, "number": 0.0, "boolean": False}.get(
                    ptype, ""
                )

        # Create a proper function with the right signature for FastMCP
        param_names = list(params_def.keys())
        req_params = [p for p in param_names if p in required]
        opt_params = [p for p in param_names if p not in required]

        # Build function code dynamically
        sig_parts = []
        for p in req_params:
            sig_parts.append(f"{p}")
        for p in opt_params:
            default_val = defaults.get(p, "")
            if isinstance(default_val, str):
                sig_parts.append(f"{p}={default_val!r}")
            else:
                sig_parts.append(f"{p}={default_val!r}")

        sig = ", ".join(sig_parts)

        # Create the function dynamically with proper annotations
        func_code = f"def {name}({sig}) -> str:\n"
        func_code += f'    """{description}"""\n'
        func_code += f"    return _handler({', '.join(f'{p}={p}' for p in param_names)})\n"

        local_ns = {"_handler": handler}
        # Add type annotations
        for pname, ptype in annotations.items():
            local_ns[f"_type_{pname}"] = ptype

        exec(func_code, local_ns)  # noqa: S102
        func = local_ns[name]
        func.__annotations__ = {**annotations, "return": str}

        server.tool()(func)

    return server


if __name__ == "__main__":
    # Parse config path from args
    config_path = DEFAULT_CONFIG
    args = sys.argv[1:]
    if args:
        candidate = args[-1] if not args[-1].startswith("-") else None
        if "--config" in args:
            idx = args.index("--config")
            if idx + 1 < len(args):
                candidate = args[idx + 1]
        if candidate:
            config_path = Path(candidate)

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        print(f"Creating default config at: {config_path}", file=sys.stderr)
        # Create a minimal default
        default = {"server_name": "custom-server", "tools": []}
        config_path.write_text(json.dumps(default, indent=2), encoding="utf-8")

    server = build_server(config_path)
    server.run(transport="stdio")
