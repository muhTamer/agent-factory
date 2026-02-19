# app/runtime/tools/adapters/http.py
"""
HttpTool — calls a customer-provided HTTP endpoint as a workflow tool.

Config example (in tools_config.json):
    {
        "name": "initiate_refund",
        "type": "http",
        "url": "https://erp.customer.com/api/refunds",
        "method": "POST",
        "headers": {
            "Authorization": "Bearer ${REFUND_API_KEY}",
            "Content-Type": "application/json"
        },
        "timeout": 15,
        "slot_map": {
            "refundId": "refund_id",
            "status":   "refund_status"
        }
    }

Behaviour:
- Sends the current FSM `slots` dict as a JSON body (POST/PUT/PATCH)
  or as query params (GET).
- ${ENV_VAR} tokens in header values are expanded from os.environ.
- The JSON response is parsed and slot_map (if provided) renames keys
  before they are merged back into the FSM.  Without slot_map, the
  full response dict is returned as-is.
- On HTTP errors, raises WorkflowExecutionError so the FSM can route
  to an escalation state.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from app.runtime.tools.interface import ITool


class HttpTool(ITool):
    """Calls a customer HTTP endpoint and returns JSON response as slot updates."""

    def __init__(
        self,
        name: str,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        slot_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.url = url
        self.method = method.upper()
        self._headers = headers or {}
        self.timeout = timeout
        self.slot_map = slot_map or {}  # {response_key: slot_key}

    # ------------------------------------------------------------------
    # ITool
    # ------------------------------------------------------------------
    def execute(self, slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._resolve_headers()
        response_data = self._call(slots, headers)
        return self._map_response(response_data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_headers(self) -> Dict[str, str]:
        """Expand ${ENV_VAR} placeholders in header values."""
        resolved: Dict[str, str] = {}
        for key, value in self._headers.items():
            resolved[key] = os.path.expandvars(value)
        return resolved

    def _call(self, slots: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Perform the HTTP request using stdlib urllib (no extra deps)."""
        body: Optional[bytes] = None
        url = self.url

        if self.method in ("POST", "PUT", "PATCH"):
            payload = {k: v for k, v in slots.items() if v is not None}
            body = json.dumps(payload).encode("utf-8")
            if "Content-Type" not in headers:
                headers["Content-Type"] = "application/json"
        elif self.method == "GET":
            params = {k: str(v) for k, v in slots.items() if v is not None}
            if params:
                url = url + "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(url, data=body, headers=headers, method=self.method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"HttpTool '{self.name}' got HTTP {exc.code} from {self.url}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"HttpTool '{self.name}' could not reach {self.url}: {exc.reason}"
            ) from exc

    def _map_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply slot_map renaming to the response dict."""
        if not self.slot_map:
            return data
        result: Dict[str, Any] = {}
        for resp_key, slot_key in self.slot_map.items():
            if resp_key in data:
                result[slot_key] = data[resp_key]
        # Pass through any keys not in slot_map
        for key, val in data.items():
            if key not in self.slot_map:
                result[key] = val
        return result

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "http",
            "url": self.url,
            "method": self.method,
        }
