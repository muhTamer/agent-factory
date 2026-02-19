# app/runtime/tools/adapters/sql.py
"""
SqlTool — runs a parameterised SQL query against a customer database.

Config example (in tools_config.json):
    {
        "name": "lookup_customer",
        "type": "sql",
        "dsn": "sqlite:///data/customers.db",
        "query": "SELECT account_status, kyc_status FROM customers WHERE id = :customer_id",
        "slot_map": {
            "account_status": "account_status",
            "kyc_status":     "kyc_status"
        }
    }

DSN formats:
    sqlite:///relative/path.db        — SQLite via stdlib (no extra deps)
    sqlite:////absolute/path/db.db    — SQLite absolute path
    postgresql://user:pass@host/db    — requires sqlalchemy + psycopg2
    mysql+pymysql://user:pass@host/db — requires sqlalchemy + pymysql
    any other SQLAlchemy-compatible DSN

Behaviour:
- Named parameters in `query` (:param_name) are resolved from FSM slots.
- The first row of the result set is returned as a dict.
- slot_map renames result columns to slot keys (optional; pass-through if absent).
- If the query returns no rows, returns an empty dict (no slot updates).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.runtime.tools.interface import ITool


class SqlTool(ITool):
    """Runs a parameterised SQL query and returns the first result row as slot updates."""

    def __init__(
        self,
        name: str,
        dsn: str,
        query: str,
        slot_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.dsn = dsn
        self.query = query
        self.slot_map = slot_map or {}

    # ------------------------------------------------------------------
    # ITool
    # ------------------------------------------------------------------
    def execute(self, slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = self._extract_params(slots)
        row = self._run_query(params)
        return self._map_row(row)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _extract_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Pull only the named params referenced in the query from slots."""
        referenced = set(re.findall(r":([a-zA-Z_][a-zA-Z0-9_]*)", self.query))
        return {k: slots.get(k) for k in referenced}

    def _run_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the query and return the first row as a dict (or {} if no rows)."""
        if self.dsn.startswith("sqlite"):
            return self._run_sqlite(params)
        return self._run_sqlalchemy(params)

    def _run_sqlite(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Use stdlib sqlite3 — no extra dependencies required."""
        import sqlite3

        # Parse path from dsn: sqlite:///path or sqlite:////abs/path
        path = re.sub(r"^sqlite:///", "", self.dsn)
        # Convert named params (:name) to ?-style for sqlite3, keeping a param list
        positional_query, ordered_params = self._named_to_positional(params)

        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(positional_query, ordered_params)
            row = cur.fetchone()

        if row is None:
            return {}
        return dict(row)

    def _run_sqlalchemy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Use SQLAlchemy for non-SQLite DSNs (optional dependency)."""
        try:
            import sqlalchemy  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "SqlTool requires 'sqlalchemy' for non-SQLite databases. "
                "Install it with: pip install sqlalchemy"
            ) from exc

        engine = sqlalchemy.create_engine(self.dsn)
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(self.query), params)
            row = result.mappings().fetchone()

        return dict(row) if row else {}

    def _named_to_positional(self, params: Dict[str, Any]):
        """
        Convert :name-style placeholders to ?-style for sqlite3,
        returning the rewritten query and an ordered list of values.
        """
        ordered: list = []

        def replace(match):
            ordered.append(params.get(match.group(1)))
            return "?"

        positional_query = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", replace, self.query)
        return positional_query, ordered

    def _map_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Apply slot_map renaming; pass through unmapped columns unchanged."""
        if not self.slot_map:
            return row
        result: Dict[str, Any] = {}
        for col, slot_key in self.slot_map.items():
            if col in row:
                result[slot_key] = row[col]
        for col, val in row.items():
            if col not in self.slot_map:
                result[col] = val
        return result

    def describe(self) -> Dict[str, Any]:
        # Omit DSN from public describe (may contain credentials)
        return {
            "name": self.name,
            "type": "sql",
            "query": self.query,
        }
