from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class PolicyPack:
    name: str = "default"
    version: str = "0"
    max_query_chars: int = 4000

    intent_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    route_to_intent: Dict[str, str] = field(default_factory=dict)

    blocked_phrases: list[str] = field(default_factory=list)
    pii_redaction: bool = False

    @staticmethod
    def load(path: str | Path) -> "PolicyPack":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))

        return PolicyPack(
            name=data.get("name", "default"),
            version=str(data.get("version", "0")),
            max_query_chars=int(data.get("max_query_chars", 4000)),
            intent_rules=dict(data.get("intent_rules", {})),
            route_to_intent=dict(data.get("route_to_intent", {})),
            blocked_phrases=list(data.get("blocked_phrases", [])),
            pii_redaction=bool(data.get("pii_redaction", False)),
        )
