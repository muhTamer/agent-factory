# app/runtime/policy_pack.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PolicyPack:
    name: str = "default"
    version: str = "0"
    # Minimal, safe to start with:
    blocked_phrases: List[str] = field(default_factory=list)
    max_query_chars: int = 4000
    allowed_tools: Optional[List[str]] = None  # None means "no restriction"
    pii_redaction: bool = False  # placeholder for later

    @staticmethod
    def load(path: str | Path) -> "PolicyPack":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return PolicyPack(
            name=data.get("name", "default"),
            version=str(data.get("version", "0")),
            blocked_phrases=list(data.get("blocked_phrases", [])),
            max_query_chars=int(data.get("max_query_chars", 4000)),
            allowed_tools=data.get("allowed_tools"),
            pii_redaction=bool(data.get("pii_redaction", False)),
        )
