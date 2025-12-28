# app/shared/schemas/validate.py
from __future__ import annotations
from pathlib import Path
import json
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).parent


def _load_schema(filename: str) -> dict:
    path = SCHEMA_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


_WORKFLOW_SCHEMA = None
_BLUEPRINT_SCHEMA = None


def validate_workflow_spec(spec: dict) -> list[str]:
    global _WORKFLOW_SCHEMA
    if _WORKFLOW_SCHEMA is None:
        _WORKFLOW_SCHEMA = _load_schema("workflow_spec.schema.json")
    v = Draft202012Validator(_WORKFLOW_SCHEMA)
    errors = sorted(v.iter_errors(spec), key=lambda e: e.path)
    return [f"path={'/'.join(map(str, e.path))} msg={e.message}" for e in errors]


def validate_agent_blueprint(bp: dict) -> list[str]:
    global _BLUEPRINT_SCHEMA
    if _BLUEPRINT_SCHEMA is None:
        _BLUEPRINT_SCHEMA = _load_schema("agent_blueprint.schema.json")
    v = Draft202012Validator(_BLUEPRINT_SCHEMA)
    errors = sorted(v.iter_errors(bp), key=lambda e: e.path)
    return [f"path={'/'.join(map(str, e.path))} msg={e.message}" for e in errors]
