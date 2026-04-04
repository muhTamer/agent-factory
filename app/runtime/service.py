# app/runtime/service.py
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.generator.generate_agent import generate_agent
from app.runtime.registry import AgentRegistry
from app.runtime.router import LLMRouter
from app.runtime.spine import RuntimeSpine
from app.runtime.routing import DefaultRouter
from app.runtime.router_adapter import LLMRouterAdapter
from app.runtime.policy_pack import PolicyPack
from app.runtime.governance_config import GovernanceConfig, GovernanceLevel
from app.runtime.governance_guardrails import GovernanceAwareGuardrails
from app.runtime.tools import DEFAULT_REGISTRY, build_registry
from app.runtime.tools.registry import ToolRegistry
from app.orchestration.performance_store import PerformanceStore
from app.orchestration.aop_coordinator import AOPCoordinator
from app.runtime.memory import ConversationMemory
from app.runtime.rate_limiter import (
    RateLimitMiddleware,
    record_llm_call,
    get_session_usage,
    get_daily_usage,
)

router: LLMRouter | None = None
spine: RuntimeSpine | None = None
tool_registry: ToolRegistry = DEFAULT_REGISTRY  # replaced at startup if config found

app = FastAPI(title="Agent Factory Runtime", version="1.0")

# CORS: allow local dev + Azure Container Apps frontend
_cors_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://agent-factory-frontend.politedune-9f1beae9.westeurope.azurecontainerapps.io",
]
# Allow extra origins via env var (comma-separated)
_extra_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _extra_origins:
    _cors_origins.extend([o.strip() for o in _extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)

THREAD_CTX: dict[str, dict] = {}

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../agent-factory
policy_path = REPO_ROOT / ".factory" / "policy_pack.json"

# Global registry and state
registry = AgentRegistry()
FACTORY_SPEC_PATH = REPO_ROOT / ".factory" / "factory_spec.json"
TOOLS_CONFIG_PATH = REPO_ROOT / ".factory" / "tools_config.json"

pack = PolicyPack.load(policy_path) if policy_path.exists() else PolicyPack()

# RQ3: Governance level from env var (default: medium = current behaviour)
_gov_level_str = os.getenv("AF_GOVERNANCE_LEVEL", "medium").lower()
_gov_level = GovernanceLevel(_gov_level_str)
_gov_config = GovernanceConfig.for_level(_gov_level)
guardrails = GovernanceAwareGuardrails(pack, _gov_config)
print(
    f"[GOVERNANCE] level={_gov_level.value} pre_checks={_gov_config.pre_checks_enabled} "
    f"hallucination={_gov_config.hallucination_detection} tone={_gov_config.tone_control_enabled}"
)

spine = RuntimeSpine(registry=registry, router=router, guardrails=guardrails)


# ---------- Models ----------
class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None
    request_id: str | None = None
    context: dict | None = None


class ToolTestRequest(BaseModel):
    slots: dict = {}
    context: dict = {}


# ---------- Spec loading (reusable for startup + /reload) ----------
def _load_spec_from_path(spec_path: Path) -> bool:
    """Load agents from a factory spec file. Returns True if successful."""
    global router, spine, tool_registry

    if not spec_path.exists():
        return False

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    print(f"[BOOT] Loading spec with {len(spec.get('agents', []))} agents...")

    # Clear existing agents
    registry._agents.clear()
    registry._meta.clear()

    for agent_spec in spec.get("agents", []):
        a_id = agent_spec["id"]
        a_type = agent_spec.get("type")

        if a_type == "guardrails":
            continue

        if a_type == "autogen":
            gen_path = generate_agent(agent_spec)

            gen_dir = gen_path if isinstance(gen_path, Path) else Path(gen_path)
            if gen_dir.suffix == ".py":
                gen_dir = gen_dir.parent

            agent = registry.import_generated_agent(a_id, gen_dir)
            agent.load(agent_spec)
            registry.register(a_id, agent, meta=agent_spec.get("blueprint_meta"))
            print(f"[BOOT] Agent ready: {a_id}")
        else:
            print(f"[BOOT] Skipping unrecognized type {a_type} ({a_id})")

    llm_router = LLMRouter(registry=registry)
    router = (
        LLMRouterAdapter(llm_router) if registry.all_ids() else DefaultRouter(registry)
    )

    # Conversation memory (shared across spine + AOP)
    memory = ConversationMemory()

    # AOP coordinator (hierarchical delegation for multi-intent queries)
    perf_store = PerformanceStore()
    aop = AOPCoordinator(registry=registry, performance_store=perf_store, memory=memory)

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=guardrails,
        aop_coordinator=aop,
        memory=memory,
    )

    print(f"[BOOT] All agents loaded: {registry.all_ids()}")
    print(
        f"[POLICY] blocked_phrases={getattr(pack, 'blocked_phrases', None)} "
        f"policy_path={policy_path.resolve()}"
    )
    print(f"[SPINE] guardrails={type(spine.guardrails).__name__}")

    # Load customer tool overrides
    if TOOLS_CONFIG_PATH.exists():
        tools_config = json.loads(TOOLS_CONFIG_PATH.read_text(encoding="utf-8"))
        tool_registry = build_registry(
            config=tools_config.get("tools", []),
            mcp_servers=tools_config.get("mcp_servers", []),
        )
        print(f"[TOOLS] Loaded customer config: {tool_registry.all_names()}")
    else:
        tool_registry = DEFAULT_REGISTRY
        print(
            f"[TOOLS] No tools_config.json found — using stubs: {tool_registry.all_names()}"
        )

    return True


def _load_spec_from_json(spec_data: dict) -> bool:
    """Load agents from a spec dict (received via /reload API)."""
    # Write spec to disk so generate_agent can find it
    FACTORY_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    FACTORY_SPEC_PATH.write_text(
        json.dumps(spec_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return _load_spec_from_path(FACTORY_SPEC_PATH)


# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    if FACTORY_SPEC_PATH.exists():
        _load_spec_from_path(FACTORY_SPEC_PATH)
    else:
        print(
            "[BOOT] No factory spec found — running in waiting mode. "
            "Complete onboarding via the concierge to load agents."
        )


# ---------- Shutdown ----------
@app.on_event("shutdown")
def shutdown_event():
    """Clean up MCP server connections on shutdown."""
    if hasattr(tool_registry, "shutdown"):
        tool_registry.shutdown()
        print("[TOOLS] MCP servers disconnected.")


# ---------- Routes ----------
@app.get("/health")
def health():
    agents_meta = registry.all_meta()
    return {
        "status": "ok" if agents_meta else "waiting",
        "agents": agents_meta,
        "dry_run": True,
        "request_id": str(uuid.uuid4()),
    }


class ReloadRequest(BaseModel):
    spec: dict | None = None
    concierge_url: str | None = None


@app.post("/reload")
def reload_spec(req: ReloadRequest):
    """
    Hot-reload agents from a factory spec.
    Either pass the spec directly or provide a concierge_url to fetch it from.
    """
    if req.spec:
        ok = _load_spec_from_json(req.spec)
        if ok:
            return {"status": "reloaded", "agents": registry.all_ids()}
        return {"status": "error", "message": "Failed to load spec"}

    if req.concierge_url:
        import requests as http_requests

        try:
            r = http_requests.get(f"{req.concierge_url}/concierge/spec", timeout=10)
            r.raise_for_status()
            data = r.json()
            if "spec" not in data or not data["spec"]:
                return {
                    "status": "error",
                    "message": "No spec available from concierge",
                }
            ok = _load_spec_from_json(data["spec"])
            if ok:
                return {"status": "reloaded", "agents": registry.all_ids()}
            return {"status": "error", "message": "Failed to load fetched spec"}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch spec: {exc}")

    if FACTORY_SPEC_PATH.exists():
        ok = _load_spec_from_path(FACTORY_SPEC_PATH)
        if ok:
            return {"status": "reloaded", "agents": registry.all_ids()}
        return {"status": "error", "message": "Failed to reload from disk"}

    raise HTTPException(
        status_code=400,
        detail="Provide 'spec' (JSON) or 'concierge_url' to reload from.",
    )


@app.get("/tools")
def list_tools():
    """List all registered tools and their descriptions."""
    return {
        "tools": tool_registry.describe_all(),
        "count": len(tool_registry.all_names()),
        "config_file": str(TOOLS_CONFIG_PATH) if TOOLS_CONFIG_PATH.exists() else None,
    }


@app.post("/tools/{name}/test")
def test_tool(name: str, req: ToolTestRequest):
    """
    Call a specific tool with the provided slots and context.
    Useful for verifying tool behaviour before deploying with real backends.
    """
    tool = tool_registry.get(name)
    if tool is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{name}' not found. Available: {tool_registry.all_names()}",
        )
    try:
        result = tool.execute(req.slots, req.context)
        return {"tool": name, "slots_in": req.slots, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------- Guardrail admin endpoints ----------


class GuardrailToggleRequest(BaseModel):
    enabled: bool


@app.get("/guardrails")
def list_guardrails():
    """List all guardrail rules with their current enabled state."""
    return {
        "rules": pack.rules_summary(),
        "transaction_slot_keys": pack.transaction_slot_keys,
        "policy_pack": pack.name,
        "version": pack.version,
    }


@app.patch("/guardrails/{rule_id}")
def toggle_guardrail(rule_id: str, req: GuardrailToggleRequest):
    """Enable or disable a specific guardrail rule at runtime."""
    success = pack.set_rule_enabled(rule_id, req.enabled)
    if not success:
        available = [r.id for r in pack.guardrail_rules]
        raise HTTPException(
            status_code=404,
            detail=f"Rule '{rule_id}' not found. Available: {available}",
        )

    # Persist change to disk so it survives restarts
    if policy_path.exists():
        pack.save(policy_path)

    # Rebuild guardrails so the inner PolicyGuardrails picks up the change
    _rebuild_guardrails()

    rule = pack.get_rule(rule_id)
    return {
        "rule_id": rule_id,
        "enabled": req.enabled,
        "rule": rule.to_dict() if rule else None,
    }


def _rebuild_guardrails():
    """Rebuild the guardrails stack after a rule toggle."""
    global guardrails
    from app.runtime.governance_guardrails import GovernanceAwareGuardrails

    guardrails = GovernanceAwareGuardrails(pack, _gov_config)
    if spine is not None:
        spine.guardrails = guardrails


# ---------- Solvability estimator admin endpoints ----------


class EstimatorSwitchRequest(BaseModel):
    kind: str  # "neural" or "tfidf"


@app.get("/solvability-estimator")
def get_estimator():
    """Return the currently active solvability estimator kind."""
    if spine is None or spine.aop_coordinator is None:
        raise HTTPException(status_code=503, detail="AOP coordinator not initialized.")
    return {
        "kind": spine.aop_coordinator.active_estimator_kind,
        "options": ["neural", "tfidf"],
    }


@app.patch("/solvability-estimator")
def switch_estimator(req: EstimatorSwitchRequest):
    """Hot-swap the solvability estimator at runtime (neural ↔ tfidf)."""
    if spine is None or spine.aop_coordinator is None:
        raise HTTPException(status_code=503, detail="AOP coordinator not initialized.")
    if req.kind not in ("neural", "tfidf"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid kind '{req.kind}'. Use 'neural' or 'tfidf'.",
        )
    try:
        active = spine.aop_coordinator.swap_estimator(req.kind)
        return {"kind": active, "message": f"Switched to {active} estimator."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query text required.")
    if spine is None:
        raise HTTPException(status_code=500, detail="Runtime spine not initialized.")

    thread_id = req.thread_id or str(uuid.uuid4())
    session_id = f"thread:{thread_id}"

    # ── Enforce LLM usage limits before calling the model ───
    usage = record_llm_call(session_id)

    ctx = THREAD_CTX.get(thread_id, {})
    ctx.update(req.context or {})
    ctx["thread_id"] = thread_id

    resp = spine.handle_chat(q, request_id=req.request_id, context=ctx)

    THREAD_CTX[thread_id] = ctx
    resp["thread_id"] = thread_id
    resp["usage"] = usage
    return resp


# ---------- Usage monitoring endpoints ----------


@app.get("/usage/session/{thread_id}", tags=["Usage"])
def session_usage(thread_id: str):
    """Return LLM usage stats for a specific session."""
    return get_session_usage(f"thread:{thread_id}")


@app.get("/usage/daily", tags=["Usage"])
def daily_usage():
    """Return global daily LLM usage stats (for monitoring/alerting)."""
    return get_daily_usage()
