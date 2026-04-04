# app/runtime/service.py
"""
Agent Factory Runtime — multi-tenant.

Each tenant gets an isolated set of: AgentRegistry, Router, RuntimeSpine.
Tenants are loaded on-demand when the concierge triggers /reload with a spec,
and evicted (LRU) when the cap is reached.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import AuthUser, get_current_user, get_optional_user
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

app = FastAPI(title="Agent Factory Runtime", version="1.0")

# CORS: allow local dev + Azure Container Apps frontend
_cors_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://agent-factory-frontend.politedune-9f1beae9.westeurope.azurecontainerapps.io",
]
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

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_TENANTS = 100

# Shared governance config (same for all tenants)
policy_path = REPO_ROOT / ".factory" / "policy_pack.json"
pack = PolicyPack.load(policy_path) if policy_path.exists() else PolicyPack()

_gov_level_str = os.getenv("AF_GOVERNANCE_LEVEL", "medium").lower()
_gov_level = GovernanceLevel(_gov_level_str)
_gov_config = GovernanceConfig.for_level(_gov_level)
guardrails = GovernanceAwareGuardrails(pack, _gov_config)
print(
    f"[GOVERNANCE] level={_gov_level.value} pre_checks={_gov_config.pre_checks_enabled} "
    f"hallucination={_gov_config.hallucination_detection} tone={_gov_config.tone_control_enabled}"
)


# ---------------------------------------------------------------------------
# Per-tenant runtime environment
# ---------------------------------------------------------------------------
@dataclass
class TenantRuntime:
    """Isolated runtime environment for one tenant."""

    tenant_id: str
    registry: AgentRegistry = field(default_factory=AgentRegistry)
    router: LLMRouter | None = None
    spine: RuntimeSpine | None = None
    tool_registry: ToolRegistry = field(default_factory=lambda: DEFAULT_REGISTRY)
    thread_ctx: dict[str, dict] = field(default_factory=dict)
    loaded: bool = False


# LRU ordered dict of tenant runtimes
_tenants: OrderedDict[str, TenantRuntime] = OrderedDict()


def _get_tenant(tenant_id: str) -> TenantRuntime:
    """Get or create a TenantRuntime, evicting oldest if over limit."""
    if tenant_id in _tenants:
        _tenants.move_to_end(tenant_id)
        return _tenants[tenant_id]
    while len(_tenants) >= MAX_TENANTS:
        evicted_id, _ = _tenants.popitem(last=False)
        print(f"[TENANT] Evicted idle tenant {evicted_id}")
    tr = TenantRuntime(tenant_id=tenant_id)
    # Create a spine in "waiting" mode (no agents loaded yet)
    tr.spine = RuntimeSpine(
        registry=tr.registry, router=tr.router, guardrails=guardrails
    )
    _tenants[tenant_id] = tr
    return tr


def _load_spec_for_tenant(tr: TenantRuntime, spec: dict) -> bool:
    """Load agents from a spec dict into the tenant's runtime."""
    agents = spec.get("agents", [])
    print(f"[TENANT:{tr.tenant_id}] Loading spec with {len(agents)} agents...")

    # Clear existing agents
    tr.registry._agents.clear()
    tr.registry._meta.clear()

    for agent_spec in agents:
        a_id = agent_spec["id"]
        a_type = agent_spec.get("type")

        if a_type == "guardrails":
            continue

        if a_type == "autogen":
            gen_path = generate_agent(agent_spec)
            gen_dir = gen_path if isinstance(gen_path, Path) else Path(gen_path)
            if gen_dir.suffix == ".py":
                gen_dir = gen_dir.parent

            agent = tr.registry.import_generated_agent(a_id, gen_dir)
            agent.load(agent_spec)
            tr.registry.register(a_id, agent, meta=agent_spec.get("blueprint_meta"))
            print(f"[TENANT:{tr.tenant_id}] Agent ready: {a_id}")
        else:
            print(
                f"[TENANT:{tr.tenant_id}] Skipping unrecognized type {a_type} ({a_id})"
            )

    llm_router = LLMRouter(registry=tr.registry)
    tr.router = (
        LLMRouterAdapter(llm_router)
        if tr.registry.all_ids()
        else DefaultRouter(tr.registry)
    )

    memory = ConversationMemory()
    perf_store = PerformanceStore()
    aop = AOPCoordinator(
        registry=tr.registry, performance_store=perf_store, memory=memory
    )

    tr.spine = RuntimeSpine(
        registry=tr.registry,
        router=tr.router,
        guardrails=guardrails,
        aop_coordinator=aop,
        memory=memory,
    )
    tr.loaded = True

    print(f"[TENANT:{tr.tenant_id}] All agents loaded: {tr.registry.all_ids()}")
    return True


# ---------------------------------------------------------------------------
# Legacy / dev-mode: load default spec at startup for local single-user dev
# ---------------------------------------------------------------------------
FACTORY_SPEC_PATH = REPO_ROOT / ".factory" / "factory_spec.json"
TOOLS_CONFIG_PATH = REPO_ROOT / ".factory" / "tools_config.json"

# Default "dev" tenant (used when AUTH_ENABLED=false)
_DEV_TENANT_ID = "dev"


@app.on_event("startup")
def startup_event():
    if FACTORY_SPEC_PATH.exists():
        spec = json.loads(FACTORY_SPEC_PATH.read_text(encoding="utf-8"))
        tr = _get_tenant(_DEV_TENANT_ID)
        _load_spec_for_tenant(tr, spec)

        # Load tools config
        if TOOLS_CONFIG_PATH.exists():
            tools_config = json.loads(TOOLS_CONFIG_PATH.read_text(encoding="utf-8"))
            tr.tool_registry = build_registry(
                config=tools_config.get("tools", []),
                mcp_servers=tools_config.get("mcp_servers", []),
            )
            print(f"[TOOLS] Loaded customer config: {tr.tool_registry.all_names()}")
    else:
        print(
            "[BOOT] No factory spec found — running in waiting mode. "
            "Complete onboarding via the concierge to load agents."
        )


@app.on_event("shutdown")
def shutdown_event():
    for tr in _tenants.values():
        if hasattr(tr.tool_registry, "shutdown"):
            tr.tool_registry.shutdown()
    print("[TOOLS] MCP servers disconnected.")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None
    request_id: str | None = None
    context: dict | None = None


class ToolTestRequest(BaseModel):
    slots: dict = {}
    context: dict = {}


class ReloadRequest(BaseModel):
    spec: dict | None = None
    concierge_url: str | None = None
    tenant_id: str | None = None


class GuardrailToggleRequest(BaseModel):
    enabled: bool


class EstimatorSwitchRequest(BaseModel):
    kind: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health(user: AuthUser | None = Depends(get_optional_user)):
    """Health check. When authenticated, returns tenant-specific agent info."""
    if user:
        tr = _get_tenant(user.tenant_id)
        agents_meta = tr.registry.all_meta()
    else:
        # Unauthenticated: check if any tenant has agents (backward compat)
        if _DEV_TENANT_ID in _tenants:
            agents_meta = _tenants[_DEV_TENANT_ID].registry.all_meta()
        else:
            agents_meta = {}

    return {
        "status": "ok" if agents_meta else "waiting",
        "agents": agents_meta,
        "dry_run": True,
        "request_id": str(uuid.uuid4()),
    }


@app.post("/reload")
def reload_spec(req: ReloadRequest):
    """
    Hot-reload agents from a factory spec for a specific tenant.
    Called by the concierge after deploy.
    """
    tenant_id = req.tenant_id or _DEV_TENANT_ID

    if req.spec:
        tr = _get_tenant(tenant_id)
        ok = _load_spec_for_tenant(tr, req.spec)
        if ok:
            return {"status": "reloaded", "agents": tr.registry.all_ids()}
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
            tr = _get_tenant(tenant_id)
            ok = _load_spec_for_tenant(tr, data["spec"])
            if ok:
                return {"status": "reloaded", "agents": tr.registry.all_ids()}
            return {"status": "error", "message": "Failed to load fetched spec"}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch spec: {exc}")

    # Fallback: try loading from disk (dev mode)
    if FACTORY_SPEC_PATH.exists():
        spec = json.loads(FACTORY_SPEC_PATH.read_text(encoding="utf-8"))
        tr = _get_tenant(tenant_id)
        ok = _load_spec_for_tenant(tr, spec)
        if ok:
            return {"status": "reloaded", "agents": tr.registry.all_ids()}
        return {"status": "error", "message": "Failed to reload from disk"}

    raise HTTPException(
        status_code=400,
        detail="Provide 'spec' (JSON) or 'concierge_url' to reload from.",
    )


@app.post("/chat")
def chat(req: ChatRequest, user: AuthUser = Depends(get_current_user)):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query text required.")

    tr = _get_tenant(user.tenant_id)
    if tr.spine is None or not tr.loaded:
        raise HTTPException(
            status_code=503,
            detail="Your agents are not loaded yet. Complete onboarding first.",
        )

    thread_id = req.thread_id or str(uuid.uuid4())
    session_id = f"tenant:{user.tenant_id}:thread:{thread_id}"

    # Enforce LLM usage limits
    usage = record_llm_call(session_id)

    ctx = tr.thread_ctx.get(thread_id, {})
    ctx.update(req.context or {})
    ctx["thread_id"] = thread_id
    ctx["tenant_id"] = user.tenant_id

    resp = tr.spine.handle_chat(q, request_id=req.request_id, context=ctx)

    tr.thread_ctx[thread_id] = ctx
    resp["thread_id"] = thread_id
    resp["usage"] = usage
    return resp


@app.get("/tools")
def list_tools(user: AuthUser = Depends(get_current_user)):
    tr = _get_tenant(user.tenant_id)
    return {
        "tools": tr.tool_registry.describe_all(),
        "count": len(tr.tool_registry.all_names()),
    }


@app.post("/tools/{name}/test")
def test_tool(
    name: str, req: ToolTestRequest, user: AuthUser = Depends(get_current_user)
):
    tr = _get_tenant(user.tenant_id)
    tool = tr.tool_registry.get(name)
    if tool is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{name}' not found. Available: {tr.tool_registry.all_names()}",
        )
    try:
        result = tool.execute(req.slots, req.context)
        return {"tool": name, "slots_in": req.slots, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------- Guardrail admin endpoints ----------


@app.get("/guardrails")
def list_guardrails():
    return {
        "rules": pack.rules_summary(),
        "transaction_slot_keys": pack.transaction_slot_keys,
        "policy_pack": pack.name,
        "version": pack.version,
    }


@app.patch("/guardrails/{rule_id}")
def toggle_guardrail(rule_id: str, req: GuardrailToggleRequest):
    success = pack.set_rule_enabled(rule_id, req.enabled)
    if not success:
        available = [r.id for r in pack.guardrail_rules]
        raise HTTPException(
            status_code=404,
            detail=f"Rule '{rule_id}' not found. Available: {available}",
        )

    if policy_path.exists():
        pack.save(policy_path)

    _rebuild_guardrails()

    rule = pack.get_rule(rule_id)
    return {
        "rule_id": rule_id,
        "enabled": req.enabled,
        "rule": rule.to_dict() if rule else None,
    }


def _rebuild_guardrails():
    global guardrails
    guardrails = GovernanceAwareGuardrails(pack, _gov_config)
    for tr in _tenants.values():
        if tr.spine is not None:
            tr.spine.guardrails = guardrails


# ---------- Solvability estimator admin endpoints ----------


@app.get("/solvability-estimator")
def get_estimator(user: AuthUser = Depends(get_current_user)):
    tr = _get_tenant(user.tenant_id)
    if tr.spine is None or tr.spine.aop_coordinator is None:
        raise HTTPException(status_code=503, detail="AOP coordinator not initialized.")
    return {
        "kind": tr.spine.aop_coordinator.active_estimator_kind,
        "options": ["neural", "tfidf"],
    }


@app.patch("/solvability-estimator")
def switch_estimator(
    req: EstimatorSwitchRequest, user: AuthUser = Depends(get_current_user)
):
    tr = _get_tenant(user.tenant_id)
    if tr.spine is None or tr.spine.aop_coordinator is None:
        raise HTTPException(status_code=503, detail="AOP coordinator not initialized.")
    if req.kind not in ("neural", "tfidf"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid kind '{req.kind}'. Use 'neural' or 'tfidf'.",
        )
    try:
        active = tr.spine.aop_coordinator.swap_estimator(req.kind)
        return {"kind": active, "message": f"Switched to {active} estimator."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------- Usage monitoring endpoints ----------


@app.get("/usage/session/{thread_id}", tags=["Usage"])
def session_usage(thread_id: str, user: AuthUser = Depends(get_current_user)):
    return get_session_usage(f"tenant:{user.tenant_id}:thread:{thread_id}")


@app.get("/usage/daily", tags=["Usage"])
def daily_usage():
    return get_daily_usage()
