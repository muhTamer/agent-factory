# app/runtime/service.py
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.generator.generate_agent import generate_agent
from app.runtime.registry import AgentRegistry
from app.runtime.router import LLMRouter
from app.runtime.spine import RuntimeSpine
from app.runtime.routing import DefaultRouter
from app.runtime.router_adapter import LLMRouterAdapter
from app.runtime.policy_pack import PolicyPack
from app.runtime.policy_guardrails import PolicyGuardrails
from app.orchestration.performance_store import PerformanceStore
from app.orchestration.aop_coordinator import AOPCoordinator
from app.runtime.memory import ConversationMemory

router: LLMRouter | None = None
spine: RuntimeSpine | None = None

app = FastAPI(title="Agent Factory Runtime", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

THREAD_CTX: dict[str, dict] = {}

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../agent-factory
policy_path = REPO_ROOT / ".factory" / "policy_pack.json"

# Global registry and state
registry = AgentRegistry()
FACTORY_SPEC_PATH = REPO_ROOT / ".factory" / "factory_spec.json"

pack = PolicyPack.load(policy_path) if policy_path.exists() else PolicyPack()
guardrails = PolicyGuardrails(pack)

spine = RuntimeSpine(registry=registry, router=router, guardrails=guardrails)


# ---------- Models ----------
class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None
    request_id: str | None = None
    context: dict | None = None


# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    global router, spine

    if not FACTORY_SPEC_PATH.exists():
        raise RuntimeError(f"Factory spec not found at {FACTORY_SPEC_PATH}")

    spec = json.loads(FACTORY_SPEC_PATH.read_text(encoding="utf-8"))
    print(f"[BOOT] Loading spec with {len(spec.get('agents', []))} agents...")

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
            registry.register(a_id, agent)
            print(f"[BOOT] Agent ready: {a_id}")
        else:
            print(f"[BOOT] Skipping unrecognized type {a_type} ({a_id})")

    llm_router = LLMRouter(registry=registry)
    router = LLMRouterAdapter(llm_router) if registry.all_ids() else DefaultRouter(registry)

    # AOP coordinator (hierarchical delegation for multi-intent queries)
    perf_store = PerformanceStore()
    aop = AOPCoordinator(registry=registry, performance_store=perf_store)

    memory = ConversationMemory()

    spine = RuntimeSpine(
        registry=registry, router=router, guardrails=guardrails, aop_coordinator=aop, memory=memory
    )

    print(f"[BOOT] All agents loaded: {registry.all_ids()}")

    print(
        f"[POLICY] blocked_phrases={getattr(pack, 'blocked_phrases', None)} policy_path={policy_path.resolve()}"
    )
    print(f"[SPINE] guardrails={type(spine.guardrails).__name__}")


# ---------- Routes ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "agents": registry.all_meta(),
        "dry_run": True,
        "request_id": str(uuid.uuid4()),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query text required.")
    if spine is None:
        raise HTTPException(status_code=500, detail="Runtime spine not initialized.")

    thread_id = req.thread_id or str(uuid.uuid4())
    ctx = THREAD_CTX.get(thread_id, {})
    ctx.update(req.context or {})
    ctx["thread_id"] = thread_id

    resp = spine.handle_chat(q, request_id=req.request_id, context=ctx)

    THREAD_CTX[thread_id] = ctx
    resp["thread_id"] = thread_id
    return resp
