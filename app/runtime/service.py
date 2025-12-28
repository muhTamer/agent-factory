# app/runtime/service.py
from __future__ import annotations
import json
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.runtime.registry import AgentRegistry
from app.generator.generate_agent import generate_agent
from app.runtime.router import LLMRouter

router: LLMRouter | None = None

app = FastAPI(title="Agent Factory Runtime", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registry and state
registry = AgentRegistry()
FACTORY_SPEC_PATH = Path(".factory/factory_spec.json")


# ---------- Models ----------
class ChatRequest(BaseModel):
    query: str


# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    global router

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
            from pathlib import Path

            gen_dir = gen_path if isinstance(gen_path, Path) else Path(gen_path)
            if gen_dir.suffix == ".py":
                gen_dir = gen_dir.parent
            agent = registry.import_generated_agent(a_id, gen_dir)
            agent.load(agent_spec)
            registry.register(a_id, agent)
            print(f"[BOOT] Agent ready: {a_id}")
        else:
            print(f"[BOOT] Skipping unrecognized type {a_type} ({a_id})")

    router = LLMRouter(registry=registry)
    print(f"[BOOT] All agents loaded: {registry.all_ids()}")


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

    rid = str(uuid.uuid4())
    print(f"[REQ] {rid}: {q}")

    if router is None:
        # trivial fallback if router failed to init
        ids = registry.all_ids()
        if not ids:
            return {"error": "No agent available.", "request_id": rid}
        aid = ids[0]
        agent = registry.get(aid)
        res = agent.handle({"query": q, "text": q})
        res["agent_id"] = aid
        res["request_id"] = rid
        return res

    # --- LLM-based routing ---
    plan = router.route(q)
    print(f"[ROUTER] plan={plan}")

    results = []
    if plan.strategy == "single":
        cand = plan.candidates[0] if plan.candidates else None
        if not cand:
            return {"error": "Router provided no candidates.", "request_id": rid}
        agent = registry.get(cand.id)
        if not agent:
            return {"error": f"Agent {cand.id} not loaded.", "request_id": rid}
        res = agent.handle({"query": q, "text": q})
        res_score = float(res.get("score", cand.score))
        results.append({"agent_id": cand.id, "score": res_score, "response": res})
    else:  # fanout
        for cand in plan.candidates:
            agent = registry.get(cand.id)
            if not agent:
                continue
            try:
                r = agent.handle({"query": q, "text": q})
                res_score = float(r.get("score", cand.score))
                results.append({"agent_id": cand.id, "score": res_score, "response": r})
            except Exception as e:
                print(f"[ERR] fanout call failed for {cand.id}: {e}")

    if not results:
        return {"error": "No agent produced a response.", "request_id": rid}

    best = max(results, key=lambda x: x["score"])
    resp = best["response"]
    resp["agent_id"] = best["agent_id"]
    resp["score"] = best["score"]
    resp["request_id"] = rid
    resp["router_plan"] = {
        "primary": plan.primary,
        "strategy": plan.strategy,
        "candidates": [{"id": c.id, "score": c.score, "reason": c.reason} for c in plan.candidates],
    }
    return resp
