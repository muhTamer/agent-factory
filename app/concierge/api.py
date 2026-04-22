# app/concierge/api.py
"""
Concierge REST API — wraps ConciergeAgent for the Next.js frontend.
Run with: python -m uvicorn app.concierge.api:app --port 8001

Multi-tenant: each authenticated user gets an isolated workspace and
ConciergeAgent instance keyed by tenant_id.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid as uuid_mod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as http_requests
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import app.llm_client as llm_client
from app.auth import AUTH_ENABLED, AUTH_SECRET, AuthUser, get_current_user
from app.concierge.concierge_agent import ConciergeAgent

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("concierge")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
WORKSPACES_ROOT = REPO_ROOT / ".workspace"

# Runtime backend URL — on Azure this is the backend container app,
# locally it's http://127.0.0.1:808
RUNTIME_BACKEND_URL = os.getenv("RUNTIME_BACKEND_URL", "http://127.0.0.1:808")
MCP_TOOLS_CONFIG = REPO_ROOT / "tests" / "fixtures" / "mcp_tools_config.json"

FINTECH_DATA_FILES = [
    DATA_DIR / "BankFAQs.csv",
    DATA_DIR / "refunds_policy.yaml",
    DATA_DIR / "complaints_policy.yaml",
    DATA_DIR / "accounts_policy.yaml",
]

# Visibility for quickstart preset files: policy YAMLs are internal
# (agent instructions only), FAQs are customer-facing (shareable).
FINTECH_DOC_VISIBILITY: Dict[str, str] = {
    "refunds_policy.yaml": "internal",
    "complaints_policy.yaml": "internal",
    "accounts_policy.yaml": "internal",
    "BankFAQs.csv": "customer_facing",
}

RETAIL_DATA_FILES = [
    DATA_DIR / "RetailFAQs.csv",
    DATA_DIR / "retail_refunds_policy.yaml",
    DATA_DIR / "retail_complaints_policy.yaml",
]

RETAIL_DOC_VISIBILITY: Dict[str, str] = {
    "retail_refunds_policy.yaml": "internal",
    "retail_complaints_policy.yaml": "internal",
    "RetailFAQs.csv": "customer_facing",
}

PREBUILT_DIR = DATA_DIR / "prebuilt"

# Max number of tenant sessions held in memory before evicting the oldest
MAX_TENANTS = 200


def _build_preset_doc_meta(
    workspace: Path, files: list, vertical: str
) -> List[Dict[str, Any]]:
    """Pre-classified document metadata for quickstart presets.
    Skips 6+ LLM calls (~60-90s) since we know exactly what these files are."""
    from app.infer_capabilities import InferCapabilities

    ic = InferCapabilities()
    docs = []
    for src in files:
        p = workspace / src.name
        content_analysis = ic._analyze_document_content(p)
        ext = p.suffix.lower()
        if ext == ".csv":
            doc_type = "faq_kb"
        elif ext in (".yaml", ".yml"):
            doc_type = "policy"
        else:
            doc_type = "other"
        docs.append(
            {
                "name": p.name,
                "path": str(p),
                "doc_type": doc_type,
                "confidence": 0.99,
                "reason": "Quickstart preset file — classification skipped",
                "vertical_fit": 0.95,
                "vertical_guess": vertical,
                "vertical_fit_reason": f"Preset data for {vertical} vertical",
                "off_vertical": False,
                "content_categories": content_analysis.get("content_categories", []),
                "content_topics": content_analysis.get("content_topics", []),
            }
        )
    return docs


def _deploy_prebuilt(
    ts: "TenantSession",
    vertical: str,
    data_files: list,
    doc_visibility: Dict[str, str],
) -> Dict[str, Any]:
    """Deploy from pre-built artifacts. Zero LLM calls.

    Copies prebuilt factory_spec, generated agents, and session data into the
    tenant workspace, fixes up paths, triggers a backend reload, and saves the
    session.  Returns a result dict compatible with the quickstart job format.
    """
    prebuilt = PREBUILT_DIR / vertical
    spec_src = prebuilt / "factory_spec.json"

    factory_dir = ts.workspace / ".factory"
    factory_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy factory_spec.json and rewrite workspace placeholders
    spec = json.loads(spec_src.read_text(encoding="utf-8"))
    ws_str = str(ts.workspace)
    _rewrite_spec_paths(spec, ws_str)
    spec_path = factory_dir / "factory_spec.json"
    spec_path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[prebuilt] wrote factory_spec.json for tenant=%s", ts.tenant_id)

    # 2) Copy doc metadata
    doc_meta_src = prebuilt / ".doc_metadata.json"
    if doc_meta_src.exists():
        shutil.copy2(doc_meta_src, factory_dir / ".doc_metadata.json")

    # 3) Copy compiled policies
    compiled_src = prebuilt / "compiled_policies"
    if compiled_src.exists():
        dest = factory_dir / "compiled_policies"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(compiled_src, dest)

    # 4) Copy pre-generated agent packages to repo-root generated/
    prebuilt_gen = prebuilt / "generated"
    if prebuilt_gen.exists():
        gen_root = REPO_ROOT / "generated"
        gen_root.mkdir(exist_ok=True)
        for agent_dir in prebuilt_gen.iterdir():
            if not agent_dir.is_dir():
                continue
            dest = gen_root / agent_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(agent_dir, dest)
            logger.info("[prebuilt] copied generated/%s", agent_dir.name)

    # 5) Mirror spec to repo-root .factory for runtime dev startup
    try:
        root_factory = REPO_ROOT / ".factory"
        root_factory.mkdir(parents=True, exist_ok=True)
        (root_factory / "factory_spec.json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass

    # 6) Save session
    agent_ids = [
        a.get("id")
        for a in spec.get("agents", [])
        if isinstance(a, dict) and a.get("id")
    ]
    deploy_result = {
        "type": "decision_result",
        "text": f"Deployment prepared (DRY): {len(agent_ids)} agents from prebuilt artifacts.",
        "deployment_request": {
            "vertical": vertical,
            "mode": "dry",
            "agents": agent_ids,
            "spec_path": str(spec_path),
        },
    }
    _save_session(ts, deploy_result)

    # 7) Trigger backend reload
    _trigger_backend_reload(ts)

    return {
        "type": "factory_plan_preview",
        "text": f"Quickstart ({vertical}) deployed from prebuilt artifacts.",
        "deployment_request": deploy_result.get("deployment_request"),
        "deploy_text": deploy_result.get("text", ""),
        "prebuilt": True,
    }


def _rewrite_spec_paths(spec: dict, workspace: str) -> None:
    """Replace __WORKSPACE__ placeholders with actual workspace path."""
    paths_block = spec.get("paths", {})
    if isinstance(paths_block.get("base_dir"), str):
        paths_block["base_dir"] = paths_block["base_dir"].replace(
            "__WORKSPACE__", workspace
        )
    for agent in spec.get("agents", []):
        inputs = agent.get("inputs") or {}
        for key in ("docs", "policies", "knowledge_sources"):
            paths = inputs.get(key)
            if not isinstance(paths, list):
                continue
            inputs[key] = [
                p.replace("__WORKSPACE__", workspace) if isinstance(p, str) else p
                for p in paths
            ]


# ---------------------------------------------------------------------------
# Async Job Store — lets long-running endpoints return immediately
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS = 500  # evict oldest when exceeded


def _create_job(tenant_id: str, kind: str) -> str:
    job_id = uuid_mod.uuid4().hex[:12]
    with _jobs_lock:
        if len(_jobs) >= _MAX_JOBS:
            oldest = next(iter(_jobs))
            del _jobs[oldest]
        _jobs[job_id] = {
            "status": "processing",
            "kind": kind,
            "tenant_id": tenant_id,
            "created": time.time(),
            "result": None,
            "error": None,
        }
    return job_id


def _finish_job(job_id: str, result: Any = None, error: str | None = None):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "error" if error else "done"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["error"] = error
            _jobs[job_id]["finished"] = time.time()


def _get_job(job_id: str) -> Dict[str, Any] | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _get_active_job(tenant_id: str) -> tuple[str, Dict[str, Any]] | None:
    """Return the most recent processing or done job for a tenant."""
    with _jobs_lock:
        best: tuple[str, Dict[str, Any]] | None = None
        for jid, job in _jobs.items():
            if job.get("tenant_id") != tenant_id:
                continue
            if job["status"] in ("processing", "done"):
                if best is None or job["created"] > best[1]["created"]:
                    best = (jid, job)
        return best


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Agent Factory Concierge API", version="1.0")

# CORS: allow local dev + Azure Container Apps frontend
_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
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


# ---------------------------------------------------------------------------
# Global exception handler — catches unhandled errors so they return JSON
# instead of crashing and lets us log them. No middleware wrapping needed.
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled %s on %s %s: %s",
        type(exc).__name__,
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


@app.on_event("startup")
def _log_startup():
    logger.info(
        "Concierge starting — auth=%s, data_dir=%s, runtime=%s, cors=%s",
        AUTH_ENABLED,
        DATA_DIR.exists(),
        RUNTIME_BACKEND_URL,
        _cors_origins,
    )
    for f in FINTECH_DATA_FILES:
        logger.info("  preset file %s: %s", f.name, "OK" if f.exists() else "MISSING")
    bp_dir = REPO_ROOT / "factory" / "blueprints"
    if bp_dir.exists():
        bps = [d.name for d in bp_dir.iterdir() if d.is_dir()]
        logger.info("  blueprints: %s", bps)
    else:
        logger.warning("  blueprints dir MISSING: %s", bp_dir)
    for v in ("fintech", "retail"):
        pb = PREBUILT_DIR / v / "factory_spec.json"
        logger.info("  prebuilt %s: %s", v, "OK" if pb.exists() else "NOT FOUND")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Job status polling
# ---------------------------------------------------------------------------


@app.get("/concierge/job/{job_id}")
def get_job_status(job_id: str):
    """Poll a background job. Returns status, and result when done."""
    job = _get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"detail": "Job not found"})
    if job["status"] == "processing":
        elapsed = time.time() - job["created"]
        return {"job_id": job_id, "status": "processing", "elapsed": round(elapsed, 1)}
    if job["status"] == "error":
        return JSONResponse(
            status_code=500,
            content={"job_id": job_id, "status": "error", "error": job["error"]},
        )
    # Done — return the full result
    return {"job_id": job_id, "status": "done", "result": job["result"]}


@app.get("/concierge/active-job")
def active_job(user: AuthUser = Depends(get_current_user)):
    """Return the most recent active (processing/done) job for this tenant."""
    hit = _get_active_job(user.tenant_id)
    if not hit:
        return {"active": False}
    jid, job = hit
    if job["status"] == "processing":
        elapsed = time.time() - job["created"]
        return {
            "active": True,
            "job_id": jid,
            "status": "processing",
            "kind": job.get("kind"),
            "elapsed": round(elapsed, 1),
        }
    return {
        "active": True,
        "job_id": jid,
        "status": job["status"],
        "kind": job.get("kind"),
        "result": job.get("result"),
    }


# ---------------------------------------------------------------------------
# Diagnostic endpoint
# ---------------------------------------------------------------------------


@app.get("/concierge/debug")
def debug_info():
    """Diagnostic endpoint — shows config status (no secret values)."""
    bp_dir = REPO_ROOT / "factory" / "blueprints"
    blueprints = (
        [d.name for d in bp_dir.iterdir() if d.is_dir()] if bp_dir.exists() else []
    )
    return {
        "auth_enabled": AUTH_ENABLED,
        "auth_secret_set": bool(AUTH_SECRET),
        "auth_secret_length": len(AUTH_SECRET),
        "runtime_backend_url": RUNTIME_BACKEND_URL,
        "data_dir_exists": DATA_DIR.exists(),
        "fintech_files": {f.name: f.exists() for f in FINTECH_DATA_FILES},
        "blueprints_dir_exists": bp_dir.exists(),
        "blueprints": blueprints,
        "workspaces_root": str(WORKSPACES_ROOT),
        "active_tenants": len(_tenants),
        "tenant_ids": list(_tenants.keys()),
        "cors_origins": _cors_origins,
        "llm_config": {
            "azure_endpoint_set": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
            "azure_api_key_set": bool(os.getenv("AZURE_OPENAI_API_KEY")),
            "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION", ""),
        },
    }


@app.get("/concierge/debug/backend")
def debug_backend():
    """Proxy to the backend's /debug endpoint."""
    try:
        r = http_requests.get(f"{RUNTIME_BACKEND_URL}/debug", timeout=10)
        if r.status_code == 200:
            return {"backend_reachable": True, **r.json()}
    except Exception as exc:
        return {"backend_reachable": False, "error": str(exc)}
    return {"backend_reachable": False, "status_code": r.status_code}


@app.post("/concierge/cors-test")
def cors_test():
    """Minimal POST endpoint to verify CORS works for cross-origin requests."""
    return {"ok": True}


# ---------------------------------------------------------------------------
# Per-tenant state (LRU eviction)
# ---------------------------------------------------------------------------


@dataclass
class TenantSession:
    """Holds the concierge agent and metadata for one tenant."""

    tenant_id: str
    vertical: str = "retail"
    model: str = "gpt-5-mini"
    agent: ConciergeAgent | None = None
    doc_visibility: Dict[str, str] | None = None

    @property
    def workspace(self) -> Path:
        return WORKSPACES_ROOT / self.tenant_id

    def get_or_create_agent(
        self, vertical: str | None = None, model: str | None = None
    ) -> ConciergeAgent:
        v = vertical or self.vertical
        m = model or self.model
        if self.agent is None or v != self.vertical:
            self.workspace.mkdir(parents=True, exist_ok=True)
            self.agent = ConciergeAgent(
                vertical=v,
                data_dir=str(self.workspace),
                llm_client=llm_client,
                model=m,
            )
            self.vertical = v
            self.model = m
        return self.agent


# Ordered dict for LRU eviction
_tenants: OrderedDict[str, TenantSession] = OrderedDict()


def _get_tenant(tenant_id: str) -> TenantSession:
    """Get or create a TenantSession, evicting the oldest if over limit."""
    if tenant_id in _tenants:
        _tenants.move_to_end(tenant_id)
        return _tenants[tenant_id]
    # Evict oldest if at capacity
    while len(_tenants) >= MAX_TENANTS:
        evicted_id, _ = _tenants.popitem(last=False)
        logger.info("Tenant evicted (LRU): %s", evicted_id)
    session = TenantSession(tenant_id=tenant_id)
    _tenants[tenant_id] = session
    logger.info("Tenant created: %s (total=%d)", tenant_id, len(_tenants))
    return session


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class InitRequest(BaseModel):
    vertical: str = "retail"
    use_llm: bool = True
    model: str = "gpt-5-mini"


class AnalyzeRequest(BaseModel):
    use_llm: bool = True
    model: str = "gpt-5-mini"


class QuickstartRequest(BaseModel):
    use_llm: bool = True
    model: str = "gpt-5-mini"
    auto_deploy: bool = False


class DeployRequest(BaseModel):
    mode: str = "dry"
    doc_visibility: Optional[Dict[str, str]] = None


class RuntimeRequest(BaseModel):
    port: int = 808


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/concierge/init")
def init_session(req: InitRequest, user: AuthUser = Depends(get_current_user)):
    logger.info(
        "[init] tenant=%s vertical=%s model=%s", user.tenant_id, req.vertical, req.model
    )
    ts = _get_tenant(user.tenant_id)
    agent = ts.get_or_create_agent(vertical=req.vertical, model=req.model)
    logger.info("[init] tenant=%s ready", user.tenant_id)
    return {"status": "ready", "vertical": agent.vertical}


@app.post("/concierge/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    vertical: str = Form("retail"),
    user: AuthUser = Depends(get_current_user),
):
    logger.info(
        "[upload] tenant=%s files=%d vertical=%s",
        user.tenant_id,
        len(files),
        vertical,
    )
    ts = _get_tenant(user.tenant_id)
    ts.workspace.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f in files:
        content = await f.read()
        dst = ts.workspace / f.filename
        dst.write_bytes(content)
        saved.append(f.filename)
        logger.info(
            "[upload] tenant=%s saved %s (%d bytes)",
            user.tenant_id,
            f.filename,
            len(content),
        )
    # Ensure agent is initialised for this vertical
    ts.get_or_create_agent(vertical=vertical)
    return {"files_saved": saved, "workspace": str(ts.workspace)}


@app.post("/concierge/quickstart-fintech")
def quickstart_fintech(
    req: QuickstartRequest, user: AuthUser = Depends(get_current_user)
):
    """Start quickstart as a background job. Returns job_id for polling."""
    logger.info(
        "[quickstart] tenant=%s model=%s use_llm=%s",
        user.tenant_id,
        req.model,
        req.use_llm,
    )
    ts = _get_tenant(user.tenant_id)
    ts.workspace.mkdir(parents=True, exist_ok=True)
    for src in FINTECH_DATA_FILES:
        if not src.exists():
            logger.error("[quickstart] preset file missing: %s", src)
            return {"error": f"Preset file not found: {src}"}
        shutil.copy2(src, ts.workspace / src.name)
        logger.info("[quickstart] copied %s", src.name)

    ts.doc_visibility = FINTECH_DOC_VISIBILITY
    use_prebuilt = (PREBUILT_DIR / "fintech" / "factory_spec.json").exists()

    job_id = _create_job(user.tenant_id, "quickstart")

    def _run():
        t0 = time.time()
        try:
            if use_prebuilt:
                logger.info(
                    "[quickstart] tenant=%s using PREBUILT artifacts", ts.tenant_id
                )
                result = _deploy_prebuilt(
                    ts, "fintech", FINTECH_DATA_FILES, FINTECH_DOC_VISIBILITY
                )
            else:
                logger.info(
                    "[quickstart] tenant=%s no prebuilt, running full pipeline",
                    ts.tenant_id,
                )
                result = _run_quickstart_pipeline(
                    ts,
                    "fintech",
                    FINTECH_DATA_FILES,
                    FINTECH_DOC_VISIBILITY,
                    req,
                )
            elapsed = time.time() - t0
            logger.info("[quickstart] tenant=%s done in %.1fs", ts.tenant_id, elapsed)
            _finish_job(job_id, result=result)
        except Exception as exc:
            logger.error(
                "[quickstart] tenant=%s FAILED: %s", ts.tenant_id, exc, exc_info=True
            )
            _finish_job(job_id, error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.post("/concierge/quickstart-retail")
def quickstart_retail(
    req: QuickstartRequest, user: AuthUser = Depends(get_current_user)
):
    """Start retail quickstart as a background job. Returns job_id for polling."""
    logger.info(
        "[quickstart-retail] tenant=%s model=%s use_llm=%s",
        user.tenant_id,
        req.model,
        req.use_llm,
    )
    ts = _get_tenant(user.tenant_id)
    ts.workspace.mkdir(parents=True, exist_ok=True)
    for src in RETAIL_DATA_FILES:
        if not src.exists():
            logger.error("[quickstart-retail] preset file missing: %s", src)
            return {"error": f"Preset file not found: {src}"}
        shutil.copy2(src, ts.workspace / src.name)
        logger.info("[quickstart-retail] copied %s", src.name)

    ts.doc_visibility = RETAIL_DOC_VISIBILITY
    use_prebuilt = (PREBUILT_DIR / "retail" / "factory_spec.json").exists()

    job_id = _create_job(user.tenant_id, "quickstart-retail")

    def _run():
        t0 = time.time()
        try:
            if use_prebuilt:
                logger.info(
                    "[quickstart-retail] tenant=%s using PREBUILT artifacts",
                    ts.tenant_id,
                )
                result = _deploy_prebuilt(
                    ts, "retail", RETAIL_DATA_FILES, RETAIL_DOC_VISIBILITY
                )
            else:
                logger.info(
                    "[quickstart-retail] tenant=%s no prebuilt, running full pipeline",
                    ts.tenant_id,
                )
                result = _run_quickstart_pipeline(
                    ts,
                    "retail",
                    RETAIL_DATA_FILES,
                    RETAIL_DOC_VISIBILITY,
                    req,
                )
            elapsed = time.time() - t0
            logger.info(
                "[quickstart-retail] tenant=%s done in %.1fs", ts.tenant_id, elapsed
            )
            _finish_job(job_id, result=result)
        except Exception as exc:
            logger.error(
                "[quickstart-retail] tenant=%s FAILED: %s",
                ts.tenant_id,
                exc,
                exc_info=True,
            )
            _finish_job(job_id, error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


def _run_quickstart_pipeline(
    ts: "TenantSession",
    vertical: str,
    data_files: list,
    doc_visibility: Dict[str, str],
    req: QuickstartRequest,
) -> Dict[str, Any]:
    """Full LLM quickstart pipeline (fallback when no prebuilt artifacts)."""
    pre_docs = _build_preset_doc_meta(ts.workspace, data_files, vertical)
    agent = ts.get_or_create_agent(vertical=vertical, model=req.model)
    logger.info("[quickstart-pipeline] tenant=%s running infer...", ts.tenant_id)
    result = agent.handle_event(
        {
            "type": "upload_docs",
            "use_llm": req.use_llm,
            "model": req.model,
            "pre_classified_docs": pre_docs,
        }
    )
    if req.auto_deploy:
        logger.info("[quickstart-pipeline] tenant=%s auto-deploying...", ts.tenant_id)
        deploy_result = agent.handle_event(
            {
                "type": "user_action",
                "action": "approve_deploy_dry",
                "doc_visibility": doc_visibility,
            }
        )
        _trigger_backend_reload(ts)
        _save_session(ts, deploy_result)
        result["deployment_request"] = deploy_result.get("deployment_request")
        result["deploy_text"] = deploy_result.get("text", "")
    return result


@app.post("/concierge/analyze")
def analyze_documents(req: AnalyzeRequest, user: AuthUser = Depends(get_current_user)):
    """Start analysis as a background job. Returns job_id for polling."""
    logger.info("[analyze] tenant=%s model=%s", user.tenant_id, req.model)
    ts = _get_tenant(user.tenant_id)
    job_id = _create_job(user.tenant_id, "analyze")

    def _run():
        t0 = time.time()
        try:
            agent = ts.get_or_create_agent(model=req.model)
            result = agent.handle_event(
                {"type": "upload_docs", "use_llm": req.use_llm, "model": req.model}
            )
            logger.info(
                "[analyze] tenant=%s done in %.1fs", ts.tenant_id, time.time() - t0
            )
            _finish_job(job_id, result=result)
        except Exception as exc:
            logger.error(
                "[analyze] tenant=%s FAILED: %s", ts.tenant_id, exc, exc_info=True
            )
            _finish_job(job_id, error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.post("/concierge/generate-templates")
def generate_templates(user: AuthUser = Depends(get_current_user)):
    ts = _get_tenant(user.tenant_id)
    agent = ts.get_or_create_agent()
    result = agent.handle_event(
        {
            "type": "user_action",
            "action": "generate_placeholders",
        }
    )
    return result


@app.post("/concierge/deploy")
def deploy_factory(req: DeployRequest, user: AuthUser = Depends(get_current_user)):
    """Start deploy as a background job. Returns job_id for polling."""
    logger.info("[deploy] tenant=%s mode=%s", user.tenant_id, req.mode)
    ts = _get_tenant(user.tenant_id)
    job_id = _create_job(user.tenant_id, "deploy")

    def _run():
        t0 = time.time()
        try:
            agent = ts.get_or_create_agent()
            action = (
                "approve_deploy_dry" if req.mode == "dry" else "approve_deploy_live"
            )
            # Use explicit visibility from request, or fall back to
            # visibility stored during quickstart/upload.
            visibility = req.doc_visibility or ts.doc_visibility
            result = agent.handle_event(
                {
                    "type": "user_action",
                    "action": action,
                    "doc_visibility": visibility,
                }
            )
            logger.info(
                "[deploy] tenant=%s spec generated in %.1fs",
                ts.tenant_id,
                time.time() - t0,
            )
            _trigger_backend_reload(ts)
            # Save session metadata so the user can resume later
            _save_session(ts, result)
            _finish_job(job_id, result=result)
        except Exception as exc:
            logger.error(
                "[deploy] tenant=%s FAILED: %s", ts.tenant_id, exc, exc_info=True
            )
            _finish_job(job_id, error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.post("/concierge/runtime/start")
def start_runtime(
    request: Request, req: RuntimeRequest, user: AuthUser = Depends(get_current_user)
):
    ts = _get_tenant(user.tenant_id)
    port = req.port
    # On Azure the runtime backend is already running as a separate container.
    # Check if it's reachable and return its status.
    if RUNTIME_BACKEND_URL.startswith("http://127.0.0.1"):
        # Local dev: launch uvicorn as a subprocess
        import platform

        if platform.system() == "Windows":
            cmd = (
                f'start "agent-factory-runtime" cmd /k '
                f"python -m uvicorn app.runtime.service:app --port {port}"
            )
        else:
            cmd = f"python -m uvicorn app.runtime.service:app --port {port} &"
        subprocess.Popen(cmd, shell=True, cwd=str(REPO_ROOT))
        return {"status": "starting", "port": port}
    else:
        # Cloud: runtime is a separate container
        fwd_headers = {}
        auth = request.headers.get("authorization")
        if auth:
            fwd_headers["Authorization"] = auth

        # Check if agents are already loaded — skip reload if so
        already_loaded = False
        try:
            r = http_requests.get(
                f"{RUNTIME_BACKEND_URL}/health", headers=fwd_headers, timeout=5
            )
            if r.status_code == 200:
                h = r.json()
                if h.get("status") == "ok" and h.get("agents"):
                    already_loaded = True
                    logger.info(
                        "[start] tenant=%s agents already loaded, skipping reload",
                        user.tenant_id,
                    )
                    return {"status": "running", "url": RUNTIME_BACKEND_URL, **h}
        except Exception:
            pass

        if not already_loaded:
            logger.info("[start] tenant=%s triggering backend reload", user.tenant_id)
            _trigger_backend_reload(ts)
            try:
                r = http_requests.get(
                    f"{RUNTIME_BACKEND_URL}/health", headers=fwd_headers, timeout=5
                )
                logger.info("[start] backend health -> %d", r.status_code)
                if r.status_code == 200:
                    return {
                        "status": "running",
                        "url": RUNTIME_BACKEND_URL,
                        **r.json(),
                    }
            except Exception as exc:
                logger.error("[start] backend health check failed: %s", exc)
        return {"status": "unreachable", "url": RUNTIME_BACKEND_URL}


@app.post("/concierge/runtime/stop")
def stop_runtime(req: RuntimeRequest, user: AuthUser = Depends(get_current_user)):
    port = req.port
    if RUNTIME_BACKEND_URL.startswith("http://127.0.0.1"):
        # Local dev: stop the process
        import platform

        if platform.system() == "Windows":
            kill_cmd = (
                f'for /f "tokens=5" %%a in '
                f"('netstat -ano ^| findstr :{port}') do taskkill /F /PID %%a"
            )
        else:
            kill_cmd = f"kill $(lsof -t -i:{port}) 2>/dev/null || true"
        subprocess.Popen(kill_cmd, shell=True)
        return {"status": "stopped", "port": port}
    else:
        # Cloud: runtime is managed by Azure, can't stop it from here
        return {
            "status": "managed",
            "message": "Runtime is managed by Azure Container Apps",
        }


@app.get("/concierge/runtime/health")
def runtime_health(request: Request):
    try:
        # Forward the Authorization header so the backend can identify the tenant
        headers = {}
        auth = request.headers.get("authorization")
        if auth:
            headers["Authorization"] = auth
        r = http_requests.get(
            f"{RUNTIME_BACKEND_URL}/health", headers=headers, timeout=5
        )
        logger.info("[health] backend %s -> %d", RUNTIME_BACKEND_URL, r.status_code)
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        logger.warning("[health] backend unreachable: %s", exc)
    return {"status": "unreachable"}


@app.get("/concierge/spec")
def get_factory_spec(
    tenant_id: str = "",
    user: AuthUser = Depends(get_current_user),
):
    """Return the generated factory spec (used by backend /reload)."""
    tid = tenant_id or user.tenant_id
    workspace = WORKSPACES_ROOT / tid
    spec_path = workspace / ".factory" / "factory_spec.json"
    if not spec_path.exists():
        return {"spec": None, "status": "not_deployed"}
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        return {"spec": spec, "status": "ready"}
    except Exception as exc:
        return {"spec": None, "status": "error", "error": str(exc)}


def _trigger_backend_reload(ts: TenantSession):
    """Tell the runtime backend to reload its spec for this tenant."""
    spec_path = ts.workspace / ".factory" / "factory_spec.json"
    if not spec_path.exists():
        logger.warning(
            "[reload] No spec file for tenant %s at %s", ts.tenant_id, spec_path
        )
        return

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    agent_ids = [a.get("id", "?") for a in spec.get("agents", [])]
    logger.info(
        "[reload] tenant=%s agents=%s url=%s",
        ts.tenant_id,
        agent_ids,
        RUNTIME_BACKEND_URL,
    )

    # Inject tenant_id into the spec so the backend knows which tenant this is for
    spec["_tenant_id"] = ts.tenant_id

    # Embed doc/policy file contents so the runtime container can access them
    # even when it doesn't share a filesystem with the concierge container.
    for agent in spec.get("agents", []):
        inputs = agent.get("inputs") or {}
        for key in ("docs", "policies", "knowledge_sources"):
            paths = inputs.get(key) or []
            if not paths:
                continue
            embedded = []
            for doc_path in paths:
                if not isinstance(doc_path, str):
                    continue
                try:
                    p = Path(doc_path)
                    if p.exists() and p.is_file():
                        content = p.read_text(encoding="utf-8", errors="ignore")
                        embedded.append({"filename": p.name, "content": content})
                except (OSError, ValueError):
                    pass
            if embedded:
                inputs[f"_embedded_{key}"] = embedded

    try:
        r = http_requests.post(
            f"{RUNTIME_BACKEND_URL}/reload",
            json={"spec": spec, "tenant_id": ts.tenant_id},
            timeout=90,
        )
        logger.info(
            "[reload] tenant=%s backend responded %d: %s",
            ts.tenant_id,
            r.status_code,
            r.text[:200],
        )
    except Exception as exc:
        logger.error("[reload] tenant=%s backend unreachable: %s", ts.tenant_id, exc)


def _save_session(ts: TenantSession, deploy_result: Dict[str, Any]):
    """Persist session metadata so the user can resume on next login."""
    factory_dir = ts.workspace / ".factory"
    factory_dir.mkdir(parents=True, exist_ok=True)
    dep = deploy_result.get("deployment_request", {})
    session_data = {
        "vertical": ts.vertical,
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agents": dep.get("agents", []),
        "spec_path": dep.get("spec_path", ""),
        "deploy_text": deploy_result.get("text", ""),
        "deployment_request": dep,
    }
    session_path = factory_dir / "session.json"
    session_path.write_text(
        json.dumps(session_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "[session] saved for tenant=%s agents=%s", ts.tenant_id, dep.get("agents")
    )


# ---------------------------------------------------------------------------
# Session persistence & reset
# ---------------------------------------------------------------------------


@app.get("/concierge/session")
def get_session(user: AuthUser = Depends(get_current_user)):
    """Return the user's last deployment session, or {status: 'new'} if none."""
    ts = _get_tenant(user.tenant_id)
    session_path = ts.workspace / ".factory" / "session.json"
    if not session_path.exists():
        return {"status": "new"}
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
        return {"status": "deployed", **data}
    except Exception as exc:
        logger.warning("[session] failed to read for tenant=%s: %s", ts.tenant_id, exc)
        return {"status": "new"}


@app.post("/concierge/reset")
def reset_session(user: AuthUser = Depends(get_current_user)):
    """Delete all data for this tenant and start fresh."""
    ts = _get_tenant(user.tenant_id)
    logger.info("[reset] tenant=%s workspace=%s", user.tenant_id, ts.workspace)
    # Remove the workspace directory
    if ts.workspace.exists():
        shutil.rmtree(ts.workspace, ignore_errors=True)
        logger.info("[reset] tenant=%s workspace deleted", user.tenant_id)
    # Clear in-memory agent
    ts.agent = None
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------


@app.get("/concierge/chat-history")
def get_chat_history(user: AuthUser = Depends(get_current_user)):
    """Return saved chat threads and messages, or empty if none."""
    ts = _get_tenant(user.tenant_id)
    history_path = ts.workspace / ".factory" / "chat_history.json"
    if not history_path.exists():
        return {"threads": [], "messagesMap": {}, "activeThreadId": None}
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        logger.warning(
            "[chat-history] failed to read for tenant=%s: %s", ts.tenant_id, exc
        )
        return {"threads": [], "messagesMap": {}, "activeThreadId": None}


@app.put("/concierge/chat-history")
def save_chat_history(request: Request, user: AuthUser = Depends(get_current_user)):
    """Save chat threads and messages."""
    import asyncio

    loop = asyncio.get_event_loop()
    body = loop.run_until_complete(request.json())
    ts = _get_tenant(user.tenant_id)
    factory_dir = ts.workspace / ".factory"
    factory_dir.mkdir(parents=True, exist_ok=True)
    history_path = factory_dir / "chat_history.json"
    history_path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    thread_count = len(body.get("threads", []))
    msg_count = sum(len(v) for v in body.get("messagesMap", {}).values())
    logger.info(
        "[chat-history] saved for tenant=%s threads=%d messages=%d",
        ts.tenant_id,
        thread_count,
        msg_count,
    )
    return {"status": "saved", "threads": thread_count, "messages": msg_count}


@app.get("/concierge/workspace/files")
def list_workspace_files(user: AuthUser = Depends(get_current_user)):
    ts = _get_tenant(user.tenant_id)
    ts.workspace.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(ts.workspace.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            files.append(
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "extension": p.suffix.lstrip("."),
                }
            )
    return files


@app.delete("/concierge/workspace/files/{filename}")
def delete_workspace_file(filename: str, user: AuthUser = Depends(get_current_user)):
    ts = _get_tenant(user.tenant_id)
    target = ts.workspace / filename
    if target.exists() and target.is_file():
        target.unlink()
        return {"deleted": filename}
    return {"error": f"File not found: {filename}"}


# ---------------------------------------------------------------------------
# MCP Tool Configuration endpoints
# ---------------------------------------------------------------------------


class McpToolUpdate(BaseModel):
    """Full replacement for the mcp_tools_config.json content."""

    tools: List[Dict[str, Any]]
    server_name: str = "demo-server"


class McpToolSingleUpdate(BaseModel):
    """Update a single tool by name."""

    tool: Dict[str, Any]


@app.get("/concierge/mcp-tools")
def get_mcp_tools_config():
    """Return the current MCP tools configuration."""
    if not MCP_TOOLS_CONFIG.exists():
        return {"error": "mcp_tools_config.json not found", "tools": []}
    try:
        data = json.loads(MCP_TOOLS_CONFIG.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        return {"error": str(exc), "tools": []}


@app.put("/concierge/mcp-tools")
def update_mcp_tools_config(req: McpToolUpdate):
    """Replace the entire MCP tools configuration."""
    try:
        existing = {}
        if MCP_TOOLS_CONFIG.exists():
            existing = json.loads(MCP_TOOLS_CONFIG.read_text(encoding="utf-8"))
        # Preserve metadata keys
        existing["server_name"] = req.server_name
        existing["tools"] = req.tools
        MCP_TOOLS_CONFIG.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {"status": "saved", "tool_count": len(req.tools)}
    except Exception as exc:
        return {"error": str(exc)}


@app.put("/concierge/mcp-tools/{tool_name}")
def update_single_mcp_tool(tool_name: str, req: McpToolSingleUpdate):
    """Update or add a single tool by name."""
    try:
        if not MCP_TOOLS_CONFIG.exists():
            return {"error": "mcp_tools_config.json not found"}
        data = json.loads(MCP_TOOLS_CONFIG.read_text(encoding="utf-8"))
        tools: list = data.get("tools", [])
        # Find and replace, or append
        found = False
        for i, t in enumerate(tools):
            if t.get("name") == tool_name:
                tools[i] = req.tool
                found = True
                break
        if not found:
            tools.append(req.tool)
        data["tools"] = tools
        MCP_TOOLS_CONFIG.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "saved",
            "tool_name": tool_name,
            "action": "updated" if found else "added",
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.delete("/concierge/mcp-tools/{tool_name}")
def delete_mcp_tool(tool_name: str):
    """Delete a tool from the MCP tools configuration."""
    try:
        if not MCP_TOOLS_CONFIG.exists():
            return {"error": "mcp_tools_config.json not found"}
        data = json.loads(MCP_TOOLS_CONFIG.read_text(encoding="utf-8"))
        tools: list = data.get("tools", [])
        before = len(tools)
        data["tools"] = [t for t in tools if t.get("name") != tool_name]
        if len(data["tools"]) == before:
            return {"error": f"Tool '{tool_name}' not found"}
        MCP_TOOLS_CONFIG.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {"status": "deleted", "tool_name": tool_name}
    except Exception as exc:
        return {"error": str(exc)}
