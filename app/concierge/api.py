# app/concierge/api.py
"""
Concierge REST API — wraps ConciergeAgent for the Next.js frontend.
Run with: python -m uvicorn app.concierge.api:app --port 8001
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import requests as http_requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import app.llm_client as llm_client
from app.concierge.concierge_agent import ConciergeAgent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
WORKSPACE = REPO_ROOT / ".workspace"

FINTECH_DATA_FILES = [
    DATA_DIR / "BankFAQs.csv",
    DATA_DIR / "refunds_policy.yaml",
]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Agent Factory Concierge API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singleton state (single-user demo)
# ---------------------------------------------------------------------------
_agent: ConciergeAgent | None = None
_vertical: str = "retail"
_model: str = "gpt-5-mini"


def _get_or_create_agent(vertical: str | None = None, model: str | None = None) -> ConciergeAgent:
    global _agent, _vertical, _model
    v = vertical or _vertical
    m = model or _model
    if _agent is None or v != _vertical:
        WORKSPACE.mkdir(exist_ok=True)
        _agent = ConciergeAgent(
            vertical=v,
            data_dir=str(WORKSPACE),
            llm_client=llm_client,
            model=m,
        )
        _vertical = v
        _model = m
    return _agent


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


class DeployRequest(BaseModel):
    mode: str = "dry"


class RuntimeRequest(BaseModel):
    port: int = 808


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/concierge/init")
def init_session(req: InitRequest):
    agent = _get_or_create_agent(vertical=req.vertical, model=req.model)
    return {"status": "ready", "vertical": agent.vertical}


@app.post("/concierge/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    vertical: str = Form("retail"),
):
    WORKSPACE.mkdir(exist_ok=True)
    saved: list[str] = []
    for f in files:
        content = await f.read()
        dst = WORKSPACE / f.filename
        dst.write_bytes(content)
        saved.append(f.filename)
    # Ensure agent is initialised for this vertical
    _get_or_create_agent(vertical=vertical)
    return {"files_saved": saved, "workspace": str(WORKSPACE)}


@app.post("/concierge/quickstart-fintech")
def quickstart_fintech(req: QuickstartRequest):
    WORKSPACE.mkdir(exist_ok=True)
    # Copy preset files
    for src in FINTECH_DATA_FILES:
        if not src.exists():
            return {"error": f"Preset file not found: {src}"}
        shutil.copy2(src, WORKSPACE / src.name)

    agent = _get_or_create_agent(vertical="fintech", model=req.model)
    result = agent.handle_event(
        {
            "type": "upload_docs",
            "use_llm": req.use_llm,
            "model": req.model,
        }
    )
    return result


@app.post("/concierge/analyze")
def analyze_documents(req: AnalyzeRequest):
    agent = _get_or_create_agent(model=req.model)
    result = agent.handle_event(
        {
            "type": "upload_docs",
            "use_llm": req.use_llm,
            "model": req.model,
        }
    )
    return result


@app.post("/concierge/generate-templates")
def generate_templates():
    agent = _get_or_create_agent()
    result = agent.handle_event(
        {
            "type": "user_action",
            "action": "generate_placeholders",
        }
    )
    return result


@app.post("/concierge/deploy")
def deploy_factory(req: DeployRequest):
    agent = _get_or_create_agent()
    action = "approve_deploy_dry" if req.mode == "dry" else "approve_deploy_live"
    result = agent.handle_event(
        {
            "type": "user_action",
            "action": action,
        }
    )
    return result


@app.post("/concierge/runtime/start")
def start_runtime(req: RuntimeRequest):
    port = req.port
    # Launch uvicorn in a new console window (Windows)
    cmd = (
        f'start "agent-factory-runtime" cmd /k '
        f"python -m uvicorn app.runtime.service:app --port {port}"
    )
    subprocess.Popen(cmd, shell=True, cwd=str(REPO_ROOT))
    return {"status": "starting", "port": port}


@app.post("/concierge/runtime/stop")
def stop_runtime(req: RuntimeRequest):
    port = req.port
    kill_cmd = (
        f'for /f "tokens=5" %%a in ' f"('netstat -ano ^| findstr :{port}') do taskkill /F /PID %%a"
    )
    subprocess.Popen(kill_cmd, shell=True)
    return {"status": "stopped", "port": port}


@app.get("/concierge/runtime/health")
def runtime_health():
    try:
        r = http_requests.get("http://127.0.0.1:808/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "unreachable"}


@app.get("/concierge/workspace/files")
def list_workspace_files():
    WORKSPACE.mkdir(exist_ok=True)
    files = []
    for p in sorted(WORKSPACE.iterdir()):
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
def delete_workspace_file(filename: str):
    target = WORKSPACE / filename
    if target.exists() and target.is_file():
        target.unlink()
        return {"deleted": filename}
    return {"error": f"File not found: {filename}"}
