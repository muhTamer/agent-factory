# ui/concierge_app.py
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path
from app.concierge.concierge_agent import ConciergeAgent
import app.llm_client as llm_client
import subprocess
import sys
import requests

# ---------------------------
# Quickstart uses repo-root /data (existing docs)
# ---------------------------
DATA_DIR = Path("data")

# Update these two filenames to match what's in your repo's /data folder
FINTECH_DATA_FILES = [
    DATA_DIR / "BankFAQs.csv",
    DATA_DIR / "refunds_policy.yaml",
]


def _load_quickstart_fintech_into_workspace(work_dir: Path) -> list[Path]:
    """Copies existing /data files into workspace and returns the created workspace paths."""
    missing = [p for p in FINTECH_DATA_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing preset file(s): " + ", ".join(str(p) for p in missing))
    work_dir.mkdir(exist_ok=True)
    written = []
    for src in FINTECH_DATA_FILES:
        dst = work_dir / src.name
        dst.write_bytes(src.read_bytes())
        written.append(dst)
    return written


def _wait_runtime_health(base_url: str, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    base = (base_url or "").rstrip("/")
    while time.time() < deadline:
        try:
            r = requests.get(base + "/health", timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


if "plan" not in st.session_state:
    st.session_state.plan = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "work_dir" not in st.session_state:
    st.session_state.work_dir = str(Path(".workspace").resolve())

st.set_page_config(page_title="Agent Factory Concierge", layout="wide")

# Sidebar: LLM settings
st.sidebar.header("LLM Settings")
consent = st.sidebar.checkbox("Use LLM for sufficiency checks (redacted snippets only)", value=True)
use_llm = consent
model = st.sidebar.text_input("Model / Deployment", value="gpt-5-mini")
st.sidebar.caption("Uses Azure/OpenAI via app.llm_client with logging disabled (per your .env).")

st.markdown(
    """
<style>
    h4 {font-family: 'Inter', sans-serif;}
    .stButton>button {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


st.title("🤖 Agent Factory Concierge")
st.write("Welcome! I’ll help you analyze your documents and design your customer-service system.")


# ---------------------------
# Quickstart (optional) — does NOT replace the normal flow
# ---------------------------
with st.expander("⚡ Quickstart (optional)", expanded=False):
    st.write("Run a ready-made **Fintech** setup (loads preset docs, analyzes, deploys dry-run).")
    st.caption(
        "Keeps the normal flow intact — you can still upload your own docs and run Analyze/Generate/Approve as before."
    )
    qs_col1, qs_col2 = st.columns([1, 2])
    with qs_col1:
        qs_fintech = st.button("🚀 Quickstart: Fintech", key="qs_fintech_btn")
    with qs_col2:
        st.caption("Loads `data/bankFAQs.md` + `data/refund_policy.yaml` into `.workspace`.")

    if qs_fintech:
        st.session_state["quickstart_active"] = True
        st.session_state["quickstart_vertical"] = "fintech"
        # Trigger the same analysis path as pressing the Analyze button
        st.session_state["quickstart_stage"] = "analyze"
        st.rerun()


# ---------------------------
# 1. Select domain vertical
# ---------------------------
vertical = st.selectbox(
    "Select your business domain:",
    ["retail", "fintech", "telco", "general_service"],
    index=(
        ["retail", "fintech", "telco", "general_service"].index(
            st.session_state.get("quickstart_vertical", "retail")
        )
        if st.session_state.get("quickstart_active")
        else 0
    ),
    key="domain_vertical",
)
st.caption("💡 This helps determine which agents are relevant by default.")

# ---------------------------
# 2. File upload section
# ---------------------------
st.subheader("📂 Upload your documents")
uploaded_files = st.file_uploader(
    "Upload FAQs, policies, or SOPs (CSV, YAML, MD, TXT supported)",
    type=["csv", "yaml", "yml", "md", "txt"],
    accept_multiple_files=True,
)

# Create temporary working directory (persistent across reruns unless we clear it)
work_dir = Path(".workspace")
work_dir.mkdir(exist_ok=True)

# If quickstart is active, populate workspace with preset docs (acts like uploaded files)
if (
    st.session_state.get("quickstart_active")
    and st.session_state.get("quickstart_vertical") == "fintech"
):
    try:
        _load_quickstart_fintech_into_workspace(work_dir)
    except Exception as e:
        st.error(f"Quickstart failed to load preset files: {e}")
        st.session_state["quickstart_active"] = False
        st.session_state.pop("quickstart_stage", None)

# ✅ If user didn't upload anything this run, clear previous workspace files
# (but do NOT clear when a Quickstart preset is active)
if (not uploaded_files) and (not st.session_state.get("quickstart_active")):
    for p in work_dir.iterdir():
        if p.is_file():
            p.unlink()

# Save newly uploaded files

for f in uploaded_files:
    content = f.read()
    (work_dir / f.name).write_bytes(content)

# ---------------------------
# 3. Concierge interaction
# ---------------------------
if "agent" not in st.session_state or st.session_state.get("agent_vertical") != vertical:
    st.session_state.agent = ConciergeAgent(
        vertical=vertical,
        data_dir=work_dir,
        llm_client=llm_client,  # pass the real client
        model=model,
    )
    st.session_state.agent_vertical = vertical

agent = st.session_state.agent

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    analyze = st.button("🔍 Analyze Documents") or (
        st.session_state.get("quickstart_stage") == "analyze"
    )
with col2:
    generate = st.button("🧾 Generate Templates")
with col3:
    approve = st.button("🚀 Approve & Deploy (Dry-Run)") or (
        st.session_state.get("quickstart_stage") == "approve"
    )

# Output area
st.subheader("🧠 Concierge Response")
st.write("---")

if analyze:
    res = agent.handle_event({"type": "upload_docs", "use_llm": use_llm, "model": model})
    st.markdown(res["text"])

    plan = res["plan"]
    st.session_state.plan = plan
    st.session_state.analyzed = True
    st.session_state.work_dir = str(work_dir.resolve())
    st.write("### 🧠 Plan Summary", plan["summary"])

    # ---- layout: cards per agent ----
    for a in plan["agents"]:
        status = a["status"]
        color = {
            "ready": "#309659",
            "partial": "#DEBD3A",
            "missing_docs": "#D05E4F",
        }.get(status, "#F3F4F6")

        icon = a.get("icon", "⚙️")
        display_name = a.get("display_name", a["id"])
        conf = int(a.get("confidence", 0) * 100)

        st.markdown(
            f"""
            <div style="
                background-color:{color};
                border-radius:12px;
                padding:15px 20px;
                margin-bottom:15px;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);
                border:1px solid rgba(0,0,0,0.05);
            ">
                <h4 style="margin-bottom:5px;">{icon} {display_name}</h4>
                <p style="margin:2px 0;"><b>Status:</b> {status.capitalize()} ({conf}% confidence)</p>
                <p style="margin:2px 0;"><b>Detected:</b> {', '.join(a['docs_detected']) or '—'}</p>
                <p style="margin:2px 0;"><b>Missing:</b> {', '.join(a['docs_missing']) or '—'}</p>
                <p style="margin:2px 0;"><b>Reason:</b> {a['why']}</p>
                <div style="background:#d0d0d0;border-radius:8px;height:8px;width:100%;">
                    <div style="background:#2E7D32;width:{conf}%;height:8px;border-radius:8px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("Choose to upload missing docs, generate templates, or approve & deploy below.")

    # Quickstart: after analysis, automatically proceed to Approve & Deploy
    if st.session_state.get("quickstart_stage") == "analyze":
        st.session_state["quickstart_stage"] = "approve"
        st.rerun()


elif generate:
    res = agent.handle_event({"type": "user_action", "action": "generate_placeholders"})
    st.success(res["text"])
    st.json(res["next"]["plan"]["summary"])

elif approve:
    if st.session_state.get("plan"):
        agent.state["plan_override"] = st.session_state.plan

    res = agent.handle_event({"type": "user_action", "action": "approve_deploy_dry"})
    if res.get("type") == "decision_result":
        st.session_state["deployment"] = res["deployment_request"]
        st.session_state["deploy_msg"] = res.get("text", "Deployment spec generated.")
        st.success(st.session_state["deploy_msg"])

        dep = st.session_state["deployment"]
        st.write("### ✅ Deployment Spec")
        st.write(f"**Spec path:** `{dep['spec_path']}`")
        st.write("**Run the runtime locally:**")
        st.code(dep["uvicorn_command"], language="bash")
        st.caption("Then open http://127.0.0.1:808/health")

        # Quickstart convenience: optionally auto-start runtime
        if st.session_state.get("quickstart_active"):
            st.info(
                "Quickstart: you can click **Start runtime** below, then test in **Try your multi-agent system**."
            )
            # Quickstart: mark deploy step complete and jump to Try section
            st.session_state.pop("quickstart_stage", None)
            st.session_state["jump_to_try"] = True

    else:
        st.warning(res.get("text") or "Could not generate deployment spec.")

else:
    st.info("Upload your files and click **Analyze Documents** to get started.")

# ----------------------------
# Always-visible after deploy: Start/Stop runtime
# ----------------------------
if st.session_state.get("deployment"):
    st.write("---")
    st.subheader("🚀 Runtime")

    if "runtime_proc" not in st.session_state:
        st.session_state["runtime_proc"] = None

    def _is_running(proc) -> bool:
        return proc is not None and proc.poll() is None

    port = 808
    dep = st.session_state["deployment"] or {}
    runtime_base = (dep.get("runtime") or {}).get("base_url") if isinstance(dep, dict) else None
    if runtime_base:
        st.session_state["runtime_url"] = runtime_base

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        start_clicked = st.button("Start runtime", key="start_runtime_btn")

    with col2:
        stop_clicked = st.button("Stop runtime", key="stop_runtime_btn")

    with col3:
        proc = st.session_state["runtime_proc"]
        if _is_running(proc):
            st.success("Runtime is running.")
        else:
            st.info("Runtime is not running.")

    if start_clicked:
        # On Windows, starting uvicorn as a child process can behave oddly with Streamlit reruns.
        # More reliable: launch uvicorn in a NEW terminal window.
        import subprocess

        cmd = (
            f'start "agent-factory-runtime" cmd /k '
            f"python -m uvicorn app.runtime.service:app  --port {port}"
        )

        # This does not block Streamlit; it opens a new console window.
        subprocess.Popen(cmd, shell=True)
        st.success("Runtime starting in a new terminal window…")

    if stop_clicked:
        # Best-effort stop: kill the process(es) listening on the port.
        # (POC-friendly; you can improve later with PID tracking)
        import subprocess

        port = 808

        # Find PID(s) bound to the port and kill them.
        # Note: double %% is required in batch context; streamlit runs via cmd, so use %%.
        kill_cmd = f"for /f \"tokens=5\" %%a in ('netstat -ano ^| findstr :{port}') do taskkill /F /PID %%a"

        subprocess.Popen(kill_cmd, shell=True)
    st.success(f"Stop command sent (killing any process bound to port {port}).")

# ----------------------------
# Always-visible after deploy: Try your system
# ----------------------------
if st.session_state.get("deployment"):
    st.write("---")
    st.subheader("🧪 Try your multi-agent system")

    dep = st.session_state.get("deployment", {}) or {}

    # Auto-fill runtime URL from deployment if available
    default_runtime = "http://127.0.0.1:808"
    runtime_from_dep = None
    if isinstance(dep, dict):
        runtime_from_dep = (dep.get("runtime") or {}).get("base_url")
    if runtime_from_dep and not st.session_state.get("runtime_url"):
        st.session_state["runtime_url"] = runtime_from_dep

    runtime_url = st.text_input(
        "Runtime base URL",
        value=st.session_state.get("runtime_url") or runtime_from_dep or default_runtime,
        help="Must match the port you run uvicorn on",
        key="runtime_url",
    )

    query = st.text_area(
        "Customer message / query",
        value=st.session_state.get("chat_query") or "What is the refund policy?",
        height=90,
        key="chat_query",
    )

    show_router = st.checkbox("Show router plan", value=True, key="show_router")
    show_raw = st.checkbox("Show raw JSON", value=False, key="show_raw")

    def _make_curl(base_url: str, q: str) -> str:
        # Use single quotes around -d JSON, and JSON-escape the query safely.
        import json as _json

        payload = _json.dumps({"query": q}, ensure_ascii=False)
        return f'curl.exe -X POST "{base_url.rstrip("/")}/chat" -H "Content-Type: application/json" -d \'{payload}\''

    def render_response(payload: dict):
        if not isinstance(payload, dict):
            st.error("Unexpected response format")
            st.write(payload)
            return

        # --- Guardrails / error response ---
        if payload.get("error"):
            st.markdown("### 🚫 Blocked")
            msg = (
                payload.get("text")
                or payload.get("response", {}).get("text")
                or payload.get("error")
            )
            st.error(msg)

            reason = payload.get("reason")
            if reason:
                st.caption(f"Reason: {reason}")

            return

        # --- FAQ / answer-style response ---
        if "answer" in payload:
            st.markdown("### 💬 Answer")
            st.success(payload.get("answer", ""))

            score = payload.get("score", None)
            agent_id = payload.get("agent_id", "—")
            if score is not None:
                try:
                    score = round(float(score), 3)
                except Exception:
                    score = score

            st.caption(f"Agent: {agent_id} · Score: {score if score is not None else '—'}")

            if payload.get("citations"):
                with st.expander("📌 Citations"):
                    st.json(payload["citations"])

        # --- Workflow-style response ---
        if "workflow_id" in payload or "current_state" in payload:
            st.markdown("### 🧭 Workflow")
            cols = st.columns(3)
            cols[0].metric("Agent", payload.get("agent_id", "—"))
            cols[1].metric("Workflow", payload.get("workflow_id", "—"))
            cols[2].metric("State", payload.get("current_state", "—"))

            mapper = payload.get("mapper")
            if mapper:
                with st.expander("🧠 Mapper"):
                    st.json(mapper)

            slots = payload.get("slots")
            if slots:
                with st.expander("🧩 Slots"):
                    st.json(slots)

            history = payload.get("history")
            if history:
                with st.expander("🕒 History"):
                    st.json(history)

        # --- Router plan ---
        if show_router and payload.get("router_plan"):
            with st.expander("🧠 Router plan"):
                st.json(payload["router_plan"])

    if st.button("Send to /chat", key="send_chat"):
        try:
            import requests

            base = runtime_url.rstrip("/")

            # ---- HARD GATE: runtime must be up ----
            try:
                with st.spinner("Checking runtime /health ..."):
                    health = requests.get(base + "/health", timeout=2)

                if health.status_code != 200:
                    st.error(
                        f"Runtime is not healthy (HTTP {health.status_code}). Start runtime first."
                    )
                    st.code(f"curl.exe -X GET '{base}/health'", language="bash")
                    st.stop()

            except Exception as e:
                st.error(
                    f"Runtime is not running or not reachable. Start runtime first.\n\nDetails: {e}"
                )
                st.code(f"curl.exe -X GET '{base}/health'", language="bash")
                st.stop()

            # ---- Call /chat only if health check passed ----
            with st.spinner("Calling runtime /chat ..."):
                resp = requests.post(
                    base + "/chat",
                    json={"query": query},
                    timeout=30,
                )

            st.caption(f"HTTP {resp.status_code}")
            if resp.status_code != 200:
                st.error(resp.text)
            else:
                data = resp.json()
                st.write("### ✅ Response")
                render_response(data)

                st.write("### 🔧 Curl (copy/paste)")
                st.code(_make_curl(runtime_url, query), language="bash")

                if show_raw:
                    with st.expander("🧾 Raw JSON"):
                        st.json(data)

        except Exception as e:
            st.error(f"Failed to call runtime: {e}")
            st.info("Make sure the runtime is running and the Runtime base URL is correct.")

# ---- Always-visible: sanitized audit samples viewer ----
st.write("---")
st.subheader("🔍 Audit: Sanitized Samples Sent to LLM")

audit_toggle = st.checkbox("Show sanitized audit samples", value=False)
if audit_toggle:
    try:
        # If you moved audits to .factory per earlier fix:
        audit_path = agent.data_dir / ".factory" / "samples_audit.json"
        # If you kept it in root, fall back:
        if not audit_path.exists():
            audit_path = agent.data_dir / "samples_audit.json"

        if audit_path.exists():
            import json

            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            # Optional: a refresh button so user can re-read file without clicking Analyze
            refresh = st.button("Refresh audit")
            # Show the last few records for brevity
            st.json(audit[-5:] if len(audit) > 5 else audit)
        else:
            st.info("No audit samples yet. Click **Analyze Documents** to generate them.")
    except Exception as e:
        st.warning(f"Could not read audit file: {e}")


# Footer
st.write("---")
st.caption("Agent Factory © 2025 — Customer Service Meta-Agent Research Prototype")
