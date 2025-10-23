# ui/concierge_app.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path
import tempfile
from app.concierge.concierge_agent import ConciergeAgent
import app.llm_client as llm_client

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
# 1. Select domain vertical
# ---------------------------
vertical = st.selectbox(
    "Select your business domain:", ["retail", "fintech", "telco", "general_service"]
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

# Create temporary working directory
work_dir = Path(tempfile.mkdtemp())
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
    analyze = st.button("🔍 Analyze Documents")
with col2:
    generate = st.button("🧾 Generate Templates")
with col3:
    approve = st.button("🚀 Approve & Deploy (Dry-Run)")

# Output area
st.subheader("🧠 Concierge Response")
st.write("---")

if analyze:
    res = agent.handle_event({"type": "upload_docs", "use_llm": use_llm, "model": model})
    st.markdown(res["text"])

    plan = res["plan"]
    st.write("### 🧠 Plan Summary", plan["summary"])

    # ---- layout: cards per agent ----
    for a in plan["agents"]:
        status = a["status"]
        color = {
            "ready": "#309659",  # soft green
            "partial": "#DEBD3A",  # soft cream
            "missing_docs": "#D05E4F",  # light rose
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

elif generate:
    res = agent.handle_event({"type": "user_action", "action": "generate_placeholders"})
    st.success(res["text"])
    st.json(res["next"]["plan"]["summary"])

elif approve:
    res = agent.handle_event({"type": "user_action", "action": "approve_deploy_dry"})
    st.success(res["text"])
    st.json(res["deployment_request"])

else:
    st.info("Upload your files and click **Analyze Documents** to get started.")


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
