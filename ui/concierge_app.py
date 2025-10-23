# ui/concierge_app.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path
import tempfile
from app.concierge.concierge_agent import ConciergeAgent

st.set_page_config(page_title="Agent Factory Concierge", layout="wide")

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
if "agent" not in st.session_state:
    st.session_state.agent = ConciergeAgent(vertical=vertical, data_dir=work_dir)

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
    res = agent.handle_event({"type": "upload_docs"})
    st.markdown(res["text"])

    plan = res["plan"]
    st.write("### 🧠 Plan Summary", plan["summary"])

    # ---- layout: cards per agent ----
    for a in plan["agents"]:
        status = a["status"]
        color = {
            "ready": "#B9F6CA",  # light green
            "partial": "#FFF59D",  # amber
            "missing_docs": "#FF8A80",  # red
        }.get(status, "#E0E0E0")

        icon = a.get("icon", "⚙️")
        display_name = a.get("display_name", a["id"])
        conf = int(a.get("confidence", 0) * 100)

        with st.container():
            st.markdown(
                f"""
                <div style="
                    background-color:{color};
                    border-radius:12px;
                    padding:15px 20px;
                    margin-bottom:10px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.1);
                    ">
                    <h4 style="margin-bottom:5px;">{icon} {display_name}</h4>
                    <p style="margin:2px 0;"><b>Status:</b> {status.capitalize()} ({conf}% confidence)</p>
                    <p style="margin:2px 0;"><b>Detected:</b> {', '.join(a['docs_detected']) or '—'}</p>
                    <p style="margin:2px 0;"><b>Missing:</b> {', '.join(a['docs_missing']) or '—'}</p>
                    <p style="margin:2px 0;"><b>Reason:</b> {a['why']}</p>
                    <div style="background:#ccc;border-radius:8px;height:10px;width:100%;">
                        <div style="background:#2E7D32;width:{conf}%;height:10px;border-radius:8px;"></div>
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

# Footer
st.write("---")
st.caption("Agent Factory © 2025 — Customer Service Meta-Agent Research Prototype")
