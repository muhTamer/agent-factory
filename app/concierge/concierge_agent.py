# app/concierge/concierge_agent.py
"""
Concierge Agent
- Chat-style orchestrator that guides users through setup.
- Integrates with PlannerInterface to analyze docs and preview system plan.
"""

from pathlib import Path
from typing import Dict, Any
from app.concierge.planner_interface import PlannerInterface
from app.deploy.spec_builder import build_factory_spec


class ConciergeAgent:
    def __init__(self, vertical: str, data_dir: str, llm_client=None, model: str | None = None):
        self.vertical = vertical
        self.data_dir = Path(data_dir)
        self.llm_client = llm_client
        self.state: Dict[str, Any] = {"last_plan": None, "awaiting_confirmation": False}
        self.model = model

    # -----------------------------
    # Public entry point
    # -----------------------------
    def handle_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main event router: decides what to do with each user/UI action.
        """
        event_type = event.get("type")
        action = event.get("action")
        cap = event.get("capability")
        use_llm = event.get("use_llm", True)
        model = event.get("model") or self.model

        if event_type in {"init", "upload_docs", "rerun_infer"}:
            return self._run_infer(use_llm=use_llm, model=model)
        if event_type in {"init", "upload_docs", "rerun_infer"}:
            return self._run_infer()

        if event_type == "user_action" and action == "ask_requirements":
            return self._explain_requirements(cap)

        if event_type == "user_action" and action == "generate_placeholders":
            return self._generate_placeholders()

        if event_type == "user_action" and action in {"approve_deploy_dry", "approve_deploy_live"}:
            return self._approve_deploy(action)

        return self._help_message()

    # -----------------------------
    # Analyze current state
    # -----------------------------
    def _run_infer(self, use_llm: bool = True, model: str | None = None) -> Dict[str, Any]:
        planner = PlannerInterface(
            self.vertical, str(self.data_dir), llm_client=self.llm_client, model=model
        )
        plan = planner.generate_plan_preview(use_llm=use_llm)
        self.state["last_plan"] = plan
        text_summary = self._textual_summary(plan)
        return {"type": "factory_plan_preview", "text": text_summary, "plan": plan}

    # -----------------------------
    # Explain requirements for a capability
    # -----------------------------
    def _explain_requirements(self, cap: str) -> Dict[str, Any]:
        kb = {
            "faq": "For FAQs, please upload a CSV or MD file with question–answer pairs.",
            "complaint": "To enable complaints, upload Refund/Returns Policy and Complaint SOP.",
            "guardrails": "Guardrails are always included. You can add extra tone or privacy policies later.",
        }
        msg = kb.get(cap, "I can’t find specific requirements for that capability yet.")
        return {"type": "text", "text": msg}

    # -----------------------------
    # Generate placeholder templates
    # -----------------------------
    def _generate_placeholders(self) -> Dict[str, Any]:
        created = []
        faq_path = self.data_dir / "sample_faqs.csv"
        if not faq_path.exists():
            faq_path.write_text("question,answer\nHow do I return an item?,Within 30 days.")
            created.append(faq_path.name)

        refund_path = self.data_dir / "refunds_policy.yaml"
        if not refund_path.exists():
            refund_path.write_text(
                "rules:\n  - id: refund_within_30\n    description: Refunds within 30 days."
            )
            created.append(refund_path.name)

        sop_path = self.data_dir / "complaint_sop.md"
        if not sop_path.exists():
            sop_path.write_text(
                "# Complaint Handling SOP\nStep 1: Record complaint\nStep 2: Acknowledge customer"
            )
            created.append(sop_path.name)

        msg = f"I created {len(created)} placeholder file(s): {', '.join(created)}. Re-running analysis..."
        return {"type": "action_result", "text": msg, "next": self._run_infer()}

    # -----------------------------
    # Approve deployment (safe gate)
    # -----------------------------
    def _approve_deploy(self, action: str) -> Dict[str, Any]:
        mode = "dry" if "dry" in action else "live"

        # 1) ensure we have a plan
        plan = self.state.get("last_plan")
        if not plan:
            return {
                "type": "text",
                "text": "Please analyze documents first, then approve deployment.",
            }

        # 2) build runtime spec (auto-creates missing blueprints)
        build_factory_spec(
            plan=plan,
            data_dir=str(self.data_dir),
            dry_run=(mode == "dry"),
            llm_client=self.llm_client,  # enables LLM blueprint creation
        )

        spec_path = self.data_dir / ".factory" / "factory_spec.json"
        # 3) return deployment info for the UI
        uvicorn_cmd = "uvicorn app.runtime.service:app --reload --port 8088"

        return {
            "type": "decision_result",
            "text": f"Deployment spec generated ({mode.upper()}).",
            "deployment_request": {
                "vertical": self.vertical,
                "mode": mode,
                "agents": [a["id"] for a in plan["agents"]],
                "spec_path": str(spec_path),
                "uvicorn_command": uvicorn_cmd,
            },
        }

    # -----------------------------
    # Helpers
    # -----------------------------
    def _textual_summary(self, plan: Dict[str, Any]) -> str:
        lines = [f"📊 Plan for *{plan['vertical']}* domain:"]
        for a in plan["agents"]:
            lines.append(
                f"{a['icon']} {a['display_name']} — {a['status'].capitalize()} "
                f"(confidence {int(a['confidence']*100)}%)"
            )
        lines.append("Choose to upload missing docs, generate templates, or approve & deploy.")
        return "\n".join(lines)

    def _build_deploy_spec(self, mode: str) -> Dict[str, Any]:
        plan = self.state.get("last_plan")
        return {
            "vertical": self.vertical,
            "mode": mode,
            "agents": [a["id"] for a in plan["agents"]],
            "timestamp": "pending-runtime",
            "notes": "Deployment spec generated by Concierge; execution handled by DeploymentService.",
        }

    def _help_message(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "text": (
                "I can analyze your uploaded documents, generate templates, "
                "or prepare your system for deployment. "
                "Try uploading docs or type 'analyze'."
            ),
        }
