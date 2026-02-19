# app/concierge/planner_interface.py
from pathlib import Path
from typing import Dict, Any
from app.infer_capabilities import InferCapabilities


class PlannerInterface:
    """
    Mediator between Concierge Agent and InferCapabilities.
    Produces structured plan previews for the Concierge to present to the user.
    """

    def __init__(self, vertical: str, data_dir: str, llm_client=None, model: str | None = None):
        self.vertical = vertical
        self.data_dir = Path(data_dir)
        self.llm_client = llm_client
        self.model = model

    from app.infer_capabilities import InferCapabilities

    def generate_plan_preview(self, use_llm: bool = True, model: str = "gpt-5-mini") -> dict:
        if use_llm:
            infer = InferCapabilities(model=model or "gpt-5-mini")
            raw = infer.infer(
                data_dir=self.data_dir,
                vertical=self.vertical,
                user_goals=getattr(self, "user_goals", "") or "",
                max_agents=6,
            )
            agents = raw.get("agents", [])
        else:
            agents = self._heuristic_agents()

        return {
            "type": "factory_plan_preview",
            "vertical": self.vertical,
            "agents": agents,
            "actions": self._suggested_actions(agents),
        }

    def _heuristic_agents(self) -> list:
        """No-LLM fallback: always include spine agents, detect RAG/workflow from files."""
        from app.infer_capabilities import InferCapabilities as IC

        ic = IC()
        files = ic._list_user_files(self.data_dir)

        agents = [
            {
                "id": "guardrails",
                "agent_kind": "guardrails",
                "status": "ready",
                "description": "Policy guardrails",
            },
            {"id": "qa", "agent_kind": "qa", "status": "ready", "description": "Quality evaluator"},
        ]

        kb_files = [f.name for f in files if f.suffix.lower() in {".csv", ".tsv"}]
        if kb_files:
            agents.append(
                {
                    "id": "customer_qa_rag",
                    "agent_kind": "rag",
                    "status": "ready",
                    "description": "FAQ / knowledge-base agent",
                    "docs_detected": kb_files,
                }
            )

        pol_files = [f.name for f in files if f.suffix.lower() in {".yaml", ".yml"}]
        if pol_files:
            agents.append(
                {
                    "id": "refunds_workflow",
                    "agent_kind": "workflow",
                    "status": "partial",
                    "description": "Refund / complaint workflow",
                    "docs_detected": pol_files,
                }
            )

        return agents

    def _format_for_ui(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform InferCapabilities output into a clean UI-facing schema
        that the Concierge chat and front-end cards can render.
        """
        ui_agents = []
        for cap in plan["capabilities"]:
            ui_agents.append(
                {
                    "id": cap["id"],
                    "display_name": self._pretty_name(cap["id"]),
                    "status": cap["status"],
                    "confidence": cap["confidence"],
                    "docs_detected": cap["docs_detected"],
                    "docs_missing": cap["docs_missing"],
                    "always_included": cap["always_included"],
                    "why": cap["why"],
                    "icon": self._icon_for(cap["id"]),
                    "color": self._color_for_status(cap["status"]),
                }
            )

        summary = plan["summary"]
        return {
            "type": "factory_plan_preview",
            "vertical": self.vertical,
            "summary": summary,
            "agents": ui_agents,
            "actions": self._suggested_actions(ui_agents),
        }

    # --------------- helper mappings ---------------
    def _pretty_name(self, cap_id: str) -> str:
        mapping = {
            "faq": "FAQ Agent",
            "complaint": "Complaint Agent",
            "guardrails": "Guardrails",
            "qa": "Quality Evaluator",
        }
        return mapping.get(cap_id, cap_id.capitalize())

    def _icon_for(self, cap_id: str) -> str:
        icons = {"faq": "❓", "complaint": "📝", "guardrails": "🛡️", "qa": "✅"}
        return icons.get(cap_id, "⚙️")

    def _color_for_status(self, status: str) -> str:
        colors = {"ready": "green", "partial": "amber", "missing_docs": "red"}
        return colors.get(status, "grey")

    def _suggested_actions(self, agents: list[dict]) -> list[dict]:
        """
        Suggest next UI actions depending on current readiness.
        """
        missing = [a for a in agents if a["status"] == "missing_docs"]
        partial = [a for a in agents if a["status"] == "partial"]

        if not missing and not partial:
            return [{"label": "Approve & Deploy", "action": "approve_deploy"}]
        else:
            return [
                {"label": "Upload Missing Documents", "action": "upload_docs"},
                {"label": "Generate Sample Templates", "action": "generate_placeholders"},
                {"label": "Re-run Analysis", "action": "rerun_infer"},
            ]
