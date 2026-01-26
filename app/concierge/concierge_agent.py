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
from app.generator.generate_agent import generate_agent
import json


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
        # Reset plan state to avoid stale docs being displayed
        self.plan = None
        self.last_plan = None  # if you have it
        self._cached_plan = None  # if you have it

        planner = PlannerInterface(
            self.vertical, str(self.data_dir), llm_client=self.llm_client, model=model
        )
        plan = planner.generate_plan_preview(use_llm=use_llm)
        self.state["last_plan"] = plan
        plan = self._normalize_plan_for_ui(plan)
        self.plan = plan
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

        # 1) ensure we have a plan (robust across Streamlit reruns)
        plan = self.state.get("last_plan")

        # UI may pass/attach plan explicitly
        if not plan and isinstance(self.state.get("plan_override"), dict):
            plan = self.state.get("plan_override")

        # As a last resort, use the normalized plan cached on the instance
        if not plan and isinstance(getattr(self, "plan", None), dict):
            plan = getattr(self, "plan")

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

        # 2.5) Pre-generate agents NOW (so uvicorn doesn't need to)
        spec_path = (self.data_dir / ".factory" / "factory_spec.json").resolve()
        try:
            spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        except Exception as e:
            return {
                "type": "text",
                "text": f"Deployment failed: could not read spec at {spec_path} ({e})",
            }

        generated = []
        errors = []
        for a in spec.get("agents", []):
            if not isinstance(a, dict):
                continue
            if a.get("type") != "autogen":
                continue

            a_id = a.get("id") or "unknown"
            try:
                gen_dir = generate_agent(a)  # expected to write to generated/<id>/
                generated.append({"id": a_id, "path": str(gen_dir)})
            except Exception as e:
                errors.append({"id": a_id, "error": str(e)})

        print(f"[DEPLOY] Spec: {spec_path}")
        print(f"[DEPLOY] Pre-generated {len(generated)} agents: {[g['id'] for g in generated]}")
        if errors:
            print(f"[DEPLOY][WARN] Generation errors: {errors}")

        # 3) return deployment info for the UI
        port = 808
        base_url = f"http://127.0.0.1:{port}"

        # Prefer python -m uvicorn for Windows reliability (PATH-safe)
        uvicorn_cmd = "python -m uvicorn app.runtime.service:app --reload --port 808"

        payload = {
            "vertical": self.vertical,
            "mode": mode,
            "agents": [
                a.get("id") for a in plan.get("agents", []) if isinstance(a, dict) and a.get("id")
            ],
            "spec_path": str(spec_path),
            "generated_agents": generated,
            "generation_errors": errors,
            "uvicorn_command": uvicorn_cmd,
            "runtime": {
                "base_url": base_url,
                "health": base_url + "/health",
                "chat": base_url + "/chat",
            },
        }

        # Cache for later UI actions (e.g., Start runtime, Send chat)
        self.state["last_deployment"] = payload

        # If any generation errors, surface it clearly in the decision_result text
        msg = f"Deployment prepared ({mode.upper()}): spec + {len(generated)} agents generated."
        if errors:
            msg += f" ⚠️ {len(errors)} agent(s) failed generation (see details)."

        return {
            "type": "decision_result",
            "text": msg,
            "deployment_request": payload,
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

    def _normalize_plan_for_ui(self, plan: dict) -> dict:
        """
        Add UI-friendly fields expected by concierge_app / textual summary.
        Keeps everything generic; does NOT hardcode FAQ/complaint.
        """
        agents = plan.get("agents", [])
        if not isinstance(agents, list):
            return plan

        kind_to_icon = {
            "rag": "📚",
            "workflow": "🧭",
            "tool": "🛠️",
            "router": "🧠",
            "qa": "✅",
            "guardrails": "🛡️",
            "other": "🤖",
        }

        # Optional: if infer_capabilities returns document metadata
        plan_docs = plan.get("documents", [])
        name_to_doc = {}
        if isinstance(plan_docs, list):
            for d in plan_docs:
                if isinstance(d, dict) and d.get("name"):
                    name_to_doc[str(d["name"])] = d

        for a in agents:
            if not isinstance(a, dict):
                continue

            # agent_kind comes from infer; fallback to type/id if missing
            kind = str(a.get("agent_kind") or a.get("type") or "other").lower()

            # icon
            a.setdefault("icon", kind_to_icon.get(kind, "🤖"))

            # display_name
            if not a.get("display_name"):
                aid = str(a.get("id", "agent"))
                a["display_name"] = aid.replace("_", " ").replace("-", " ").title()

            # status
            status = str(a.get("status", "partial")).lower()
            if status not in {"ready", "partial", "missing_docs"}:
                status = "partial"
            a["status"] = status

            # summary (agent-level)
            if "summary" not in a:
                desc = a.get("description", "")
                a["summary"] = desc if isinstance(desc, str) else ""

            # docs_detected (UI expects it)
            det = a.get("docs_detected", [])
            if not isinstance(det, list):
                det = []
            a["docs_detected"] = det

            # inputs_typed (normalize: buckets)
            typed = a.get("inputs_typed")
            if not isinstance(typed, dict):
                typed = {}
            for bucket in ("knowledge_base", "policy", "procedure", "tool_spec"):
                v = typed.get(bucket, [])
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    v = []
                typed[bucket] = v
            a["inputs_typed"] = typed

            # confidence: UI expects float 0..1
            if "confidence" not in a:
                if a["status"] == "ready":
                    a["confidence"] = 0.9
                elif a["status"] == "partial":
                    a["confidence"] = 0.7
                else:
                    a["confidence"] = 0.4
            else:
                try:
                    a["confidence"] = float(a["confidence"])
                except Exception:
                    a["confidence"] = 0.7
            a["confidence"] = max(0.0, min(1.0, a["confidence"]))

            # docs_missing: UI expects list[str]
            if "docs_missing" not in a or not isinstance(a["docs_missing"], list):
                if a["status"] == "missing_docs":
                    missing = []
                    for k, v in typed.items():
                        if isinstance(v, list) and not v:
                            missing.append(k)
                    a["docs_missing"] = missing
                else:
                    a["docs_missing"] = []

            # why: UI expects string
            if not a.get("why"):
                if isinstance(a.get("reason"), str) and a["reason"].strip():
                    a["why"] = a["reason"].strip()
                else:
                    parts = []
                    if typed.get("knowledge_base"):
                        parts.append("uses knowledge base")
                    if typed.get("policy"):
                        parts.append("uses policy")
                    if typed.get("procedure"):
                        parts.append("uses procedures")
                    if typed.get("tool_spec"):
                        parts.append("uses tool specs")

                    tail = ", ".join(parts) if parts else "no supporting docs attached yet"
                    desc = a.get("description", "")
                    if not isinstance(desc, str):
                        desc = ""
                    a["why"] = (
                        desc.strip() or f"Proposed as part of the customer-service system; {tail}."
                    )[:240]

            # Optional: append off-vertical doc warnings (if available)
            off = []
            for bucket in ("knowledge_base", "policy", "procedure", "tool_spec"):
                for dn in typed.get(bucket, []):
                    dd = name_to_doc.get(str(dn))
                    if isinstance(dd, dict) and dd.get("off_vertical"):
                        vg = dd.get("vertical_guess", "")
                        off.append(f"{dn}" + (f" (likely {vg})" if vg else ""))
            if off:
                a["why"] = (a["why"] + " Off-vertical docs: " + ", ".join(off))[:240]

        # Top-level plan summary for UI (MUST be outside the loop)
        if not plan.get("summary"):
            vertical = plan.get("vertical", "customer_service")
            if agents:
                lines = [
                    f"• {x.get('display_name', x.get('id','agent'))}: {x.get('summary','')}".strip()
                    for x in agents
                    if isinstance(x, dict)
                ]
                plan["summary"] = f"Vertical: {vertical}\n\n" + "\n".join(lines)
            else:
                plan["summary"] = (
                    f"Vertical: {vertical}\n\nNo agents proposed yet. Upload more documents or refine goals."
                )

        plan["agents"] = agents
        return plan
