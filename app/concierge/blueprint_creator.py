# app/concierge/blueprint_creator.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.llm_client import chat_json
from app.shared.schemas.validate import validate_agent_blueprint, validate_workflow_spec


DEFAULT_MODEL = "gpt-5-mini"


@dataclass
class BlueprintPlan:
    vertical: str
    blueprints: list[dict]
    missing_docs: list[str]
    warnings: list[str]
    rationale: str


class BlueprintCreatorAgent:
    """
    Consumes an existing 'plan' (from Concierge/InferCapabilities/etc.) and returns:
      - a list of validated AgentBlueprint objects (blueprints[])
      - missing_docs/warnings/rationale for UI preview

    No hardcoded FAQ/complaint. LLM decides the full agent set.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def _normalize_workflow_spec(self, wf: dict, agent_id: str) -> dict:
        """
        Coerce LLM-generated workflow_spec into executable form.
        Does NOT invent logic, only fixes structure.
        """
        wf = dict(wf)

        # Required top-level fields
        wf.setdefault("id", agent_id)
        wf.setdefault("description", f"Workflow for {agent_id}")
        wf.setdefault("engine", "fsm")

        # Normalize slots
        slots = wf.get("slots", {})
        norm_slots = {}

        if isinstance(slots, dict):
            for name, val in slots.items():
                if isinstance(val, dict):
                    norm_slots[name] = val
                else:
                    # val is likely "string", true, 1, etc.
                    if isinstance(val, bool):
                        t = "boolean"
                    elif isinstance(val, int):
                        t = "integer"
                    elif isinstance(val, float):
                        t = "number"
                    else:
                        t = "string"

                    norm_slots[name] = {
                        "type": t,
                        "required": True,
                        "description": f"Auto-normalized slot '{name}'",
                    }

        wf["slots"] = norm_slots

        # Ensure states exist
        wf.setdefault("states", {})
        wf.setdefault("initial_state", next(iter(wf["states"]), "start"))

        # ---- Normalize states (schema compliance) ----
        states = wf.get("states")

        # Dict form (LLM-friendly): {state_name: {..}}
        if isinstance(states, dict):
            for _state_name, s in list(states.items()):
                if not isinstance(s, dict):
                    continue

                # If LLM produced action as a list → rename to actions
                if isinstance(s.get("action"), list):
                    s["actions"] = [str(x) for x in s["action"] if x is not None]
                    del s["action"]

                # Remove empty action strings — schema requires minLength: 1 if field is present
                if "action" in s and isinstance(s.get("action"), str) and not s["action"].strip():
                    del s["action"]

                # If LLM produced actions as a string → wrap into list
                if isinstance(s.get("actions"), str) and s["actions"].strip():
                    s["actions"] = [s["actions"].strip()]

                # If both exist, keep actions list and drop invalid action
                if "actions" in s and isinstance(s.get("action"), str):
                    del s["action"]

        wf["states"] = states
        return wf

    def generate_plan_from_existing_plan(
        self,
        plan: dict,
        user_goals: str | None = None,
    ) -> BlueprintPlan:
        """
        Use the already-produced plan (passed from build_factory_spec) as context for blueprint generation.
        """
        if not isinstance(plan, dict):
            raise ValueError("plan must be a dict")

        vertical = str(
            plan.get("vertical") or plan.get("primary_vertical") or "generic_customer_service"
        ).strip()
        user_goals = (user_goals or str(plan.get("user_goals") or "")).strip()

        docs_summary = self._docs_summary_from_plan(plan)

        llm_out = self._llm_generate_blueprint_plan(
            vertical=vertical,
            docs_summary=docs_summary,
            user_goals=user_goals,
            existing_plan=plan,
        )

        blueprints, missing_docs, warnings, rationale = self._normalize_plan(
            llm_out, vertical=vertical
        )

        EXECUTABLE_KINDS = {"knowledge_rag", "workflow_runner", "tool_operator"}

        filtered = []
        for bp in blueprints:
            kind = bp.get("agent_kind")
            if kind not in EXECUTABLE_KINDS:
                print(
                    f"[WARN] Skipping non-executable blueprint '{bp.get('id')}' (agent_kind={kind})"
                )
                continue
            filtered.append(bp)

        blueprints = filtered

        # fail fast: validate everything, no silent fallback
        hard_errors: list[str] = []
        for bp in blueprints:
            bp_errs = validate_agent_blueprint(bp)
            if bp_errs:
                hard_errors.append(
                    f"Blueprint '{bp.get('id')}' invalid: " + " | ".join(bp_errs[:5])
                )

            if bp.get("agent_kind") == "workflow_runner":
                wf = (bp.get("inputs") or {}).get("workflow_spec")
                if isinstance(wf, dict):
                    wf = self._normalize_workflow_spec(wf, agent_id=bp.get("id"))
                    bp["inputs"]["workflow_spec"] = wf  # IMPORTANT: write back
                    wf_errs = validate_workflow_spec(wf)
                    if wf_errs:
                        hard_errors.append(
                            f"workflow_spec for '{bp.get('id')}' invalid: "
                            + " | ".join(wf_errs[:5])
                        )
                else:
                    hard_errors.append(
                        f"workflow_runner '{bp.get('id')}' missing inputs.workflow_spec"
                    )

        if hard_errors:
            raise ValueError("Blueprint plan validation failed:\n" + "\n".join(hard_errors))

        return BlueprintPlan(
            vertical=vertical,
            blueprints=blueprints,
            missing_docs=missing_docs,
            warnings=warnings,
            rationale=rationale,
        )

    # -------------------------
    # Plan → docs summary (privacy-safe)
    # -------------------------

    def _docs_summary_from_plan(self, plan: dict) -> dict:
        """
        Produce a privacy-safe summary for the LLM:
          - counts, doc kinds/roles, filenames only
          - NO raw document contents
        """
        doc_items = (
            plan.get("documents")
            or plan.get("detected_documents")
            or plan.get("doc_bindings")
            or {}
        )
        kinds: list[str] = []
        filenames: list[str] = []

        if isinstance(doc_items, dict):
            # e.g. {"faq_csv": {"path": "...", ...}, "policy_yaml": {...}}
            kinds = list(doc_items.keys())
            for k, v in doc_items.items():
                if isinstance(v, dict) and v.get("path"):
                    filenames.append(str(v["path"]).split("\\")[-1].split("/")[-1])
                elif isinstance(v, str):
                    filenames.append(v.split("\\")[-1].split("/")[-1])
        elif isinstance(doc_items, list):
            # e.g. [{"kind":"faq_csv","path":"..."}, ...]
            for x in doc_items:
                if isinstance(x, dict):
                    if x.get("kind"):
                        kinds.append(str(x["kind"]))
                    if x.get("path"):
                        filenames.append(str(x["path"]).split("\\")[-1].split("/")[-1])
                elif isinstance(x, str):
                    filenames.append(x.split("\\")[-1].split("/")[-1])

        missing = plan.get("missing_docs") or plan.get("missing") or []
        caps = (
            plan.get("capabilities")
            or plan.get("inferred_capabilities")
            or plan.get("requested_capabilities")
            or []
        )

        return {
            "count": (
                len(filenames)
                if filenames
                else (len(doc_items) if isinstance(doc_items, (list, dict)) else 0)
            ),
            "kinds": sorted(list(set([str(k) for k in kinds if k]))),
            "filenames": sorted(list(set([str(f) for f in filenames if f]))),
            "missing_docs": ([str(x) for x in missing] if isinstance(missing, list) else []),
            "capability_signals": ([str(x) for x in caps] if isinstance(caps, list) else []),
        }

    # -------------------------
    # LLM generation
    # -------------------------

    def _llm_generate_blueprint_plan(
        self,
        vertical: str,
        docs_summary: dict,
        user_goals: str,
        existing_plan: dict,
    ) -> dict:
        system = (
            "You are an architect of a customer-service agent factory.\n"
            "You are given an EXISTING PLAN (capability signals, document summary, missing docs).\n"
            "Your task is to produce a COMPLETE set of EXECUTABLE AgentBlueprints.\n\n"
            "================ EXECUTION CONTRACT (MANDATORY) ================\n"
            "- Every blueprint MUST be directly executable by the factory.\n"
            "- Every blueprint MUST include ALL of the following fields:\n"
            "    - id: short unique string\n"
            "    - agent_kind: one of [knowledge_rag, workflow_runner, tool_operator]\n"
            "    - description: string\n"
            "    - capabilities: list of strings\n"
            "    - inputs: object (see rules below)\n\n"
            "AGENT_KIND → REQUIRED INPUTS:\n"
            "- knowledge_rag:\n"
            "    inputs.docs = array of document paths or placeholders like '<UPLOAD:faq>'\n"
            "- workflow_runner:\n"
            "    inputs.workflow_spec = object defining an FSM workflow (REQUIRED)\n"
            "- tool_operator:\n"
            "    inputs.tool = string (tool name)\n\n"
            "STRICT RULES:\n"
            "- Do NOT output abstract roles or conceptual agents.\n"
            "- Do NOT output agents without agent_kind.\n"
            "- Do NOT output guardrails (guardrails are part of the runtime spine).\n"
            "- Do NOT omit required inputs.\n"
            "- Do NOT include executable code.\n"
            "- Tools must be referenced by name only.\n"
            "- Use placeholder paths if documents are missing.\n\n"
            "================ WORKFLOW_SPEC STRUCTURE (MANDATORY) ================\n"
            "If agent_kind == workflow_runner, inputs.workflow_spec MUST contain:\n"
            "    - id: string\n"
            "    - description: string\n"
            "    - engine: 'fsm'\n"
            "    - slots: object (see SLOT FORMAT below)\n"
            "    - states: FSM definition (see STATE FORMAT below)\n"
            "    - initial_state: string\n\n"
            "SLOT FORMAT (MANDATORY):\n"
            "slots: {\n"
            "  '<slot_name>': {\n"
            "      'type': 'string | boolean | integer | number',\n"
            "      'required': true | false,\n"
            "      'description': 'string'\n"
            "  }\n"
            "}\n\n"
            "STATE FORMAT (OBJECT FORM – PREFERRED):\n"
            "states: {\n"
            "  '<state_name>': {\n"
            "    'description': 'string',\n"
            "    'on': { '<event>': '<next_state>' },\n"
            "    'on_enter': 'string OR [string]',\n"
            "    'action': 'optional shorthand action',\n"
            "    'terminal': true | false\n"
            "  }\n"
            "}\n\n"
            "ALTERNATIVE STATE FORMAT:\n"
            "- You may use 'transitions': [{event, next_state}] instead of 'on'\n\n"
            "INVALID EXAMPLES (DO NOT DO THIS):\n"
            "- slots: { order_id: 'string' }\n"
            "- states missing transitions\n"
            "- missing agent_kind\n\n"
            "================ OUTPUT FORMAT =================\n"
            "Return STRICT JSON ONLY with this exact shape:\n"
            "{\n"
            '  "blueprints": [AgentBlueprint, ...],\n'
            '  "missing_docs": [string, ...],\n'
            '  "warnings": [string, ...],\n'
            '  "rationale": "string"\n'
            "}\n"
        )

        user_payload = {
            "vertical": vertical,
            "docs_summary": docs_summary,
            "user_goals": user_goals,
            "existing_plan": existing_plan,  # include plan context; do not include raw doc contents
            "available_tools": [
                "create_ticket",
                "update_ticket",
                "get_ticket_status",
                "lookup_customer",
                "reset_password",
                "verify_identity",
                "lookup_order",
                "lookup_payment",
                "initiate_refund",
                "cancel_order",
                "track_shipment",
                "create_return_label",
                "handoff_to_human",
            ],
            "baseline_guardrails": [
                "No PII leakage",
                "No hallucinated policy claims; cite documents when possible",
                "Escalate when unsure or when policy missing for sensitive actions",
            ],
            "note": "Do not include raw document content. Use placeholder paths like '<UPLOAD:policy>' when needed.",
        }

        return chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            model=self.model,  # always gpt-5-mini
        )

    # -------------------------
    # Normalization
    # -------------------------

    def _normalize_plan(
        self, raw: Any, vertical: str
    ) -> tuple[list[dict], list[str], list[str], str]:
        if not isinstance(raw, dict):
            raise ValueError(f"LLM output must be JSON object. Got: {type(raw)}")

        blueprints = raw.get("blueprints") or []
        missing_docs = raw.get("missing_docs") or []
        warnings = raw.get("warnings") or []
        rationale = raw.get("rationale") or ""

        if not isinstance(blueprints, list):
            raise ValueError("LLM output: 'blueprints' must be a list")

        norm_bps: list[dict] = []
        seen_ids: set[str] = set()

        for i, bp in enumerate(blueprints):
            if not isinstance(bp, dict):
                continue

            bp_id = str(bp.get("id") or f"agent_{i}").strip() or f"agent_{i}"
            if bp_id in seen_ids:
                bp_id = f"{bp_id}_{i}"
            seen_ids.add(bp_id)
            bp["id"] = bp_id

            bp.setdefault("vertical", vertical)
            bp.setdefault("tools", [])
            bp.setdefault("knowledge_sources", [])
            bp.setdefault("policies", [])
            bp.setdefault("inputs", {})

            caps = bp.get("capabilities")
            if not isinstance(caps, list) or not caps:
                bp["capabilities"] = [bp_id]

            if isinstance(bp.get("tools"), list):
                bp["tools"] = [str(t) for t in bp["tools"]]

            norm_bps.append(bp)

        missing_docs = [str(x) for x in missing_docs] if isinstance(missing_docs, list) else []
        warnings = [str(x) for x in warnings] if isinstance(warnings, list) else []
        rationale = str(rationale)

        return norm_bps, missing_docs, warnings, rationale
