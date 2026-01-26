# app/infer_capabilities.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.llm_client import chat_json

MODEL = "gpt-5-mini"


# Generic document taxonomy (NOT agent taxonomy)
DOC_TYPES = {
    "knowledge_base",  # FAQs, Q&A, help center, KB articles, product guides
    "policy",  # refunds policy, compliance, privacy, eligibility rules
    "procedure",  # SOPs, onboarding playbooks, workflow steps
    "tool_spec",  # API docs/specs/endpoints, tool adapters info
    "other",
}


AGENT_KINDS = {
    "rag",  # retrieval-based answering / lookup
    "workflow",  # multi-step state machine / process runner
    "tool",  # tool operator / action executor
    "router",  # intent routing / agent selection
    "qa",  # evaluation / monitoring / scoring
    "guardrails",  # policy enforcement / safety constraints
    "other",
}


@dataclass
class InferOutput:
    vertical: str
    documents: List[Dict[str, Any]]
    agents: List[Dict[str, Any]]
    notes: List[str]


class InferCapabilities:
    """
    Concierge-side inference that remains generic:
      1) Collect user-uploaded files
      2) Classify docs into generic doc types (knowledge_base/policy/procedure/tool_spec/other)
      3) Ask LLM to propose agent set + per-agent inputs (by doc type)
      4) Normalize + produce plan structure used downstream (spec_builder)
    """

    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    # -----------------------------
    # Public API
    # -----------------------------
    def infer(
        self,
        *,
        data_dir: str | Path,
        vertical: str,
        user_goals: str = "",
        max_agents: int = 6,
    ) -> Dict[str, Any]:
        base_dir = Path(data_dir).resolve()
        files = self._list_user_files(base_dir)

        documents = self._classify_documents(files, vertical=vertical)

        # LLM proposes agents + uses documents (by name) as inputs
        llm_plan = self._propose_agents_llm(
            vertical=vertical,
            user_goals=user_goals,
            documents=documents,
            max_agents=max_agents,
        )

        agents = self._normalize_agents_plan(
            llm_plan=llm_plan,
            documents=documents,
        )

        # Legacy back-compat: docs_detected union field (used by older spec_builder logic)
        for a in agents:
            a["docs_detected"] = self._legacy_docs_detected(a)

        out = InferOutput(
            vertical=vertical,
            documents=documents,
            agents=agents,
            notes=llm_plan.get("notes", []) if isinstance(llm_plan.get("notes"), list) else [],
        )
        return {
            "vertical": out.vertical,
            "documents": out.documents,
            "agents": out.agents,
            "notes": out.notes,
        }

    # -----------------------------
    # File enumeration
    # -----------------------------
    def _list_user_files(self, base_dir: Path) -> List[Path]:
        if not base_dir.exists():
            return []

        out: List[Path] = []
        for p in base_dir.iterdir():
            if not p.is_file():
                continue

            # Ignore internal/system artifacts (common culprits)
            name_l = p.name.lower()
            if name_l.startswith("."):
                continue
            if name_l in {"samples_audit.json"}:
                continue
            if name_l.endswith(".log"):
                continue

            out.append(p)

        return out

    # -----------------------------
    # Document classification
    # -----------------------------
    def _classify_documents(self, files: List[Path], vertical: str) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for p in files:
            prior = self._heuristic_doc_type(p)
            snippet = self._safe_snippet(p)

            llm = self._classify_doc_llm(
                filename=p.name,
                prior=prior,
                snippet=snippet,
            )

            doc_type = str(llm.get("doc_type", prior)).strip().lower()
            if doc_type not in DOC_TYPES:
                doc_type = prior

            confidence = llm.get("confidence", 0.75)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.75
            confidence = max(0.0, min(1.0, confidence))

            reason = llm.get("reason", "")
            if not isinstance(reason, str):
                reason = ""

            # ✅ NEW: vertical fit scoring
            fit = self._assess_vertical_fit_llm(
                vertical=vertical,
                filename=p.name,
                snippet=snippet,
            )
            fit_score = fit.get("fit_score", 0.6)
            try:
                fit_score = float(fit_score)
            except Exception:
                fit_score = 0.6
            fit_score = max(0.0, min(1.0, fit_score))

            vertical_guess = fit.get("vertical_guess", "")
            if not isinstance(vertical_guess, str):
                vertical_guess = ""

            fit_reason = fit.get("reason", "")
            if not isinstance(fit_reason, str):
                fit_reason = ""

            docs.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "doc_type": doc_type,
                    "confidence": confidence,
                    "reason": reason[:300],
                    # ✅ NEW fields
                    "vertical_fit": fit_score,
                    "vertical_guess": vertical_guess[:60],
                    "vertical_fit_reason": fit_reason[:240],
                    "off_vertical": bool(fit_score < 0.5),
                }
            )
        return docs

    def _heuristic_doc_type(self, p: Path) -> str:
        ext = p.suffix.lower()
        name = p.name.lower()

        if ext in {".csv", ".tsv"}:
            return "knowledge_base"
        if ext in {".yaml", ".yml"}:
            # could be policy/procedure/tool_spec; LLM will confirm
            if (
                "policy" in name
                or "refund" in name
                or "terms" in name
                or "compliance" in name
                or "privacy" in name
            ):
                return "policy"
            if "sop" in name or "process" in name or "onboard" in name or "workflow" in name:
                return "procedure"
            return "policy"
        if ext in {".md", ".txt"}:
            if "sop" in name or "process" in name or "onboard" in name:
                return "procedure"
            return "other"
        if ext in {".json"}:
            if "openapi" in name or "swagger" in name or "tool" in name or "api" in name:
                return "tool_spec"
            return "other"

        return "other"

    def _safe_snippet(self, p: Path) -> str:
        """
        Privacy-minimizing snippet:
          - CSV/TSV: header + 3 rows
          - YAML/MD/TXT/JSON: first ~40 lines
        """
        ext = p.suffix.lower()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

        lines = text.splitlines()
        if ext in {".csv", ".tsv"}:
            return "\n".join(lines[:5])
        return "\n".join(lines[:40])

    def _classify_doc_llm(self, *, filename: str, prior: str, snippet: str) -> Dict[str, Any]:
        system = (
            "You classify uploaded customer-service documents into ONE doc_type:\n"
            "knowledge_base, policy, procedure, tool_spec, other.\n"
            "Use filename + snippet. If it contains eligibility/rules/terms, choose policy.\n"
            "If it describes steps/process/onboarding, choose procedure.\n"
            "If it looks like API/tool specs (endpoints/params), choose tool_spec.\n"
            "Return strict JSON: {doc_type, confidence, reason}. confidence is 0..1.\n"
        )
        user = {
            "filename": filename,
            "prior": prior,
            "snippet": snippet[:2000],
        }

        return chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            model=self.model,
            temperature=1.0,
        )

    # -----------------------------
    # LLM agent proposal (generic)
    # -----------------------------
    def _propose_agents_llm(
        self,
        *,
        vertical: str,
        user_goals: str,
        documents: List[Dict[str, Any]],
        max_agents: int,
    ) -> Dict[str, Any]:
        """
        LLM proposes the agent set and per-agent doc usage, without hardcoding
        any specific agents (FAQ/complaint/etc.) in code.
        """
        system = (
            "You are designing a CUSTOMER-SERVICE multi-agent system plan.\n"
            "You must propose a small set of agents that can handle the user's goals.\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Do NOT invent documents. Use only the provided document names.\n"
            "- Keep it minimal (<= max_agents). Prefer reusable agents.\n"
            "- Each agent must specify inputs by DOC TYPE buckets:\n"
            "  knowledge_base, policy, procedure, tool_spec\n"
            "- Documents have vertical_fit/off_vertical flags.\n"
            "  By default, DO NOT attach off_vertical documents to agents.\n"
            "  If there are NO on-vertical documents for a needed bucket, you may attach off_vertical docs,\n"
            "  but then set status='partial' and explain the mismatch in notes.\n"
            "- If a policy doc exists and is on-vertical, share it across relevant agents.\n"
            "Return STRICT JSON with this shape:\n"
            "{\n"
            '  "agents": [\n'
            "    {\n"
            '      "id": string,\n'
            '      "agent_kind": one of [rag, workflow, tool, router, qa, guardrails, other],\n'
            '      "description": string,\n'
            '      "status": one of [ready, partial, missing_docs],\n'
            '      "inputs": {\n'
            '        "knowledge_base": [doc_name...],\n'
            '        "policy": [doc_name...],\n'
            '        "procedure": [doc_name...],\n'
            '        "tool_spec": [doc_name...]\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "notes": [string...]\n'
            "}\n"
        )

        doc_summary = [
            {
                "name": d["name"],
                "doc_type": d["doc_type"],
                "confidence": d.get("confidence", 0.0),
                "reason": d.get("reason", ""),
                # ✅ NEW: vertical fit signals for planning
                "vertical_fit": d.get("vertical_fit", 0.6),
                "vertical_guess": d.get("vertical_guess", ""),
                "off_vertical": bool(d.get("off_vertical", False)),
                "vertical_fit_reason": d.get("vertical_fit_reason", ""),
                "snippet": self._doc_snippet_for_planning(d, max_chars=600),
            }
            for d in documents
        ]

        user = {
            "vertical": vertical,
            "user_goals": user_goals,
            "max_agents": max_agents,
            "documents": doc_summary,
        }

        return chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            model=self.model,
            temperature=1.0,
        )

    def _doc_snippet_for_planning(self, d: Dict[str, Any], max_chars: int = 600) -> str:
        """
        Planning snippet comes from classification snippet already (reason + type).
        Avoid re-reading file here; keep minimal.
        """
        # If you later store snippets in documents, prefer that.
        s = f"{d.get('doc_type','')}: {d.get('reason','')}"
        return (s or "")[:max_chars]

    # -----------------------------
    # Normalize + bridge to runtime keys
    # -----------------------------
    def _normalize_agents_plan(
        self,
        *,
        llm_plan: Dict[str, Any],
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        doc_names = {d["name"] for d in documents}

        raw_agents = llm_plan.get("agents", [])
        if not isinstance(raw_agents, list):
            raw_agents = []

        agents: List[Dict[str, Any]] = []
        for a in raw_agents:
            if not isinstance(a, dict):
                continue

            agent_id = str(a.get("id", "")).strip()
            if not agent_id:
                continue

            agent_kind = str(a.get("agent_kind", "other")).strip().lower()
            if agent_kind not in AGENT_KINDS:
                agent_kind = "other"

            description = a.get("description", "")
            if not isinstance(description, str):
                description = ""

            status = str(a.get("status", "partial")).strip().lower()
            if status not in {"ready", "partial", "missing_docs"}:
                status = "partial"

            inputs = a.get("inputs", {})
            if not isinstance(inputs, dict):
                inputs = {}

            # Ensure list fields and only include known doc names
            typed = {
                "knowledge_base": self._normalize_doc_list(inputs.get("knowledge_base"), doc_names),
                "policy": self._normalize_doc_list(inputs.get("policy"), doc_names),
                "procedure": self._normalize_doc_list(inputs.get("procedure"), doc_names),
                "tool_spec": self._normalize_doc_list(inputs.get("tool_spec"), doc_names),
            }

            # Bridge to runtime/spec_builder-friendly keys (generic mapping)
            # NOTE: this is not hardcoding agents; it's mapping doc TYPES to input KEYS.
            runtime_inputs = {
                "docs": typed["knowledge_base"],
                "policies": typed["policy"],
                "procedures": typed["procedure"],
                "tools": typed["tool_spec"],
            }

            agents.append(
                {
                    "id": agent_id,
                    "agent_kind": agent_kind,
                    "description": description,
                    "status": status,
                    # Keep both for transparency/debugging
                    "inputs_typed": typed,
                    "inputs": runtime_inputs,  # what spec_builder should consume
                }
            )

        # If LLM returns nothing, keep system usable (minimal, generic fallback)
        if not agents:
            agents = self._minimal_generic_fallback(documents)

        return agents

    def _normalize_doc_list(self, v: Any, allowed: set[str]) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for x in v:
            if not isinstance(x, str):
                continue
            x = x.strip()
            if not x:
                continue
            if x in allowed:
                out.append(x)
        # stable unique
        return sorted(set(out))

    def _minimal_generic_fallback(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Only used if LLM fails completely. Still generic: no FAQ/complaint naming.
        """
        kb = [d["name"] for d in documents if d.get("doc_type") == "knowledge_base"]
        pol = [d["name"] for d in documents if d.get("doc_type") == "policy"]
        proc = [d["name"] for d in documents if d.get("doc_type") == "procedure"]
        tool = [d["name"] for d in documents if d.get("doc_type") == "tool_spec"]

        return [
            {
                "id": "customer_service_assistant",
                "agent_kind": "rag" if kb else "other",
                "description": "Generic customer service assistant grounded in available documents.",
                "status": "partial" if documents else "missing_docs",
                "inputs_typed": {
                    "knowledge_base": kb,
                    "policy": pol,
                    "procedure": proc,
                    "tool_spec": tool,
                },
                "inputs": {
                    "docs": kb,
                    "policies": pol,
                    "procedures": proc,
                    "tools": tool,
                },
                "docs_detected": sorted(set(kb + pol + proc + tool)),
            }
        ]

    def _legacy_docs_detected(self, agent: Dict[str, Any]) -> List[str]:
        inputs = agent.get("inputs", {})
        if not isinstance(inputs, dict):
            return []
        legacy: List[str] = []
        for k in ("docs", "policies", "procedures", "tools"):
            v = inputs.get(k)
            if isinstance(v, str):
                legacy.append(v)
            elif isinstance(v, list):
                legacy.extend([x for x in v if isinstance(x, str)])
        return sorted(set(legacy))

    def _assess_vertical_fit_llm(
        self, *, vertical: str, filename: str, snippet: str
    ) -> Dict[str, Any]:
        """
        Returns JSON: {fit_score: 0..1, vertical_guess: str, reason: str}
        fit_score is how well the document matches the selected vertical.
        """
        system = (
            "You assess whether an uploaded customer-service document matches a target vertical.\n"
            "Use filename + snippet to guess the document's likely vertical (e.g., fintech, retail, telecom, insurance, travel).\n"
            "Then score how well it fits the TARGET vertical.\n"
            "Return STRICT JSON: {fit_score, vertical_guess, reason}.\n"
            "fit_score must be a number between 0 and 1.\n"
            "Keep reason short.\n"
        )
        user = {
            "target_vertical": vertical,
            "filename": filename,
            "snippet": snippet[:2000],
        }

        return chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            model=self.model,
            temperature=1.0,
        )
