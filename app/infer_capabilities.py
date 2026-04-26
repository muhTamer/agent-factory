# app/infer_capabilities.py
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml as _yaml

from app.llm_client import chat_json
from app.runtime.tools.stub_tools import STUB_TOOLS

import logging as _logging

_log = _logging.getLogger(__name__)


def _discover_tool_names(data_dir: Path | None = None) -> List[str]:
    """Return canonical tool names from stubs + tools_config (HTTP/MCP)."""
    names = set(STUB_TOOLS.keys())

    cfg_paths = []
    if data_dir:
        cfg_paths.append(Path(data_dir) / ".factory" / "tools_config.json")
    cfg_paths.append(Path(".factory") / "tools_config.json")

    for cfg_path in cfg_paths:
        if not cfg_path.exists():
            continue
        try:
            tc = json.loads(cfg_path.read_text(encoding="utf-8"))
            for tool in tc.get("tools", []):
                if tool.get("_disabled") or not tool.get("name"):
                    continue
                names.add(tool["name"])
            for srv in tc.get("mcp_servers", []):
                if not srv.get("enabled", True) or srv.get("_disabled"):
                    continue
                try:
                    from app.runtime.tools.mcp_manager import MCPManager
                    mgr = MCPManager.get_instance()
                    if not mgr.is_connected():
                        mgr.connect_servers([srv])
                    for tool_name in mgr.get_tools():
                        names.add(tool_name)
                except Exception as exc:
                    _log.debug("MCP discovery skipped for %s: %s", srv.get("id"), exc)
        except Exception:
            pass
    return sorted(names)

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
    "domain_agent",  # unified domain specialist (RAG + tools + ReAct reasoning)
    "rag",  # retrieval-based answering / lookup (legacy)
    "workflow",  # multi-step state machine / process runner (legacy)
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
        pre_classified_docs: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        base_dir = Path(data_dir).resolve()
        files = self._list_user_files(base_dir)

        if pre_classified_docs:
            documents = pre_classified_docs
        else:
            documents = self._classify_documents(files, vertical=vertical)

        # Persist document metadata for downstream components
        self._save_doc_metadata(base_dir, documents)

        # Discover all available tool names (stubs + HTTP + MCP)
        tool_names = _discover_tool_names(base_dir)

        # LLM proposes agents + uses documents (by name) as inputs
        llm_plan = self._propose_agents_llm(
            vertical=vertical,
            user_goals=user_goals,
            documents=documents,
            max_agents=max_agents,
            tool_names=tool_names,
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
            notes=(
                llm_plan.get("notes", [])
                if isinstance(llm_plan.get("notes"), list)
                else []
            ),
        )
        return {
            "vertical": out.vertical,
            "documents": out.documents,
            "agents": out.agents,
            "notes": out.notes,
        }

    # -----------------------------
    # Document metadata persistence
    # -----------------------------
    _DOC_META_FILE = ".doc_metadata.json"

    def _save_doc_metadata(
        self, base_dir: Path, documents: List[Dict[str, Any]]
    ) -> None:
        """Persist document content metadata so downstream components can use it."""
        meta_path = base_dir / self._DOC_META_FILE
        try:
            meta_path.write_text(
                json.dumps(documents, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[INFER] Document metadata saved to {meta_path}")
        except Exception as e:
            print(f"[WARN] Could not save document metadata: {e}")

    @staticmethod
    def load_doc_metadata(data_dir: str | Path) -> List[Dict[str, Any]]:
        """Load persisted document metadata (used by spec_builder, estimators).

        Returns empty list if no metadata file exists yet.
        """
        meta_path = Path(data_dir).resolve() / InferCapabilities._DOC_META_FILE
        if not meta_path.exists():
            return []
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return []

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
    def _classify_documents(
        self, files: List[Path], vertical: str
    ) -> List[Dict[str, Any]]:
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

            # Deep content analysis: categories, topics
            content_analysis = self._analyze_document_content(p)

            docs.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "doc_type": doc_type,
                    "confidence": confidence,
                    "reason": reason[:300],
                    # Vertical fit fields
                    "vertical_fit": fit_score,
                    "vertical_guess": vertical_guess[:60],
                    "vertical_fit_reason": fit_reason[:240],
                    "off_vertical": bool(fit_score < 0.5),
                    # Content analysis fields
                    "content_categories": content_analysis.get(
                        "content_categories", []
                    ),
                    "content_topics": content_analysis.get("content_topics", []),
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
            if (
                "sop" in name
                or "process" in name
                or "onboard" in name
                or "workflow" in name
            ):
                return "procedure"
            return "policy"
        if ext in {".md", ".txt"}:
            if "sop" in name or "process" in name or "onboard" in name:
                return "procedure"
            return "other"
        if ext in {".json"}:
            if (
                "openapi" in name
                or "swagger" in name
                or "tool" in name
                or "api" in name
            ):
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

    # -----------------------------
    # Document content analysis
    # -----------------------------
    def _analyze_document_content(self, p: Path) -> Dict[str, Any]:
        """Deep content analysis: extract categories and topics from document.

        For CSV files: detect category/class columns and extract unique values.
        For YAML files: extract top-level and second-level headers.

        NOTE: Visibility (customer_facing vs internal) is NOT determined here.
        It is a user-provided setting from the onboarding wizard, passed via
        the doc_visibility map to spec_builder.

        Returns dict with:
          - content_categories: list of category values (from CSV Class column)
          - content_topics: list of topic/header strings (from YAML structure)
        """
        ext = p.suffix.lower()
        content_categories: List[str] = []
        content_topics: List[str] = []

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return {"content_categories": [], "content_topics": []}

        if ext in {".csv", ".tsv"}:
            content_categories, _ = self._extract_csv_categories(text, ext)
        elif ext in {".yaml", ".yml"}:
            content_topics, _ = self._extract_yaml_headers(text)
        elif ext in {".md", ".txt"}:
            content_topics, _ = self._extract_text_headers(text)

        return {
            "content_categories": content_categories,
            "content_topics": content_topics,
        }

    def _extract_csv_categories(self, text: str, ext: str) -> tuple[List[str], str]:
        """Extract unique category values from CSV category/class columns.

        Detects columns named: Class, Category, Type, Topic, Department, etc.
        Returns (categories_list, structure_summary_string).
        """
        delimiter = "\t" if ext == ".tsv" else ","
        try:
            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            if reader.fieldnames is None:
                return [], ""

            # Find category-like columns
            _CAT_HINTS = {
                "class",
                "category",
                "type",
                "topic",
                "department",
                "group",
                "section",
            }
            cat_col = None
            for col in reader.fieldnames:
                if col.strip().lower() in _CAT_HINTS:
                    cat_col = col
                    break

            # Also detect Q/A columns for structure summary
            q_col = None
            a_col = None
            _Q_HINTS = {"question", "q", "prompt", "query", "faq_question", "title"}
            _A_HINTS = {"answer", "a", "response", "reply", "content", "text"}
            for col in reader.fieldnames:
                cl = col.strip().lower()
                if cl in _Q_HINTS and not q_col:
                    q_col = col
                if cl in _A_HINTS and not a_col:
                    a_col = col

            categories: set[str] = set()
            row_count = 0
            sample_rows: List[str] = []
            for row in reader:
                row_count += 1
                if cat_col and row.get(cat_col):
                    categories.add(row[cat_col].strip().lower())
                if row_count <= 3 and q_col:
                    q_text = (row.get(q_col) or "")[:100]
                    sample_rows.append(q_text)

            cat_list = sorted(categories)

            # Build structure summary for LLM visibility classification
            summary_parts = [
                f"CSV file with {row_count} rows.",
                f"Columns: {', '.join(reader.fieldnames)}.",
            ]
            if q_col and a_col:
                summary_parts.append(
                    f"Has Q&A structure (question column: '{q_col}', answer column: '{a_col}')."
                )
            if cat_list:
                summary_parts.append(
                    f"Categories found in '{cat_col}' column: {', '.join(cat_list)}."
                )
            if sample_rows:
                summary_parts.append(f"Sample questions: {'; '.join(sample_rows[:3])}")

            return cat_list, " ".join(summary_parts)

        except Exception:
            return [], ""

    def _extract_yaml_headers(self, text: str) -> tuple[List[str], str]:
        """Extract top-level and second-level headers from YAML policy docs.

        Returns (topics_list, structure_summary_string).
        """
        try:
            data = _yaml.safe_load(text)
            if not isinstance(data, dict):
                return [], ""

            topics: List[str] = []
            summary_parts = ["YAML document with sections:"]

            for key in data:
                key_str = str(key)
                topics.append(key_str.lower())
                sub_keys: List[str] = []
                if isinstance(data[key], dict):
                    sub_keys = [str(k) for k in list(data[key].keys())[:10]]
                    topics.extend(sk.lower() for sk in sub_keys)

                if sub_keys:
                    summary_parts.append(f"  {key_str}: {', '.join(sub_keys)}")
                else:
                    val_preview = str(data[key])[:80] if data[key] else ""
                    summary_parts.append(f"  {key_str}: {val_preview}")

            return topics, "\n".join(summary_parts)

        except Exception:
            return [], ""

    def _extract_text_headers(self, text: str) -> tuple[List[str], str]:
        """Extract markdown-style headers from .md/.txt files."""
        topics: List[str] = []
        for line in text.splitlines()[:200]:
            stripped = line.strip()
            if stripped.startswith("#"):
                header = stripped.lstrip("#").strip().lower()
                if header:
                    topics.append(header)
        summary = (
            f"Text document with headers: {', '.join(topics[:20])}" if topics else ""
        )
        return topics, summary

    def _classify_doc_llm(
        self, *, filename: str, prior: str, snippet: str
    ) -> Dict[str, Any]:
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
            temperature=0.2,
            timeout=120,
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
        tool_names: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        LLM proposes the agent set and per-agent doc usage, without hardcoding
        any specific agents (FAQ/complaint/etc.) in code.
        """
        system = (
            "You are designing a CUSTOMER-SERVICE multi-agent system plan.\n"
            "You must propose a small set of DOMAIN AGENTS that can handle the user's goals.\n\n"
            "AGENT TYPE:\n"
            "Every agent MUST be agent_kind='domain_agent'.\n"
            "A domain_agent is a unified domain specialist that combines:\n"
            "  - Knowledge retrieval (RAG) from domain documents\n"
            "  - Tool selection and execution\n"
            "  - ReAct reasoning loop for autonomous decision-making\n"
            "  - Policy enforcement via prompt constraints\n"
            "Each agent specializes in one domain (e.g. refunds, orders, accounts, FAQ).\n\n"
            "REQUIRED FIELDS per agent:\n"
            "  - domain: string (e.g. 'refunds', 'orders', 'faq', 'accounts')\n"
            "  - goal: string (e.g. 'Help customers with refund requests')\n"
            "  - required_tools: list of tool names the agent needs "
            "(MUST use exact names from the available tools list below)\n"
            "  - policies_text: list of natural language policy constraints\n\n"
            "AVAILABLE TOOLS (use these exact names in required_tools):\n"
            f"  {', '.join(tool_names or sorted(STUB_TOOLS.keys()))}\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Do NOT invent tool names. Use ONLY names from the list above.\n"
            "- Do NOT invent documents. Use only the provided document names.\n"
            "- Keep it minimal (<= max_agents). Prefer reusable agents.\n"
            "- Each agent must specify inputs by DOC TYPE buckets:\n"
            "  knowledge_base, policy, procedure, tool_spec\n"
            "- Documents have vertical_fit/off_vertical flags.\n"
            "  By default, DO NOT attach off_vertical documents to agents.\n"
            "  If there are NO on-vertical documents for a needed bucket, you may attach off_vertical docs,\n"
            "  but then set status='partial' and explain the mismatch in notes.\n"
            "- Policy documents (doc_type='policy') provide INTERNAL rules that constrain the agent's\n"
            "  reasoning (eligibility rules, refund thresholds, KYC requirements).\n"
            "  They become 'policies_text' constraints in the agent's ReAct prompt.\n"
            "- Knowledge_base documents (FAQs, help articles) become the agent's retrieval corpus.\n"
            "DOCUMENT CONTENT METADATA:\n"
            "- Documents may have 'content_categories' (e.g., ['accounts','loans','insurance'])\n"
            "  showing the topics they cover. Use this to assign docs to the right agents.\n"
            "- Documents may have 'content_topics' showing structural headers/sections.\n"
            "- The SAME document MUST NOT be assigned to multiple agents (non-redundancy principle).\n"
            "  The orchestrator handles cross-domain routing.\n"
            "AGENT NAMING:\n"
            "- Agent IDs MUST follow the pattern: {domain}_agent (lowercase, underscores).\n"
            "- The 'domain' part should reflect what the agent DOES, not just one document it uses.\n"
            "- If a knowledge_base document covers MULTIPLE categories (e.g. accounts, loans, insurance, cards),\n"
            "  name the agent after its primary FUNCTION, not a single category.\n"
            "  Example: an agent using a multi-category FAQ file → 'faq_agent' (not 'accounts_agent').\n"
            "- If an agent handles a specific policy domain → name it after that domain.\n"
            "  Example: refunds policy → 'refunds_agent', complaints policy → 'complaints_agent'.\n\n"
            "AOP ELIGIBILITY — aop_eligible field:\n"
            "Each agent MUST include an 'aop_eligible' boolean field.\n"
            "aop_eligible=true means the agent can be assigned user-facing subtasks by the orchestrator.\n"
            "Set aop_eligible=true for all customer-facing domain agents.\n"
            "Set aop_eligible=false only for internal-only agents (if any).\n\n"
            "Return STRICT JSON with this shape:\n"
            "{\n"
            '  "agents": [\n'
            "    {\n"
            '      "id": string,\n'
            '      "agent_kind": "domain_agent",\n'
            '      "description": string,\n'
            '      "domain": string,\n'
            '      "goal": string,\n'
            '      "required_tools": [string...],\n'
            '      "policies_text": [string...],\n'
            '      "aop_eligible": boolean,\n'
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
                # Vertical fit signals
                "vertical_fit": d.get("vertical_fit", 0.6),
                "vertical_guess": d.get("vertical_guess", ""),
                "off_vertical": bool(d.get("off_vertical", False)),
                "vertical_fit_reason": d.get("vertical_fit_reason", ""),
                "snippet": self._doc_snippet_for_planning(d, max_chars=600),
                # Content analysis signals
                "content_categories": d.get("content_categories", []),
                "content_topics": d.get("content_topics", []),
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
            temperature=0.2,
            timeout=120,
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
                "knowledge_base": self._normalize_doc_list(
                    inputs.get("knowledge_base"), doc_names
                ),
                "policy": self._normalize_doc_list(inputs.get("policy"), doc_names),
                "procedure": self._normalize_doc_list(
                    inputs.get("procedure"), doc_names
                ),
                "tool_spec": self._normalize_doc_list(
                    inputs.get("tool_spec"), doc_names
                ),
            }

            # Bridge to runtime/spec_builder-friendly keys (generic mapping)
            # NOTE: this is not hardcoding agents; it's mapping doc TYPES to input KEYS.
            runtime_inputs: Dict[str, Any] = {
                "docs": typed["knowledge_base"],
                "policies": typed["policy"],
                "procedures": typed["procedure"],
                "tools": typed["tool_spec"],
            }

            # Domain agent specific fields
            if agent_kind == "domain_agent":
                runtime_inputs["domain"] = str(a.get("domain", agent_id)).strip()
                runtime_inputs["goal"] = str(a.get("goal", description)).strip()
                runtime_inputs["available_tools"] = (
                    a.get("required_tools") or a.get("available_tools") or []
                )
                runtime_inputs["policies_text"] = a.get("policies_text") or []
                # knowledge_sources = knowledge_base + policy docs for RAG corpus
                runtime_inputs["knowledge_sources"] = (
                    typed["knowledge_base"] + typed["policy"]
                )

            # aop_eligible: declarative flag from LLM (default True for
            # backward compat — the spec_builder/aop_coordinator apply
            # their own fallback logic when the flag is absent).
            aop_eligible = a.get("aop_eligible")
            if isinstance(aop_eligible, bool):
                pass  # keep as-is
            elif isinstance(aop_eligible, str):
                aop_eligible = aop_eligible.lower() in ("true", "1", "yes")
            else:
                aop_eligible = None  # unknown — let downstream decide

            agents.append(
                {
                    "id": agent_id,
                    "agent_kind": agent_kind,
                    "description": description,
                    "aop_eligible": aop_eligible,
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

    def _minimal_generic_fallback(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
                "agent_kind": "domain_agent",
                "description": "Generic customer service assistant grounded in available documents.",
                "aop_eligible": True,
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
                    "domain": "general",
                    "goal": "Help customers with general inquiries",
                    "knowledge_sources": kb + pol,
                    "available_tools": [],
                    "policies_text": [],
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

    # -----------------------------
    # Guardrail rule generation
    # -----------------------------
    def generate_guardrail_rules(
        self,
        *,
        vertical: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to generate vertical-specific guardrail rules based on
        the uploaded documents. Returns a dict with 'guardrail_rules' and
        'transaction_slot_keys' ready for PolicyPack.
        """
        doc_summary = []
        for d in documents:
            doc_summary.append(
                {
                    "name": d.get("name", ""),
                    "doc_type": d.get("doc_type", ""),
                    "snippet": d.get("reason", "")[:400],
                }
            )

        system = (
            "You are generating GUARDRAIL RULES for a customer-service AI system.\n"
            "The system uses these rules to detect hallucinated action claims,\n"
            "strip inappropriate tone, and hide internal system details.\n\n"
            "Based on the VERTICAL and DOCUMENTS provided, generate rules in these categories:\n\n"
            "1. hallucination_action_claims (category: safety, severity: high)\n"
            "   - Regex patterns that detect when the agent falsely claims to have\n"
            "     performed a domain action (e.g. 'refund processed', 'prescription issued').\n"
            "   - These patterns should match the ACTION VERBS and DOMAIN NOUNS from the policies.\n\n"
            "2. transaction_context (category: safety, severity: high)\n"
            "   - Regex pattern that detects when the user's query contains a real\n"
            "     transaction/case/record identifier (e.g. order #123, patient MRN).\n"
            "   - Used to avoid false positives in hallucination detection.\n\n"
            "3. tone rules (category: tone, severity: low-medium)\n"
            "   - Patterns for promises the system cannot fulfil (async follow-ups, escalation claims).\n"
            "   - These are usually GENERIC across verticals.\n\n"
            "4. internal rules (category: internal, severity: medium)\n"
            "   - Patterns to strip system jargon and internal file references.\n"
            "   - These are usually GENERIC across verticals.\n\n"
            "5. transaction_slot_keys (list of strings)\n"
            "   - Slot field names that indicate a legitimate multi-turn workflow.\n"
            "   - e.g. ['payment_id', 'transaction_id', 'refund_id'] for fintech.\n\n"
            "IMPORTANT:\n"
            "- Patterns must be valid Python regex (re module, case-insensitive).\n"
            "- Each rule needs: id, label, description, category, severity, enabled (bool), "
            "patterns (list of regex strings).\n"
            "- Generate 4-8 rules total. Focus on domain-specific hallucination and "
            "transaction patterns.\n"
            "- Tone and internal rules can be generic.\n\n"
            "Return STRICT JSON:\n"
            "{\n"
            '  "guardrail_rules": [\n'
            "    {"
            '"id": "...", "label": "...", "description": "...", '
            '"category": "...", "severity": "...", "enabled": true, '
            '"patterns": ["regex..."]'
            "}\n"
            "  ],\n"
            '  "transaction_slot_keys": ["slot_key_1", "slot_key_2"]\n'
            "}\n"
        )

        user = {
            "vertical": vertical,
            "documents": doc_summary,
        }

        try:
            result = chat_json(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                model=self.model,
                temperature=1.0,
                timeout=120,
            )

            # Validate structure
            rules = result.get("guardrail_rules", [])
            if not isinstance(rules, list):
                rules = []

            validated_rules = []
            for r in rules:
                if not isinstance(r, dict):
                    continue
                if not r.get("id") or not r.get("patterns"):
                    continue
                validated_rules.append(
                    {
                        "id": str(r.get("id", "")),
                        "label": str(r.get("label", r.get("id", ""))),
                        "description": str(r.get("description", "")),
                        "category": str(r.get("category", "general")),
                        "severity": str(r.get("severity", "medium")),
                        "enabled": bool(r.get("enabled", True)),
                        "patterns": [
                            str(p) for p in r["patterns"] if isinstance(p, str)
                        ],
                    }
                )

            tx_keys = result.get("transaction_slot_keys", [])
            if not isinstance(tx_keys, list):
                tx_keys = []

            return {
                "guardrail_rules": validated_rules,
                "transaction_slot_keys": [
                    str(k) for k in tx_keys if isinstance(k, str)
                ],
            }

        except Exception as e:
            print(f"[INFER] Guardrail generation failed, using defaults: {e}")
            return {"guardrail_rules": [], "transaction_slot_keys": []}

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
            temperature=0.2,
            timeout=120,
        )
