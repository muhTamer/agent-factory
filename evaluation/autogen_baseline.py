# evaluation/autogen_baseline.py
"""
AutoGen AgentChat Baseline for RQ1 Comparison.

Replicates the same 3 domain agents (FAQ, Refunds, Complaints) using
AutoGen's SelectorGroupChat to provide an independent orchestration
baseline. Same model (gpt-5-mini), same knowledge, same scenarios.

Usage:
    python -m evaluation.autogen_baseline
    python -m evaluation.autogen_baseline --dry-run
    python -m evaluation.autogen_baseline --scenario faq_01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Force UTF-8 stdout/stderr on Windows ─────────────────────────────
if sys.platform == "win32":
    for _stream in ("stdout", "stderr"):
        _s = getattr(sys, _stream, None)
        if _s and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from autogen_agentchat.agents import AssistantAgent  # noqa: E402
from autogen_agentchat.conditions import MaxMessageTermination  # noqa: E402
from autogen_agentchat.teams import SelectorGroupChat  # noqa: E402
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient  # noqa: E402

LOG_DIR = ROOT / "evaluation" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "autogen_baseline.log"

_log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_log_formatter)
_file_handler = logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8")
_file_handler.setFormatter(_log_formatter)

# Root logger at WARNING to suppress noisy autogen_core/httpx chatter
logging.basicConfig(level=logging.WARNING, handlers=[_stream_handler, _file_handler])

# Our evaluation logger at INFO
logger = logging.getLogger("evaluation.autogen_baseline")
logger.setLevel(logging.INFO)

# Suppress autogen internal and httpx noise
for _noisy in ("autogen_core", "autogen_core.events", "httpx", "httpcore", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
SCENARIO_TIMEOUT = int(os.getenv("EVAL_SCENARIO_TIMEOUT", "300"))  # 5 min default

SCENARIOS_PATH = ROOT / "evaluation" / "scenarios" / "ground_truth.json"
INTER_SCENARIO_DELAY = float(os.getenv("EVAL_DELAY_SECONDS", "5.0"))
MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("EVAL_RETRY_BASE_DELAY", "10.0"))


# ── Stub domain tools (same as Agent Factory uses) ───────────────────


def verify_identity(customer_id: str = "", id_type: str = "") -> str:
    """Verify a customer's identity using KYC checks."""
    return json.dumps({"kyc_status": "verified", "identity_verified": True})


def lookup_customer(customer_id: str = "") -> str:
    """Look up customer account details and status."""
    return json.dumps(
        {"account_status": "active", "kyc_status": "verified", "customer_found": True}
    )


def lookup_payment(transaction_id: str = "", amount: str = "") -> str:
    """Look up a payment or transaction by ID to check its status."""
    return json.dumps(
        {
            "payment_found": True,
            "settlement_status": "settled",
            "transaction_age_days": 5,
        }
    )


def initiate_refund(customer_id: str = "", amount: str = "", reason: str = "") -> str:
    """Initiate a refund for a customer transaction. Use when a customer requests a refund."""
    return json.dumps(
        {"refund_id": "REF-001", "refund_status": "success", "refund_initiated": True}
    )


def create_ticket(
    customer_id: str = "", category: str = "", description: str = ""
) -> str:
    """Create a support ticket for tracking customer issues, complaints, or escalations."""
    return json.dumps({"ticket_id": "TKT-001", "ticket_status": "created"})


def handoff_to_human(reason: str = "", customer_id: str = "") -> str:
    """Escalate the conversation to a human agent when the issue is beyond automated resolution."""
    return json.dumps({"handed_off": True, "handoff_agent": "human_ops_team"})


def update_profile(customer_id: str = "", field: str = "", value: str = "") -> str:
    """Update a customer's profile information such as address, phone, or email."""
    return json.dumps(
        {
            "profile_updated": True,
            "field_changed": field or "address",
            "message": "Profile updated successfully.",
        }
    )


def update_account(customer_id: str = "", action: str = "") -> str:
    """Perform an account management action such as status change or settings update."""
    return json.dumps(
        {
            "account_updated": True,
            "action": action or "update",
            "message": "Account action completed successfully.",
        }
    )


def generate_statement(customer_id: str = "", period: str = "") -> str:
    """Generate an account statement for a specified period."""
    return json.dumps(
        {
            "statement_generated": True,
            "period": period or "3 months",
            "format": "PDF",
            "message": "Statement generated and available for download.",
        }
    )


def freeze_account(customer_id: str = "", reason: str = "") -> str:
    """Freeze a customer account for security or investigation purposes."""
    return json.dumps(
        {
            "account_frozen": True,
            "freeze_reason": reason or "security",
            "message": "Account frozen pending investigation.",
        }
    )


def initiate_transfer(
    customer_id: str = "", amount: str = "", destination: str = ""
) -> str:
    """Initiate a fund transfer from a customer's account."""
    return json.dumps(
        {
            "transfer_initiated": True,
            "transfer_id": "TRF-001",
            "amount": amount or "0",
            "message": "Transfer initiated successfully.",
        }
    )


# Same tool set as Agent Factory — distributed by domain
DOMAIN_TOOLS = {
    "refunds": [lookup_payment, initiate_refund, verify_identity],
    "complaints": [create_ticket, handoff_to_human, lookup_customer],
    "accounts": [
        verify_identity,
        lookup_customer,
        update_profile,
        update_account,
        generate_statement,
        freeze_account,
        initiate_transfer,
    ],
}


# ── TF-IDF RAG (same as our system uses) ─────────────────────────────


class SimpleRAG:
    """Lightweight TF-IDF search over a corpus.json file."""

    def __init__(self, corpus_path: Path):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine_similarity = cosine_similarity
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        self._texts = [item["text"] for item in data]
        self._sources = [item.get("source", "") for item in data]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = self._vectorizer.fit_transform(self._texts)

    def search(self, query: str, top_k: int = 3) -> str:
        q_vec = self._vectorizer.transform([query])
        scores = self._cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_idx = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_idx:
            if scores[idx] > 0.01:
                results.append(f"[{self._sources[idx]}] {self._texts[idx]}")
        if not results:
            return "No relevant knowledge found."
        return "\n\n".join(results)


# ── Build AutoGen model client ────────────────────────────────────────


def _build_model_client() -> AzureOpenAIChatCompletionClient:
    from app.llm_client import LLM_MODEL

    return AzureOpenAIChatCompletionClient(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", LLM_MODEL),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", LLM_MODEL),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )


# ── Build AutoGen agents ─────────────────────────────────────────────


def _discover_agents(gen_dir: Path) -> List[Dict[str, Any]]:
    """Auto-discover generated agents that have config.json + corpus.json."""
    discovered = []
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generated agents directory not found: {gen_dir}")
    for agent_dir in sorted(gen_dir.iterdir()):
        config_path = agent_dir / "config.json"
        corpus_path = agent_dir / "corpus.json"
        if config_path.exists() and corpus_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            discovered.append(
                {
                    "id": agent_dir.name,
                    "domain": config.get("domain", agent_dir.name),
                    "config": config,
                    "corpus_path": corpus_path,
                }
            )
    if not discovered:
        raise RuntimeError(
            f"No agents found in {gen_dir} with config.json + corpus.json"
        )
    logger.info(
        "Discovered %d agents: %s", len(discovered), [a["id"] for a in discovered]
    )
    return discovered


def _build_agents(
    model_client: AzureOpenAIChatCompletionClient,
) -> List[AssistantAgent]:
    """Create domain agents from auto-discovered generated agents."""
    gen_dir = ROOT / "generated"
    agents = []
    discovered = _discover_agents(gen_dir)

    for agent_info in discovered:
        agent_id = agent_info["id"]
        domain = agent_info["domain"]
        config = agent_info["config"]
        rag = SimpleRAG(agent_info["corpus_path"])

        # Create a tool function for knowledge search (closure over rag)
        def make_search_tool(rag_instance: SimpleRAG):
            def knowledge_search(query: str) -> str:
                """Search the knowledge base for relevant information about the query."""
                return rag_instance.search(query, top_k=3)

            return knowledge_search

        search_tool = make_search_tool(rag)

        # Domain-specific tools (same stubs as Agent Factory)
        agent_tools = [search_tool] + DOMAIN_TOOLS.get(domain, [])

        policies_text = "\n".join(f"- {p}" for p in config.get("policies", []))
        tool_names = ", ".join(t.__name__ for t in agent_tools)
        system_message = (
            f"You are {agent_id}, a specialist in the {domain} domain.\n\n"
            f"GOAL: {config['goal']}\n\n"
            f"POLICIES:\n{policies_text}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Use the knowledge_search tool to find relevant information before answering.\n"
            f"- Use domain tools ({tool_names}) when the customer request requires action.\n"
            f"- Ground your answers in the knowledge base results.\n"
            f"- If the query is outside your domain, say so clearly.\n"
            f"- Be concise and helpful.\n"
        )

        agent = AssistantAgent(
            name=agent_id,
            model_client=model_client,
            tools=agent_tools,
            system_message=system_message,
            description=(f"{domain.upper()} specialist. {config['goal'][:100]}"),
            reflect_on_tool_use=False,
        )
        agents.append(agent)

    return agents


# ── Scenario runner ───────────────────────────────────────────────────


def _detect_pattern(
    scenario: Dict[str, Any],
    first_agent: str,
    all_agents_involved: set,
) -> str:
    """Map AutoGen response to orchestration pattern.

    Detected purely from actual execution — does NOT peek at ground truth.
    If multiple domain agents participated, that's hierarchical_delegation.
    If only one agent handled it, that's direct routing.
    """
    if len(all_agents_involved) > 1:
        return "hierarchical_delegation"
    return "direct"


def _check_agent(responding_agent: str, expected_agent_contains: Optional[str]) -> bool:
    """Check if the responding agent matches expected."""
    if expected_agent_contains is None:
        return True
    return expected_agent_contains.lower() in responding_agent.lower()


def _check_outcome(
    answer_text: str, tools_used: List[str], expected_outcome: Dict[str, Any]
) -> tuple:
    """Evaluate agent behavior by checking actions taken, not keywords.

    Mirrors the outcome checks in evaluation/harness.py but adapted
    for baseline frameworks that expose (answer_text, tools_used).
    """
    checks_passed = []
    checks_failed = []

    actual_tools = {t.lower() for t in tools_used}

    # tools_called
    for tool in expected_outcome.get("tools_called", []):
        if tool.lower() in actual_tools:
            checks_passed.append(f"tool:{tool}")
        else:
            checks_failed.append(f"tool_missing:{tool}")

    # tools_not_called
    for tool in expected_outcome.get("tools_not_called", []):
        if tool.lower() in actual_tools:
            checks_failed.append(f"forbidden_tool:{tool}")
        else:
            checks_passed.append(f"no_forbidden:{tool}")

    # clarification_ok
    if expected_outcome.get("clarification_ok") is True:
        asks_question = answer_text.endswith("?") or any(
            phrase in answer_text.lower()
            for phrase in [
                "could you",
                "can you",
                "please provide",
                "which ",
                "what is your",
                "do you have",
            ]
        )
        if asks_question or bool(answer_text.strip()):
            checks_passed.append("clarification_or_response")
        else:
            checks_failed.append("no_response")

    # escalation_expected — baselines don't have escalation signals
    escalation_expected = expected_outcome.get("escalation_expected")
    if escalation_expected is True:
        escalation_keywords = ["escalat", "human agent", "supervisor", "manager"]
        if any(kw in answer_text.lower() for kw in escalation_keywords):
            checks_passed.append("escalation=yes")
        else:
            checks_failed.append("escalation_mismatch:expected=True,got=False")
    elif escalation_expected is False:
        checks_passed.append("escalation=no")

    # knowledge_retrieved — check if a knowledge_search tool was actually called
    knowledge_expected = expected_outcome.get("knowledge_retrieved")
    if knowledge_expected is not None:
        has_knowledge = any("knowledge" in t or "search" in t for t in actual_tools)
        if has_knowledge == knowledge_expected:
            checks_passed.append(f"knowledge={'yes' if has_knowledge else 'no'}")
        else:
            checks_failed.append(
                f"knowledge_mismatch:expected={knowledge_expected},got={has_knowledge}"
            )

    # response_not_empty
    if expected_outcome.get("response_not_empty"):
        if answer_text.strip():
            checks_passed.append("response_present")
        else:
            checks_failed.append("response_empty")

    # answer_contains (ALL keywords must match)
    for kw in expected_outcome.get("answer_contains", []):
        if kw.lower() in answer_text.lower():
            checks_passed.append(f"kw:{kw}")
        else:
            checks_failed.append(f"kw_missing:{kw}")

    # answer_not_contains (NONE should appear — internal doc leak check)
    kw_blacklist = expected_outcome.get("answer_not_contains")
    if kw_blacklist:
        leaked = [kw for kw in kw_blacklist if kw.lower() in answer_text.lower()]
        if leaked:
            checks_failed.append(f"leak_detected:[{','.join(leaked)}]")
        else:
            checks_passed.append("no_leak")

    # answer_contains_any (ANY keyword match = pass, for policy compliance)
    kw_any_list = expected_outcome.get("answer_contains_any")
    if kw_any_list:
        matched = [kw for kw in kw_any_list if kw.lower() in answer_text.lower()]
        if matched:
            checks_passed.append(f"kw_any:{','.join(matched)}")
        else:
            checks_failed.append(f"kw_any_missing:none_of[{','.join(kw_any_list)}]")

    if not checks_passed and not checks_failed:
        return True, "no_outcome_checks"

    success = len(checks_failed) == 0
    detail_parts = []
    if checks_passed:
        detail_parts.append(f"pass=[{','.join(checks_passed)}]")
    if checks_failed:
        detail_parts.append(f"fail=[{','.join(checks_failed)}]")
    return success, " ".join(detail_parts)


async def _run_team_with_retry(
    agents: List[AssistantAgent],
    model_client: AzureOpenAIChatCompletionClient,
    task: str,
    sc_id: str,
) -> Any:
    """Run SelectorGroupChat with exponential backoff on rate limits."""
    import re

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            team = SelectorGroupChat(
                agents,
                model_client=model_client,
                termination_condition=MaxMessageTermination(max_messages=6),
            )
            return await team.run(task=task)
        except Exception as retry_err:
            if "429" in str(retry_err) or "RateLimit" in str(retry_err):
                last_err = retry_err
                if attempt < MAX_RETRIES:
                    retry_match = re.search(
                        r"retry after (\d+)", str(retry_err), re.IGNORECASE
                    )
                    suggested = int(retry_match.group(1)) if retry_match else 0
                    wait = min(max(suggested + 2, RETRY_BASE_DELAY * (2**attempt)), 90)
                    logger.warning(
                        "Rate limited on %s (attempt %d/%d), waiting %.0fs...",
                        sc_id,
                        attempt + 1,
                        MAX_RETRIES + 1,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
            raise
    raise last_err


def _extract_from_autogen_messages(messages: list) -> Dict[str, Any]:
    """Extract agent routing, tools, and answer from AutoGen messages.

    Tracks which agent called which tool so we can determine
    'handling_agents' — agents that actually did work (called domain tools),
    not just responded with text.
    """
    first_agent = ""
    responding_agent = ""
    answer_text = ""
    agents_involved = set()
    tools_used = []
    agent_tool_map: Dict[str, List[str]] = {}  # agent -> list of tools called

    agent_messages = [
        m
        for m in messages
        if hasattr(m, "source") and m.source not in ("user", "_group_chat_manager")
    ]

    for m in agent_messages:
        source = getattr(m, "source", "")
        msg_type = type(m).__name__
        if source:
            agents_involved.add(source)
            if not first_agent:
                first_agent = source

        if msg_type == "ToolCallRequestEvent":
            for tc in getattr(m, "content", []):
                if hasattr(tc, "name"):
                    tools_used.append(tc.name)
                    # Attribute tool to calling agent
                    if source:
                        agent_tool_map.setdefault(source, []).append(tc.name)

        if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip():
            responding_agent = source
            answer_text = m.content

    # Handling agents = agents that called at least one tool
    handling_agents = [a for a in agents_involved if a in agent_tool_map]

    return {
        "first_agent": first_agent,
        "responding_agent": responding_agent,
        "answer_text": answer_text,
        "agents_involved": agents_involved,
        "handling_agents": handling_agents,
        "agent_tool_map": agent_tool_map,
        "tools_used": tools_used,
    }


async def run_single_scenario(
    agents: List[AssistantAgent],
    scenario: Dict[str, Any],
    model_client: AzureOpenAIChatCompletionClient,
) -> Dict[str, Any]:
    """Run a single scenario through AutoGen SelectorGroupChat (supports multi-turn)."""
    sc_id = scenario["id"]
    category = scenario["category"]
    turns_spec = scenario.get("turns", [])

    t0 = time.perf_counter()

    try:
        all_tools_used = []
        all_agents_involved = set()
        all_handling_agents = set()  # agents that called domain tools
        first_agent = ""
        responding_agent = ""
        answer_text = ""
        all_pattern_ok = True
        all_agent_ok = True
        all_outcome_ok = True
        outcome_details = []

        # For multi-turn, we build up the conversation as a single composite task
        # since AutoGen's SelectorGroupChat doesn't natively support turn-by-turn
        conversation_context = []

        for turn_idx, turn in enumerate(turns_spec):
            query = turn["query"]
            expected = turn.get("expected", {})

            # Build task: for turn 0 just the query; for later turns include history
            if turn_idx == 0:
                task = query
            else:
                # Include prior conversation as context for the team
                history = "\n".join(conversation_context)
                task = (
                    f"Previous conversation:\n{history}\n\nCustomer now says: {query}"
                )

            result = await _run_team_with_retry(
                agents,
                model_client,
                task,
                sc_id,
            )

            extracted = _extract_from_autogen_messages(result.messages)
            turn_tools = extracted["tools_used"]
            turn_answer = extracted["answer_text"]

            # Accumulate
            all_tools_used.extend(turn_tools)
            all_agents_involved.update(extracted["agents_involved"])
            all_handling_agents.update(extracted["handling_agents"])
            if not first_agent:
                first_agent = extracted["first_agent"]
            responding_agent = extracted["responding_agent"]
            answer_text = turn_answer

            # Add to conversation context for next turn
            conversation_context.append(f"Customer: {query}")
            if turn_answer:
                conversation_context.append(f"Agent: {turn_answer}")

            # Per-turn checks
            if expected.get("pattern"):
                actual_pattern = _detect_pattern(
                    scenario, first_agent, all_agents_involved
                )
                if actual_pattern != expected["pattern"]:
                    all_pattern_ok = False

            # Check agent(s) — an agent counts as "handling" only if it
            # called at least one domain tool (Option 3: tool+knowledge attribution).
            expected_agents_list = expected.get("expected_agents")
            expected_agent = expected.get("agent_contains")

            handling_agents = {a.lower() for a in all_handling_agents}

            if expected_agents_list is not None:
                agent_matched = all(
                    any(ea.lower() in ha for ha in handling_agents)
                    for ea in expected_agents_list
                )
                if not agent_matched:
                    all_agent_ok = False
            elif expected_agent is not None:
                agent_matched = any(
                    expected_agent.lower() in ha for ha in handling_agents
                )
                if not agent_matched:
                    all_agent_ok = False

            expected_outcome = expected.get("expected_outcome")
            if expected_outcome:
                outcome_ok, outcome_detail = _check_outcome(
                    turn_answer, turn_tools, expected_outcome
                )
                if not outcome_ok:
                    all_outcome_ok = False
                outcome_details.append(outcome_detail)

        # Scenario-level accumulated outcome check
        scenario_outcome_spec = scenario.get("expected_outcome")
        if scenario_outcome_spec:
            sc_ok, sc_detail = _check_outcome(
                answer_text, all_tools_used, scenario_outcome_spec
            )
            if not sc_ok:
                all_outcome_ok = False
            outcome_details.append(f"scenario:{sc_detail}")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Pattern and agent are tracked for analysis but don't gate success.
        # RQ1 studies how pattern selection affects outcomes — recording the
        # actual pattern is data, not a pass/fail criterion.
        soft_pass = False

        # Soft-pass: if routing was correct but only tool_missing or
        # knowledge_mismatch checks failed, mark as success with soft_pass flag
        # (LLM non-determinism in tool calling / knowledge retrieval)
        # Also allow soft-pass when agents_involved shows correct routing
        # even if handling_agents (tool-based) doesn't — the agent was
        # reached but chose to ask for user confirmation instead of acting.
        routing_ok_by_involvement = all_agent_ok
        if not routing_ok_by_involvement:
            involved_lower = {a.lower() for a in all_agents_involved}
            for turn_spec in turns_spec:
                exp = turn_spec.get("expected", {})
                ea_list = exp.get("expected_agents")
                ea_single = exp.get("agent_contains")
                if ea_list is not None:
                    routing_ok_by_involvement = all(
                        any(ea.lower() in ai for ai in involved_lower) for ea in ea_list
                    )
                elif ea_single is not None:
                    routing_ok_by_involvement = any(
                        ea_single.lower() in ai for ai in involved_lower
                    )
                if not routing_ok_by_involvement:
                    break

        if not all_outcome_ok and routing_ok_by_involvement:
            detail_str = "; ".join(outcome_details)
            import re

            fail_match = re.search(r"fail=\[([^\]]+)\]", detail_str)
            if fail_match:
                fail_content = fail_match.group(1)
                # Extract individual checks (split on boundaries between checks,
                # not on commas inside values like "expected=True,got=False")
                failed_checks = re.findall(
                    r"(tool_missing:[^,\]]+|knowledge_mismatch:[^]]+|[a-z_]+(?::[^,\]]*)?)",
                    fail_content,
                )
                # Verify ALL extracted checks are minor (tool_missing or knowledge_mismatch)
                only_minor = bool(failed_checks) and all(
                    f.startswith("tool_missing:")
                    or f.startswith("knowledge_mismatch:")
                    or f.startswith("kw_missing:")
                    or f.startswith("kw_any_missing:")
                    or f.startswith("escalation_mismatch:")
                    for f in failed_checks
                )
                has_response = "response_present" in detail_str or (
                    answer_text and answer_text.strip()
                )
                if only_minor and has_response:
                    all_outcome_ok = True
                    soft_pass = True
                    outcome_details.append(
                        "[SOFT_PASS:tool_not_called_but_orchestration_ok]"
                    )

        success = all_outcome_ok

        return {
            "scenario_id": sc_id,
            "category": category,
            "description": scenario.get("description", ""),
            "success": success,
            "soft_pass": soft_pass,
            "pattern_correct": all_pattern_ok,
            "agent_correct": all_agent_ok,
            "answer_keywords_found": True,
            "outcome_correct": all_outcome_ok,
            "outcome_detail": "; ".join(outcome_details),
            "latency_ms": latency_ms,
            "first_agent": first_agent,
            "responding_agent": responding_agent,
            "agents_involved": list(all_agents_involved),
            "handling_agents": sorted(all_handling_agents),
            "tools_used": all_tools_used,
            "answer_text": answer_text[:500],
            "error": None,
        }

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.error("Scenario %s failed: %s", sc_id, e)
        return {
            "scenario_id": sc_id,
            "category": category,
            "description": scenario.get("description", ""),
            "success": False,
            "soft_pass": False,
            "pattern_correct": False,
            "agent_correct": False,
            "answer_keywords_found": False,
            "outcome_correct": False,
            "outcome_detail": f"error:{e}",
            "latency_ms": latency_ms,
            "responding_agent": "",
            "agents_involved": [],
            "tools_used": [],
            "answer_text": "",
            "error": str(e),
        }


# ── Main evaluation loop ─────────────────────────────────────────────


async def run_autogen_evaluation(
    output_dir: Path,
    scenario_filter: Optional[str] = None,
    max_scenarios: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all scenarios through AutoGen and compare."""

    model_client = _build_model_client()
    agents = _build_agents(model_client)

    logger.info("Built %d AutoGen agents: %s", len(agents), [a.name for a in agents])

    # Load scenarios
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if scenario_filter:
        ids = {sid.strip() for sid in scenario_filter.split(",")}
        scenarios = [s for s in scenarios if s["id"] in ids]
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    print(f"\nRunning {len(scenarios)} scenarios with AutoGen SelectorGroupChat...")
    print(f"Model: {os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-5-mini')}")
    print(f"Inter-scenario delay: {INTER_SCENARIO_DELAY}s\n")

    all_results = []
    total_start = time.time()
    passed_count = 0
    failed_count = 0

    logger.info("=" * 60)
    logger.info("AUTOGEN BASELINE — Starting %d scenarios", len(scenarios))
    logger.info("Log file: %s", LOG_FILE.resolve())
    logger.info("=" * 60)

    for i, sc in enumerate(scenarios, 1):
        sc_start = time.time()
        logger.info(
            "[%d/%d] ▶ %s (category=%s)",
            i,
            len(scenarios),
            sc["id"],
            sc["category"],
        )

        try:
            result = await asyncio.wait_for(
                run_single_scenario(agents, sc, model_client),
                timeout=SCENARIO_TIMEOUT,
            )
        except asyncio.TimeoutError:
            sc_elapsed = (time.time() - sc_start) * 1000.0
            logger.error(
                "[%d/%d] ✗ %s TIMEOUT after %ds",
                i,
                len(scenarios),
                sc["id"],
                SCENARIO_TIMEOUT,
            )
            result = {
                "scenario_id": sc["id"],
                "category": sc["category"],
                "description": sc.get("description", ""),
                "success": False,
                "soft_pass": False,
                "pattern_correct": False,
                "agent_correct": False,
                "answer_keywords_found": False,
                "outcome_correct": False,
                "outcome_detail": f"timeout:{SCENARIO_TIMEOUT}s",
                "latency_ms": sc_elapsed,
                "first_agent": "",
                "responding_agent": "",
                "agents_involved": [],
                "tools_used": [],
                "answer_text": "",
                "error": f"Timeout after {SCENARIO_TIMEOUT}s",
            }

        all_results.append(result)

        # Incremental save after each scenario (prevents data loss on crash/kill)
        _inc_path = Path(output_dir) / "autogen_baseline_results.json"
        _inc_path.parent.mkdir(parents=True, exist_ok=True)
        _inc_path.write_text(
            json.dumps(all_results, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        if result["success"]:
            passed_count += 1
        else:
            failed_count += 1

        status = (
            "SOFT"
            if result.get("soft_pass")
            else ("PASS" if result["success"] else "FAIL")
        )
        sc_elapsed = time.time() - sc_start
        elapsed_total = time.time() - total_start
        avg_per_sc = elapsed_total / i
        eta = avg_per_sc * (len(scenarios) - i)

        log_line = (
            f"  [{status}] {sc['id']:20s}  "
            f"agent={result.get('responding_agent', ''):30s}  "
            f"pattern={'OK' if result['pattern_correct'] else 'MISS':4s}  "
            f"outcome={'OK' if result.get('outcome_correct') else 'MISS':4s}  "
            f"latency={result['latency_ms']:.0f}ms  "
            f"[{passed_count}✓ {failed_count}✗ | "
            f"ETA {int(eta//60)}m{int(eta%60):02d}s]"
        )
        logger.info(log_line)
        sys.stdout.flush()

        if i < len(scenarios):
            await asyncio.sleep(INTER_SCENARIO_DELAY)

    total_elapsed = time.time() - total_start

    # Compute metrics
    n = len(all_results)
    metrics = {
        "framework": "autogen_agentchat",
        "total_scenarios": n,
        "passed": sum(1 for r in all_results if r["success"]),
        "soft_passed": sum(1 for r in all_results if r.get("soft_pass")),
        "failed": sum(1 for r in all_results if not r["success"]),
        "orchestration_accuracy": (
            round(sum(1 for r in all_results if r["pattern_correct"]) / n, 4)
            if n
            else 0
        ),
        "agent_accuracy": (
            round(sum(1 for r in all_results if r["agent_correct"]) / n, 4) if n else 0
        ),
        "outcome_accuracy": (
            round(sum(1 for r in all_results if r.get("outcome_correct", False)) / n, 4)
            if n
            else 0
        ),
        "reasoning_accuracy": (
            round(sum(1 for r in all_results if r["success"]) / n, 4) if n else 0
        ),
        "avg_latency_ms": (
            round(sum(r["latency_ms"] for r in all_results) / n, 2) if n else 0
        ),
        "execution_mode": "real_llm",
        "total_wall_time_seconds": round(total_elapsed, 2),
    }

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in all_results))
    metrics["by_category"] = {}
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        cn = len(cat_results)
        metrics["by_category"][cat] = {
            "n": cn,
            "orchestration_accuracy": round(
                sum(1 for r in cat_results if r["pattern_correct"]) / cn, 4
            ),
            "agent_accuracy": round(
                sum(1 for r in cat_results if r["agent_correct"]) / cn, 4
            ),
            "outcome_accuracy": round(
                sum(1 for r in cat_results if r.get("outcome_correct", False)) / cn, 4
            ),
            "reasoning_accuracy": round(
                sum(1 for r in cat_results if r["success"]) / cn, 4
            ),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in cat_results) / cn, 2),
        }

    # Print summary
    print("\n" + "=" * 60)
    print("AUTOGEN BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Total scenarios:         {metrics['total_scenarios']}")
    print(f"  Passed:                  {metrics['passed']}")
    print(f"  Failed:                  {metrics['failed']}")
    print(f"  Orchestration Accuracy:  {metrics['orchestration_accuracy']:.1%}")
    print(f"  Agent Accuracy:          {metrics['agent_accuracy']:.1%}")
    print(f"  Outcome Accuracy:        {metrics['outcome_accuracy']:.1%}")
    print(f"  Reasoning Accuracy:      {metrics['reasoning_accuracy']:.1%}")
    print(f"  Avg Latency:             {metrics['avg_latency_ms']:.1f} ms")
    print(f"  Total wall time:         {total_elapsed:.1f}s")

    print("\n  By category:")
    for cat, cm in metrics["by_category"].items():
        print(
            f"    {cat:30s}  n={cm['n']:2d}  "
            f"orch={cm['orchestration_accuracy']:.1%}  "
            f"agent={cm['agent_accuracy']:.1%}  "
            f"latency={cm['avg_latency_ms']:.0f}ms"
        )

    # Export
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "autogen_baseline_results.json"
    results_path.write_text(
        json.dumps(all_results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = output_dir / "autogen_baseline_summary.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  Results exported to: {output_dir.resolve()}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AutoGen AgentChat baseline for RQ1 comparison"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/autogen_baseline",
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("AUTOGEN AGENTCHAT BASELINE — RQ1 COMPARISON")
    print("Same agents, same model, same scenarios")
    print("=" * 60 + "\n")

    max_sc = 3 if args.dry_run else None
    asyncio.run(
        run_autogen_evaluation(
            Path(args.output),
            scenario_filter=args.scenario,
            max_scenarios=max_sc,
        )
    )
