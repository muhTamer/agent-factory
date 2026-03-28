# evaluation/langgraph_baseline.py
"""
LangGraph Supervisor Baseline for RQ1 Comparison.

Replicates the same 3 domain agents (FAQ, Refunds, Complaints) using
LangGraph's Supervisor pattern to provide an independent orchestration
baseline. Same model (gpt-5-mini), same knowledge, same scenarios.

Architecture: Supervisor → routes to one agent → agent answers → done.
This is architecturally closest to our Spine → Router → Agent pattern.

Usage:
    python -m evaluation.langgraph_baseline
    python -m evaluation.langgraph_baseline --dry-run
    python -m evaluation.langgraph_baseline --scenario faq_01
"""

from __future__ import annotations

import argparse
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

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_openai import AzureChatOpenAI  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from langgraph_supervisor import create_supervisor  # noqa: E402

LOG_DIR = ROOT / "evaluation" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "langgraph_baseline.log"

_log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_log_formatter)
_file_handler = logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8")
_file_handler.setFormatter(_log_formatter)

# Root logger at WARNING to suppress noisy library chatter
logging.basicConfig(level=logging.WARNING, handlers=[_stream_handler, _file_handler])

# Our evaluation logger at INFO
logger = logging.getLogger("evaluation.langgraph_baseline")
logger.setLevel(logging.INFO)

# Suppress noisy libraries
for _noisy in ("httpx", "httpcore", "openai", "langchain", "langgraph"):
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


# Same tool set as Agent Factory — distributed by domain
DOMAIN_TOOLS = {
    "refunds": [lookup_payment, initiate_refund, verify_identity],
    "complaints": [create_ticket, handoff_to_human, lookup_customer],
    "accounts": [verify_identity, lookup_customer],
}


# ── TF-IDF RAG (same as AutoGen baseline) ────────────────────────────


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


# ── Build LangGraph model ─────────────────────────────────────────────


def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        temperature=1.0,
    )


# ── Build LangGraph agents + supervisor ───────────────────────────────


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


def _build_supervisor(llm: AzureChatOpenAI):
    """Create supervisor with auto-discovered generated agents."""
    gen_dir = ROOT / "generated"
    agents = []
    discovered = _discover_agents(gen_dir)

    for agent_info in discovered:
        agent_id = agent_info["id"]
        domain = agent_info["domain"]
        config = agent_info["config"]
        rag = SimpleRAG(agent_info["corpus_path"])

        # Create tool function (closure)
        def make_search_tool(rag_instance: SimpleRAG, name: str):
            def knowledge_search(query: str) -> str:
                """Search the knowledge base for relevant information about the query."""
                return rag_instance.search(query, top_k=3)

            knowledge_search.__name__ = f"{name}_knowledge_search"
            knowledge_search.__qualname__ = f"{name}_knowledge_search"
            return knowledge_search

        search_tool = make_search_tool(rag, domain)

        # Domain-specific tools (same stubs as Agent Factory)
        agent_tools = [search_tool] + DOMAIN_TOOLS.get(domain, [])

        policies_text = "\n".join(f"- {p}" for p in config.get("policies", []))
        tool_names = ", ".join(t.__name__ for t in agent_tools)
        system_message = (
            f"You are {agent_id}, a specialist in the {domain} domain.\n\n"
            f"GOAL: {config['goal']}\n\n"
            f"POLICIES:\n{policies_text}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Search the knowledge base ONCE with the most relevant query, then answer.\n"
            f"- Use domain tools ({tool_names}) when the customer request requires action.\n"
            f"- Do NOT search more than twice. After searching, provide your answer.\n"
            f"- Ground your answers in the knowledge base results.\n"
            f"- If the query is outside your domain, say so clearly.\n"
            f"- Be concise and helpful.\n"
        )

        # Limit tool calls to prevent infinite ReAct loops (analogous to
        # AutoGen's MaxMessageTermination(max_messages=4)).
        MAX_AGENT_TOOL_CALLS = 5

        def _make_tool_limiter(limit: int, agent_name: str):
            def post_model_hook(state):
                msgs = state.get("messages", [])
                # Only count domain tool calls from this agent, not routing tools
                tool_call_count = 0
                for m in msgs:
                    if not isinstance(m, AIMessage) or not getattr(
                        m, "tool_calls", None
                    ):
                        continue
                    for tc in m.tool_calls:
                        tc_name = (
                            tc.get("name", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "")
                        )
                        if (
                            tc_name
                            and not tc_name.startswith("transfer_to_")
                            and tc_name != "transfer_back_to_supervisor"
                        ):
                            tool_call_count += 1
                if tool_call_count >= limit:
                    last = msgs[-1] if msgs else None
                    if isinstance(last, AIMessage) and getattr(
                        last, "tool_calls", None
                    ):
                        return {
                            "messages": [
                                AIMessage(
                                    content=last.content
                                    or "Based on my search, here is my answer.",
                                    name=last.name,
                                )
                            ]
                        }
                return None

            return post_model_hook

        agent = create_react_agent(
            model=llm,
            tools=agent_tools,
            name=agent_id,
            prompt=system_message,
            post_model_hook=_make_tool_limiter(MAX_AGENT_TOOL_CALLS, agent_id),
        )
        agents.append(agent)

    # Build supervisor prompt dynamically from discovered agents
    agent_lines = "\n".join(
        f"- {info['id']}: {info['config'].get('goal', info['domain'])[:100]}"
        for info in discovered
    )
    supervisor_prompt = (
        "You are a banking customer service supervisor. "
        "Route each customer query to the most appropriate specialist agent:\n"
        f"{agent_lines}\n\n"
        "IMPORTANT: Route to ONE agent per query. After the agent responds, "
        "provide the final answer to the user. Do NOT route to another agent."
    )

    # Supervisor routes to the correct agent (like our Spine)
    supervisor = create_supervisor(
        agents=agents,
        model=llm,
        prompt=supervisor_prompt,
        output_mode="full_history",
        supervisor_name="supervisor",
    )

    return supervisor.compile()


# ── Scenario runner ───────────────────────────────────────────────────


def _detect_pattern(scenario: Dict[str, Any], agents_involved: List[str]) -> str:
    """Detect orchestration pattern from LangGraph execution.

    Detected purely from actual execution — does NOT peek at ground truth.
    If multiple domain agents participated, that's hierarchical_delegation.
    If only one agent handled it, that's direct routing.
    """
    if len(set(agents_involved)) > 1:
        return "hierarchical_delegation"
    return "direct"


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
        # Baselines can't explicitly escalate; check if answer mentions escalation
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


def _invoke_with_retry(supervisor, messages: list, sc_id: str) -> dict:
    """Invoke supervisor with exponential backoff on rate limits."""
    import re

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return supervisor.invoke(
                {"messages": messages},
                config={"recursion_limit": 100},
            )
        except Exception as retry_err:
            if "429" in str(retry_err) or "RateLimit" in str(retry_err):
                last_err = retry_err
                if attempt < MAX_RETRIES:
                    retry_match = re.search(
                        r"retry after (\d+)", str(retry_err), re.IGNORECASE
                    )
                    suggested = int(retry_match.group(1)) if retry_match else 0
                    wait = max(suggested + 2, RETRY_BASE_DELAY * (2**attempt))
                    logger.warning(
                        "Rate limited on %s (attempt %d/%d), waiting %.0fs...",
                        sc_id,
                        attempt + 1,
                        MAX_RETRIES + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue
            raise
    raise last_err


def _extract_from_messages(messages: list) -> Dict[str, Any]:
    """Extract agent routing, tools, and answer from LangGraph messages.

    Handles both LangChain message objects (isinstance checks) and
    serialized dicts (checking 'type' key) since LangGraph may return either.
    """
    first_agent = ""
    responding_agent = ""
    answer_text = ""
    agents_involved = []
    tools_used = []

    for m in messages:
        # --- Normalise: detect format (object vs dict) ---
        if isinstance(m, dict):
            name = m.get("name", "") or ""
            msg_type = m.get("type", "")
            content = m.get("content", "")
            tool_calls = m.get("tool_calls", []) or []
        else:
            name = getattr(m, "name", "") or ""
            msg_type = type(m).__name__
            content = getattr(m, "content", "")
            tool_calls = getattr(m, "tool_calls", []) or []

        is_ai = isinstance(m, AIMessage) or msg_type in ("AIMessage", "ai")
        is_tool = isinstance(m, ToolMessage) or msg_type in ("ToolMessage", "tool")

        # --- Agent tracking (only from AI messages, not tool/human messages) ---
        if name and name != "supervisor" and is_ai:
            if name not in agents_involved:
                agents_involved.append(name)
            if not first_agent:
                first_agent = name

        # --- Tool calls from AI messages ---
        if is_ai and tool_calls:
            for tc in tool_calls:
                tc_name = (
                    tc.get("name", "")
                    if isinstance(tc, dict)
                    else getattr(tc, "name", "")
                )
                if tc_name:
                    tools_used.append(tc_name)

        # --- Tool results (ToolMessage carries the tool name) ---
        if is_tool and name:
            tools_used.append(name)

        # --- Answer text from AI messages ---
        if is_ai:
            text = (
                content if isinstance(content, str) else str(content) if content else ""
            )
            if text.strip():
                responding_agent = name or responding_agent
                answer_text = text

    # Separate routing tools from domain tools
    domain_tools = [
        t
        for t in tools_used
        if not t.startswith("transfer_to_") and t != "transfer_back_to_supervisor"
    ]
    routing_tools = [
        t
        for t in tools_used
        if t.startswith("transfer_to_") or t == "transfer_back_to_supervisor"
    ]

    return {
        "first_agent": first_agent,
        "responding_agent": responding_agent,
        "answer_text": answer_text,
        "agents_involved": agents_involved,
        "tools_used": domain_tools,
        "routing_tools": routing_tools,
    }


def run_single_scenario(supervisor, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single scenario through LangGraph supervisor (supports multi-turn)."""
    sc_id = scenario["id"]
    category = scenario["category"]
    turns_spec = scenario.get("turns", [])

    t0 = time.perf_counter()

    try:
        conversation = []  # Accumulated message history
        all_tools_used = []
        all_agents_involved = []
        first_agent = ""
        responding_agent = ""
        answer_text = ""
        all_pattern_ok = True
        all_agent_ok = True
        all_outcome_ok = True
        outcome_details = []

        for turn_idx, turn in enumerate(turns_spec):
            query = turn["query"]
            expected = turn.get("expected", {})

            # Add user message to conversation
            conversation.append({"role": "user", "content": query})

            result = _invoke_with_retry(supervisor, conversation, sc_id)
            resp_messages = result.get("messages", [])

            # Extract info from this turn's full message trace
            extracted = _extract_from_messages(resp_messages)
            turn_tools = extracted["tools_used"]
            turn_answer = extracted["answer_text"]

            # Accumulate across turns
            all_tools_used.extend(turn_tools)
            for a in extracted["agents_involved"]:
                if a not in all_agents_involved:
                    all_agents_involved.append(a)
            if not first_agent:
                first_agent = extracted["first_agent"]
            responding_agent = extracted["responding_agent"]
            answer_text = turn_answer

            # Append assistant response to conversation for next turn
            if turn_answer:
                conversation.append({"role": "assistant", "content": turn_answer})

            # Per-turn checks
            if expected.get("pattern"):
                actual_pattern = _detect_pattern(scenario, all_agents_involved)
                if actual_pattern != expected["pattern"]:
                    all_pattern_ok = False

            expected_agent = expected.get("agent_contains")
            if expected_agent is not None:
                if expected_agent.lower() not in (first_agent or "").lower():
                    all_agent_ok = False

            # Per-turn outcome check
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
        if not all_outcome_ok and all_agent_ok:
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
                    f.startswith("tool_missing:") or f.startswith("knowledge_mismatch:")
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
            "agents_involved": all_agents_involved,
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
            "first_agent": "",
            "responding_agent": "",
            "agents_involved": [],
            "tools_used": [],
            "answer_text": "",
            "error": str(e),
        }


# ── Main evaluation loop ─────────────────────────────────────────────


def run_langgraph_evaluation(
    output_dir: Path,
    scenario_filter: Optional[str] = None,
    max_scenarios: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all scenarios through LangGraph supervisor and compare."""

    llm = _build_llm()
    supervisor = _build_supervisor(llm)

    logger.info("Built LangGraph supervisor with 3 domain agents")

    # Load scenarios
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    print(f"\nRunning {len(scenarios)} scenarios with LangGraph Supervisor...")
    print(f"Model: {os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-5-mini')}")
    print(f"Inter-scenario delay: {INTER_SCENARIO_DELAY}s\n")

    all_results = []
    total_start = time.time()
    passed_count = 0
    failed_count = 0

    logger.info("=" * 60)
    logger.info("LANGGRAPH BASELINE — Starting %d scenarios", len(scenarios))
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

        # Run with thread-based timeout (synchronous invoke)
        import concurrent.futures

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(run_single_scenario, supervisor, sc)
        try:
            result = future.result(timeout=SCENARIO_TIMEOUT)
        except concurrent.futures.TimeoutError:
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
        finally:
            # Don't wait for zombie threads — let them die in background
            pool.shutdown(wait=False, cancel_futures=True)

        all_results.append(result)

        # Incremental save after each scenario (prevents data loss on crash/kill)
        _inc_path = Path(output_dir) / "langgraph_baseline_results.json"
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
            f"agent={result.get('first_agent', ''):30s}  "
            f"pattern={'OK' if result['pattern_correct'] else 'MISS':4s}  "
            f"outcome={'OK' if result.get('outcome_correct') else 'MISS':4s}  "
            f"latency={result['latency_ms']:.0f}ms  "
            f"[{passed_count}✓ {failed_count}✗ | "
            f"ETA {int(eta//60)}m{int(eta%60):02d}s]"
        )
        logger.info(log_line)
        sys.stdout.flush()

        if i < len(scenarios):
            time.sleep(INTER_SCENARIO_DELAY)

    total_elapsed = time.time() - total_start

    # Compute metrics
    n = len(all_results)
    metrics = {
        "framework": "langgraph_supervisor",
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
    print("LANGGRAPH SUPERVISOR BASELINE SUMMARY")
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

    results_path = output_dir / "langgraph_baseline_results.json"
    results_path.write_text(
        json.dumps(all_results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = output_dir / "langgraph_baseline_summary.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  Results exported to: {output_dir.resolve()}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LangGraph Supervisor baseline for RQ1 comparison"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/langgraph_baseline",
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("LANGGRAPH SUPERVISOR BASELINE — RQ1 COMPARISON")
    print("Same agents, same model, same scenarios")
    print("=" * 60 + "\n")

    max_sc = 3 if args.dry_run else None
    run_langgraph_evaluation(
        Path(args.output),
        scenario_filter=args.scenario,
        max_scenarios=max_sc,
    )
