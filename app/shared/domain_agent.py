# app/shared/domain_agent.py
"""
Domain Agent builder — generates a complete domain agent package.

Entrypoint: build_agent(agent_id, inputs, gen_dir) -> Path

Each domain agent is a specialist that combines:
  - Knowledge retrieval (RAG) from domain docs
  - Tool execution via ITool interface
  - ReAct reasoning loop for autonomous decision-making
  - Policy enforcement via prompt constraints
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from app.shared.rag import load_corpus, CorpusItem


def build_agent(
    agent_id: str,
    inputs: dict,
    gen_dir: Path,
) -> Path:
    """
    Domain agent generator.

    Expected inputs:
      - domain: str (e.g. "refunds", "orders", "accounts")
      - goal: str (e.g. "Help customers with refund requests")
      - knowledge_sources: list[str] (file paths: CSV, YAML, MD)
      - available_tools: list[str] (tool names from ToolRegistry)
      - policies: list[str] (natural language policy constraints)
      - max_steps: int (optional, default 10)
      - model: str (optional, default "gpt-5-mini")

    Output:
      - generated/<agent_id>/config.json
      - generated/<agent_id>/corpus.json
      - generated/<agent_id>/agent.py
    """
    gen_dir.mkdir(parents=True, exist_ok=True)

    domain = str(inputs.get("domain", "general")).strip()
    goal = str(inputs.get("goal", f"Help customers in the {domain} domain")).strip()
    knowledge_sources = inputs.get("knowledge_sources") or inputs.get("docs") or []
    available_tools = inputs.get("available_tools") or inputs.get("tools") or []
    policies = inputs.get("policies") or []
    max_steps = int(inputs.get("max_steps", 10))
    model = str(inputs.get("model", "gpt-5-mini"))

    # Ensure lists
    if isinstance(knowledge_sources, str):
        knowledge_sources = [knowledge_sources]
    if isinstance(available_tools, str):
        available_tools = [available_tools]
    if isinstance(policies, str):
        policies = [policies]

    # Document visibility map: {filename: "internal"|"customer_facing"}
    # Set by user during onboarding. Internal docs are agent instructions
    # only; customer-facing docs can be shared with customers.
    doc_vis_map: Dict[str, str] = inputs.get("doc_visibility_map") or {}

    # When knowledge sources are embedded in the spec (cross-container deployment),
    # write them to gen_dir so load_corpus can access them.
    embedded_ks = inputs.get("_embedded_knowledge_sources") or []
    if embedded_ks:
        for ed in embedded_ks:
            (gen_dir / ed["filename"]).write_text(ed["content"], encoding="utf-8")
        knowledge_sources = [str(gen_dir / ed["filename"]) for ed in embedded_ks]

    # ---- Build corpus from knowledge sources ----
    # Skip re-parsing if corpus.json already exists (prebuilt artifacts)
    corpus_path = gen_dir / "corpus.json"
    if corpus_path.exists() and not embedded_ks:
        pass  # reuse existing corpus.json
    else:
        corpus_items: List[CorpusItem] = []
        if knowledge_sources:
            corpus_items = load_corpus(knowledge_sources)

        corpus_data = [
            {
                "text": item.text,
                "source": item.source,
                "kind": item.kind,
                "meta": {
                    **(item.meta or {}),
                    "visibility": doc_vis_map.get(
                        Path(item.source).name, "customer_facing"
                    ),
                },
            }
            for item in corpus_items
        ]
        corpus_path.write_text(
            json.dumps(corpus_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- Write config ----
    cfg: Dict[str, Any] = {
        "id": agent_id,
        "domain": domain,
        "goal": goal,
        "knowledge_sources": [str(p) for p in knowledge_sources],
        "available_tools": [str(t) for t in available_tools],
        "policies": [str(p) for p in policies],
        "max_steps": max_steps,
        "model": model,
    }
    (gen_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- Generate agent.py ----
    agent_src = _generate_agent_source(agent_id)
    (gen_dir / "agent.py").write_text(agent_src, encoding="utf-8")

    return gen_dir


def _generate_agent_source(agent_id: str) -> str:
    """Generate the agent.py wrapper source code."""
    # NOTE: This is a code template, not runtime code.
    # The generated agent.py will import DomainAgentEngine at runtime.
    return textwrap.dedent(f"""\
        # Auto-generated Domain Agent ({agent_id})
        from __future__ import annotations

        import json
        import logging
        from pathlib import Path
        from typing import Dict, Any, Optional

        from app.runtime.interfaces import IAgent
        from app.runtime.domain_agent_engine import DomainAgentEngine, DomainAgentConfig
        from app.shared.rag import CorpusItem, build_index, Index

        _log = logging.getLogger(__name__)


        class Agent(IAgent):
            def __init__(self) -> None:
                self.ready = False
                self.cfg: Dict[str, Any] = {{}}
                self._engine: Optional[DomainAgentEngine] = None

            def load(self, spec: Dict[str, Any]) -> None:
                gen_dir = Path(__file__).parent

                # Load config
                cfg_path = gen_dir / "config.json"
                self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

                # Load corpus and build index
                corpus_path = gen_dir / "corpus.json"
                corpus_data = (
                    json.loads(corpus_path.read_text(encoding="utf-8"))
                    if corpus_path.exists()
                    else []
                )
                corpus_items = [
                    CorpusItem(
                        text=item["text"],
                        source=item["source"],
                        kind=item["kind"],
                        meta=item.get("meta", {{}}),
                    )
                    for item in corpus_data
                ]
                index = build_index(corpus_items)

                # Load tools — register ALL stubs so the policy workflow
                # has every tool it needs available.
                tools = {{}}
                try:
                    from app.runtime.tools.registry import ToolRegistry
                    from app.runtime.tools.adapters.stub import StubTool
                    from app.runtime.tools.stub_tools import STUB_TOOLS, TOOL_DESCRIPTIONS

                    registry = ToolRegistry()
                    for tool_name, stub_fn in STUB_TOOLS.items():
                        desc = TOOL_DESCRIPTIONS.get(tool_name, "")
                        registry.register(tool_name, StubTool(tool_name, stub_fn, description=desc))
                    tools = {{name: registry.get(name) for name in registry.all_names()}}
                except Exception:
                    pass

                # Also try loading HTTP tools from tools_config.json
                try:
                    from app.runtime.tools.registry import ToolRegistry, build_registry

                    factory_dir = Path(".factory")
                    tools_config_path = factory_dir / "tools_config.json"
                    if tools_config_path.exists():
                        tc = json.loads(tools_config_path.read_text(encoding="utf-8"))
                        http_registry = build_registry(tc.get("tools", []))
                        for name in http_registry.all_names():
                            tools[name] = http_registry.get(name)
                except Exception:
                    pass

                # Also try loading MCP tools (shared manager across agents)
                try:
                    _tools_cfg_path = Path(".factory") / "tools_config.json"
                    if _tools_cfg_path.exists():
                        _tc = json.loads(_tools_cfg_path.read_text(encoding="utf-8"))
                        _mcp_configs = _tc.get("mcp_servers", [])
                        if _mcp_configs:
                            from app.runtime.tools.mcp_manager import MCPManager
                            _mcp_mgr = MCPManager.get_instance()
                            if not _mcp_mgr.is_connected():
                                _mcp_mgr.connect_servers(_mcp_configs)
                            _mcp_tools = _mcp_mgr.get_tools()
                            tools.update(_mcp_tools)
                            _log.info("Loaded %d MCP tools", len(_mcp_tools))
                except Exception as exc:
                    _log.warning("MCP tools unavailable: %s", exc)

                # Load LLM function
                llm_fn = None
                try:
                    from app.llm_client import chat_json
                    llm_fn = chat_json
                except ImportError:
                    pass

                # Load memory
                memory = None
                try:
                    from app.runtime.memory import ConversationMemory
                    memory = ConversationMemory()
                except ImportError:
                    pass

                # Load embedding function for dense retrieval (text-embedding-3-small)
                embed_fn = None
                enable_dense = False
                try:
                    from app.runtime.embeddings import get_embed_fn
                    embed_fn = get_embed_fn()
                    enable_dense = True
                    _log.info("Embedding function loaded for agent %s", self.cfg.get("id"))
                except Exception as exc:
                    _log.warning(
                        "Dense retrieval disabled for agent %s: %s",
                        self.cfg.get("id"), exc,
                    )

                # Create engine immediately (TF-IDF works without embeddings)
                config = DomainAgentConfig(
                    agent_id=self.cfg["id"],
                    domain=self.cfg.get("domain", "general"),
                    goal=self.cfg.get("goal", ""),
                    policies=self.cfg.get("policies", []),
                    max_steps=self.cfg.get("max_steps", 10),
                    model=self.cfg.get("model", "gpt-5-mini"),
                    enable_dense_retrieval=enable_dense,
                )

                self._engine = DomainAgentEngine(
                    config=config,
                    index=index,
                    tools=tools,
                    llm_fn=llm_fn,
                    memory=memory,
                    embed_fn=embed_fn,
                    dense_vecs=None,
                )
                self.ready = True

                # Pre-compute embeddings in background so load() returns fast
                if embed_fn and corpus_items:
                    import threading
                    engine_ref = self._engine
                    agent_id_ref = self.cfg.get("id", "?")

                    def _bg_embed():
                        try:
                            _log.info(
                                "Background: computing embeddings for %d items (agent %s)...",
                                len(corpus_items), agent_id_ref,
                            )
                            texts = [item.text for item in corpus_items]
                            vecs = embed_fn(texts)
                            engine_ref._dense_vecs = vecs
                            _log.info(
                                "Background: embeddings ready for agent %s (%d vectors)",
                                agent_id_ref, len(vecs),
                            )
                        except Exception as exc:
                            _log.warning(
                                "Background: embedding failed for agent %s: %s",
                                agent_id_ref, exc,
                            )

                    threading.Thread(target=_bg_embed, daemon=True).start()

            def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
                text = (request.get("text") or request.get("query") or "").strip()
                if not text:
                    return {{"answer": "Please provide a question.", "score": 0.0}}

                # Extract thread_id
                ctx = request.get("context") if isinstance(request, dict) else None
                thread_id = "default"
                if isinstance(ctx, dict) and ctx.get("thread_id"):
                    thread_id = str(ctx["thread_id"])
                elif isinstance(request, dict) and request.get("thread_id"):
                    thread_id = str(request["thread_id"])

                return self._engine.handle(query=text, thread_id=thread_id, context=ctx)

            def metadata(self) -> Dict[str, Any]:
                return {{
                    "id": self.cfg.get("id", "{agent_id}"),
                    "type": "domain_agent",
                    "agent_kind": "domain_agent",
                    "ready": self.ready,
                    "domain": self.cfg.get("domain", "general"),
                    "goal": self.cfg.get("goal", ""),
                    "description": (
                        f"Domain specialist for {{self.cfg.get('domain', 'general')}}. "
                        f"Goal: {{self.cfg.get('goal', 'assist customers')}}. "
                        f"Has {{len(self.cfg.get('available_tools', []))}} tools and "
                        f"knowledge from {{len(self.cfg.get('knowledge_sources', []))}} sources."
                    ),
                    "capabilities": [
                        self.cfg.get("domain", "general"),
                        "multi_turn",
                        "tool_use",
                        "knowledge_retrieval",
                        "reasoning",
                        "domain_agent",
                    ],
                    "available_tools": self.cfg.get("available_tools", []),
                    "vertical": "generic_customer_service",
                }}
    """)
