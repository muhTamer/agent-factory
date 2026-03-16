# app/shared/workflow.py
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from string import Template

# Directories to search when resolving file references from slot descriptions
_DATA_DIRS = [Path("data"), Path(".workspace"), Path("data/uploads")]


def _resolve_file_hint(hint: str) -> str | None:
    """Find a file in _DATA_DIRS whose name matches the hint (case-insensitive)."""
    hint_lower = hint.lower()
    # Also try segments: "refunds_policy" → ["refunds_policy", "refunds", "policy"]
    segments = [hint_lower] + [
        s for s in hint_lower.replace(".", "_").split("_") if len(s) > 3
    ]
    for d in _DATA_DIRS:
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            fname = f.name.lower()
            if any(seg in fname for seg in segments):
                return str(f.resolve())
    return None


def _extract_file_refs_from_slots(
    workflow_spec: dict, docs: list, policies: list
) -> None:
    """Scan workflow slot descriptions for file references and add to context.

    Mutates ``docs`` and ``policies`` in-place so resolved paths are
    available to the generated agent as internal context.
    """
    _FILE_RE = re.compile(r"\b([\w.-]+\.(?:yaml|yml|json|csv|txt|md))\b", re.IGNORECASE)
    _UPLOAD_RE = re.compile(r"<UPLOAD:([^>]+)>", re.IGNORECASE)

    for slot_name, meta in (workflow_spec.get("slots") or {}).items():
        if not isinstance(meta, dict):
            continue
        desc = str(meta.get("description", ""))
        is_policy_slot = "policy" in slot_name.lower()

        # Collect file hints from <UPLOAD:...> tags and literal filenames
        hints: list[str] = []
        for m in _UPLOAD_RE.finditer(desc):
            hints.append(m.group(1).strip())
        for m in _FILE_RE.finditer(desc):
            hints.append(m.group(1).strip())

        for hint in hints:
            resolved = _resolve_file_hint(hint)
            if not resolved:
                continue
            target = policies if (is_policy_slot or "policy" in hint.lower()) else docs
            if resolved not in target:
                target.append(resolved)
                print(
                    f"[GEN][AUTO-REF] slot '{slot_name}' -> {resolved} ({'policy' if target is policies else 'doc'})"
                )


def build_agent(agent_id: str, inputs: dict, gen_dir: Path) -> Path:
    """
    Generic workflow agent generator.

    Expected inputs:
      - workflow_spec: dict (LLM-generated, validated earlier)
      - docs: optional list[str]
      - policies: optional list[str]
      - tools: optional list[str]
      - policy_config: optional dict (policy auto-event resolution config)

    Output:
      - generated/<agent_id>/workflow_spec.json
      - generated/<agent_id>/config.json
      - generated/<agent_id>/agent.py (thin wrapper around GenericWorkflowEngine + LLM event mapper)
    """
    gen_dir.mkdir(parents=True, exist_ok=True)

    workflow_spec = inputs.get("workflow_spec")
    if not isinstance(workflow_spec, dict):
        raise ValueError(
            "workflow.build_agent requires inputs['workflow_spec'] as dict"
        )

    docs = inputs.get("docs") or []
    policies = inputs.get("policies") or []
    tools = inputs.get("tools") or []
    policy_config = inputs.get("policy_config") or {}

    if isinstance(docs, str):
        docs = [docs]
    if isinstance(policies, str):
        policies = [policies]
    if isinstance(tools, str):
        tools = [tools]

    # ── Auto-discover internal docs/policies from slot descriptions ──
    # LLM-generated workflow specs often reference files (e.g. refunds_policy.yaml)
    # in slot descriptions but the factory spec may not list them as explicit inputs.
    # Scan slots for file hints and resolve them into the agent's context so they
    # are available internally (never exposed to customers).
    _extract_file_refs_from_slots(workflow_spec, docs, policies)

    # Write workflow spec beside the agent for runtime loading
    wf_path = gen_dir / "workflow_spec.json"
    wf_path.write_text(json.dumps(workflow_spec, indent=2), encoding="utf-8")

    # Write a small config that can include doc/policy/tool context
    cfg = {
        "id": agent_id,
        "workflow_spec_path": str(wf_path),
        "docs": docs,
        "policies": policies,
        "tools": tools,
        "policy_config": policy_config,
    }
    cfg_path = gen_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Generate the wrapper agent.py
    # NOTE: use Template ($agent_id) to avoid f-string brace collisions inside generated code
    agent_src = Template(
        textwrap.dedent(
            """\
        # Auto-generated Workflow Runner agent ($agent_id)
        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import Dict, Any, List, Optional

        from app.runtime.interfaces import IAgent
        from app.runtime.workflow_engine import GenericWorkflowEngine
        from app.runtime.workflow_mapper import map_query_to_event_and_slots

        try:
            from app.runtime.policy.policy_compiler import PolicyCompiler
            from app.runtime.policy.workflow_policy_bridge import WorkflowPolicyBridge
            _POLICY_BRIDGE_AVAILABLE = True
        except ImportError:
            _POLICY_BRIDGE_AVAILABLE = False

        try:
            from app.runtime.memory import ConversationMemory as _ConversationMemory
            _MEMORY_AVAILABLE = True
        except ImportError:
            _ConversationMemory = None
            _MEMORY_AVAILABLE = False


        class Agent(IAgent):
            def __init__(self) -> None:
                self.ready = False

                # Thread-aware: one engine per conversation thread
                self.engines: Dict[str, GenericWorkflowEngine] = {}

                # Shared, immutable inputs loaded once
                self.workflow_spec: Dict[str, Any] = {}
                self.context: Dict[str, Any] = {}

                # Conversation memory — the "M" in PMPA (Wang et al. 2024)
                self._memory: Optional[object] = None

                # Policy auto-event resolution (optional, loaded from policy_config)
                self.policy_bridge: Optional[object] = None
                self.policy_state_map: Dict[str, Any] = {}
                self.policy_slot_map: Dict[str, str] = {}
                self.policy_slot_computed: Dict[str, Any] = {}
                self.policy_slot_defaults: Dict[str, Any] = {}

                # Action prefixes that indicate user-input collection states
                # (these should NOT auto-advance through the auto-chain).
                self._USER_ACTION_PREFIXES = ("collect_", "ask_", "prompt_")

                # Keywords indicating system-internal slots (never ask customer)
                self._SYSTEM_SLOT_DESC_KW = (
                    "path", "file path", "yaml", "yml", "json",
                    "upload", "config", "<upload:",
                )
                self._SYSTEM_SLOT_NAME_KW = (
                    "_reference", "_path", "_config", "override",
                )

            def load(self, spec: Dict[str, Any]) -> None:
                cfg_path = Path(__file__).parent / "config.json"
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

                # Always resolve workflow_spec.json relative to this agent.py file
                # so the path is correct regardless of the server's working directory.
                wf_path = Path(__file__).parent / "workflow_spec.json"
                self.workflow_spec = json.loads(wf_path.read_text(encoding="utf-8"))

                self.context = {
                    "docs": cfg.get("docs", []),
                    "policies": cfg.get("policies", []),
                    "tools": cfg.get("tools", []),
                }

                # Engines are created lazily per thread_id in handle().
                self.engines = {}

                # Load policy bridge if policy_config is provided
                policy_config = cfg.get("policy_config") or {}
                if policy_config and _POLICY_BRIDGE_AVAILABLE:
                    pack_path_str = policy_config.get("policy_pack_path")
                    if pack_path_str:
                        try:
                            pack_path = Path(pack_path_str)
                            # Resolve relative paths against repo root (2 levels above generated/<id>/agent.py)
                            if not pack_path.is_absolute() and not pack_path.exists():
                                repo_root = Path(__file__).resolve().parents[2]
                                pack_path = repo_root / pack_path_str
                            compiler = PolicyCompiler()
                            policy_pack = compiler.load_pack(pack_path)
                            self.policy_bridge = WorkflowPolicyBridge(policy_pack)
                            self.policy_state_map = policy_config.get("state_auto_events") or {}
                            self.policy_slot_map = policy_config.get("slot_map") or {}
                            self.policy_slot_computed = policy_config.get("slot_computed") or {}
                            self.policy_slot_defaults = policy_config.get("slot_defaults") or {}
                            print(
                                f"[WF:$agent_id] Policy bridge loaded: "
                                f"{len(policy_pack.rules)} rules, "
                                f"{len(self.policy_state_map)} auto-event states"
                            )
                        except Exception as e:
                            print(f"[WF:$agent_id] Policy bridge load failed: {e}")

                # Initialize conversation memory if available
                if _MEMORY_AVAILABLE and _ConversationMemory is not None:
                    self._memory = _ConversationMemory()

                self.ready = True

            def _get_thread_id(self, request: Dict[str, Any]) -> str:
                ctx = request.get("context") if isinstance(request, dict) else None
                if isinstance(ctx, dict) and ctx.get("thread_id"):
                    return str(ctx["thread_id"])
                if isinstance(request, dict) and request.get("thread_id"):
                    return str(request["thread_id"])
                return "default"

            def _engine_for(self, thread_id: str) -> GenericWorkflowEngine:
                eng = self.engines.get(thread_id)
                if eng is None:
                    try:
                        from app.runtime.tools.stub_tools import STUB_TOOLS as _stubs
                    except ImportError:
                        _stubs = {}
                    eng = GenericWorkflowEngine(
                        agent_id="$agent_id",
                        workflow_spec=self.workflow_spec,
                        context=self.context,
                        tools=_stubs,
                    )
                    self.engines[thread_id] = eng
                return eng

            def _allowed_events(self, engine: GenericWorkflowEngine) -> List[str]:
                state_name = engine.current_state
                state = engine.states.get(state_name, {})
                on = state.get("on") or {}
                if isinstance(on, dict):
                    return list(on.keys())
                return []

            def _build_policy_slots(self, fsm_slots: Dict[str, Any]) -> Dict[str, Any]:
                '''
                Build a policy-compatible slot dict from FSM slots using:
                  1. slot_defaults  - baseline values (lowest priority)
                  2. slot_computed  - transform FSM values (e.g. bool -> enum string)
                  3. slot_map       - direct renames (highest priority)
                '''
                policy_slots: Dict[str, Any] = {}

                # 1. Apply defaults first
                for k, v in self.policy_slot_defaults.items():
                    policy_slots[k] = v

                # 2. Apply computed transformations
                for target_slot, compute_cfg in self.policy_slot_computed.items():
                    from_slot = compute_cfg.get("from")
                    value_map = compute_cfg.get("values") or {}
                    if from_slot and from_slot in fsm_slots and fsm_slots[from_slot] is not None:
                        raw_key = str(fsm_slots[from_slot]).lower()
                        if raw_key in value_map:
                            policy_slots[target_slot] = value_map[raw_key]

                # 3. Apply direct renames / pass-through (highest priority)
                for fsm_key, fsm_val in fsm_slots.items():
                    if fsm_val is None:
                        continue
                    policy_key = self.policy_slot_map.get(fsm_key, fsm_key)
                    policy_slots[policy_key] = fsm_val

                return policy_slots

            def _is_system_slot(self, slot_name: str, slot_defs: Dict[str, Any]) -> bool:
                '''Check if a slot is system-internal (not customer-facing).'''
                meta = (slot_defs or {}).get(slot_name)
                if not isinstance(meta, dict):
                    return False
                name_lower = slot_name.lower()
                if any(kw in name_lower for kw in self._SYSTEM_SLOT_NAME_KW):
                    return True
                desc = str(meta.get("description", "")).lower()
                return any(kw in desc for kw in self._SYSTEM_SLOT_DESC_KW)

            def _auto_fill_system_slots(self, engine: GenericWorkflowEngine, ctx: Dict[str, Any]) -> None:
                '''Auto-fill system-internal slots from agent context.

                LLM-generated workflow specs may include required slots like
                "policy_reference" that point to internal config (policy YAML
                paths, file references).  These must never be surfaced to the
                customer as clarification questions.
                '''
                policies = ctx.get("policies") or []
                docs = ctx.get("docs") or []
                for slot_name in list(engine.slot_defs or {}):
                    if engine.slots.get(slot_name):
                        continue
                    if not self._is_system_slot(slot_name, engine.slot_defs):
                        continue
                    name_lower = slot_name.lower()
                    if "policy" in name_lower and policies:
                        engine.slots[slot_name] = policies[0] if len(policies) == 1 else str(policies)
                    elif "doc" in name_lower and docs:
                        engine.slots[slot_name] = docs[0] if len(docs) == 1 else str(docs)
                    else:
                        engine.slots[slot_name] = "N/A"
                    print(f"[SYS-SLOT:$agent_id] auto-filled {slot_name}={engine.slots[slot_name]}")

            def _try_policy_auto_event(self, engine: GenericWorkflowEngine) -> Optional[str]:
                '''
                Deterministically resolve an FSM event for system states.

                Three paths:
                  - tool_exec (explicit or inferred): stub tools always succeed;
                    fire pass_event / first happy-path event immediately.
                    Does NOT require the policy bridge.
                  - eligibility / approval_needed: run compiled policy rules.
                    Requires the policy bridge.
                  - inferred system state: if the state spec has an
                    ``action`` starting with ``invoke_tool:`` but no explicit
                    config in policy_state_map, auto-fire the first non-cancel
                    event so the auto-chain can advance through tool states.

                Returns None to fall back to LLM mapper (user-input states).
                '''
                # ── Guard: don't auto-advance while required slots are missing ──
                # System states (tool exec, policy checks, etc.) need slot data to
                # produce meaningful results.  If the user hasn't provided required
                # information yet, fall back to the LLM mapper / request_clarification
                # path so the workflow can collect the data first.
                for slot_name, meta in (engine.slot_defs or {}).items():
                    if isinstance(meta, dict) and meta.get("required") and not engine.slots.get(slot_name):
                        return None

                state_cfg = (self.policy_state_map or {}).get(engine.current_state)

                # ── Inferred auto-advance for unconfigured system states ──
                # States whose ``action`` is a system operation (tool invocation,
                # internal agent query, ticket creation, handoff, etc.) should
                # auto-advance without waiting for user input.
                # Only ``collect_*`` / ``ask_*`` actions require user input.
                if state_cfg is None:
                    state_spec = engine.states.get(engine.current_state, {})
                    action = state_spec.get("action") or ""
                    actions = state_spec.get("actions") or []
                    all_actions = ([action] if action else []) + (
                        actions if isinstance(actions, list) else []
                    )
                    all_actions = [a for a in all_actions if isinstance(a, str) and a.strip()]

                    is_system = (
                        bool(all_actions)
                        and not any(
                            a.startswith(pfx)
                            for a in all_actions
                            for pfx in self._USER_ACTION_PREFIXES
                        )
                    )
                    if is_system:
                        on = state_spec.get("on") or {}
                        # Pick first happy-path event (skip cancel/error/failure)
                        _skip = {"cancel", "error", "failure", "timeout"}
                        event = next(
                            (e for e in on if e.lower() not in _skip), None
                        )
                        if event:
                            print(
                                f"[SYSTEM-AUTO-INFER:$agent_id] state={engine.current_state} "
                                f"-> event={event} (inferred from action: {all_actions})"
                            )
                            return event
                    return None

                if not state_cfg:
                    return None

                check_type = state_cfg.get("check")

                # tool_exec: no policy bridge needed — stub always returns happy path
                if check_type == "tool_exec":
                    event = state_cfg["pass_event"]
                    print(
                        f"[TOOL-AUTO:$agent_id] state={engine.current_state} "
                        f"-> event={event} (stub: tool always succeeds)"
                    )
                    return event

                # eligibility / approval_needed: policy bridge required
                if not self.policy_bridge:
                    return None

                policy_slots = self._build_policy_slots(engine.slots or {})
                check_type = state_cfg.get("check")

                try:
                    if check_type == "eligibility":
                        is_eligible, reason, _ = self.policy_bridge.check_eligibility(policy_slots)
                        event = state_cfg["pass_event"] if is_eligible else state_cfg["fail_event"]
                        print(
                            f"[POLICY-AUTO:$agent_id] state={engine.current_state} "
                            f"eligible={is_eligible} reason={reason!r} -> event={event}"
                        )
                        return event

                    elif check_type == "combined_eligibility_approval":
                        # Runs both checks; picks the right pass event
                        is_eligible, reason, _ = self.policy_bridge.check_eligibility(policy_slots)
                        if not is_eligible:
                            event = state_cfg["fail_event"]
                        else:
                            needs_approval, appr_reason, _ = self.policy_bridge.check_approval_needed(policy_slots)
                            event = state_cfg["approval_required_event"] if needs_approval else state_cfg["no_approval_event"]
                        print(
                            f"[POLICY-AUTO:$agent_id] state={engine.current_state} "
                            f"eligible={is_eligible} -> event={event}"
                        )
                        return event

                    elif check_type == "approval_needed":
                        needed, reason, _ = self.policy_bridge.check_approval_needed(policy_slots)
                        event = state_cfg["pass_event"] if needed else state_cfg["fail_event"]
                        print(
                            f"[POLICY-AUTO:$agent_id] state={engine.current_state} "
                            f"approval_needed={needed} reason={reason!r} -> event={event}"
                        )
                        return event

                except Exception as e:
                    print(f"[POLICY-AUTO:$agent_id] Error at state={engine.current_state}: {e}")

                return None

            def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
                if not self.ready:
                    return {"error": "engine_not_loaded", "agent_id": "$agent_id"}

                # Incoming request: {"query": "...", "context": {"thread_id": "..."}}
                query = ""
                if isinstance(request, dict):
                    query = str(request.get("query", "") or "")
                else:
                    query = str(request or "")

                thread_id = self._get_thread_id(request if isinstance(request, dict) else {})
                engine = self._engine_for(thread_id)

                # Merge request context (runtime) into engine.context
                req_ctx = request.get("context") if isinstance(request, dict) else None
                merged_ctx = dict(self.context or {})

                if isinstance(req_ctx, dict):
                    # Merge lists (docs/policies/tools) without duplicates
                    for k in ("docs", "policies", "tools"):
                        base = merged_ctx.get(k) or []
                        extra = req_ctx.get(k) or []
                        if not isinstance(base, list):
                            base = [base]
                        if not isinstance(extra, list):
                            extra = [extra]
                        merged_ctx[k] = list(dict.fromkeys([*base, *extra]))

                    # Carry over other keys (thread_id, intent, domain, etc.)
                    for k, v in req_ctx.items():
                        if k not in ("docs", "policies", "tools"):
                            merged_ctx[k] = v

                engine.context = merged_ctx

                # Pre-populate engine slots from accumulated cross-agent context.
                # When a previous agent (e.g. router-intent) already extracted
                # customer_id or transaction_id, carry those forward so the
                # workflow doesn't re-ask the user for information they already provided.
                _acc = merged_ctx.get("_accumulated_slots")
                if isinstance(_acc, dict) and _acc:
                    for k, v in _acc.items():
                        if k in (engine.slot_defs or {}) and not engine.slots.get(k):
                            engine.slots[k] = v

                # Auto-fill system-internal slots (policy paths, file refs, etc.)
                # so the customer is never asked for internal config values.
                self._auto_fill_system_slots(engine, merged_ctx)

                allowed_events = self._allowed_events(engine)

                # ── Policy auto-event: deterministic resolution for policy-evaluation states ──
                # This prevents the LLM mapper from getting stuck on internal system events
                # (e.g. eligible/ineligible, approval_needed/no_approval_needed) that the
                # user will never say.  The rule engine decides these based on compiled rules.
                auto_event = self._try_policy_auto_event(engine)

                if auto_event is not None:
                    # Deterministic path — skip LLM mapper entirely
                    wf_res = engine.handle({
                        "event": auto_event,
                        "slots": {},
                        "query": query,
                    })
                    wf_res["mapper"] = {
                        "thread_id": thread_id,
                        "state": engine.current_state,
                        "allowed_events": allowed_events,
                        "event": auto_event,
                        "slots": {},
                        "confidence": 1.0,
                        "rationale": (
                            f"Policy auto-event for state '{engine.current_state}' "
                            "(deterministic rule engine — no LLM guess)"
                        ),
                        "policy_auto_resolved": True,
                    }
                else:
                    # Inject conversation context from memory (PMPA memory-assisted planning)
                    _conv_ctx: List[Dict[str, Any]] = []
                    if self._memory:
                        try:
                            _conv_ctx = self._memory.get_conversation_context(thread_id, limit=3)
                        except Exception:
                            pass

                    # LLM mapper path — for user-driven events (e.g. begin, info_provided)
                    mr = map_query_to_event_and_slots(
                        query=query,
                        current_state=engine.current_state,
                        allowed_events=allowed_events,
                        slot_defs=engine.slot_defs,
                        model="gpt-5-mini",
                        current_slots=getattr(engine, "slots", {}) or {},
                        conversation_context=_conv_ctx if _conv_ctx else None,
                    )

                    # Hard guard: if required slots are still missing after applying the
                    # extracted slots, force event=None so the engine stays in
                    # request_clarification mode.  Prevents the LLM from picking an
                    # error/escalation event just because a required field is absent.
                    #
                    # The guard only fires when the event would transition to a
                    # NON-collection target (i.e., a system state or processing
                    # state).  This allows incremental slot collection within
                    # user-facing collection states: the mapper can return a
                    # partial-slot event, the engine saves the new slots, and
                    # on the next turn the remaining slots are collected.
                    #
                    # Exception: "N/A" is a valid sentinel set by the mapper when the
                    # user explicitly declines to provide a field.  Treat it as satisfied
                    # so the workflow can progress past an unavailable required slot.
                    _DECLINED_SENTINELS = {"N/A", "n/a", "unknown", "not_available", "none"}
                    _mapped_event = mr.event
                    if _mapped_event is not None:
                        _tentative = dict(engine.slots or {})
                        _tentative.update(mr.slots or {})
                        _still_missing = [
                            k for k, meta in (engine.slot_defs or {}).items()
                            if isinstance(meta, dict) and meta.get("required")
                            and not _tentative.get(k)
                            and str(_tentative.get(k, "")).strip() not in _DECLINED_SENTINELS
                        ]
                        if _still_missing:
                            # Check if the target state is a system/processing
                            # state.  If so, block — we need all data first.
                            # If the target is another user-input collection
                            # state (or we can't determine), allow the event so
                            # the mapper's extracted slots get committed and the
                            # engine can narrow the clarification.
                            _target = (
                                engine.states
                                .get(engine.current_state, {})
                                .get("on", {})
                                .get(_mapped_event)
                            )
                            _target_spec = engine.states.get(_target, {}) if _target else {}
                            _target_action = _target_spec.get("action") or ""
                            _target_is_system = (
                                _target_action
                                and not any(
                                    _target_action.startswith(pfx)
                                    for pfx in self._USER_ACTION_PREFIXES
                                )
                            )
                            if _target_is_system:
                                _mapped_event = None

                    # Drive the workflow engine
                    wf_res = engine.handle({
                        "event": _mapped_event,
                        "slots": mr.slots,
                        "query": query,
                    })

                    # Attach explainability/debug info
                    wf_res["mapper"] = {
                        "thread_id": thread_id,
                        "state": engine.current_state,
                        "allowed_events": allowed_events,
                        "event": _mapped_event,
                        "llm_event": mr.event,
                        "slots": mr.slots,
                        "confidence": mr.confidence,
                        "rationale": mr.rationale,
                    }

                # ── Auto-chain: advance through consecutive system states ──
                # After any user-driven transition the engine may land in a
                # system state (tool-exec / policy auto-event).  Keep advancing
                # until we hit a state that needs user input or reach terminal.
                _MAX_CHAIN = 8
                _chain_events: List[str] = []
                for _c in range(_MAX_CHAIN):
                    if engine.states.get(engine.current_state, {}).get("terminal"):
                        break
                    _chain_event = self._try_policy_auto_event(engine)
                    if _chain_event is None:
                        break
                    print(f"[CHAIN:$agent_id] auto-chain {_c+1} state={engine.current_state} event={_chain_event}")
                    _chain_events.append(f"{engine.current_state}->{_chain_event}")
                    wf_res = engine.handle({"event": _chain_event, "slots": {}, "query": query})

                if _chain_events:
                    wf_res.setdefault("mapper", {})["auto_chain"] = _chain_events

                # Record turn + state snapshot in conversation memory (PMPA audit trail)
                if self._memory:
                    try:
                        _policy_decisions = []
                        _mapper = wf_res.get("mapper") or {}
                        if _mapper.get("policy_auto_resolved"):
                            _policy_decisions.append({
                                "type": "policy_auto_event",
                                "state": _mapper.get("state"),
                                "event": _mapper.get("event"),
                            })
                        self._memory.record_turn(
                            thread_id=thread_id,
                            query=query,
                            response=wf_res,
                            agent_id="$agent_id",
                            fsm_state=engine.current_state,
                            slots=dict(engine.slots or {}),
                            policy_decisions=_policy_decisions if _policy_decisions else None,
                        )
                        self._memory.record_state_snapshot(
                            thread_id=thread_id,
                            state=engine.current_state,
                            slots=dict(engine.slots or {}),
                            agent_id="$agent_id",
                            trigger="state_entry",
                        )
                    except Exception:
                        pass

                # Echo merged context for visibility/debugging
                wf_res["context"] = engine.context

                # Safety net: strip any system-internal slots that still appear
                # in clarification requests (e.g. if auto-fill missed them).
                if wf_res.get("action") == "request_clarification":
                    wf_res["missing_slots"] = [
                        s for s in (wf_res.get("missing_slots") or [])
                        if not self._is_system_slot(s, engine.slot_defs)
                    ]

                return wf_res


            def metadata(self) -> Dict[str, Any]:
                return {
                    "id": "$agent_id",
                    "type": "workflow_runner",
                    "ready": self.ready,
                    "description": (
                        "Multi-turn workflow agent with policy-driven auto-transitions. "
                        "Deterministic rule engine resolves policy states; LLM handles user input."
                    ),
                    "capabilities": ["multi_turn", "followups", "workflow", "policy_auto_events"],
                }
        """
        )
    ).substitute(agent_id=agent_id)

    (gen_dir / "agent.py").write_text(agent_src, encoding="utf-8")
    return gen_dir
