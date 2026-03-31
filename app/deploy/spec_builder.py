# app/deploy/spec_builder.py
from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.concierge.blueprint_creator import BlueprintCreatorAgent
from app.infer_capabilities import InferCapabilities


# ------------------------------------------------------------
# 🔍 Blueprint Discovery (legacy: still supported)
# ------------------------------------------------------------
def discover_blueprints(bp_dir: Path) -> Dict[str, dict]:
    """
    Scan factory/blueprints for available (file-based) blueprints and return:
        { capability_id: blueprint_metadata }
    Each blueprint.yaml should contain:
        id, capabilities, description, entrypoint, (optional) inputs/output/version
    """
    mapping: Dict[str, dict] = {}
    if not bp_dir.exists():
        print(f"[WARN] No blueprints folder found at {bp_dir}")
        return mapping

    for sub in bp_dir.iterdir():
        if not sub.is_dir():
            continue
        bp_yaml = sub / "blueprint.yaml"
        if not bp_yaml.exists():
            continue
        try:
            meta = yaml.safe_load(bp_yaml.read_text(encoding="utf-8")) or {}
            bp_id = meta.get("id", sub.name)
            caps = meta.get("capabilities", [bp_id])
            for cap in caps:
                mapping[str(cap).lower()] = meta
        except Exception as e:
            print(f"[WARN] Could not parse blueprint {bp_yaml}: {e}")

    return mapping


# ------------------------------------------------------------
# 📦 Helper: absolute path resolution
# ------------------------------------------------------------
def _abs(p: Path | str) -> str:
    return str(Path(p).resolve())


# ------------------------------------------------------------
# 🧠 Helper: build inputs from a Blueprint (NOT hardcoded per capability)
# ------------------------------------------------------------
def _is_internal(filename: str, doc_visibility: Optional[Dict[str, str]]) -> bool:
    """
    Returns True if a file should be treated as internal (excluded from customer-facing RAG).

    Uses the user-provided visibility map from the onboarding wizard.
    Only documents explicitly marked "internal" are excluded; everything else
    (including unmarked files and files with no visibility map) defaults to
    customer-facing.
    """
    if doc_visibility:
        return doc_visibility.get(filename, "").lower() == "internal"
    return False


def _inputs_from_blueprint(
    bp: Dict[str, Any], data_dir: Path, doc_visibility: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Build runtime 'inputs' for an AgentBlueprint.

    Rules:
    - knowledge_rag expects inputs.docs (list of absolute paths)
    - workflow_runner expects inputs.workflow_spec (dict) + optional docs/policies/tools context
    - tool_operator expects inputs.tool (string) + optional defaults
    - If blueprint already contains inputs.docs etc, we resolve relative/placeholder paths if possible.
    - We do NOT hardcode FAQ/complaint.
    """
    agent_kind = str(bp.get("agent_kind", "")).strip()
    inputs: Dict[str, Any] = dict(bp.get("inputs") or {})

    # Helper: resolve "<UPLOAD:...>" placeholders to any matching file, best-effort
    def _resolve_placeholder(value: str) -> Optional[str]:
        if not isinstance(value, str):
            return None
        v = value.strip()
        if not v.startswith("<UPLOAD:"):
            return None
        hint = v[len("<UPLOAD:") :].rstrip(">").strip().lower()
        # try match by hint in filename
        matches = [
            f
            for f in data_dir.iterdir()
            if f.is_file() and hint and hint in f.name.lower()
        ]
        return _abs(matches[0]) if matches else None

    # knowledge sources / policies may include placeholder paths
    # We'll try to resolve and also expose them as docs/policies in inputs for builders.
    knowledge_sources = bp.get("knowledge_sources") or []
    policies = bp.get("policies") or []

    resolved_docs: List[str] = []
    for ks in knowledge_sources:
        if isinstance(ks, dict) and ks.get("path"):
            p = str(ks["path"])
            rp = _resolve_placeholder(p) or (
                p if Path(p).is_absolute() else str((data_dir / p).resolve())
            )
            if Path(rp).exists():
                resolved_docs.append(_abs(rp))

    resolved_policies: List[str] = []
    for pol in policies:
        if isinstance(pol, dict) and pol.get("path"):
            p = str(pol["path"])
            rp = _resolve_placeholder(p) or (
                p if Path(p).is_absolute() else str((data_dir / p).resolve())
            )
            if Path(rp).exists():
                resolved_policies.append(_abs(rp))

    # If blueprint didn't explicitly set docs, use resolved docs for RAG
    if agent_kind == "knowledge_rag":
        if "docs" not in inputs or not inputs.get("docs"):
            # fallback: use resolved_docs; if empty, take any csv/md/txt files
            if resolved_docs:
                inputs["docs"] = resolved_docs
            else:
                candidates = [
                    f
                    for f in data_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {".csv", ".md", ".txt"}
                ]
                inputs["docs"] = [_abs(f) for f in candidates] if candidates else []

        # Ensure policy docs are available (needed as agent instructions)
        if resolved_policies:
            existing = set(inputs.get("policies") or [])
            for rp in resolved_policies:
                if rp not in existing:
                    inputs.setdefault("policies", []).append(rp)

    # For workflow_runner, always pass workflow_spec through (already LLM-generated dict)
    # plus attach docs/policies if available (useful for future checks)
    if agent_kind == "workflow_runner":
        if resolved_docs and "docs" not in inputs:
            inputs["docs"] = resolved_docs
        if resolved_policies and "policies" not in inputs:
            inputs["policies"] = resolved_policies
        # Auto-inject policy_config by scanning workflow states for known policy patterns
        if "policy_config" not in inputs:
            pc = _build_policy_config(inputs.get("workflow_spec") or {}, data_dir)
            if pc:
                inputs["policy_config"] = pc

    # For domain_agent: set domain, goal, knowledge_sources, available_tools, policies
    if agent_kind == "domain_agent":
        inputs.setdefault("domain", bp.get("domain") or bp.get("id", "general"))
        inputs.setdefault("goal", bp.get("goal") or bp.get("description", ""))

        # Resolve any <UPLOAD:...> placeholders in existing knowledge_sources
        if inputs.get("knowledge_sources"):
            _resolved_ks: List[str] = []
            for ks_entry in inputs["knowledge_sources"]:
                if isinstance(ks_entry, str):
                    rp = _resolve_placeholder(ks_entry)
                    if rp and Path(rp).exists():
                        _resolved_ks.append(rp)
                    elif Path(ks_entry).exists():
                        _resolved_ks.append(_abs(ks_entry))
                    else:
                        # Try relative to data_dir
                        candidate = data_dir / Path(ks_entry).name
                        if candidate.exists():
                            _resolved_ks.append(_abs(candidate))
            if _resolved_ks:
                inputs["knowledge_sources"] = _resolved_ks

        # knowledge_sources: merge resolved docs + policy docs for the RAG corpus.
        # ALL documents are included — both customer-facing and internal.
        # Internal docs serve as agent instructions; customer-facing docs
        # provide content the agent can share with customers.
        # The doc_visibility_map (below) tells the agent which is which.
        if "knowledge_sources" not in inputs or not inputs.get("knowledge_sources"):
            ks = list(resolved_docs)
            ks.extend(resolved_policies)
            if not ks:
                # Fallback: use any csv/md/txt files
                candidates = [
                    f
                    for f in data_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {".csv", ".md", ".txt"}
                ]
                ks = [_abs(f) for f in candidates]
            inputs["knowledge_sources"] = ks

        # Build per-document visibility map so the agent knows which docs
        # are internal (instructions only) vs customer-facing (shareable).
        # Only include actual file paths (knowledge_sources, docs), not
        # natural-language policy strings.
        _vis_map: Dict[str, str] = {}
        for _doc_key in ("knowledge_sources", "docs"):
            if _doc_key in inputs and isinstance(inputs[_doc_key], list):
                for d in inputs[_doc_key]:
                    dname = Path(d).name
                    _vis_map[dname] = (
                        "internal"
                        if _is_internal(dname, doc_visibility)
                        else "customer_facing"
                    )
        if _vis_map:
            inputs["doc_visibility_map"] = _vis_map

        # available_tools: from blueprint inputs or bp["tools"]
        if "available_tools" not in inputs or not inputs.get("available_tools"):
            tools_list = bp.get("tools") or []
            if isinstance(tools_list, list):
                inputs["available_tools"] = [str(t) for t in tools_list]
            else:
                inputs["available_tools"] = []

        # policies: natural language constraints from policy docs or explicit policies_text
        if "policies" not in inputs or not inputs.get("policies"):
            policies_text = bp.get("policies_text") or inputs.get("policies_text") or []
            if isinstance(policies_text, list):
                inputs["policies"] = [str(p) for p in policies_text]
            else:
                inputs["policies"] = []

    # For tool_operator, allow a default tool name from blueprint inputs or bp["tools"][0]
    if agent_kind == "tool_operator":
        if "tool" not in inputs or not inputs.get("tool"):
            tools = bp.get("tools") or []
            if isinstance(tools, list) and tools:
                inputs["tool"] = str(tools[0])

    # Best-effort resolve any string placeholders inside inputs (shallow).
    # Skip strings that look like natural-language text (contain spaces,
    # are very long, etc.) — they are policy constraints, not file paths.
    def _looks_like_path(s: str) -> bool:
        """Return True if *s* could plausibly be a filesystem path."""
        return (
            len(s) <= 255
            and " " not in s
            and not s.startswith(("When ", "If ", "Do ", "Ensure ", "Always "))
        )

    for k, v in list(inputs.items()):
        if isinstance(v, str):
            rp = _resolve_placeholder(v)
            if rp:
                inputs[k] = rp
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, str):
                    rp = _resolve_placeholder(item)
                    if rp:
                        new_list.append(rp)
                    elif _looks_like_path(item):
                        # resolve relative to data_dir
                        p = Path(item)
                        if p.is_absolute():
                            new_list.append(_abs(p))
                        elif (data_dir / p).exists():
                            new_list.append(_abs(data_dir / p))
                        else:
                            new_list.append(item)
                    else:
                        new_list.append(item)
                else:
                    new_list.append(item)
            inputs[k] = new_list

    return inputs


# ------------------------------------------------------------
# 🔗 Policy config auto-detection for workflow_runner agents
# ------------------------------------------------------------
def _build_policy_config(
    workflow_spec: Dict[str, Any], base_dir: Path
) -> Dict[str, Any]:
    """
    Scan a workflow_spec for states whose event sets match known policy check patterns,
    then wire them to the compiled policy pack automatically.

    Detected patterns:
      {eligible, ineligible}               -> eligibility check
      {approval_required, auto_approve}    -> approval_needed check
      {approval_needed, no_approval_needed}-> approval_needed check

    Returns an empty dict if no policy pack is found or no matching states exist.
    """
    # Locate compiled policy pack (check repo-root .factory first, then workspace)
    candidates = [
        Path(".factory") / "compiled_policies",
        base_dir / ".factory" / "compiled_policies",
    ]
    pack_path: Optional[Path] = None
    for d in candidates:
        if d.exists():
            packs = sorted(d.glob("*.json"))
            if packs:
                pack_path = packs[0]
                break
    # Note: we continue even without a pack_path so tool_exec states are always detected

    # Normalise states dict (handles both dict and list forms)
    raw_states = workflow_spec.get("states") or {}
    if isinstance(raw_states, list):
        raw_states = {
            s["name"]: s for s in raw_states if isinstance(s, dict) and "name" in s
        }

    # Eligibility events: vocab-based (superset-safe — event_set may include "error" etc.)
    _ELIG_PASS = {
        "eligible",
        "eligibility_pass",
        "eligibility_met",
        "eligibility_passed",
    }
    _ELIG_FAIL = {
        "ineligible",
        "eligibility_fail",
        "eligibility_failed",
        "not_eligible",
    }

    # Approval events: vocab-based
    _APPROV_PASS = {"approval_required", "approval_needed", "requires_approval"}
    _APPROV_FAIL = {"auto_approve", "no_approval_needed", "auto_approved"}

    # Tool-execution system-state detection: suffix-based.
    # A state qualifies if ALL non-neutral events match known system suffixes.
    _PASS_SFXS = (
        "_verified",
        "_pass",
        "_success",
        "_found",
        "_initiated",
        "_created",
        "_ok",
        "_met",
    )
    _FAIL_SFXS = ("_failed", "_fail", "_rejected", "_denied")
    # Events that are treated as neutral (don't disqualify a system state)
    _NEUTRAL_EVENTS = {"error", "timeout"}

    state_auto_events: Dict[str, Any] = {}
    for state_name, state_def in raw_states.items():
        on = state_def.get("on") or {}
        if not isinstance(on, dict):
            continue
        event_set = set(on.keys())
        if not event_set:
            continue

        # 1. Eligibility check (vocabulary match — superset-safe).
        # Also handles compound events like eligible_no_approval / eligible_requires_approval
        # by using prefix matching: any event starting with "eligible" (not "ineligible") is a pass.
        def _elig_pass_events(es):
            exact = [e for e in es if e in _ELIG_PASS]
            if exact:
                return exact
            # prefix-based: starts with "eligible" but not "ineligible"
            return [
                e
                for e in es
                if e.lower().startswith("eligible")
                and not e.lower().startswith("ineligible")
            ]

        def _elig_fail_events(es):
            exact = [e for e in es if e in _ELIG_FAIL]
            if exact:
                return exact
            return [
                e
                for e in es
                if e.lower().startswith("ineligible") or e.lower() == "not_eligible"
            ]

        pass_candidates = _elig_pass_events(event_set)
        fail_candidates = _elig_fail_events(event_set)

        if pass_candidates and fail_candidates:
            # Single pass event → simple eligibility check
            # Two pass events (e.g. eligible_no_approval + eligible_requires_approval) →
            # combined eligibility+approval check that picks the right pass event
            if len(pass_candidates) == 1:
                state_auto_events[state_name] = {
                    "check": "eligibility",
                    "pass_event": pass_candidates[0],
                    "fail_event": fail_candidates[0],
                }
            else:
                # Compound: find which pass event signals "no approval needed"
                no_appr = next(
                    (e for e in pass_candidates if "no_approval" in e or "auto" in e),
                    pass_candidates[0],
                )
                req_appr = next(
                    (
                        e
                        for e in pass_candidates
                        if "require" in e or "approval" in e and "no" not in e
                    ),
                    pass_candidates[-1],
                )
                state_auto_events[state_name] = {
                    "check": "combined_eligibility_approval",
                    "no_approval_event": no_appr,
                    "approval_required_event": req_appr,
                    "fail_event": fail_candidates[0],
                }
            continue

        # 2. Approval check (vocabulary match — superset-safe)
        pass_e = next((e for e in event_set if e in _APPROV_PASS), None)
        fail_e = next((e for e in event_set if e in _APPROV_FAIL), None)
        if pass_e and fail_e:
            state_auto_events[state_name] = {
                "check": "approval_needed",
                "pass_event": pass_e,
                "fail_event": fail_e,
            }
            continue

        # 3. General tool-execution state (suffix-based).
        # Register if ALL non-neutral events look like system-generated events
        # (no plain user-input words like "submit", "close", "yes").
        non_neutral = {e for e in event_set if e.lower() not in _NEUTRAL_EVENTS}
        if non_neutral:
            pass_events = [
                e for e in non_neutral if any(e.lower().endswith(s) for s in _PASS_SFXS)
            ]
            fail_events = [
                e for e in non_neutral if any(e.lower().endswith(s) for s in _FAIL_SFXS)
            ]
            all_matched = all(
                any(e.lower().endswith(s) for s in _PASS_SFXS + _FAIL_SFXS)
                for e in non_neutral
            )
            if all_matched and pass_events:
                state_auto_events[state_name] = {
                    "check": "tool_exec",
                    "pass_event": pass_events[0],
                    "fail_event": fail_events[0] if fail_events else pass_events[0],
                }

    if not state_auto_events:
        return {}

    # tool_exec states need no policy pack — return a minimal config for them
    if not pack_path:
        return {
            "state_auto_events": state_auto_events,
            "slot_map": {},
            "slot_computed": {},
            "slot_defaults": {},
        }

    # Map FSM amount slot name -> policy slot name
    slot_map: Dict[str, str] = {}
    for slot_name in workflow_spec.get("slots") or {}:
        if slot_name in ("amount", "amount_requested", "refund_amount"):
            slot_map[slot_name] = "refund_amount_requested"
            break

    # Build relative path (forward slashes, repo-root relative)
    try:
        rel = pack_path.resolve().relative_to(Path(".").resolve())
        rel_str = str(rel).replace("\\", "/")
    except ValueError:
        rel_str = str(pack_path).replace("\\", "/")

    return {
        "policy_pack_path": rel_str,
        "state_auto_events": state_auto_events,
        "slot_map": slot_map,
        "slot_computed": {},
        "slot_defaults": {
            "kyc_status": "verified",
            "account_status": "active",
            "investigation_status": "none",
        },
    }


def _compile_policy_files(base_dir: Path, factory_dir: Path) -> None:
    """Compile policy YAML files into a JSON policy pack for workflow agents.

    The compiled pack is stored at .factory/compiled_policies/ where
    _build_policy_config() can find it and inject policy_pack_path into
    the workflow agent config.  Without this step, the policy bridge
    never loads and internal states (eligibility, approval) can't
    auto-resolve — causing the workflow to ask the customer about
    internal process decisions.
    """
    policy_files = list(base_dir.glob("*.yaml")) + list(base_dir.glob("*.yml"))
    if not policy_files:
        return

    try:
        from app.runtime.policy.policy_compiler import PolicyCompiler
    except ImportError:
        print("[SPEC] PolicyCompiler not available — skipping policy compilation")
        return

    out_dir = factory_dir / "compiled_policies"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        compiler = PolicyCompiler(domain="fintech")
        pack = compiler.compile_policies(policy_files)
        out_path = out_dir / f"{pack.policy_id}.json"
        compiler.save_pack(pack, out_path)
        print(
            f"[SPEC] Compiled {len(policy_files)} policy files -> {out_path.name} "
            f"({len(pack.rules)} rules)"
        )
    except Exception as e:
        print(f"[SPEC] Policy compilation failed: {e}")


# ------------------------------------------------------------
# 📄 Document content metadata for agents
# ------------------------------------------------------------
def _derive_doc_content_meta(
    agent_inputs: Dict[str, Any],
    doc_metadata: List[Dict[str, Any]],
    doc_visibility: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Derive document content metadata for an agent from its assigned docs.

    Uses the persisted .doc_metadata.json for content analysis (categories,
    topics) and the user-provided doc_visibility map for visibility
    (customer_facing vs internal).

    Fields produced:
    - has_customer_facing_docs: agent has at least one customer-facing document
    - has_internal_policy: agent has at least one internal policy document
    - document_categories: aggregated content categories from customer-facing docs
    - coverage_topics: aggregated topic headers from all docs

    These fields are stored in blueprint_meta and used by the solvability
    estimator for intent-aware routing.
    """
    # Build lookup by filename
    meta_by_name: Dict[str, Dict[str, Any]] = {
        d["name"]: d for d in doc_metadata if isinstance(d, dict) and "name" in d
    }

    # Collect the agent's assigned docs
    ks = agent_inputs.get("knowledge_sources") or []
    if not isinstance(ks, list):
        ks = []

    has_customer_facing = False
    has_internal = False
    categories: List[str] = []
    topics: List[str] = []

    for doc_path in ks:
        doc_name = Path(doc_path).name
        dm = meta_by_name.get(doc_name, {})

        # Visibility comes from the user-provided doc_visibility map
        # (set during onboarding). Documents not in the map default to
        # customer-facing (same convention as _is_internal()).
        is_internal = _is_internal(doc_name, doc_visibility)

        if is_internal:
            has_internal = True
        else:
            has_customer_facing = True
            # Only aggregate categories from customer-facing docs
            categories.extend(dm.get("content_categories", []))

        topics.extend(dm.get("content_topics", []))

    # Deduplicate
    categories = sorted(set(categories))
    topics = sorted(set(topics))

    return {
        "has_customer_facing_docs": has_customer_facing,
        "has_internal_policy": has_internal,
        "document_categories": categories,
        "coverage_topics": topics,
    }


# ------------------------------------------------------------
# 🧰 Spec Builder (NEW: uses LLM-generated AgentBlueprints, not per-capability hardcoding)
# ------------------------------------------------------------
def build_factory_spec(
    plan: Dict[str, Any],
    data_dir: str,
    dry_run: bool = True,
    llm_client: Optional[object] = None,
    doc_visibility: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convert Concierge plan preview → runtime factory_spec.json.

    NEW behavior:
      - Consumes the existing 'plan' passed from Concierge (already computed).
      - Calls BlueprintCreatorAgent.generate_plan_from_existing_plan(plan)
      - Uses returned AgentBlueprints (N agents) to build a dynamic factory spec.
      - No hardcoded FAQ/complaint logic.

    We keep discover_blueprints() around for compatibility, but the primary path is blueprints[].
    """
    base_dir = Path(data_dir).resolve()
    factory_dir = base_dir / ".factory"
    factory_dir.mkdir(parents=True, exist_ok=True)
    spec_path = factory_dir / "factory_spec.json"

    # Load persisted document content metadata (categories, visibility, topics)
    doc_metadata = InferCapabilities.load_doc_metadata(base_dir)

    # Compile policy YAML files into .factory/compiled_policies/ so that
    # workflow agents can load the policy bridge for auto-event resolution.
    _compile_policy_files(base_dir, factory_dir)

    # Always include guardrails (spine requirement)
    agents_block: List[Dict[str, Any]] = [
        {
            "id": "guardrails",
            "type": "guardrails",
            "config": "spec/base_policy_pack.yaml",
        }
    ]

    # Generate N AgentBlueprints from the existing plan (LLM)
    bp_creator = BlueprintCreatorAgent(model="gpt-5-mini")
    bp_plan = bp_creator.generate_plan_from_existing_plan(
        plan=plan, user_goals=plan.get("user_goals", "")
    )

    # Store plan info for UI/debugging
    print(
        f"[SPEC] Blueprint plan: {len(bp_plan.blueprints)} agents | vertical={bp_plan.vertical}"
    )
    if bp_plan.missing_docs:
        print(f"[SPEC] missing_docs: {bp_plan.missing_docs}")
    if bp_plan.warnings:
        print(f"[SPEC] warnings: {bp_plan.warnings}")

    # Map AgentBlueprint.agent_kind -> generic builder blueprint id
    # These are NOT domain-specific; they're engine-level templates.
    # You'll create these generic blueprints once:
    #   - factory/blueprints/knowledge_rag/blueprint.yaml   (entrypoint: app.shared.rag.build_agent)
    #   - factory/blueprints/workflow_runner/blueprint.yaml (entrypoint: app.shared.workflow.build_agent)  <-- next step
    #   - factory/blueprints/tool_operator/blueprint.yaml   (entrypoint: app.shared.toolop.build_agent)   <-- later
    kind_to_blueprint = {
        "domain_agent": "domain_agent",
        "knowledge_rag": "knowledge_rag",
        "workflow_runner": "workflow_runner",
        "tool_operator": "tool_operator",
    }

    for bp in bp_plan.blueprints:
        agent_id = str(bp.get("id")).strip()
        if not agent_id or agent_id.lower() in {"guardrails", "qa"}:
            continue

        agent_kind = str(bp.get("agent_kind", "")).strip()
        # Only generate domain_agent types; skip legacy/tool types
        if agent_kind != "domain_agent":
            print(f"[SPEC] Skipping non-domain_agent: {agent_id} ({agent_kind})")
            continue
        blueprint_id = kind_to_blueprint.get(agent_kind)
        if not blueprint_id:
            raise ValueError(
                f"Unsupported agent_kind '{agent_kind}' for blueprint id='{agent_id}'"
            )

        agent_inputs = _inputs_from_blueprint(bp, base_dir, doc_visibility)

        # Derive document content metadata for intent-aware routing
        doc_content_meta = _derive_doc_content_meta(
            agent_inputs, doc_metadata, doc_visibility
        )

        agents_block.append(
            {
                "id": agent_id,
                "type": "autogen",
                "blueprint": blueprint_id,
                "status": "ready",  # this is plan-level; can be refined later
                "inputs": agent_inputs,
                "blueprint_meta": {
                    # keep the whole declarative blueprint for routing + explainability
                    "agent_kind": agent_kind,
                    "description": bp.get("description", ""),
                    "capabilities": bp.get("capabilities", []),
                    "tools": bp.get("tools", []),
                    "vertical": bp.get("vertical", bp_plan.vertical),
                    # Agents that must not execute without concrete user-provided transaction
                    # details (order ID, amount, etc.).  The AOPCoordinator reads this flag
                    # to decide whether to block execution until the user supplies context.
                    # Add new action agent kinds here as the platform grows.
                    "requires_user_context": agent_kind
                    in {"workflow_runner", "domain_agent"},
                    # customer_facing is now derived from doc_content_meta
                    # (has_customer_facing_docs / has_internal_policy)
                    # based on user-provided doc_visibility, not heuristics.
                    # Declarative AOP eligibility: set by the planning LLMs
                    # (infer_capabilities → blueprint_creator).  AOPCoordinator
                    # uses this to decide which agents can receive user-facing
                    # subtasks.  When absent (legacy specs), AOP falls back to
                    # a description-based heuristic.
                    "aop_eligible": bp.get("aop_eligible"),
                    # Document content metadata for intent-aware routing.
                    # Derived from persisted .doc_metadata.json content analysis.
                    **doc_content_meta,
                },
            }
        )

    # final spec structure
    spec: Dict[str, Any] = {
        "version": "1.0",
        "vertical": bp_plan.vertical,
        "modes": {"dry_run": bool(dry_run)},
        "paths": {
            "base_dir": _abs(base_dir),
            "policy_pack": "spec/base_policy_pack.yaml",
        },
        "agents": agents_block,
        "tools": [
            # We'll replace this with a real tool registry later.
            {"id": "ticketing", "type": "dummy", "base_url": None}
        ],
        "plan_preview": {
            # helpful for debugging and UI
            "missing_docs": bp_plan.missing_docs,
            "warnings": bp_plan.warnings,
            "rationale": bp_plan.rationale,
        },
    }

    # --- write primary spec inside workspace/.factory ---
    spec_json = json.dumps(spec, indent=2)
    spec_path.write_text(spec_json, encoding="utf-8")

    # --- mirror spec to repo-root .factory for runtime startup ---
    try:
        root_spec_dir = Path(".factory")
        root_spec_dir.mkdir(parents=True, exist_ok=True)
        mirror_path = root_spec_dir / "factory_spec.json"
        mirror_path.write_text(spec_json, encoding="utf-8")
        print(
            f"[INFO] Factory spec written to both:\n"
            f"  - {spec_path}\n"
            f"  - {mirror_path}"
        )
    except Exception as e:
        print(f"[WARN] Could not mirror spec to repo root: {e}")

    return spec
