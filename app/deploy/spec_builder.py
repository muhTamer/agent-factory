# app/deploy/spec_builder.py
from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.concierge.blueprint_creator import BlueprintCreatorAgent


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
def _inputs_from_blueprint(bp: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
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
        matches = [f for f in data_dir.iterdir() if f.is_file() and hint and hint in f.name.lower()]
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

    # For tool_operator, allow a default tool name from blueprint inputs or bp["tools"][0]
    if agent_kind == "tool_operator":
        if "tool" not in inputs or not inputs.get("tool"):
            tools = bp.get("tools") or []
            if isinstance(tools, list) and tools:
                inputs["tool"] = str(tools[0])

    # Best-effort resolve any string placeholders inside inputs (shallow)
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
                    else:
                        # resolve relative to data_dir if looks like a path
                        if ("/" in item) or ("\\" in item):
                            p = Path(item)
                            new_list.append(_abs(p) if p.is_absolute() else _abs(data_dir / p))
                        else:
                            new_list.append(item)
                else:
                    new_list.append(item)
            inputs[k] = new_list

    return inputs


# ------------------------------------------------------------
# 🔗 Policy config auto-detection for workflow_runner agents
# ------------------------------------------------------------
def _build_policy_config(workflow_spec: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
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
        raw_states = {s["name"]: s for s in raw_states if isinstance(s, dict) and "name" in s}

    # Eligibility events: vocab-based (superset-safe — event_set may include "error" etc.)
    _ELIG_PASS = {"eligible", "eligibility_pass", "eligibility_met", "eligibility_passed"}
    _ELIG_FAIL = {"ineligible", "eligibility_fail", "eligibility_failed", "not_eligible"}

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
                if e.lower().startswith("eligible") and not e.lower().startswith("ineligible")
            ]

        def _elig_fail_events(es):
            exact = [e for e in es if e in _ELIG_FAIL]
            if exact:
                return exact
            return [
                e for e in es if e.lower().startswith("ineligible") or e.lower() == "not_eligible"
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
            pass_events = [e for e in non_neutral if any(e.lower().endswith(s) for s in _PASS_SFXS)]
            fail_events = [e for e in non_neutral if any(e.lower().endswith(s) for s in _FAIL_SFXS)]
            all_matched = all(
                any(e.lower().endswith(s) for s in _PASS_SFXS + _FAIL_SFXS) for e in non_neutral
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


# ------------------------------------------------------------
# 🧰 Spec Builder (NEW: uses LLM-generated AgentBlueprints, not per-capability hardcoding)
# ------------------------------------------------------------
def build_factory_spec(
    plan: Dict[str, Any],
    data_dir: str,
    dry_run: bool = True,
    llm_client: Optional[object] = None,
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

    # optional: legacy blueprint directory (still useful later)
    # bp_dir = Path("factory/blueprints")

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
    print(f"[SPEC] Blueprint plan: {len(bp_plan.blueprints)} agents | vertical={bp_plan.vertical}")
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
        "knowledge_rag": "knowledge_rag",
        "workflow_runner": "workflow_runner",
        "tool_operator": "tool_operator",
    }

    for bp in bp_plan.blueprints:
        agent_id = str(bp.get("id")).strip()
        if not agent_id or agent_id.lower() in {"guardrails", "qa"}:
            continue

        agent_kind = str(bp.get("agent_kind", "")).strip()
        blueprint_id = kind_to_blueprint.get(agent_kind)
        if not blueprint_id:
            raise ValueError(f"Unsupported agent_kind '{agent_kind}' for blueprint id='{agent_id}'")

        agents_block.append(
            {
                "id": agent_id,
                "type": "autogen",
                "blueprint": blueprint_id,
                "status": "ready",  # this is plan-level; can be refined later
                "inputs": _inputs_from_blueprint(bp, base_dir),
                "blueprint_meta": {
                    # keep the whole declarative blueprint for routing + explainability
                    "agent_kind": agent_kind,
                    "description": bp.get("description", ""),
                    "capabilities": bp.get("capabilities", []),
                    "tools": bp.get("tools", []),
                    "vertical": bp.get("vertical", bp_plan.vertical),
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
        print(f"[INFO] Factory spec written to both:\n" f"  - {spec_path}\n" f"  - {mirror_path}")
    except Exception as e:
        print(f"[WARN] Could not mirror spec to repo root: {e}")

    return spec
