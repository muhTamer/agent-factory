# app/runtime/policy/integration_example.py
"""
Policy Compilation Integration Example

This shows how to integrate the policy compilation pipeline into your
existing agent factory workflow system.

BEFORE (LLM guesses decisions):
    workflow_mapper.py → LLM decides next event → might get stuck
    
AFTER (Rules make decisions):
    policy_compiler.py → compile rules → workflow uses rules → deterministic
"""
import json
from pathlib import Path
from typing import Any, Dict, List

# Your existing imports
from app.runtime.workflow_mapper import map_query_to_event_and_slots

# New policy imports
from app.runtime.policy.policy_compiler import PolicyCompiler
from app.runtime.policy.workflow_policy_bridge import WorkflowPolicyBridge


# ============================================================================
# STEP 1: Compile Policies (ONE-TIME or on change)
# ============================================================================


def compile_policies_for_domain(domain: str, policy_files: List[Path]):
    """
    Compile policies for a domain.

    Run this:
    - At startup
    - When policy files change
    - During agent generation
    """
    import app.llm_client as llm_client

    compiler = PolicyCompiler(
        llm_client=llm_client,
        domain=domain,
        model="gpt-4o-mini",
    )

    # Compile all policy files
    pack = compiler.compile_policies(
        policy_files=policy_files,
        policy_id=f"{domain}_policies",
        version="1.0",
    )

    # Validate
    issues = compiler.validate_pack(pack)
    if issues:
        print(f"[POLICY] Validation warnings: {issues}")

    # Save to .factory directory
    output_path = Path(".factory") / f"{domain}_policy_pack.json"
    compiler.save_pack(pack, output_path)

    print(f"[POLICY] Compiled {len(pack.rules)} rules for {domain}")
    print(f"[POLICY] Saved to {output_path}")

    return pack, output_path


# ============================================================================
# STEP 2: Enhanced Workflow Runner (with Policy Bridge)
# ============================================================================


class PolicyAwareWorkflowRunner:
    """
    Enhanced workflow runner that uses compiled policies for decisions.

    This REPLACES LLM guessing with deterministic rule evaluation.
    """

    def __init__(
        self,
        workflow_id: str,
        fsm_config: Dict[str, Any],
        policy_pack_path: Path,
        llm_client,
    ):
        self.workflow_id = workflow_id
        self.fsm = fsm_config
        self.llm_client = llm_client

        # Load compiled policy pack
        compiler = PolicyCompiler(llm_client=llm_client, domain="")
        self.policy_pack = compiler.load_pack(policy_pack_path)

        # Create policy bridge
        self.policy_bridge = WorkflowPolicyBridge(self.policy_pack)

        print(f"[WORKFLOW] Loaded {len(self.policy_pack.rules)} policy rules")

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main workflow handler with policy integration."""
        query = request.get("query", "")
        context = request.get("context", {})

        # Get current state and slots from context
        current_state = context.get("current_state", "start")
        slots = context.get("slots", {})

        # STEP 1: Extract slots from query (still uses LLM, but only for extraction)
        allowed_events = self._get_allowed_events(current_state)
        slot_defs = self.policy_pack.slot_schema

        map_result = map_query_to_event_and_slots(
            query=query,
            current_state=current_state,
            allowed_events=allowed_events,
            slot_defs=slot_defs,
            current_slots=slots,
        )

        # Update slots
        slots.update(map_result.slots)

        # STEP 2: Check eligibility (RULE-BASED, not LLM)
        is_eligible, eligibility_reason, _ = self.policy_bridge.check_eligibility(slots)

        if not is_eligible:
            return {
                "workflow_id": self.workflow_id,
                "current_state": current_state,
                "status": "blocked",
                "reason": eligibility_reason,
                "slots": slots,
                "terminal": True,
            }

        # STEP 3: Check approval needed (RULE-BASED, not LLM)
        approval_needed, approval_reason, _ = self.policy_bridge.check_approval_needed(slots)

        # STEP 4: Check risk controls (RULE-BASED)
        has_risks, risk_reasons, _ = self.policy_bridge.check_risk_controls(slots)

        # STEP 5: Determine next state (RULE-BASED if possible, LLM fallback)
        next_state = self.policy_bridge.get_next_state(
            current_state=current_state,
            slots=slots,
            allowed_transitions=allowed_events,
        )

        # Fallback to LLM-mapped event if no rule matched
        if not next_state and map_result.event:
            next_state = map_result.event

        # STEP 6: Check for missing required slots
        missing_slots = self.policy_bridge.get_missing_slots(current_state, slots)

        if missing_slots:
            return {
                "workflow_id": self.workflow_id,
                "current_state": current_state,
                "status": "awaiting_info",
                "missing_slots": missing_slots,
                "slots": slots,
                "terminal": False,
            }

        # STEP 7: Check if tools should be called (RULE-BASED)
        tools_to_call = []
        for tool_name in [
            "payment_processor",
            "compliance_check",
            "notification_service",
        ]:
            should_call, tool_params, _ = self.policy_bridge.should_call_tool(tool_name, slots)
            if should_call:
                tools_to_call.append({"tool": tool_name, "params": tool_params})

        # STEP 8: Build response
        response = {
            "workflow_id": self.workflow_id,
            "current_state": current_state,
            "next_state": next_state,
            "slots": slots,
            "status": "in_progress",
            "approval_needed": approval_needed,
            "terminal": next_state in self._get_terminal_states(),
        }

        if approval_needed:
            response["approval_reason"] = approval_reason
            response["approval_teams"] = self.policy_bridge.get_required_approvals(slots)

        if has_risks:
            response["risk_flags"] = risk_reasons

        if tools_to_call:
            response["tools_to_call"] = tools_to_call

        # STEP 9: Generate audit trail (for compliance)
        response["audit_trail"] = self.policy_bridge.get_audit_trail(slots)

        return response

    def _get_allowed_events(self, state: str) -> List[str]:
        """Get allowed transitions from FSM config."""
        state_config = self.fsm.get("states", {}).get(state, {})
        return list(state_config.get("on", {}).keys())

    def _get_terminal_states(self) -> List[str]:
        """Get terminal states from FSM config."""
        return [
            state_name
            for state_name, state_config in self.fsm.get("states", {}).items()
            if state_config.get("terminal", False)
        ]


# ============================================================================
# STEP 3: Integration into Service Startup
# ============================================================================


def startup_with_policy_compilation():
    """
    Enhanced startup that compiles policies before loading agents.

    Add this to your service.py startup_event() function.
    """
    from pathlib import Path
    import json

    REPO_ROOT = Path(__file__).resolve().parents[3]
    FACTORY_DIR = REPO_ROOT / ".factory"
    FACTORY_DIR.mkdir(exist_ok=True)

    # Load factory spec
    spec_path = FACTORY_DIR / "factory_spec.json"
    if not spec_path.exists():
        print("[BOOT] No factory spec found, skipping policy compilation")
        return

    spec = json.loads(spec_path.read_text())
    domain = spec.get("domain", "general")

    # Find policy files
    policy_files = []

    # From environment variable
    import os

    env_policies = os.getenv("AF_POLICIES", "").strip()
    if env_policies:
        for p in env_policies.split(","):
            p = p.strip()
            if p and Path(p).exists():
                policy_files.append(Path(p))

    # From data directory
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        for ext in ["*.yaml", "*.yml", "*.pdf", "*.txt"]:
            policy_files.extend(data_dir.glob(ext))

    # From spec (if agents reference policies)
    for agent_spec in spec.get("agents", []):
        agent_policies = agent_spec.get("policies", [])
        for pol_path in agent_policies:
            p = Path(pol_path)
            if p.exists() and p not in policy_files:
                policy_files.append(p)

    if not policy_files:
        print("[BOOT] No policy files found")
        return

    print(f"[BOOT] Found {len(policy_files)} policy files")

    # Compile policies
    pack, pack_path = compile_policies_for_domain(domain, policy_files)

    # Store pack path in spec for agents to reference
    spec["policy_pack"] = str(pack_path)
    spec_path.write_text(json.dumps(spec, indent=2))

    print(f"[BOOT] Policy compilation complete: {pack_path}")


# ============================================================================
# STEP 4: Usage Example
# ============================================================================


def example_usage():
    """Complete example of using the policy system."""

    # 1. Compile policies (do this once at startup or when policies change)
    policy_files = [
        Path("data/refunds_policy.yaml"),
        # Path("data/compliance_guide.pdf"),  # If you have PDFs
    ]

    pack, pack_path = compile_policies_for_domain("fintech", policy_files)

    # 2. Create workflow runner with policy bridge
    import app.llm_client as llm_client

    fsm_config = {
        "states": {
            "start": {"on": {"submit_request": "validating"}},
            "validating": {"on": {"valid": "processing", "invalid": "rejected"}},
            "processing": {"on": {"approved": "executing", "needs_approval": "approval_pending"}},
            "approval_pending": {"on": {"approved": "executing", "rejected": "rejected"}},
            "executing": {"on": {"success": "completed", "failed": "failed"}},
            "completed": {"terminal": True},
            "rejected": {"terminal": True},
            "failed": {"terminal": True},
        }
    }

    runner = PolicyAwareWorkflowRunner(
        workflow_id="refund_processor",
        fsm_config=fsm_config,
        policy_pack_path=pack_path,
        llm_client=llm_client,
    )

    # 3. Handle request
    request = {
        "query": "I want a refund for transaction TX123, amount is 3000 EUR",
        "context": {
            "current_state": "start",
            "slots": {
                "kyc_status": "verified",
                "account_status": "active",
            },
        },
    }

    response = runner.handle(request)

    # 4. Response will include:
    # - approval_needed: True/False (from rules, not LLM guess)
    # - next_state: determined by rules
    # - missing_slots: what's still needed
    # - audit_trail: which rules matched and why

    print(json.dumps(response, indent=2))

    """
    Expected output:
    {
      "workflow_id": "refund_processor",
      "current_state": "start",
      "next_state": "validating",
      "slots": {
        "kyc_status": "verified",
        "account_status": "active",
        "refund_amount_requested": 3000,
        "transaction_id": "TX123"
      },
      "status": "in_progress",
      "approval_needed": false,  // <-- DETERMINED BY RULE (amount <= 5000)
      "terminal": false,
      "audit_trail": [
        {
          "matched": true,
          "rule_id": "verified_customer_required",
          "explanation": "kyc_status=verified satisfies kyc_status == verified",
          "citations": [...]
        }
      ]
    }
    """


if __name__ == "__main__":
    example_usage()
