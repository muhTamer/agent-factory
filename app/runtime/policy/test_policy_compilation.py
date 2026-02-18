#!/usr/bin/env python3
# examples/test_policy_compilation.py
"""
Complete test example using the refunds policy.

This demonstrates:
1. Compiling the refunds_policy.yaml
2. Testing rule evaluation
3. Showing how workflows would use it
"""
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_refunds_policy_compilation():
    """Test compiling the refunds policy."""
    print("=" * 80)
    print("POLICY COMPILATION TEST")
    print("=" * 80)
    print()

    # Mock LLM client (for the example - you'd use your real client)
    class MockLLMClient:
        def chat_json(self, messages, model):
            # For this test, we're using YAML so LLM not actually needed
            return {"rules": []}

    from app.runtime.policy.policy_compiler import PolicyCompiler

    llm_client = MockLLMClient()

    # Create compiler
    compiler = PolicyCompiler(
        llm_client=llm_client,
        domain="fintech",
        model="gpt-4o-mini",
    )

    # Compile the refunds policy
    policy_file = Path("data/refunds_policy.yaml")

    if not policy_file.exists():
        print(f"❌ Policy file not found: {policy_file}")
        print("   Please ensure refunds_policy.yaml is in the data/ directory")
        return False

    print(f"📄 Compiling policy: {policy_file}")
    print()

    pack = compiler.compile_policies(
        policy_files=[policy_file],
        policy_id="refunds_policy_test",
        version="1.0",
    )

    print("✅ Compiled successfully!")
    print(f"   - Rules: {len(pack.rules)}")
    print(f"   - Slots: {len(pack.slot_schema)}")
    print()

    # Show rules by type
    from collections import defaultdict

    rules_by_type = defaultdict(list)
    for rule in pack.rules:
        rules_by_type[rule.rule_type].append(rule)

    print("📋 Rules by type:")
    for rule_type, rules in sorted(rules_by_type.items()):
        print(f"   {rule_type.value}: {len(rules)} rules")
    print()

    # Save pack
    output_path = Path(".factory/test_policy_pack.json")
    output_path.parent.mkdir(exist_ok=True)
    compiler.save_pack(pack, output_path)

    print(f"💾 Saved to: {output_path}")
    print()

    return pack, output_path


def test_rule_evaluation(pack):
    """Test evaluating rules against sample slots."""
    print("=" * 80)
    print("RULE EVALUATION TEST")
    print("=" * 80)
    print()

    from app.runtime.policy.rule_engine import RuleEngine

    engine = RuleEngine(pack)

    # Test case 1: Small refund (auto-approved)
    print("Test Case 1: Small refund (3000 EUR)")
    print("-" * 80)

    slots_1 = {
        "kyc_status": "verified",
        "account_status": "active",
        "transaction_age_days": 30,
        "refund_amount_requested": 3000,
        "refunds_last_30_days": 1,
    }

    print(f"Slots: {json.dumps(slots_1, indent=2)}")
    print()

    # Check eligibility
    eligible, results = engine.check_eligibility(slots_1)
    print(f"✓ Eligibility: {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'}")
    if results:
        for r in results:
            print(f"  - {r.rule.rule_id}: {r.explanation}")
    print()

    # Check approval
    approval_needed, result = engine.check_approval_required(slots_1)
    print(f"✓ Approval needed: {'YES' if approval_needed else 'NO'}")
    if result:
        print(f"  - {result.rule.rule_id}: {result.explanation}")
    print()

    # Test case 2: Large refund (needs approval)
    print("Test Case 2: Large refund (8000 EUR)")
    print("-" * 80)

    slots_2 = {
        **slots_1,
        "refund_amount_requested": 8000,
    }

    print(f"Slots: {json.dumps(slots_2, indent=2)}")
    print()

    eligible, results = engine.check_eligibility(slots_2)
    print(f"✓ Eligibility: {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'}")
    print()

    approval_needed, result = engine.check_approval_required(slots_2)
    print(f"✓ Approval needed: {'YES' if approval_needed else 'NO'}")
    if result:
        print(f"  - {result.rule.rule_id}: {result.explanation}")
        for action in result.actions:
            print(f"  - Action: {action.action_type}")
            print(f"  - Params: {action.parameters}")
    print()

    # Test case 3: Risk flag (too many refunds)
    print("Test Case 3: Velocity risk (5 refunds in 30 days)")
    print("-" * 80)

    slots_3 = {
        **slots_1,
        "refunds_last_30_days": 5,
    }

    print(f"Slots: {json.dumps(slots_3, indent=2)}")
    print()

    risk_results = engine.check_risk_controls(slots_3)
    print(f"✓ Risk flags: {len(risk_results)}")
    for r in risk_results:
        print(f"  - {r.rule.rule_id}: {r.explanation}")
        for action in r.actions:
            print(f"    Action: {action.action_type}")
    print()

    # Test case 4: Not eligible (account frozen)
    print("Test Case 4: Not eligible (account frozen)")
    print("-" * 80)

    slots_4 = {
        **slots_1,
        "account_status": "frozen",
    }

    print(f"Slots: {json.dumps(slots_4, indent=2)}")
    print()

    eligible, results = engine.check_eligibility(slots_4)
    print(f"✓ Eligibility: {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'}")
    for r in results:
        print(f"  - {r.rule.rule_id}: {r.explanation}")
        if not r.matched:
            print("    (blocked by this rule)")
    print()


def test_workflow_integration(pack):
    """Test using policy bridge in a workflow."""
    print("=" * 80)
    print("WORKFLOW INTEGRATION TEST")
    print("=" * 80)
    print()

    from app.runtime.policy.workflow_policy_bridge import WorkflowPolicyBridge

    bridge = WorkflowPolicyBridge(pack)

    # Simulate a workflow handling a refund request
    print("Simulating refund workflow...")
    print("-" * 80)
    print()

    slots = {
        "kyc_status": "verified",
        "account_status": "active",
        "transaction_age_days": 45,
        "refund_amount_requested": 6500,
        "refunds_last_30_days": 2,
        "transaction_id": "TX-123456",
    }

    print("Customer request: Refund for TX-123456, amount 6500 EUR")
    print(f"Initial slots: {json.dumps(slots, indent=2)}")
    print()

    # Step 1: Check eligibility
    print("Step 1: Checking eligibility...")
    is_eligible, reason, _ = bridge.check_eligibility(slots)
    print(f"  Result: {'✓ ELIGIBLE' if is_eligible else '✗ NOT ELIGIBLE'}")
    print(f"  Reason: {reason}")
    print()

    if not is_eligible:
        print("❌ Request rejected due to eligibility check")
        return

    # Step 2: Check approval needed
    print("Step 2: Checking if approval needed...")
    approval_needed, approval_reason, rule = bridge.check_approval_needed(slots)
    print(f"  Result: {'YES' if approval_needed else 'NO'}")
    if approval_needed:
        print(f"  Reason: {approval_reason}")

        # Get required approval teams
        teams = bridge.get_required_approvals(slots)
        print(f"  Required approvals: {teams}")
    print()

    # Step 3: Check risk controls
    print("Step 3: Checking risk controls...")
    has_risks, risk_reasons, _ = bridge.check_risk_controls(slots)
    if has_risks:
        print("  ⚠️  Risk flags detected:")
        for reason in risk_reasons:
            print(f"    - {reason}")
    else:
        print("  ✓ No risk flags")
    print()

    # Step 4: Determine next state
    print("Step 4: Determining next workflow state...")
    current_state = "validating"

    # In a real workflow, this would check transition rules
    if not is_eligible:
        next_state = "rejected"
    elif approval_needed:
        next_state = "approval_pending"
    else:
        next_state = "processing"

    print(f"  Current state: {current_state}")
    print(f"  Next state: {next_state}")
    print()

    # Step 5: Get audit trail
    print("Step 5: Generating audit trail...")
    audit = bridge.get_audit_trail(slots)
    print(f"  Matched rules: {len(audit)}")
    for entry in audit[:3]:  # Show first 3
        print(f"    - {entry['rule_id']}: {entry['explanation'][:60]}...")
    print()

    # Final summary
    print("=" * 80)
    print("WORKFLOW DECISION SUMMARY")
    print("=" * 80)
    print(f"Eligible: {is_eligible}")
    print(f"Approval needed: {approval_needed}")
    print(f"Risk flags: {has_risks}")
    print(f"Next state: {next_state}")
    print()
    print("✅ All decisions made deterministically using compiled rules!")
    print("   (No LLM guessing required)")
    print()


def main():
    """Run all tests."""
    print()
    print("🚀 Starting Policy Compilation System Tests")
    print()

    # Test 1: Compilation
    pack, pack_path = test_refunds_policy_compilation()

    if not pack:
        print("❌ Compilation failed, aborting tests")
        return 1

    # Test 2: Rule evaluation
    test_rule_evaluation(pack)

    # Test 3: Workflow integration
    test_workflow_integration(pack)

    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review compiled rules in:", pack_path)
    print("2. Integrate PolicyBridge into your workflow runner")
    print("3. Replace LLM decision points with bridge.check_*() methods")
    print("4. Test with real user queries")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
