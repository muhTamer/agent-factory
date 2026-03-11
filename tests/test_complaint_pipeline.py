# tests/test_complaint_pipeline.py
"""
Full pipeline test for the complaint domain.

Proves the agent-factory system is domain-agnostic by compiling a complaint
policy YAML (not refund) through the same pipeline and verifying:

  Layer 1 — Compilation: YAML → CompiledPolicyPack (correct rule types)
  Layer 2 — Policy bridge: eligibility, approval, risk control evaluation
  Layer 3 — Spec builder: auto-wiring of policy config (slot_map, domain)
"""
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLAINTS_YAML = REPO_ROOT / "data" / "complaints_policy.yaml"

pytestmark = pytest.mark.skipif(
    not COMPLAINTS_YAML.exists(),
    reason="complaints_policy.yaml not found in data/",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_pack():
    """Compile complaints_policy.yaml into a CompiledPolicyPack."""
    from app.runtime.policy.policy_compiler import PolicyCompiler

    compiler = PolicyCompiler(domain="fintech")
    pack = compiler.compile_policies([COMPLAINTS_YAML])
    return pack


@pytest.fixture(scope="module")
def bridge(compiled_pack):
    """Build a WorkflowPolicyBridge from the compiled complaint pack."""
    from app.runtime.policy.workflow_policy_bridge import WorkflowPolicyBridge

    return WorkflowPolicyBridge(compiled_pack)


# Baseline: active customer, recent complaint, no duplicates
ELIGIBLE_SLOTS = {
    "account_status": "active",
    "complaint_age_days": 30,
    "duplicate_complaint_open": False,
    "compensation_amount": 200,
    "complaints_last_30_days": 1,
    "complaint_category": "billing",
}


# ===================================================================
# Layer 1 — Compilation
# ===================================================================


def test_complaints_policy_compiles(compiled_pack):
    """PolicyCompiler produces a non-empty CompiledPolicyPack."""
    assert compiled_pack is not None
    assert compiled_pack.policy_id


def test_complaints_pack_has_rules(compiled_pack):
    """Pack contains a meaningful number of rules."""
    assert len(compiled_pack.rules) >= 5


def test_complaints_pack_has_eligibility_rules(compiled_pack):
    from app.runtime.policy.policy_ast import RuleType

    eligibility_rules = [r for r in compiled_pack.rules if r.rule_type == RuleType.ELIGIBILITY]
    assert len(eligibility_rules) >= 2, "Expected at least 2 eligibility rules"


def test_complaints_pack_has_amount_threshold_rules(compiled_pack):
    from app.runtime.policy.policy_ast import RuleType

    amount_rules = [r for r in compiled_pack.rules if r.rule_type == RuleType.AMOUNT_THRESHOLD]
    assert len(amount_rules) >= 1, "Expected at least 1 amount threshold rule"


# ===================================================================
# Layer 2 — Policy Bridge Evaluation
# ===================================================================


def test_active_customer_eligible(bridge):
    """Active account + within time limit → eligible."""
    is_eligible, reason, _ = bridge.check_eligibility(ELIGIBLE_SLOTS)
    assert is_eligible is True
    assert reason


def test_inactive_account_ineligible(bridge):
    """Suspended account → ineligible."""
    slots = {**ELIGIBLE_SLOTS, "account_status": "suspended"}
    is_eligible, reason, _ = bridge.check_eligibility(slots)
    assert is_eligible is False


def test_frozen_account_ineligible(bridge):
    """Frozen account → ineligible."""
    slots = {**ELIGIBLE_SLOTS, "account_status": "frozen"}
    is_eligible, reason, _ = bridge.check_eligibility(slots)
    assert is_eligible is False


def test_expired_complaint_ineligible(bridge):
    """Complaint older than 180 days → ineligible."""
    slots = {**ELIGIBLE_SLOTS, "complaint_age_days": 200}
    is_eligible, reason, _ = bridge.check_eligibility(slots)
    assert is_eligible is False


def test_low_compensation_no_approval(bridge):
    """EUR 200 compensation → auto-approved, no manager needed."""
    slots = {**ELIGIBLE_SLOTS, "compensation_amount": 200}
    needed, reason, _ = bridge.check_approval_needed(slots)
    assert needed is False


def test_high_compensation_requires_approval(bridge):
    """EUR 1000 compensation → manager approval required."""
    slots = {**ELIGIBLE_SLOTS, "compensation_amount": 1000}
    needed, reason, _ = bridge.check_approval_needed(slots)
    assert needed is True


# ===================================================================
# Layer 3 — Spec Builder Integration
# ===================================================================


def test_compile_produces_slot_schema(compiled_pack):
    """Compiled pack has a slot_schema with domain-relevant slot names."""
    schema = compiled_pack.slot_schema
    assert isinstance(schema, dict)
    # Should contain complaint-domain slots, not just refund slots
    slot_names = set(schema.keys())
    # At minimum, account_status and compensation_amount should be detected
    assert "account_status" in slot_names or "compensation_amount" in slot_names


def test_amount_slot_uses_compensation_name(compiled_pack):
    """Amount threshold rules use 'compensation_amount', not 'refund_amount_requested'."""
    from app.runtime.policy.policy_ast import RuleType

    amount_rules = [r for r in compiled_pack.rules if r.rule_type == RuleType.AMOUNT_THRESHOLD]
    for rule in amount_rules:
        if rule.conditions:
            slot_names = _extract_slot_names(rule.conditions)
            if slot_names:
                assert "refund_amount_requested" not in slot_names, (
                    f"Rule {rule.rule_id} should use 'compensation_amount', "
                    f"not 'refund_amount_requested'"
                )


def test_pack_domain_is_set(compiled_pack):
    """Compiled pack has a domain field set."""
    assert compiled_pack.domain


# ===================================================================
# Helpers
# ===================================================================


def _extract_slot_names(conditions) -> set:
    """Recursively extract slot names from a Condition or ConditionGroup."""
    from app.runtime.policy.policy_ast import Condition, ConditionGroup

    names = set()
    if isinstance(conditions, Condition):
        names.add(conditions.slot_name)
    elif isinstance(conditions, ConditionGroup):
        for c in conditions.conditions:
            names.update(_extract_slot_names(c))
    return names
