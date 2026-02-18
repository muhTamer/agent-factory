# examples/test_policy_refinement.py
"""
Policy Refinement Examples

Demonstrates how messy customer-provided policies are refined
before compilation into executable rules.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Example 1: Messy formatting and missing metadata
MESSY_POLICY_1 = """
# Customer's refund rules

eligibility:
  - when customer verified and account is active
  - must request within 90 days
  
amounts:
  small refunds under 5000 are auto approved
  large refunds need manager approval
  
risks:
  if customer has more than 3 refunds in 30 days flag for review
"""

EXPECTED_REFINEMENT_1 = """
metadata:
  policy_id: customer_refunds
  domain: fintech
  version: "1.0"
  description: Customer refund policy

eligibility:
  - id: verified_customer
    description: Customer must be verified and account active
    conditions:
      - kyc_status == "verified"
      - account_status == "active"
    outcome: eligible

  - id: time_limit
    description: Must request within 90 days
    conditions:
      - transaction_age_days <= 90
    outcome: eligible

amount_rules:
  - id: small_refund_auto
    description: Small refunds auto-approved
    threshold: 5000
    currency: EUR
    approval: auto

  - id: large_refund_manual
    description: Large refunds need manager approval
    threshold: 5000
    currency: EUR
    approval: manager_required

risk_controls:
  - id: velocity_check
    description: Flag high refund velocity
    condition:
      - refunds_last_30_days > 3
    action: escalate_risk_review
"""


# Example 2: Inconsistent slot names and operators
MESSY_POLICY_2 = """
eligibility:
  - if KYCStatus equals verified
  - if AccountStatus is active or pending
  - transaction age should be less than 90 days
  
amount_rules:
  - for amounts up to EUR 5000 approve automatically
  - amounts over EUR 5000 require manager sign-off
"""

EXPECTED_REFINEMENT_2 = """
metadata:
  policy_id: customer_eligibility
  domain: fintech
  version: "1.0"

eligibility:
  - id: kyc_verified
    description: KYC must be verified
    conditions:
      - kyc_status == "verified"
    outcome: eligible

  - id: account_active
    description: Account must be active or pending
    conditions:
      - account_status in ["active", "pending"]
    outcome: eligible

  - id: transaction_age_limit
    description: Transaction must be recent
    conditions:
      - transaction_age_days < 90
    outcome: eligible

amount_rules:
  - id: auto_approval_limit
    description: Auto-approve small amounts
    threshold: 5000
    currency: EUR
    approval: auto

  - id: manager_approval_limit
    description: Manager approval for large amounts
    threshold: 5000
    currency: EUR
    approval: manager_required
"""


# Example 3: Natural language conditions
MESSY_POLICY_3 = """
rules:
  - customers who have been with us for more than a year and haven't had any chargebacks can get instant refunds up to $10,000
  
  - new customers (less than 6 months) need manual review for any refund over $500
  
  - if we detect the customer is from a high-risk country, escalate to compliance team regardless of amount
"""

EXPECTED_REFINEMENT_3 = """
metadata:
  policy_id: tiered_refund_policy
  domain: fintech
  version: "1.0"

eligibility:
  - id: established_customer_instant
    description: Established customers with clean history get instant refunds
    conditions:
      - customer_tenure_months > 12
      - chargeback_count == 0
      - refund_amount_requested <= 10000
    outcome: eligible

amount_rules:
  - id: established_customer_limit
    description: Instant approval for established customers
    threshold: 10000
    currency: USD
    approval: auto
    conditions:
      - customer_tenure_months > 12
      - chargeback_count == 0

  - id: new_customer_limit
    description: Manual review for new customers over threshold
    threshold: 500
    currency: USD
    approval: manager_required
    conditions:
      - customer_tenure_months < 6

risk_controls:
  - id: high_risk_country_check
    description: Escalate high-risk countries to compliance
    condition:
      - customer_country in ["sanctioned_list"]
    action: compliance_hold
"""


def demonstrate_refinement():
    """Demonstrate policy refinement with examples."""

    print("=" * 80)
    print("POLICY REFINEMENT DEMONSTRATION")
    print("=" * 80)
    print()

    examples = [
        ("Messy Formatting", MESSY_POLICY_1, EXPECTED_REFINEMENT_1),
        ("Inconsistent Naming", MESSY_POLICY_2, EXPECTED_REFINEMENT_2),
        ("Natural Language", MESSY_POLICY_3, EXPECTED_REFINEMENT_3),
    ]

    for title, messy, expected in examples:
        print(f"Example: {title}")
        print("-" * 80)
        print()

        print("BEFORE (Customer-provided):")
        print("─" * 40)
        print(messy.strip())
        print()

        print("AFTER (Refined):")
        print("─" * 40)
        print(expected.strip())
        print()
        print("=" * 80)
        print()


def test_refinement_live():
    """Test refinement with actual LLM (requires llm_client)."""

    try:
        import app.llm_client as llm_client
        from app.runtime.policy.policy_refiner import PolicyRefiner
    except ImportError:
        print("⚠️  Cannot import llm_client - skipping live test")
        return

    print("=" * 80)
    print("LIVE REFINEMENT TEST")
    print("=" * 80)
    print()

    refiner = PolicyRefiner(llm_client)

    # Test with messy policy
    print("Refining messy policy...")
    print()

    result = refiner.refine_policy(
        policy_yaml=MESSY_POLICY_1,
        domain="fintech",
    )

    print("Refinement complete!")
    print()

    if result.issues_found:
        print(f"Issues found ({len(result.issues_found)}):")
        for issue in result.issues_found:
            print(f"  - {issue}")
        print()

    if result.fixes_applied:
        print(f"Fixes applied ({len(result.fixes_applied)}):")
        for fix in result.fixes_applied:
            print(f"  ✓ {fix}")
        print()

    print(f"Confidence: {result.confidence:.1%}")
    print(f"Needs review: {result.needs_review}")
    print()

    print("Refined YAML:")
    print("-" * 80)
    print(result.refined_yaml)
    print()


def main():
    """Run all demonstrations."""

    # Show examples
    demonstrate_refinement()

    # Try live refinement if available
    test_refinement_live()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("The policy refiner handles:")
    print("  ✓ Missing metadata")
    print("  ✓ Inconsistent formatting")
    print("  ✓ Natural language conditions")
    print("  ✓ Ambiguous slot names")
    print("  ✓ Invalid operators")
    print("  ✓ Incomplete rule definitions")
    print()
    print("Usage:")
    print("  # Validate customer policy")
    print("  python -m app.runtime.policy.policy_refiner_cli validate customer.yaml")
    print()
    print("  # Refine customer policy")
    print("  python -m app.runtime.policy.policy_refiner_cli refine customer.yaml")
    print()
    print("  # Batch refine multiple policies")
    print("  python -m app.runtime.policy.policy_refiner_cli batch-refine input/ output/")
    print()


if __name__ == "__main__":
    main()
