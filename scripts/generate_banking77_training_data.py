# scripts/generate_banking77_training_data.py
"""
Generate contrastive training data for the Neural Solvability Estimator
using the BANKING77 dataset (Casanueva et al., 2020).

Maps 77 banking intents → 3 agents (customer_faq_agent_v1, refunds_agent_v1,
complaints_agent_v1), then creates contrastive (subtask, agent_desc, score)
triples where:
  - correct agent pairing → score 1.0
  - wrong agent pairing  → score 0.0

This produces ~30K training pairs (10K utterances × 3 agents) with strong
discriminative signal, replacing the original 42-example single-pairing dataset.

Usage:
    PYTHONPATH=. python scripts/generate_banking77_training_data.py
    PYTHONPATH=. python scripts/generate_banking77_training_data.py --max-per-intent 50
    PYTHONPATH=. python scripts/generate_banking77_training_data.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path
from typing import Dict, List

# ── BANKING77 intent → agent mapping ──────────────────────────────
#
# Each of the 77 intents is mapped to one of three agents based on
# the intent's semantic category:
#
# customer_faq_agent_v1  — informational queries about products, accounts,
#                          cards, transfers, identity, policies
# refunds_agent_v1       — refund requests, reversals, chargebacks,
#                          unrecognised charges, wrong amounts
# complaints_agent_v1    — complaints, disputes, service quality issues,
#                          escalations, unresolved problems

INTENT_TO_AGENT: Dict[str, str] = {
    # ── FAQ / Informational intents → customer_faq_agent_v1 ──────
    "activate_my_card": "customer_faq_agent_v1",
    "age_limit": "customer_faq_agent_v1",
    "apple_pay_or_google_pay": "customer_faq_agent_v1",
    "atm_support": "customer_faq_agent_v1",
    "automatic_top_up": "customer_faq_agent_v1",
    "beneficiary_not_allowed": "customer_faq_agent_v1",
    "card_about_to_expire": "customer_faq_agent_v1",
    "card_acceptance": "customer_faq_agent_v1",
    "card_arrival": "customer_faq_agent_v1",
    "card_delivery_estimate": "customer_faq_agent_v1",
    "card_linking": "customer_faq_agent_v1",
    "change_pin": "customer_faq_agent_v1",
    "contactless_not_working": "customer_faq_agent_v1",
    "country_support": "customer_faq_agent_v1",
    "disposable_card_limits": "customer_faq_agent_v1",
    "edit_personal_details": "customer_faq_agent_v1",
    "exchange_charge": "customer_faq_agent_v1",
    "exchange_rate": "customer_faq_agent_v1",
    "exchange_via_app": "customer_faq_agent_v1",
    "fiat_currency_support": "customer_faq_agent_v1",
    "get_disposable_virtual_card": "customer_faq_agent_v1",
    "get_physical_card": "customer_faq_agent_v1",
    "getting_spare_card": "customer_faq_agent_v1",
    "getting_virtual_card": "customer_faq_agent_v1",
    "lost_or_stolen_phone": "customer_faq_agent_v1",
    "order_physical_card": "customer_faq_agent_v1",
    "passcode_forgotten": "customer_faq_agent_v1",
    "pending_card_payment": "customer_faq_agent_v1",
    "pending_cash_withdrawal": "customer_faq_agent_v1",
    "pending_top_up": "customer_faq_agent_v1",
    "pending_transfer": "customer_faq_agent_v1",
    "pin_blocked": "customer_faq_agent_v1",
    "receiving_money": "customer_faq_agent_v1",
    "supported_cards_and_currencies": "customer_faq_agent_v1",
    "terminate_account": "customer_faq_agent_v1",
    "top_up_by_bank_transfer_charge": "customer_faq_agent_v1",
    "top_up_by_card_charge": "customer_faq_agent_v1",
    "top_up_by_cash_or_cheque": "customer_faq_agent_v1",
    "top_up_failed": "customer_faq_agent_v1",
    "top_up_limits": "customer_faq_agent_v1",
    "topping_up_by_card": "customer_faq_agent_v1",
    "transfer_fee_charged": "customer_faq_agent_v1",
    "transfer_into_account": "customer_faq_agent_v1",
    "transfer_timing": "customer_faq_agent_v1",
    "unable_to_verify_identity": "customer_faq_agent_v1",
    "verify_my_identity": "customer_faq_agent_v1",
    "verify_source_of_funds": "customer_faq_agent_v1",
    "verify_top_up": "customer_faq_agent_v1",
    "virtual_card_not_working": "customer_faq_agent_v1",
    "visa_or_mastercard": "customer_faq_agent_v1",
    "why_verify_identity": "customer_faq_agent_v1",
    # ── Refund / Reversal intents → refunds_agent_v1 ─────────────
    "Refund_not_showing_up": "refunds_agent_v1",
    "balance_not_updated_after_bank_transfer": "refunds_agent_v1",
    "balance_not_updated_after_cheque_or_cash_deposit": "refunds_agent_v1",
    "cancel_transfer": "refunds_agent_v1",
    "card_payment_fee_charged": "refunds_agent_v1",
    "card_payment_not_recognised": "refunds_agent_v1",
    "card_payment_wrong_exchange_rate": "refunds_agent_v1",
    "cash_withdrawal_charge": "refunds_agent_v1",
    "cash_withdrawal_not_recognised": "refunds_agent_v1",
    "direct_debit_payment_not_recognised": "refunds_agent_v1",
    "extra_charge_on_statement": "refunds_agent_v1",
    "failed_transfer": "refunds_agent_v1",
    "request_refund": "refunds_agent_v1",
    "reverted_card_payment?": "refunds_agent_v1",
    "top_up_reverted": "refunds_agent_v1",
    "transaction_charged_twice": "refunds_agent_v1",
    "transfer_not_received_by_recipient": "refunds_agent_v1",
    "wrong_amount_of_cash_received": "refunds_agent_v1",
    "wrong_exchange_rate_for_cash_withdrawal": "refunds_agent_v1",
    # ── Complaint / Dispute intents → complaints_agent_v1 ────────
    "card_not_working": "complaints_agent_v1",
    "card_swallowed": "complaints_agent_v1",
    "compromised_card": "complaints_agent_v1",
    "declined_card_payment": "complaints_agent_v1",
    "declined_cash_withdrawal": "complaints_agent_v1",
    "declined_transfer": "complaints_agent_v1",
    "lost_or_stolen_card": "complaints_agent_v1",
}

# Verify all 77 intents are mapped
assert len(INTENT_TO_AGENT) == 77, f"Expected 77 mappings, got {len(INTENT_TO_AGENT)}"

BANKING77_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
    "/master/banking_data/train.csv"
)
BANKING77_TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
    "/master/banking_data/test.csv"
)

OUTPUT_PATH = Path("data/training_data/reward_training.json")


def fetch_banking77(url: str) -> List[Dict[str, str]]:
    """Download BANKING77 CSV and return list of {text, category} dicts."""
    print(f"  Fetching: {url}")
    req = urllib.request.urlopen(url)
    data = req.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(data))
    rows = list(reader)
    print(f"  Loaded {len(rows)} examples")
    return rows


def get_agent_descriptions() -> Dict[str, str]:
    """Load agent descriptions from the factory spec via bootstrap."""
    from scripts._bootstrap import bootstrap_registry

    registry, _ = bootstrap_registry()
    catalog = registry.all_meta()

    descriptions = {}
    for agent_id, meta in catalog.items():
        parts = []
        desc = meta.get("description", "")
        if desc:
            parts.append(str(desc))
        caps = meta.get("capabilities", [])
        if isinstance(caps, list):
            parts.append(" ".join(str(c) for c in caps))
        atype = meta.get("type", "")
        if atype:
            parts.append(str(atype))
        akind = meta.get("agent_kind", "")
        if akind and akind != atype:
            parts.append(str(akind))
        descriptions[agent_id] = " ".join(parts)

    return descriptions


def generate_contrastive_pairs(
    utterances: List[Dict[str, str]],
    agent_descriptions: Dict[str, str],
    max_per_intent: int | None = None,
) -> List[Dict]:
    """Generate contrastive training triples from BANKING77 utterances.

    For each utterance:
      - Correct agent → score 1.0
      - Each wrong agent → score 0.0

    Returns list of {subtask, agent_description, score, agent_id, source_intent}.
    """
    agent_ids = sorted(agent_descriptions.keys())
    pairs: List[Dict] = []

    # Group by intent for optional per-intent limiting
    by_intent: Dict[str, List[str]] = {}
    for row in utterances:
        intent = row["category"]
        by_intent.setdefault(intent, []).append(row["text"])

    for intent, texts in sorted(by_intent.items()):
        correct_agent = INTENT_TO_AGENT.get(intent)
        if not correct_agent:
            print(f"  WARNING: unmapped intent '{intent}', skipping")
            continue

        # Limit examples per intent if requested
        subset = texts[:max_per_intent] if max_per_intent else texts

        for text in subset:
            for agent_id in agent_ids:
                score = 1.0 if agent_id == correct_agent else 0.0
                pairs.append(
                    {
                        "subtask": text,
                        "agent_description": agent_descriptions[agent_id],
                        "score": score,
                        "agent_id": agent_id,
                        "source_intent": intent,
                    }
                )

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive training data from BANKING77"
    )
    parser.add_argument(
        "--max-per-intent",
        type=int,
        default=None,
        help="Max examples per intent (default: all ~130 each)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include BANKING77 test split too",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Limit to 5 examples per intent for testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    max_per = 5 if args.dry_run else args.max_per_intent

    print("=" * 60)
    print("BANKING77 -> Contrastive Training Data Generator")
    print("=" * 60)

    # 1. Fetch BANKING77
    print("\n[1/3] Fetching BANKING77 dataset...")
    utterances = fetch_banking77(BANKING77_URL)
    if args.include_test:
        utterances += fetch_banking77(BANKING77_TEST_URL)

    # 2. Load agent descriptions
    print("\n[2/3] Loading agent descriptions from factory spec...")
    agent_descs = get_agent_descriptions()
    for aid, desc in agent_descs.items():
        print(f"  {aid}: {desc[:80]}...")

    # Distribution check
    agent_counts = {a: 0 for a in agent_descs}
    for row in utterances:
        agent = INTENT_TO_AGENT.get(row["category"])
        if agent:
            agent_counts[agent] += 1
    print("\n  Intent distribution:")
    for a, c in sorted(agent_counts.items()):
        print(f"    {a}: {c} utterances")

    # 3. Generate contrastive pairs
    print("\n[3/3] Generating contrastive training pairs...")
    pairs = generate_contrastive_pairs(utterances, agent_descs, max_per_intent=max_per)

    # Stats
    positives = sum(1 for p in pairs if p["score"] == 1.0)
    negatives = sum(1 for p in pairs if p["score"] == 0.0)
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Positive (score=1.0): {positives}")
    print(f"  Negative (score=0.0): {negatives}")
    print(f"  Ratio: 1:{negatives // max(1, positives)}")

    # 4. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
