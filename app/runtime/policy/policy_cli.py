#!/usr/bin/env python3
# app/runtime/policy/policy_cli.py
"""
Policy Management CLI

Commands:
    compile - Compile policies from sources
    validate - Validate compiled policy pack
    inspect - Inspect rules in a policy pack
    diff - Compare two policy packs
    recompile - Recompile if sources changed
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.runtime.policy.policy_compiler import PolicyCompiler
from app.runtime.policy.rule_engine import RuleEngine


def cmd_compile(args):
    """Compile policies from source files."""
    print(f"[COMPILE] Compiling {len(args.sources)} policy files...")
    print(f"[COMPILE] Domain: {args.domain}")

    # Mock LLM client for now (you'll inject your real client)
    import app.llm_client as llm_client

    compiler = PolicyCompiler(
        llm_client=llm_client,
        domain=args.domain,
        model=args.model,
    )

    pack = compiler.compile_policies(
        policy_files=args.sources,
        policy_id=args.policy_id,
        version=args.version,
    )

    # Validate
    issues = compiler.validate_pack(pack)
    if issues:
        print("\n[VALIDATE] Validation warnings:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n[VALIDATE] ✓ No issues found")

    # Save
    output_path = compiler.save_pack(pack, args.output)

    print(f"\n[SUCCESS] Compiled policy pack saved to: {output_path}")
    print(f"  - Policy ID: {pack.policy_id}")
    print(f"  - Rules: {len(pack.rules)}")
    print(f"  - Slots: {len(pack.slot_schema)}")
    print(f"  - Source hash: {pack.source_hash[:16]}...")


def cmd_validate(args):
    """Validate a compiled policy pack."""
    print(f"[VALIDATE] Loading policy pack: {args.pack}")

    import app.llm_client as llm_client

    compiler = PolicyCompiler(llm_client=llm_client, domain="")

    pack = compiler.load_pack(args.pack)

    print(f"  - Policy ID: {pack.policy_id}")
    print(f"  - Domain: {pack.domain}")
    print(f"  - Rules: {len(pack.rules)}")

    issues = compiler.validate_pack(pack)

    if issues:
        print("\n[ISSUES] Validation warnings:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        return 1
    else:
        print("\n[SUCCESS] ✓ Policy pack is valid")
        return 0


def cmd_inspect(args):
    """Inspect rules in a policy pack."""
    print(f"[INSPECT] Loading policy pack: {args.pack}")

    import app.llm_client as llm_client

    compiler = PolicyCompiler(llm_client=llm_client, domain="")

    pack = compiler.load_pack(args.pack)

    print(f"\n{'='*80}")
    print(f"Policy Pack: {pack.policy_id}")
    print(f"Domain: {pack.domain}")
    print(f"Version: {pack.version}")
    print(f"Compiled: {pack.compiled_at}")
    print(f"Source files: {', '.join(pack.source_files)}")
    print(f"{'='*80}\n")

    # Group by type
    from collections import defaultdict

    by_type = defaultdict(list)

    for rule in pack.rules:
        by_type[rule.rule_type].append(rule)

    for rule_type, rules in sorted(by_type.items()):
        print(f"\n{rule_type.value.upper()} ({len(rules)} rules)")
        print("-" * 80)

        for rule in sorted(rules, key=lambda r: r.priority):
            print(f"\n  [{rule.priority:3d}] {rule.rule_id}")
            print(f"       {rule.description[:100]}")

            if args.verbose:
                if rule.conditions:
                    print(f"       Conditions: {rule.conditions}")
                print(f"       Actions: {[a.action_type for a in rule.actions]}")

                if rule.citations:
                    print("       Citations:")
                    for cite in rule.citations:
                        print(f"         - {cite.source_file}:{cite.section}:{cite.rule_id}")

    print(f"\n{'='*80}")
    print(f"Slot Schema ({len(pack.slot_schema)} slots)")
    print(f"{'='*80}\n")

    for slot_name, schema in sorted(pack.slot_schema.items()):
        print(f"  {slot_name}:")
        print(f"    Type: {schema.get('type', 'any')}")
        if schema.get("required"):
            print("    Required: yes")
        if schema.get("possible_values"):
            vals = schema["possible_values"]
            if isinstance(vals, (list, set)):
                vals_str = ", ".join(str(v) for v in list(vals)[:5])
                if len(vals) > 5:
                    vals_str += f"... ({len(vals)} total)"
                print(f"    Values: {vals_str}")


def cmd_diff(args):
    """Compare two policy packs."""
    print("[DIFF] Comparing policy packs...")
    print(f"  Old: {args.old}")
    print(f"  New: {args.new}")

    import app.llm_client as llm_client

    compiler = PolicyCompiler(llm_client=llm_client, domain="")

    old_pack = compiler.load_pack(args.old)
    new_pack = compiler.load_pack(args.new)

    print(f"\n{'='*80}")
    print("POLICY PACK DIFF")
    print(f"{'='*80}\n")

    # Compare metadata
    print(f"Policy ID:   {old_pack.policy_id} -> {new_pack.policy_id}")
    print(f"Version:     {old_pack.version} -> {new_pack.version}")
    print(f"Domain:      {old_pack.domain} -> {new_pack.domain}")
    print(f"Rules:       {len(old_pack.rules)} -> {len(new_pack.rules)}")
    print(f"Source hash: {old_pack.source_hash[:16]}... -> {new_pack.source_hash[:16]}...")

    # Compare rules
    old_ids = {r.rule_id for r in old_pack.rules}
    new_ids = {r.rule_id for r in new_pack.rules}

    added = new_ids - old_ids
    removed = old_ids - new_ids
    common = old_ids & new_ids

    print("\nRule changes:")
    print(f"  Added:   {len(added)}")
    print(f"  Removed: {len(removed)}")
    print(f"  Common:  {len(common)}")

    if added:
        print("\n  Added rules:")
        for rule_id in sorted(added):
            rule = new_pack.get_rule_by_id(rule_id)
            print(f"    + {rule_id} ({rule.rule_type.value})")

    if removed:
        print("\n  Removed rules:")
        for rule_id in sorted(removed):
            rule = old_pack.get_rule_by_id(rule_id)
            print(f"    - {rule_id} ({rule.rule_type.value})")

    # Check for modified rules
    modified = []
    for rule_id in common:
        old_rule = old_pack.get_rule_by_id(rule_id)
        new_rule = new_pack.get_rule_by_id(rule_id)

        if old_rule.to_dict() != new_rule.to_dict():
            modified.append(rule_id)

    if modified:
        print(f"\n  Modified rules ({len(modified)}):")
        for rule_id in sorted(modified):
            print(f"    ~ {rule_id}")


def cmd_recompile(args):
    """Recompile policy pack if sources changed."""
    print(f"[RECOMPILE] Checking if recompilation needed: {args.pack}")

    import app.llm_client as llm_client

    compiler = PolicyCompiler(llm_client=llm_client, domain="", model=args.model)

    pack = compiler.load_pack(args.pack)

    needs_recompile = compiler.needs_recompilation(pack)

    if needs_recompile:
        print("[RECOMPILE] Source files have changed, recompiling...")
        new_pack = compiler.recompile(args.pack)
        print("[SUCCESS] Recompiled successfully")
        print(f"  Old hash: {pack.source_hash[:16]}...")
        print(f"  New hash: {new_pack.source_hash[:16]}...")
        print(f"  Rules: {len(pack.rules)} -> {len(new_pack.rules)}")
    else:
        print("[SUCCESS] ✓ Policy pack is up to date (no changes)")


def cmd_test(args):
    """Test rule evaluation with sample slots."""
    print(f"[TEST] Loading policy pack: {args.pack}")

    import app.llm_client as llm_client

    compiler = PolicyCompiler(llm_client=llm_client, domain="")

    pack = compiler.load_pack(args.pack)
    engine = RuleEngine(pack)

    # Load test slots
    if args.slots_file:
        slots = json.loads(Path(args.slots_file).read_text())
    else:
        # Example slots for testing
        slots = {
            "kyc_status": "verified",
            "account_status": "active",
            "transaction_age_days": 30,
            "refund_amount_requested": 3000,
            "refunds_last_30_days": 1,
        }
        print("\n[TEST] Using sample slots:")
        print(json.dumps(slots, indent=2))

    print(f"\n{'='*80}")
    print("RULE EVALUATION RESULTS")
    print(f"{'='*80}\n")

    # Test eligibility
    eligible, results = engine.check_eligibility(slots)
    print(f"Eligibility: {'✓ ELIGIBLE' if eligible else '✗ NOT ELIGIBLE'}")
    if results:
        for result in results:
            print(f"  - {result.rule.rule_id}: {result.explanation}")

    # Test approval
    approval_needed, result = engine.check_approval_required(slots)
    print(f"\nApproval needed: {'YES' if approval_needed else 'NO'}")
    if result:
        print(f"  - {result.rule.rule_id}: {result.explanation}")

    # Test risk controls
    risk_results = engine.check_risk_controls(slots)
    print(f"\nRisk controls triggered: {len(risk_results)}")
    for result in risk_results:
        print(f"  - {result.rule.rule_id}: {result.explanation}")

    # Show all matched rules
    all_results = engine.evaluate_all(slots)
    matched = [r for r in all_results if r.matched]

    print(f"\n{'='*80}")
    print(f"All matched rules: {len(matched)}")
    print(f"{'='*80}\n")

    for result in matched:
        print(f"[{result.rule.rule_type.value}] {result.rule.rule_id}")
        print(f"  Priority: {result.rule.priority}")
        print(f"  Matched: {result.explanation}")
        print(f"  Actions: {[a.action_type for a in result.actions]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Policy compilation and management CLI")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile policies from sources")
    compile_parser.add_argument("sources", nargs="+", help="Policy source files")
    compile_parser.add_argument(
        "-o", "--output", required=True, help="Output path for compiled pack"
    )
    compile_parser.add_argument("-d", "--domain", default="general", help="Business domain")
    compile_parser.add_argument("-p", "--policy-id", help="Policy pack ID")
    compile_parser.add_argument("-v", "--version", default="1.0", help="Version")
    compile_parser.add_argument(
        "-m", "--model", default="gpt-4o-mini", help="LLM model for compilation"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate compiled policy pack")
    validate_parser.add_argument("pack", help="Path to policy pack JSON")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect policy pack")
    inspect_parser.add_argument("pack", help="Path to policy pack JSON")
    inspect_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")

    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare two policy packs")
    diff_parser.add_argument("old", help="Old policy pack")
    diff_parser.add_argument("new", help="New policy pack")

    # Recompile command
    recompile_parser = subparsers.add_parser("recompile", help="Recompile if sources changed")
    recompile_parser.add_argument("pack", help="Path to policy pack JSON")
    recompile_parser.add_argument("-m", "--model", default="gpt-4o-mini", help="LLM model")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test rule evaluation")
    test_parser.add_argument("pack", help="Path to policy pack JSON")
    test_parser.add_argument("-s", "--slots-file", help="JSON file with test slots")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    handlers = {
        "compile": cmd_compile,
        "validate": cmd_validate,
        "inspect": cmd_inspect,
        "diff": cmd_diff,
        "recompile": cmd_recompile,
        "test": cmd_test,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args) or 0
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
