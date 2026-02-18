# app/runtime/policy/policy_refiner_cli.py
"""
Policy Refinement CLI Extension

Commands for refining customer-provided policies before compilation.

Usage:
    python -m app.runtime.policy.policy_refiner_cli refine customer_policy.yaml
    python -m app.runtime.policy.policy_refiner_cli validate customer_policy.yaml
    python -m app.runtime.policy.policy_refiner_cli batch-refine input_dir/ output_dir/
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.runtime.policy.policy_refiner import (
    PolicyRefiner,
    refine_policy_file,
    validate_customer_policy,
)


def cmd_refine(args):
    """Refine a customer-provided policy."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_refined.yaml"

    print(f"[REFINE] Input: {input_path}")
    print(f"[REFINE] Output: {output_path}")
    print(f"[REFINE] Domain: {args.domain}")
    print()

    # Import LLM client
    import app.llm_client as llm_client

    # Refine
    result = refine_policy_file(
        input_path=input_path,
        output_path=output_path,
        llm_client=llm_client,
        domain=args.domain,
    )

    # Display results
    print("=" * 80)
    print("REFINEMENT RESULTS")
    print("=" * 80)
    print()

    print(f"✅ Refined policy saved to: {output_path}")
    print()

    if result.issues_found:
        print(f"Issues found ({len(result.issues_found)}):")
        for issue in result.issues_found:
            print(f"  - {issue}")
        print()
    else:
        print("✓ No issues found")
        print()

    if result.fixes_applied:
        print(f"Fixes applied ({len(result.fixes_applied)}):")
        for fix in result.fixes_applied:
            print(f"  ✓ {fix}")
        print()

    print(f"Confidence: {result.confidence:.1%}")
    print(f"Needs review: {'YES' if result.needs_review else 'NO'}")
    print()

    if result.needs_review:
        print("⚠️  MANUAL REVIEW RECOMMENDED")
        print("   Please review the refined policy before compilation.")
        print()

    # Save detailed report
    report_path = output_path.parent / f"{output_path.stem}_report.json"
    print(f"📊 Detailed report: {report_path}")
    print()

    return 0 if not result.needs_review or args.allow_review else 1


def cmd_validate(args):
    """Validate a customer policy without refinement."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    print(f"[VALIDATE] Checking: {input_path}")
    print()

    # Import LLM client
    import app.llm_client as llm_client

    # Read policy
    policy_yaml = input_path.read_text(encoding="utf-8")

    # Validate
    is_valid, errors, warnings = validate_customer_policy(
        policy_yaml=policy_yaml,
        llm_client=llm_client,
        domain=args.domain,
    )

    # Display results
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    if is_valid:
        print("✅ VALID - Policy can be compiled as-is")
        print()
    else:
        print("❌ INVALID - Policy needs refinement")
        print()

    if errors:
        print(f"Errors ({len(errors)}):")
        for error in errors:
            print(f"  ❌ {error}")
        print()

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
        print()

    if not is_valid:
        print("💡 Suggestion: Run 'refine' command to fix issues automatically")
        print(f"   python -m app.runtime.policy.policy_refiner_cli refine {input_path}")
        print()

    return 0 if is_valid else 1


def cmd_batch_refine(args):
    """Refine multiple policies in batch."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    # Find all YAML files
    policy_files = list(input_dir.glob("*.yaml")) + list(input_dir.glob("*.yml"))

    if not policy_files:
        print(f"❌ No YAML files found in {input_dir}")
        return 1

    print(f"[BATCH] Found {len(policy_files)} policy files")
    print(f"[BATCH] Input: {input_dir}")
    print(f"[BATCH] Output: {output_dir}")
    print(f"[BATCH] Domain: {args.domain}")
    print()

    # Import LLM client
    import app.llm_client as llm_client

    # Create refiner
    refiner = PolicyRefiner(llm_client, model=args.model)

    # Refine all
    results = refiner.batch_refine(
        policy_files=policy_files,
        output_dir=output_dir,
        domain=args.domain,
    )

    # Summary
    print()
    print("=" * 80)
    print("BATCH REFINEMENT SUMMARY")
    print("=" * 80)
    print()

    needs_review_count = sum(1 for _, r in results if r.needs_review)
    high_confidence_count = sum(1 for _, r in results if r.confidence >= 0.9)

    print(f"Total processed: {len(results)}")
    print(f"High confidence (≥90%): {high_confidence_count}")
    print(f"Needs review: {needs_review_count}")
    print()

    # Details
    for policy_file, result in results:
        status = "✓" if result.confidence >= 0.9 else "⚠️"
        review = " [REVIEW]" if result.needs_review else ""
        print(f"{status} {policy_file.name}: {result.confidence:.1%}{review}")

    print()
    print(f"📁 Refined policies saved to: {output_dir}")
    print()

    if needs_review_count > 0:
        print(f"⚠️  {needs_review_count} policies need manual review")
        return 1

    return 0


def cmd_compare(args):
    """Compare original and refined policies."""
    original_path = Path(args.original)
    refined_path = Path(args.refined)

    if not original_path.exists():
        print(f"❌ Original file not found: {original_path}")
        return 1

    if not refined_path.exists():
        print(f"❌ Refined file not found: {refined_path}")
        return 1

    original = original_path.read_text(encoding="utf-8")
    refined = refined_path.read_text(encoding="utf-8")

    print("=" * 80)
    print("POLICY COMPARISON")
    print("=" * 80)
    print()

    print(f"Original: {original_path}")
    print(f"Refined:  {refined_path}")
    print()

    # Basic stats
    orig_lines = len(original.split("\n"))
    refined_lines = len(refined.split("\n"))

    print(f"Original lines: {orig_lines}")
    print(f"Refined lines:  {refined_lines}")
    print(f"Difference:     {refined_lines - orig_lines:+d}")
    print()

    # Try to load report
    report_path = refined_path.parent / f"{refined_path.stem}_report.json"

    if report_path.exists():
        report = json.loads(report_path.read_text())

        print("Changes made:")
        for fix in report.get("fixes_applied", []):
            print(f"  ✓ {fix}")
        print()

        print(f"Confidence: {report.get('confidence', 0):.1%}")
        print()

    # Show side-by-side (first 20 lines)
    if args.verbose:
        print("=" * 80)
        print("SIDE-BY-SIDE PREVIEW (first 20 lines)")
        print("=" * 80)
        print()

        orig_preview = original.split("\n")[:20]
        refined_preview = refined.split("\n")[:20]

        max_len = max(len(orig_preview), len(refined_preview))

        for i in range(max_len):
            orig_line = orig_preview[i] if i < len(orig_preview) else ""
            refined_line = refined_preview[i] if i < len(refined_preview) else ""

            marker = "│" if orig_line == refined_line else "┊"
            print(f"{orig_line[:38]:38s} {marker} {refined_line}")

        print()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Policy refinement tools for customer-provided policies"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Refine command
    refine_parser = subparsers.add_parser(
        "refine", help="Refine a customer policy to make it compilable"
    )
    refine_parser.add_argument("input", help="Input policy file")
    refine_parser.add_argument("-o", "--output", help="Output path (default: input_refined.yaml)")
    refine_parser.add_argument("-d", "--domain", default="general", help="Business domain")
    refine_parser.add_argument(
        "--allow-review", action="store_true", help="Exit 0 even if review needed"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a customer policy without refinement"
    )
    validate_parser.add_argument("input", help="Input policy file")
    validate_parser.add_argument("-d", "--domain", default="general", help="Business domain")

    # Batch refine command
    batch_parser = subparsers.add_parser("batch-refine", help="Refine multiple policies in batch")
    batch_parser.add_argument("input_dir", help="Directory with policy files")
    batch_parser.add_argument("output_dir", help="Output directory")
    batch_parser.add_argument("-d", "--domain", default="general", help="Business domain")
    batch_parser.add_argument("-m", "--model", default="gpt-4o", help="LLM model")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare original and refined policies")
    compare_parser.add_argument("original", help="Original policy file")
    compare_parser.add_argument("refined", help="Refined policy file")
    compare_parser.add_argument("-v", "--verbose", action="store_true", help="Show side-by-side")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    handlers = {
        "refine": cmd_refine,
        "validate": cmd_validate,
        "batch-refine": cmd_batch_refine,
        "compare": cmd_compare,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args) or 0
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
