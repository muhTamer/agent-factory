# app/dua_v0.py
"""
Domain Understanding Agent (DUA) v0 — Step 1 (user-driven vertical + LLM signals)
- Takes --vertical from the user (Factory UI will pass this)
- Lists files in ./data
- Uses an LLM to guess the likely vertical from filenames (advisory)
- Falls back to heuristics if no LLM is configured/reachable
"""

from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from typing import List, Dict
from app.llm_client import chat_json

# --- Optional: OpenAI-style client (works with OpenAI or Azure OpenAI if envs are set) ---
_OPENAI_ERR = None
try:
    from openai import OpenAI  # pip install openai>=1.40.0
except Exception as e:
    OpenAI = None  # type: ignore
    _OPENAI_ERR = e


def _file_list(root: Path) -> List[Path]:
    """Return a flat list of files directly under `root` (non-recursive for now)."""
    return [p for p in root.iterdir() if p.is_file()]


def _heuristic_counts(files: List[Path]) -> Dict[str, int]:
    """Simple keyword counts by vertical (fallback if LLM not available)."""
    signals = {"fintech": 0, "retail": 0, "telco": 0, "generic_cs": 0}
    names = " ".join([p.name.lower() for p in files])

    fintech_terms = ("kyc", "aml", "iban", "swift", "card", "refund", "chargeback", "bank")
    retail_terms = ("order", "return", "rma", "shipment", "warehouse", "sku")
    telco_terms = ("sim", "plan", "bundle", "roaming", "outage")
    generic_terms = ("faq", "policy", "sop", "onboarding", "complaint", "ticket")

    signals["fintech"] += sum(t in names for t in fintech_terms)
    signals["retail"] += sum(t in names for t in retail_terms)
    signals["telco"] += sum(t in names for t in telco_terms)
    signals["generic_cs"] += sum(t in names for t in generic_terms)
    return signals


# ------------------------------------------------------------
# Helper: LLM-based advisory vertical classification (simplified)
# ------------------------------------------------------------


def detect_signals_llm(filenames: list[str]) -> dict:
    """
    Uses the shared LLM client to classify the customer-service vertical from filenames.
    Returns a structured dict with keys: primary, scores, explanation.
    """
    system = (
        "You classify customer-service verticals from file names only. "
        "Choose one: fintech, retail, telco, or general_service. "
        "Return STRICT JSON with keys: primary, scores, explanation."
    )
    user = "Filenames:\n" + "\n".join(f"- {f}" for f in filenames)
    return chat_json(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model="gpt-5-mini",  # or read from env / args
    )


# ------------------------------------------------------------
# Capability inference (deterministic v0)
# ------------------------------------------------------------


def infer_capabilities(files: List[Path]) -> List[str]:
    """
    Detect which high-level capabilities are implied by the documents.
    v0: filename-based heuristics only (no LLM).
    """
    caps = set()
    names = " ".join([f.name.lower() for f in files])

    # FAQ if there is a CSV or explicit FAQ mention
    if "faq" in names or any(f.suffix.lower() == ".csv" for f in files):
        caps.add("faq")

    # Complaint / refund workflow
    complaint_terms = ("complaint", "dispute", "refund", "chargeback", "issue", "ticket")
    if any(t in names for t in complaint_terms):
        caps.add("complaint")

    # Guardrails / policies present
    policy_terms = ("policy", "policies", "privacy", "tone", "refund")
    if any(t in names for t in policy_terms):
        caps.add("guardrails")

    # Default to faq if nothing else detected (keeps the demo moving)
    if not caps:
        caps.add("faq")

    return sorted(list(caps))


# ------------------------------------------------------------
# Requirements writer + optional schema validation
# ------------------------------------------------------------


def build_requirements(vertical: str, capabilities: List[str], files: List[Path]) -> dict:
    """
    Build a minimal requirements.json payload (v0).
    Entities + workflows are minimal right now; we’ll enrich later.
    """
    # quick entities guess from filenames
    base_entities = {"customer", "ticket", "refund", "account", "card", "order"}
    existing_names = " ".join([f.name.lower() for f in files])
    entities = [e for e in base_entities if e in existing_names] or ["customer"]

    # quick policy presence flags
    policies_present = []
    if "refund" in existing_names:
        policies_present.append("refunds")
    if "privacy" in existing_names:
        policies_present.append("privacy")
    if "tone" in existing_names:
        policies_present.append("tone")

    req = {
        "vertical": vertical,
        "capabilities": capabilities,
        "entities": sorted(list(set(entities))),
        "workflows": [],
        "policies_present": sorted(list(set(policies_present))),
        "gaps": [],
        "constraints": {},
        "kpis": {
            "faq_accuracy_target": 0.90 if "faq" in capabilities else None,
            "guardrail_block_rate_target": 0.99 if "guardrails" in capabilities else None,
        },
    }

    # minimal complaint flow if present
    if "complaint" in capabilities:
        req["workflows"].append(
            {
                "name": "complaint_refund",
                "steps": [
                    "ticket.create",
                    "kyc.verify",
                    "policy.refund_check",
                    "refund.apply",
                    "notify.send",
                ],
            }
        )

    return req


def validate_against_schema(obj: dict, spec_dir: Path) -> None:
    """
    If spec/requirements.schema.json exists and jsonschema is installed, validate.
    """
    try:
        from jsonschema import Draft202012Validator  # optional dependency
    except Exception:
        return  # silently skip if not available

    schema_path = spec_dir / "requirements.schema.json"
    if not schema_path.exists():
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(obj)


def write_requirements(req: dict, out_path: Path, spec_dir: Path) -> None:
    """Validate (if schema present) and write to disk."""
    validate_against_schema(req, spec_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(req, indent=2), encoding="utf-8")


def main() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Domain Understanding Agent (DUA) v0")
    parser.add_argument("--data", type=str, default="./data", help="Path to docs folder")
    parser.add_argument(
        "--vertical",
        type=str,
        required=True,
        help="User-selected vertical (fintech, retail, telco, general_service)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5-mini",
        help="LLM deployment/model for advisory classification",
    )
    parser.add_argument("--spec", type=str, default="./spec", help="Path to spec folder (schemas)")
    parser.add_argument(
        "--out", type=str, default="./data/requirements.json", help="Output JSON path"
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    spec_dir = Path(args.spec)
    out_path = Path(args.out)

    data_dir.mkdir(parents=True, exist_ok=True)
    files = _file_list(data_dir)
    filenames = [f.name for f in files]

    print("\n[DUA] 🧩 Step 2: writing requirements.json")
    for f in files:
        print(f"  - {f.name}")

    # advisory (LLM + heuristic)
    llm_guess = None
    try:
        llm_guess = detect_signals_llm(filenames)
    except Exception as e:
        print(f"[DUA][WARN] LLM classification failed: {e}")
    heuristics = _heuristic_counts(files)
    print(f"[DUA] user-selected vertical: {args.vertical}")
    if llm_guess:
        print(f"[DUA] LLM advisory → primary: {llm_guess.get('primary')}")
    else:
        print("[DUA] LLM unavailable → using heuristic only.")
    print(f"[DUA] heuristic counts: {heuristics}")

    strongest = llm_guess.get("primary") if llm_guess else max(heuristics, key=heuristics.get)
    if files and strongest and args.vertical.lower() != strongest:
        print(
            f"[DUA][ADVISORY] Docs look like '{strongest}', "
            f"but keeping user-selected '{args.vertical}'."
        )

    # NEW: capabilities + requirements write
    caps = infer_capabilities(files)
    req = build_requirements(vertical=args.vertical, capabilities=caps, files=files)
    write_requirements(req, out_path, spec_dir)

    print(f"[DUA] ✅ Wrote {out_path} with capabilities={caps}\n")


if __name__ == "__main__":
    main()
