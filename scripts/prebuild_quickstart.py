#!/usr/bin/env python3
"""
Pre-build quickstart artifacts for fintech and retail verticals.

Runs the full pipeline (infer → spec → generate agents) once and saves all
artifacts to data/prebuilt/{fintech,retail}/.  Subsequent quickstarts copy
these pre-built files instead of running the LLM pipeline — making them
near-instant.

Usage:
    python -m scripts.prebuild_quickstart            # both verticals
    python -m scripts.prebuild_quickstart fintech     # fintech only
    python -m scripts.prebuild_quickstart retail      # retail only

Requires LLM credentials (AZURE_OPENAI_ENDPOINT, etc.) to be set.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
PREBUILT_DIR = DATA_DIR / "prebuilt"

VERTICALS = {
    "fintech": {
        "files": [
            DATA_DIR / "BankFAQs.csv",
            DATA_DIR / "refunds_policy.yaml",
            DATA_DIR / "complaints_policy.yaml",
            DATA_DIR / "accounts_policy.yaml",
        ],
        "doc_visibility": {
            "refunds_policy.yaml": "internal",
            "complaints_policy.yaml": "internal",
            "accounts_policy.yaml": "internal",
            "BankFAQs.csv": "customer_facing",
        },
    },
    "retail": {
        "files": [
            DATA_DIR / "RetailFAQs.csv",
            DATA_DIR / "retail_refunds_policy.yaml",
            DATA_DIR / "retail_complaints_policy.yaml",
        ],
        "doc_visibility": {
            "retail_refunds_policy.yaml": "internal",
            "retail_complaints_policy.yaml": "internal",
            "RetailFAQs.csv": "customer_facing",
        },
    },
}


def prebuild(vertical: str) -> None:
    cfg = VERTICALS[vertical]
    print(f"\n{'='*60}")
    print(f"Pre-building {vertical} quickstart artifacts")
    print(f"{'='*60}")
    t0 = time.time()

    # Temporary workspace
    workspace = REPO_ROOT / ".workspace" / f"_prebuild_{vertical}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    # Copy data files
    for src in cfg["files"]:
        if not src.exists():
            print(f"[ERROR] Missing preset file: {src}")
            return
        shutil.copy2(src, workspace / src.name)
        print(f"  Copied {src.name}")

    # Run infer pipeline with pre-classified docs
    from app.infer_capabilities import InferCapabilities
    from app.concierge.concierge_agent import ConciergeAgent
    import app.llm_client as llm_client

    # Build pre-classified doc metadata (skips LLM classification)
    ic = InferCapabilities()
    pre_docs = []
    for src in cfg["files"]:
        p = workspace / src.name
        content_analysis = ic._analyze_document_content(p)
        ext = p.suffix.lower()
        if ext == ".csv":
            doc_type = "faq_kb"
        elif ext in (".yaml", ".yml"):
            doc_type = "policy"
        else:
            doc_type = "other"
        pre_docs.append(
            {
                "name": p.name,
                "path": str(p),
                "doc_type": doc_type,
                "confidence": 0.99,
                "reason": "Quickstart preset file",
                "vertical_fit": 0.95,
                "vertical_guess": vertical,
                "vertical_fit_reason": f"Preset data for {vertical} vertical",
                "off_vertical": False,
                "content_categories": content_analysis.get("content_categories", []),
                "content_topics": content_analysis.get("content_topics", []),
            }
        )

    # Run concierge agent: infer + deploy
    agent = ConciergeAgent(
        vertical=vertical,
        data_dir=str(workspace),
        llm_client=llm_client,
    )

    print("\n[1/3] Running inference pipeline (LLM agent proposal)...")
    result = agent.handle_event(
        {
            "type": "upload_docs",
            "use_llm": True,
            "pre_classified_docs": pre_docs,
        }
    )
    print(f"  Plan: {len(result.get('plan', {}).get('agents', []))} agents proposed")

    print("\n[2/3] Building factory spec + generating agents (LLM blueprints)...")
    deploy_result = agent.handle_event(
        {
            "type": "user_action",
            "action": "approve_deploy_dry",
            "doc_visibility": cfg["doc_visibility"],
        }
    )
    print(f"  Deploy: {deploy_result.get('text', '')}")

    # Collect artifacts
    factory_dir = workspace / ".factory"
    spec_path = factory_dir / "factory_spec.json"
    if not spec_path.exists():
        print(f"[ERROR] No factory_spec.json at {spec_path}")
        return

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    # Save to prebuilt dir
    prebuilt = PREBUILT_DIR / vertical
    if prebuilt.exists():
        shutil.rmtree(prebuilt)
    prebuilt.mkdir(parents=True)

    # Make paths in the spec relative (they'll be rewritten at quickstart time)
    _relativize_spec_paths(spec, workspace)

    # Save factory_spec.json
    (prebuilt / "factory_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n  Saved factory_spec.json")

    # Save doc metadata
    doc_meta_path = factory_dir / ".doc_metadata.json"
    if doc_meta_path.exists():
        shutil.copy2(doc_meta_path, prebuilt / ".doc_metadata.json")
        print("  Saved .doc_metadata.json")

    # Save compiled policies
    compiled_dir = factory_dir / "compiled_policies"
    if compiled_dir.exists():
        dest_compiled = prebuilt / "compiled_policies"
        shutil.copytree(compiled_dir, dest_compiled)
        print("  Saved compiled_policies/")

    # Save generated agents
    print("\n[3/3] Capturing generated agent artifacts...")
    gen_root = REPO_ROOT / "generated"
    prebuilt_gen = prebuilt / "generated"
    prebuilt_gen.mkdir(exist_ok=True)

    for agent_spec in spec.get("agents", []):
        if agent_spec.get("type") != "autogen":
            continue
        a_id = agent_spec["id"]
        src_dir = gen_root / a_id
        if src_dir.exists():
            dest_dir = prebuilt_gen / a_id
            shutil.copytree(src_dir, dest_dir)
            files = [f.name for f in dest_dir.iterdir()]
            print(f"  Saved generated/{a_id}/ ({', '.join(files)})")
        else:
            print(f"  [WARN] No generated dir for {a_id}")

    # Save session template
    session_data = {
        "vertical": vertical,
        "deployed_at": "__PLACEHOLDER__",
        "agents": [
            a.get("id")
            for a in spec.get("agents", [])
            if isinstance(a, dict) and a.get("id")
        ],
        "spec_path": "__PLACEHOLDER__",
        "deploy_text": deploy_result.get("text", ""),
        "deployment_request": {
            "vertical": vertical,
            "mode": "dry",
            "agents": [
                a.get("id")
                for a in spec.get("agents", [])
                if isinstance(a, dict) and a.get("id")
            ],
        },
    }
    (prebuilt / "session.json").write_text(
        json.dumps(session_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("  Saved session.json")

    # Cleanup temp workspace
    shutil.rmtree(workspace, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"\n[DONE] {vertical} prebuilt in {elapsed:.1f}s")
    print(f"  Artifacts: {prebuilt}")


def _relativize_spec_paths(spec: dict, workspace: Path) -> None:
    """Replace absolute workspace paths with placeholders for portability."""
    ws_str = str(workspace)
    for agent in spec.get("agents", []):
        inputs = agent.get("inputs") or {}
        for key in ("docs", "policies", "knowledge_sources"):
            paths = inputs.get(key)
            if not isinstance(paths, list):
                continue
            inputs[key] = [
                p.replace(ws_str, "__WORKSPACE__") if isinstance(p, str) else p
                for p in paths
            ]
        # Relativize base_dir in paths
        paths_block = spec.get("paths", {})
        if isinstance(paths_block.get("base_dir"), str):
            paths_block["base_dir"] = "__WORKSPACE__"


if __name__ == "__main__":
    targets = sys.argv[1:] or ["fintech", "retail"]
    for v in targets:
        if v not in VERTICALS:
            print(f"Unknown vertical: {v}. Choose from: {list(VERTICALS.keys())}")
            sys.exit(1)
        prebuild(v)
    print("\nAll done! Commit data/prebuilt/ to the repo.")
