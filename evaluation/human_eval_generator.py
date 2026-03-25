# evaluation/human_eval_generator.py
"""
Generate a self-contained HTML human evaluation survey for RQ2.

Runs a stratified sample of scenarios through the system, captures
explanations at all 3 levels, and produces an HTML file that evaluators
can open in any browser. Ratings are stored in localStorage and can be
exported as JSON with one click.

Usage:
    python -m evaluation.human_eval_generator
    python -m evaluation.human_eval_generator --sample 15
    python -m evaluation.human_eval_generator --output evaluation/results/rq2/human_eval.html
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Force UTF-8 stdout/stderr on Windows ─────────────────────────────
if sys.platform == "win32":
    for _stream in ("stdout", "stderr"):
        _s = getattr(sys, _stream, None)
        if _s and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluation.human_eval")

SCENARIOS_PATH = ROOT / "evaluation" / "scenarios" / "ground_truth.json"


def _stratified_sample(scenarios: List[Dict], n: int) -> List[Dict]:
    """Select a stratified sample ensuring each category is represented."""
    by_cat: Dict[str, List[Dict]] = {}
    for sc in scenarios:
        cat = sc["category"]
        by_cat.setdefault(cat, []).append(sc)

    # At least 1 per category, rest distributed proportionally
    selected = []
    cats = sorted(by_cat.keys())
    per_cat = max(1, n // len(cats))
    remainder = n - per_cat * len(cats)

    for cat in cats:
        pool = by_cat[cat]
        k = min(per_cat, len(pool))
        selected.extend(random.sample(pool, k))

    # Fill remainder from largest categories
    remaining = [s for s in scenarios if s not in selected]
    if remainder > 0 and remaining:
        selected.extend(random.sample(remaining, min(remainder, len(remaining))))

    return selected[:n]


def _collect_explanations(sample: List[Dict]) -> List[Dict[str, Any]]:
    """Run scenarios and capture explanations."""
    import tempfile

    from app.governance.explainability import ExplainabilityEngine
    from app.runtime.trace import Trace
    from evaluation.run_evaluation import build_eval_spine

    tmp_dir = Path(tempfile.mkdtemp(prefix="human_eval_"))
    spine = build_eval_spine(tmp_dir)
    explainer = ExplainabilityEngine()

    items = []
    for i, sc in enumerate(sample, 1):
        sc_id = sc["id"]
        query = sc["turns"][0]["query"]
        logger.info("[%d/%d] Running %s: %s", i, len(sample), sc_id, query[:60])

        try:
            resp = spine.handle_chat(
                query,
                request_id=f"human_eval_{sc_id}",
                context={"thread_id": f"human_eval_{sc_id}"},
            )

            # Build trace
            trace = Trace.start(query=query, request_id=f"human_eval_{sc_id}")
            trace.add("request_received")
            if resp.get("orchestration_pattern"):
                trace.add(
                    "orchestration_pattern", pattern=resp["orchestration_pattern"]
                )
            if resp.get("agent_id"):
                trace.add("execute", agent_id=resp["agent_id"])
                trace.add(
                    "select",
                    selected_agent=resp["agent_id"],
                    score=resp.get("score", 0),
                )
            if resp.get("delegated_agents"):
                trace.add("aop_delegation", delegated_agents=resp["delegated_agents"])
            if resp.get("subtask_results"):
                trace.add(
                    "aop_execute",
                    results=[
                        {
                            "subtask": st.get("subtask"),
                            "agent": st.get("agent_id"),
                            "success": st.get("success"),
                        }
                        for st in resp["subtask_results"]
                    ],
                )
            trace.add("guard_post_ok")

            # Generate explanations
            explanations = explainer.generate_all_levels(trace, resp)
            expl_dicts = {k: v.to_dict() for k, v in explanations.items()}

            # Extract answer text
            answer = resp.get("answer") or resp.get("text") or ""

            items.append(
                {
                    "id": sc_id,
                    "category": sc["category"],
                    "description": sc.get("description", ""),
                    "query": query,
                    "answer": (
                        answer[:1000] if isinstance(answer, str) else str(answer)[:1000]
                    ),
                    "agent_id": resp.get("agent_id", ""),
                    "delegated_agents": resp.get("delegated_agents", []),
                    "explanations": {
                        level: {
                            "narrative": exp.get("narrative", ""),
                            "agents_involved": exp.get("agents_involved", []),
                            "decisions": exp.get("decisions", []),
                            "provenance": exp.get("provenance", []),
                        }
                        for level, exp in expl_dicts.items()
                    },
                    "trace_events": [
                        {"event": e.stage, "data": e.data} for e in trace.events
                    ],
                }
            )

        except Exception as e:
            logger.error("Failed %s: %s", sc_id, e)
            items.append(
                {
                    "id": sc_id,
                    "category": sc["category"],
                    "description": sc.get("description", ""),
                    "query": query,
                    "answer": "",
                    "agent_id": "",
                    "delegated_agents": [],
                    "explanations": {},
                    "trace_events": [],
                    "error": str(e),
                }
            )

        time.sleep(1.0)

    return items


def _generate_html(items: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate a self-contained HTML evaluation survey with onboarding wizard."""
    items_json = json.dumps(items, ensure_ascii=False, indent=2, default=str)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RQ2 Human Evaluation — Explanation Quality</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
  .container {{ max-width: 920px; margin: 0 auto; padding: 20px; }}
  h1 {{ text-align: center; margin: 20px 0 8px; color: #1a1a2e; }}
  .subtitle {{ text-align: center; color: #666; font-size: 14px; margin-bottom: 20px; }}

  /* ── Wizard styles ── */
  .wizard-overlay {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; display: flex; align-items: center; justify-content: center; }}
  .wizard {{ background: white; border-radius: 12px; max-width: 700px; width: 90%; max-height: 85vh; overflow-y: auto; padding: 32px; box-shadow: 0 8px 32px rgba(0,0,0,0.2); }}
  .wizard h2 {{ color: #1a1a2e; margin-bottom: 16px; }}
  .wizard h3 {{ color: #2d6a4f; margin: 16px 0 8px; font-size: 17px; }}
  .wizard p {{ margin: 8px 0; }}
  .wizard .example-box {{ background: #f8f9fa; border-left: 4px solid #2d6a4f; padding: 12px 16px; margin: 12px 0; border-radius: 0 6px 6px 0; font-size: 14px; }}
  .wizard .bad-example {{ border-left-color: #d63031; background: #fff5f5; }}
  .wizard .score-example {{ display: flex; gap: 8px; margin: 4px 0; align-items: center; }}
  .wizard .score-badge {{ width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; color: white; flex-shrink: 0; }}
  .wizard .s1 {{ background: #d63031; }}
  .wizard .s2 {{ background: #e17055; }}
  .wizard .s3 {{ background: #fdcb6e; color: #333; }}
  .wizard .s4 {{ background: #00b894; }}
  .wizard .s5 {{ background: #2d6a4f; }}
  .wizard .step-dots {{ display: flex; gap: 8px; justify-content: center; margin: 20px 0 12px; }}
  .wizard .dot {{ width: 10px; height: 10px; border-radius: 50%; background: #ddd; }}
  .wizard .dot.active {{ background: #2d6a4f; }}
  .wizard .dot.done {{ background: #00b894; }}
  .wizard-nav {{ display: flex; justify-content: space-between; margin-top: 20px; }}
  .wizard-nav button {{ padding: 10px 28px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }}
  .wizard-nav .wprev {{ background: #eee; color: #333; }}
  .wizard-nav .wnext {{ background: #2d6a4f; color: white; }}
  .wizard-nav button:disabled {{ opacity: 0.3; }}

  /* ── Main eval styles ── */
  .hidden {{ display: none; }}
  .progress {{ text-align: center; margin: 10px 0 20px; font-size: 14px; color: #666; }}
  .progress-bar {{ width: 100%; height: 6px; background: #ddd; border-radius: 3px; margin: 8px 0; }}
  .progress-fill {{ height: 100%; background: #2d6a4f; border-radius: 3px; transition: width 0.3s; }}
  .card {{ background: white; border-radius: 8px; padding: 24px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card h2 {{ color: #1a1a2e; margin-bottom: 8px; font-size: 18px; }}
  .meta {{ font-size: 13px; color: #888; margin-bottom: 16px; }}
  .section {{ margin: 16px 0; }}
  .section h3 {{ font-size: 15px; color: #555; margin-bottom: 6px; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  .query {{ background: #e8f4f8; padding: 12px; border-radius: 6px; font-style: italic; }}
  .answer {{ background: #f0f7f0; padding: 12px; border-radius: 6px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }}
  .explanation {{ background: #fff9e6; padding: 12px; border-radius: 6px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }}
  .trace {{ background: #f5f0ff; padding: 12px; border-radius: 6px; font-size: 12px; font-family: monospace; white-space: pre-wrap; max-height: 150px; overflow-y: auto; }}
  .rating-group {{ margin: 12px 0; padding: 14px; background: #fafafa; border-radius: 6px; border: 1px solid #eee; }}
  .rating-group label {{ display: block; font-weight: 600; margin-bottom: 4px; font-size: 14px; }}
  .rating-group .hint {{ font-size: 12px; color: #888; margin-bottom: 8px; }}
  .stars {{ display: flex; gap: 4px; }}
  .stars button {{ width: 40px; height: 40px; border: 2px solid #ddd; border-radius: 6px; background: white; cursor: pointer; font-size: 16px; font-weight: bold; transition: all 0.15s; }}
  .stars button:hover {{ border-color: #2d6a4f; background: #e8f4e8; }}
  .stars button.active {{ border-color: #2d6a4f; background: #2d6a4f; color: white; }}
  .star-labels {{ display: flex; justify-content: space-between; font-size: 11px; color: #999; margin-top: 4px; width: 216px; }}
  .level-tabs {{ display: flex; gap: 8px; margin: 12px 0; }}
  .level-tabs button {{ padding: 8px 18px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer; font-size: 13px; }}
  .level-tabs button.active {{ background: #2d6a4f; color: white; border-color: #2d6a4f; }}
  .level-desc {{ font-size: 12px; color: #666; font-style: italic; margin-bottom: 8px; }}
  .nav {{ display: flex; justify-content: space-between; margin: 20px 0; }}
  .nav button {{ padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }}
  .nav .prev {{ background: #ddd; color: #333; }}
  .nav .next {{ background: #2d6a4f; color: white; }}
  .nav button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .export-bar {{ text-align: center; margin: 20px 0; }}
  .export-bar button {{ padding: 12px 32px; background: #1a1a2e; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }}
  .evaluator-info input {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; width: 300px; }}
  .help-toggle {{ position: fixed; bottom: 20px; right: 20px; width: 44px; height: 44px; border-radius: 50%; background: #2d6a4f; color: white; border: none; font-size: 20px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.2); z-index: 100; }}
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- ONBOARDING WIZARD                                          -->
<!-- ═══════════════════════════════════════════════════════════ -->
<div class="wizard-overlay" id="wizardOverlay">
<div class="wizard">
  <div class="step-dots" id="stepDots"></div>
  <div id="wizardContent"></div>
  <div class="wizard-nav">
    <button class="wprev" id="wPrev" onclick="wizardNav(-1)">Back</button>
    <button class="wnext" id="wNext" onclick="wizardNav(1)">Next</button>
  </div>
</div>
</div>

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- MAIN EVALUATION (hidden until wizard completes)            -->
<!-- ═══════════════════════════════════════════════════════════ -->
<div class="container hidden" id="mainApp">
  <h1>RQ2 Explanation Quality Evaluation</h1>
  <p class="subtitle">Rate how well the AI system explains its decisions</p>

  <div class="card evaluator-info">
    <label><strong>Your Name:</strong></label>
    <input type="text" id="evaluatorName" placeholder="Enter your name" style="margin-top:8px;">
  </div>

  <div class="progress">
    <span id="progressText"></span>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
  </div>

  <div id="scenarioCard" class="card"></div>

  <div class="nav">
    <button class="prev" id="prevBtn" onclick="navigate(-1)">&#8592; Previous</button>
    <button class="next" id="nextBtn" onclick="navigate(1)">Next &#8594;</button>
  </div>

  <div class="export-bar">
    <button onclick="exportResults()">Export Results as JSON</button>
    <span id="exportStatus" style="margin-left:12px;color:#2d6a4f;"></span>
  </div>
</div>

<button class="help-toggle" onclick="showWizard()" title="Review guide">?</button>

<script>
// ── DATA ──
const ITEMS = {items_json};

// ── WIZARD STEPS ──
const WIZARD_STEPS = [
  {{
    title: "Welcome!",
    html: `
      <h2>Welcome to the Explanation Quality Evaluation</h2>
      <p>Thank you for helping evaluate how well an AI system explains its decisions to users.</p>
      <p>This takes about <strong>15-20 minutes</strong>. You'll review ${{ITEMS.length}} scenarios and rate the explanations the system generates.</p>
      <p style="margin-top:16px;"><strong>What you'll see for each scenario:</strong></p>
      <ul style="margin:8px 0 0 20px;">
        <li>A customer's question to the system</li>
        <li>The system's answer</li>
        <li>What actually happened behind the scenes (the "execution trace")</li>
        <li>The system's explanation of what it did and why</li>
      </ul>
      <p style="margin-top:16px;">You'll rate each explanation on <strong>3 quality dimensions</strong> derived from IEEE standards for AI transparency. The next steps will introduce each dimension.</p>
    `
  }},
  {{
    title: "Background: IEEE AI Transparency Standards",
    html: `
      <h2>Background: IEEE Standards for AI Transparency</h2>
      <p>International standards bodies have defined requirements for how AI systems should explain themselves. The three dimensions you'll rate come from two IEEE standards:</p>

      <h3>IEEE 2894-2024 — Guide for AI Explainability</h3>
      <div class="example-box">
        This standard defines requirements for AI systems to provide explanations that are:
        <ul style="margin:8px 0 0 16px;">
          <li><strong>Traceable</strong> (R7) — explanations should link back to actual processing steps</li>
          <li><strong>User-appropriate</strong> (R2) — non-technical summaries for end users</li>
          <li><strong>Auditor-appropriate</strong> (R3) — detailed explanations for compliance review</li>
          <li><strong>Grounded in provenance</strong> (R4) — citing data sources used</li>
          <li><strong>Including decision rationale</strong> (R5) — explaining why a particular decision was made</li>
        </ul>
      </div>

      <h3>IEEE 3152-2024 — Standard for Transparent AI Agent Interactions</h3>
      <div class="example-box">
        This standard requires AI agents to:
        <ul style="margin:8px 0 0 16px;">
          <li><strong>Disclose AI nature</strong> (R1) — make clear the response is AI-generated</li>
          <li><strong>Disclose identity</strong> (R2) — indicate which AI agent(s) handled the request</li>
        </ul>
      </div>

      <p style="margin-top:16px;">These are the principles you'll use to judge explanation quality. You do <strong>not</strong> need to memorise them — the rating form includes reminders.</p>
    `
  }},
  {{
    title: "Explanation Levels",
    html: `
      <h2>Three Levels of Explanation</h2>
      <p>IEEE 2894 recognises that different audiences need different levels of detail. The system produces three explanation levels:</p>

      <h3>Summary (IEEE 2894-R2: User-Appropriate)</h3>
      <div class="example-box">
        Intended for <strong>end users / customers</strong>. Should use plain language, explain what happened and what it means for them, without technical jargon.
      </div>

      <h3>Detailed (IEEE 2894-R3: Auditor-Appropriate)</h3>
      <div class="example-box">
        Intended for <strong>compliance officers / auditors</strong>. Should include decision points, which rules or checks were applied, and whether standards were met.
      </div>

      <h3>Full (Developer Trace)</h3>
      <div class="example-box">
        Intended for <strong>developers / technical staff</strong>. Should provide event-level detail — agent identifiers, processing steps, timing, and scores.
      </div>

      <p style="margin-top:12px;">You'll rate each level separately using tabs. Keep the intended audience in mind when scoring.</p>
    `
  }},
  {{
    title: "Dimension 1: Faithfulness",
    html: `
      <h2>Dimension 1: Faithfulness</h2>
      <p style="color:#555;">Based on <strong>IEEE 2894-2024, Requirement R7 (Traceability)</strong>:</p>
      <div class="example-box">
        "Explanations shall be traceable to specific processing steps performed by the AI system."
      </div>

      <p style="margin-top:16px;"><strong>Core question:</strong> Can each claim in the explanation be verified against what actually happened?</p>
      <p>You'll see an execution trace showing the actual steps the system took. Compare the explanation against this trace.</p>

      <h3>Example of high faithfulness:</h3>
      <div class="example-box">
        "Your query was classified as a refund request and routed to a specialist agent, which consulted the refunds policy to determine eligibility."
      </div>
      <p style="font-size:13px;color:#666;">If the trace confirms: classification occurred, refund agent was selected, and policy was consulted — this is faithful.</p>

      <h3>Example of low faithfulness:</h3>
      <div class="example-box bad-example">
        "Multiple experts reviewed your case across several departments before reaching a consensus."
      </div>
      <p style="font-size:13px;color:#666;">If the trace shows a single agent handled it — this explanation fabricates a process that didn't happen.</p>

      <h3>Scoring:</h3>
      <div class="score-example"><span class="score-badge s1">1</span> Fabricated or contradicts what happened</div>
      <div class="score-example"><span class="score-badge s2">2</span> Vague references with no grounding in actual steps</div>
      <div class="score-example"><span class="score-badge s3">3</span> Mentions key decisions but doesn't link to specific steps</div>
      <div class="score-example"><span class="score-badge s4">4</span> Most claims traceable, minor gaps</div>
      <div class="score-example"><span class="score-badge s5">5</span> Every claim verifiable in the trace</div>
    `
  }},
  {{
    title: "Dimension 2: Completeness",
    html: `
      <h2>Dimension 2: Completeness</h2>
      <p style="color:#555;">Based on <strong>IEEE 2894-R4/R5</strong> (provenance, decision rationale) and <strong>IEEE 3152-R1/R2</strong> (AI disclosure, agent identity).</p>
      <div class="example-box">
        A complete explanation, per these standards, should address four things:
        <ol style="margin:8px 0 0 16px;">
          <li><strong>Data provenance</strong> (2894-R4) — what data sources were consulted</li>
          <li><strong>Decision rationale</strong> (2894-R5) — why this path was taken</li>
          <li><strong>AI nature disclosure</strong> (3152-R1) — is it clear this is AI-generated</li>
          <li><strong>Agent identity</strong> (3152-R2) — which AI component(s) handled the request</li>
        </ol>
      </div>

      <p style="margin-top:16px;"><strong>Core question:</strong> Does the explanation provide enough information for the reader to understand and trust the response?</p>

      <h3>Check for each element:</h3>
      <div class="example-box">
        <strong>Provenance:</strong> "Answer sourced from [specific database/file]"<br>
        <strong>Rationale:</strong> "Classified as [type], routed to [agent] because..."<br>
        <strong>AI disclosure:</strong> "This response was generated by an AI system"<br>
        <strong>Agent identity:</strong> "Handled by [specific agent name/type]"
      </div>

      <h3>Scoring:</h3>
      <div class="score-example"><span class="score-badge s1">1</span> None of the 4 elements present</div>
      <div class="score-example"><span class="score-badge s2">2</span> Missing provenance and identity; only superficial rationale</div>
      <div class="score-example"><span class="score-badge s3">3</span> Has rationale but lacks provenance, or vice versa; partial disclosure</div>
      <div class="score-example"><span class="score-badge s4">4</span> Covers most elements, minor omissions</div>
      <div class="score-example"><span class="score-badge s5">5</span> All 4 elements clearly present</div>
    `
  }},
  {{
    title: "Dimension 3: Clarity",
    html: `
      <h2>Dimension 3: Clarity</h2>
      <p style="color:#555;">Based on <strong>IEEE 2894-R2</strong> (user-appropriate explanations) and <strong>IEEE 2894-R3</strong> (auditor-appropriate explanations).</p>
      <div class="example-box">
        IEEE 2894 recognises that a single explanation style does not serve all stakeholders. Explanations should be calibrated to the knowledge level and needs of the intended reader.
      </div>

      <p style="margin-top:16px;"><strong>Core question:</strong> Is the depth and language appropriate for the intended audience?</p>

      <h3>What "appropriate" means per level:</h3>

      <p><strong>Summary (for customers):</strong></p>
      <div class="example-box">
        <strong>Appropriate:</strong> "I found your answer by searching our FAQ database. The information comes from our account transfer policies."<br><br>
        <strong>Not appropriate:</strong> "TF-IDF vector similarity score 0.94 against corpus entry #847, routed via LLM intent classifier with confidence threshold 0.7"
      </div>

      <p><strong>Detailed (for compliance officers):</strong></p>
      <div class="example-box">
        <strong>Appropriate:</strong> Structured decision points, governance checks performed, data sources referenced, agent selection rationale.<br><br>
        <strong>Not appropriate:</strong> "We helped you with your question!" (too simple to audit)
      </div>

      <p><strong>Full (for developers):</strong></p>
      <div class="example-box">
        <strong>Appropriate:</strong> Event-level processing steps with identifiers, timing, parameters, and outcomes.<br><br>
        <strong>Not appropriate:</strong> High-level prose that omits technical specifics.
      </div>

      <h3>Scoring:</h3>
      <div class="score-example"><span class="score-badge s1">1</span> Completely wrong audience (e.g., raw technical detail in a customer summary)</div>
      <div class="score-example"><span class="score-badge s3">3</span> Right information but wrong framing for the audience</div>
      <div class="score-example"><span class="score-badge s5">5</span> Perfectly calibrated — the intended reader would understand immediately</div>
    `
  }},
  {{
    title: "Practice example",
    html: `
      <h2>Quick Practice</h2>
      <p>Here's a worked example. Try scoring it mentally first, then check below.</p>

      <p><strong>Customer query:</strong> "Can I transfer my account to another branch?"</p>
      <p><strong>Execution trace:</strong> request received &#8594; intent classified &#8594; agent selected &#8594; knowledge base searched &#8594; answer returned</p>

      <h3>Summary-level explanation to rate:</h3>
      <div class="example-box">
        "I found the answer to your question by searching our knowledge base. A specialist agent handled your request."
      </div>

      <p style="margin-top:16px;"><strong>Suggested scoring (yours may differ):</strong></p>
      <ul style="margin:8px 0 0 20px;">
        <li><strong>Faithfulness: 4</strong> — claims are traceable (agent selected, knowledge searched) but doesn't mention which specific source was consulted</li>
        <li><strong>Completeness: 3</strong> — mentions agent identity and partial rationale, but no specific data source provenance, no explicit AI disclosure</li>
        <li><strong>Clarity: 4</strong> — plain language appropriate for a customer, could be slightly more informative</li>
      </ul>
      <p style="margin-top:16px;color:#666;font-size:13px;">There are no "right" answers — reasonable people may differ by 1 point. What matters is <strong>consistent</strong> application of the criteria across scenarios.</p>
    `
  }},
  {{
    title: "Ready!",
    html: `
      <h2>You're Ready!</h2>
      <p>Quick recap of the three dimensions:</p>
      <table style="width:100%;border-collapse:collapse;margin:12px 0;">
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:8px;font-weight:bold;">Faithfulness</td>
          <td style="padding:8px;">Can each claim be verified against what actually happened? (IEEE 2894-R7)</td>
        </tr>
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:8px;font-weight:bold;">Completeness</td>
          <td style="padding:8px;">Are data sources, rationale, AI disclosure, and agent identity present? (IEEE 2894-R4/R5, 3152-R1/R2)</td>
        </tr>
        <tr>
          <td style="padding:8px;font-weight:bold;">Clarity</td>
          <td style="padding:8px;">Is it written at the right level for the intended audience? (IEEE 2894-R2/R3)</td>
        </tr>
      </table>
      <p><strong>Practical tips:</strong></p>
      <ul style="margin:8px 0 0 20px;">
        <li>Rate each explanation level (Summary, Detailed, Full) using the tabs</li>
        <li>Your ratings auto-save in your browser — you can close and return later</li>
        <li>Click the <strong>?</strong> button at any time to review this guide</li>
        <li>When finished, click <strong>Export Results</strong> to download your ratings as a file</li>
      </ul>
      <p style="margin-top:16px;">Click <strong>Start Evaluating</strong> to begin!</p>
    `
  }}
];

let wizardStep = 0;

function renderWizard() {{
  const step = WIZARD_STEPS[wizardStep];
  document.getElementById('wizardContent').innerHTML = step.html;

  // Dots
  let dots = '';
  for (let i = 0; i < WIZARD_STEPS.length; i++) {{
    const cls = i === wizardStep ? 'dot active' : (i < wizardStep ? 'dot done' : 'dot');
    dots += `<span class="${{cls}}"></span>`;
  }}
  document.getElementById('stepDots').innerHTML = dots;

  // Nav
  document.getElementById('wPrev').style.display = wizardStep === 0 ? 'none' : '';
  const isLast = wizardStep === WIZARD_STEPS.length - 1;
  document.getElementById('wNext').textContent = isLast ? 'Start Evaluating' : 'Next';
}}

function wizardNav(dir) {{
  if (dir > 0 && wizardStep === WIZARD_STEPS.length - 1) {{
    document.getElementById('wizardOverlay').classList.add('hidden');
    document.getElementById('mainApp').classList.remove('hidden');
    localStorage.setItem('rq2_wizard_done', '1');
    return;
  }}
  wizardStep = Math.max(0, Math.min(WIZARD_STEPS.length - 1, wizardStep + dir));
  renderWizard();
}}

function showWizard() {{
  wizardStep = 0;
  document.getElementById('wizardOverlay').classList.remove('hidden');
  renderWizard();
}}

// Skip wizard if already completed
if (localStorage.getItem('rq2_wizard_done')) {{
  document.getElementById('wizardOverlay').classList.add('hidden');
  document.getElementById('mainApp').classList.remove('hidden');
}} else {{
  renderWizard();
}}

// ── RATING DIMENSIONS ──
const LEVEL_DESCS = {{
  summary: "For end users: plain language, what happened and why, no technical jargon.",
  detailed: "For compliance officers: decision points, governance checks, standards.",
  full: "For developers: event-level trace, timing, agent IDs, scores."
}};

const DIMS = [
  {{
    key: "faithfulness",
    label: "Faithfulness",
    hint: "Does the explanation match the execution trace?",
    lo: "Made up", hi: "Fully traceable"
  }},
  {{
    key: "completeness",
    label: "Completeness",
    hint: "Data source, reasoning, AI disclosure, agent identity present?",
    lo: "Nothing", hi: "All 4 elements"
  }},
  {{
    key: "clarity",
    label: "Clarity",
    hint: "Right depth for the audience (customer / officer / developer)?",
    lo: "Wrong audience", hi: "Perfect fit"
  }}
];

let currentIdx = 0;
let ratings = JSON.parse(localStorage.getItem('rq2_human_ratings') || '{{}}');
let currentLevel = 'summary';

function saveRatings() {{
  localStorage.setItem('rq2_human_ratings', JSON.stringify(ratings));
}}

function render() {{
  const item = ITEMS[currentIdx];
  const key = item.id;
  if (!ratings[key]) ratings[key] = {{}};

  // Progress
  const rated = Object.keys(ratings).filter(k => {{
    const r = ratings[k];
    return ['summary','detailed','full'].some(l => r[l] && r[l].faithfulness && r[l].completeness && r[l].clarity);
  }}).length;
  document.getElementById('progressText').textContent =
    `Scenario ${{currentIdx+1}} of ${{ITEMS.length}} | ${{rated}} of ${{ITEMS.length}} rated`;
  document.getElementById('progressFill').style.width =
    `${{(rated / ITEMS.length) * 100}}%`;

  const expls = item.explanations || {{}};
  const levels = Object.keys(expls);
  if (levels.length && !levels.includes(currentLevel)) currentLevel = levels[0];

  const expl = expls[currentLevel] || {{}};
  const trace = (item.trace_events || []).map(e =>
    `[${{e.event}}] ${{JSON.stringify(e.data)}}`
  ).join('\\n');

  let html = `
    <h2>${{item.id}} — ${{item.description}}</h2>
    <div class="meta">Category: ${{item.category}} | Agent: ${{item.agent_id || (item.delegated_agents || []).join(', ') || 'N/A'}}</div>

    <div class="section">
      <h3>Customer Query</h3>
      <div class="query">${{escHtml(item.query)}}</div>
    </div>

    <div class="section">
      <h3>System Answer</h3>
      <div class="answer">${{escHtml(item.answer || '(no answer)')}}</div>
    </div>

    <div class="section">
      <h3>What Actually Happened (execution trace)</h3>
      <div class="trace">${{escHtml(trace || '(no trace)')}}</div>
    </div>

    <div class="section">
      <h3>Explanation Level</h3>
      <div class="level-tabs">
        ${{levels.map(l => `<button class="${{l===currentLevel?'active':''}}" onclick="switchLevel('${{l}}')">${{l.charAt(0).toUpperCase()+l.slice(1)}}</button>`).join('')}}
      </div>
      <div class="level-desc">${{LEVEL_DESCS[currentLevel] || ''}}</div>
      <div class="explanation">
        <strong>Narrative:</strong>\\n${{escHtml(expl.narrative || '(none)')}}
        \\n\\n<strong>Agents involved:</strong> ${{(expl.agents_involved || []).join(', ') || 'N/A'}}
        \\n<strong>Decisions:</strong> ${{(expl.decisions || []).map(d => typeof d === 'string' ? d : JSON.stringify(d)).join('; ') || 'N/A'}}
        \\n<strong>Provenance:</strong> ${{(expl.provenance || []).map(p => typeof p === 'string' ? p : JSON.stringify(p)).join('; ') || 'N/A'}}
      </div>
    </div>

    <h3 style="margin:16px 0 8px;">Rate this <em>${{currentLevel}}</em> explanation</h3>
  `;

  for (const dim of DIMS) {{
    const saved = ratings[key]?.[currentLevel]?.[dim.key] || 0;
    html += `
      <div class="rating-group">
        <label>${{dim.label}}</label>
        <div class="hint">${{dim.hint}}</div>
        <div class="stars">
          ${{[1,2,3,4,5].map(n => `<button class="${{n===saved?'active':''}}" onclick="rate('${{key}}','${{currentLevel}}','${{dim.key}}',${{n}})">${{n}}</button>`).join('')}}
        </div>
        <div class="star-labels"><span>${{dim.lo}}</span><span>${{dim.hi}}</span></div>
      </div>
    `;
  }}

  document.getElementById('scenarioCard').innerHTML = html;

  document.getElementById('prevBtn').disabled = currentIdx === 0;
  document.getElementById('nextBtn').textContent =
    currentIdx === ITEMS.length - 1 ? 'Done' : 'Next \\u2192';
}}

function escHtml(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

function switchLevel(level) {{
  currentLevel = level;
  render();
}}

function rate(scId, level, dim, score) {{
  if (!ratings[scId]) ratings[scId] = {{}};
  if (!ratings[scId][level]) ratings[scId][level] = {{}};
  ratings[scId][level][dim] = score;
  saveRatings();
  render();
}}

function navigate(dir) {{
  currentIdx = Math.max(0, Math.min(ITEMS.length - 1, currentIdx + dir));
  currentLevel = 'summary';
  render();
  window.scrollTo(0, 0);
}}

function exportResults() {{
  const name = document.getElementById('evaluatorName').value || 'anonymous';
  const output = {{
    evaluator: name,
    timestamp: new Date().toISOString(),
    total_scenarios: ITEMS.length,
    ratings: ratings,
    summary: {{}}
  }};

  let counts = {{}};
  for (const [scId, levels] of Object.entries(ratings)) {{
    for (const [level, dims] of Object.entries(levels)) {{
      for (const [dim, score] of Object.entries(dims)) {{
        const k = `${{level}}_${{dim}}`;
        if (!counts[k]) counts[k] = [];
        counts[k].push(score);
      }}
    }}
  }}
  for (const [k, scores] of Object.entries(counts)) {{
    output.summary[k] = {{
      mean: (scores.reduce((a,b) => a+b, 0) / scores.length).toFixed(2),
      n: scores.length
    }};
  }}

  const blob = new Blob([JSON.stringify(output, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `rq2_human_eval_${{name}}_${{new Date().toISOString().slice(0,10)}}.json`;
  a.click();
  document.getElementById('exportStatus').textContent = 'Exported!';
}}

render();
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML survey written to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RQ2 human evaluation survey")
    parser.add_argument(
        "--sample", type=int, default=15, help="Number of scenarios to sample"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation/results/rq2/human_eval.html"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    sample = _stratified_sample(scenarios, args.sample)
    print(f"Sampled {len(sample)} scenarios from {len(scenarios)} total:")
    for cat in sorted(set(s["category"] for s in sample)):
        n = sum(1 for s in sample if s["category"] == cat)
        print(f"  {cat}: {n}")

    print("\nRunning scenarios to capture explanations...")
    items = _collect_explanations(sample)

    _generate_html(items, Path(args.output))
    print("\nDone! Open the HTML file in a browser to start evaluating.")
    print("Ratings are saved in browser localStorage and can be exported as JSON.")
