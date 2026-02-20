# thesis/architecture_summary.py
"""
Generate architecture inventory from the codebase using AST introspection.

Usage:
    python -m thesis.architecture_summary

Outputs (to thesis/output/architecture/):
    module_inventory.md   — table: path, lines, description
    class_inventory.md    — table: class, module, methods
    component_map.md      — 4-layer architecture mapping
    codebase_stats.md     — summary statistics
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_DIR = ROOT / "app"
OUTPUT_DIR = ROOT / "thesis" / "output" / "architecture"

# 4-layer architecture mapping (matches Theory Chapter)
LAYERS = {
    "Meta-Agent (AOP)": [
        "app/orchestration/aop_coordinator.py",
        "app/orchestration/solvability_estimator.py",
        "app/orchestration/completeness_detector.py",
        "app/orchestration/performance_store.py",
    ],
    "Orchestration Patterns (Spine)": [
        "app/runtime/spine.py",
        "app/runtime/routing.py",
        "app/runtime/router.py",
        "app/runtime/router_adapter.py",
        "app/runtime/registry.py",
        "app/runtime/guardrails.py",
        "app/runtime/policy_guardrails.py",
        "app/runtime/trace.py",
        "app/runtime/audit_writer.py",
        "app/runtime/voice.py",
        "app/runtime/memory.py",
    ],
    "Agent Execution": [
        "app/runtime/rag_fsm.py",
        "app/runtime/workflow_engine.py",
        "app/runtime/workflow_mapper.py",
        "app/runtime/interfaces.py",
        "app/shared/rag.py",
        "app/shared/workflow.py",
        "app/shared/tool_operator.py",
    ],
    "Governance": [
        "app/runtime/policy/rule_engine.py",
        "app/runtime/policy/policy_compiler.py",
        "app/runtime/policy/policy_parser.py",
        "app/runtime/policy/workflow_policy_bridge.py",
        "app/runtime/policy_pack.py",
    ],
}


def _analyze_module(filepath: Path) -> dict:
    """Parse a Python file and extract docstring, classes, line count."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception:
        return None
    lines = len(source.splitlines())
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "path": str(filepath.relative_to(ROOT)),
            "lines": lines,
            "docstring": "",
            "classes": [],
        }

    docstring = ast.get_docstring(tree) or ""
    first_line = docstring.split("\n")[0].strip() if docstring else ""

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("_")
            ]
            classes.append(
                {
                    "name": node.name,
                    "methods": methods,
                    "lineno": node.lineno,
                }
            )
    return {
        "path": str(filepath.relative_to(ROOT)),
        "lines": lines,
        "docstring": first_line,
        "classes": classes,
    }


def generate_module_inventory() -> str:
    """Table of all app/ Python modules with line counts."""
    rows = []
    for pyfile in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(pyfile):
            continue
        info = _analyze_module(pyfile)
        if info is None:
            continue
        rows.append(info)

    lines = [
        "# Module Inventory",
        "",
        "| Module | Lines | Description |",
        "|--------|------:|-------------|",
    ]
    for r in sorted(rows, key=lambda x: x["path"]):
        lines.append(f"| `{r['path']}` | {r['lines']} | {r['docstring'][:80]} |")

    lines.append(f"\n**Total**: {len(rows)} modules, " f"{sum(r['lines'] for r in rows):,} lines")
    return "\n".join(lines)


def generate_class_inventory() -> str:
    """Table of key classes with their public methods."""
    rows = []
    for pyfile in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(pyfile):
            continue
        info = _analyze_module(pyfile)
        if info is None:
            continue
        for cls in info["classes"]:
            rows.append(
                {
                    "class": cls["name"],
                    "module": info["path"],
                    "methods": ", ".join(cls["methods"][:6]),
                    "method_count": len(cls["methods"]),
                }
            )

    lines = [
        "# Class Inventory",
        "",
        "| Class | Module | Public Methods | Count |",
        "|-------|--------|----------------|------:|",
    ]
    for r in sorted(rows, key=lambda x: x["module"]):
        lines.append(f"| `{r['class']}` | `{r['module']}` | {r['methods']} | {r['method_count']} |")
    lines.append(f"\n**Total**: {len(rows)} classes")
    return "\n".join(lines)


def generate_component_map() -> str:
    """4-layer architecture mapping with line counts per layer."""
    lines = [
        "# Component Architecture Map",
        "",
        "4-layer architecture aligned with Theory Chapter " "(Wang et al. PMPA + Li et al. AOP).",
        "",
    ]

    for layer, modules in LAYERS.items():
        layer_lines = 0
        mod_rows = []
        for mod_path in modules:
            fpath = ROOT / mod_path
            if fpath.exists():
                info = _analyze_module(fpath)
                if info:
                    layer_lines += info["lines"]
                    cls_names = [c["name"] for c in info["classes"]]
                    mod_rows.append(
                        f"  - `{mod_path}` ({info['lines']} lines)"
                        + (f" — {', '.join(cls_names)}" if cls_names else "")
                    )

        lines.append(f"## {layer} ({layer_lines:,} lines)")
        lines.extend(mod_rows)
        lines.append("")

    return "\n".join(lines)


def generate_stats() -> str:
    """Codebase summary statistics."""
    total_files = 0
    total_lines = 0
    layer_totals = {}

    # All app/ files
    for pyfile in APP_DIR.rglob("*.py"):
        if "__pycache__" in str(pyfile):
            continue
        try:
            total_lines += len(pyfile.read_text(encoding="utf-8").splitlines())
            total_files += 1
        except Exception:
            pass

    # Per layer
    for layer, modules in LAYERS.items():
        layer_sum = 0
        for mod_path in modules:
            fpath = ROOT / mod_path
            if fpath.exists():
                try:
                    layer_sum += len(fpath.read_text(encoding="utf-8").splitlines())
                except Exception:
                    pass
        layer_totals[layer] = layer_sum

    # Tests + evaluation
    test_files = list((ROOT / "tests").rglob("*.py")) if (ROOT / "tests").exists() else []
    test_lines = sum(
        len(f.read_text(encoding="utf-8").splitlines())
        for f in test_files
        if "__pycache__" not in str(f)
    )
    eval_files = list((ROOT / "evaluation").rglob("*.py"))
    eval_lines = sum(
        len(f.read_text(encoding="utf-8").splitlines())
        for f in eval_files
        if "__pycache__" not in str(f)
    )

    lines = [
        "# Codebase Statistics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Application modules | {total_files} |",
        f"| Application lines | {total_lines:,} |",
        f"| Test files | {len(test_files)} |",
        f"| Test lines | {test_lines:,} |",
        f"| Evaluation files | {len(eval_files)} |",
        f"| Evaluation lines | {eval_lines:,} |",
        "",
        "## Lines by Architecture Layer",
        "",
        "| Layer | Lines | % |",
        "|-------|------:|--:|",
    ]
    mapped = sum(layer_totals.values())
    for layer, count in layer_totals.items():
        pct = (count / mapped * 100) if mapped else 0
        lines.append(f"| {layer} | {count:,} | {pct:.0f}% |")
    lines.append(f"| **Mapped total** | **{mapped:,}** | **100%** |")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "module_inventory.md": generate_module_inventory,
        "class_inventory.md": generate_class_inventory,
        "component_map.md": generate_component_map,
        "codebase_stats.md": generate_stats,
    }
    for filename, fn in artifacts.items():
        content = fn()
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")
        print(f"  [OK] {filename}")

    print(f"\n  Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
