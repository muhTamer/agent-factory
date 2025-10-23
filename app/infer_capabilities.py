# app/infer_capabilities.py
"""
InferCapabilities class
- Determines which capabilities can be generated based on uploaded documents, vertical, and optional LLM checks.
- Always includes baseline guardrails and QA evaluator.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from jsonschema import Draft202012Validator, ValidationError

import re
import random
import json
import csv
import yaml


class InferCapabilities:
    def __init__(self, vertical: str, data_dir: str, llm_client=None):
        """
        Args:
            vertical: user-selected domain (e.g., 'retail', 'fintech', 'telco', 'general_service')
            data_dir: path to folder with uploaded documents
            llm_client: optional object providing .chat_json() interface (e.g., app.llm_client.chat_json)
        """
        self.vertical = vertical.lower()
        self.data_dir = Path(data_dir)
        self.llm_client = llm_client
        self.docs = self._list_files()
        self.capabilities = ["faq", "complaint", "guardrails", "qa"]

    # -----------------------------
    # 1. File discovery
    # -----------------------------
    def _list_files(self) -> List[Path]:
        """List all files under the data directory (non-recursive for now)."""
        if not self.data_dir.exists():
            return []
        return [p for p in self.data_dir.iterdir() if p.is_file()]

    # -----------------------------
    # 2. Public method: infer()
    # -----------------------------
    def infer(self, use_llm: bool = True) -> Dict[str, Any]:
        """
        Infer capabilities based on vertical, available docs, and optional LLM assessment.
        Returns structured dict (capabilities + summary).
        """
        results = []

        for cap in self.capabilities:
            if cap in {"guardrails", "qa"}:
                results.append(self._always_included(cap))
                continue

            if use_llm and self.llm_client:
                res = self._llm_assess(cap)
            else:
                res = self._heuristic_assess(cap)
            results.append(res)

        summary = self._summarize(results)
        return {"capabilities": results, "summary": summary}

    # -----------------------------
    # 3. Heuristic fallback (safe baseline)
    # -----------------------------
    def _heuristic_assess(self, cap: str) -> Dict[str, Any]:
        """Fallback deterministic logic when LLM not available."""
        names = " ".join([f.name.lower() for f in self.docs])
        detected, missing, conf = [], [], 0.5

        if cap == "faq":
            if "faq" in names or any(f.suffix.lower() == ".csv" for f in self.docs):
                detected = ["faq_csv"]
                conf = 0.9
            else:
                missing = ["faqs.csv"]
                conf = 0.5
        elif cap == "complaint":
            if any("refund" in f.name.lower() or "complaint" in f.name.lower() for f in self.docs):
                detected = ["refund_policy.yaml"]
                conf = 0.8
            else:
                missing = ["refunds_policy.yaml", "complaint_sop.md"]
                conf = 0.4

        status = "ready" if detected else "partial" if conf >= 0.6 else "missing_docs"
        return {
            "id": cap,
            "status": status,
            "confidence": round(conf, 2),
            "docs_detected": detected,
            "docs_missing": missing,
            "always_included": False,
            "why": "Heuristic fallback evaluation.",
        }

    # -----------------------------
    # 4. Always-on baseline
    # -----------------------------
    def _always_included(self, cap: str) -> Dict[str, Any]:
        if cap == "guardrails":
            return {
                "id": "guardrails",
                "status": "ready",
                "confidence": 1.0,
                "docs_detected": ["base_policy_pack.yaml"],
                "docs_missing": [],
                "always_included": True,
                "why": "Guardrails are mandatory runtime spine component.",
            }
        if cap == "qa":
            return {
                "id": "qa",
                "status": "ready",
                "confidence": 1.0,
                "docs_detected": [],
                "docs_missing": [],
                "always_included": True,
                "why": "QA evaluator always included for monitoring factuality and tone.",
            }

    # -----------------------------
    # 5. LLM-assisted assessment
    # -----------------------------
    def _llm_assess(self, cap: str) -> Dict[str, Any]:
        """
        Ask LLM if the currently uploaded docs are sufficient for capability 'cap'.
        Uses privacy-preserving sampling.
        """
        try:
            samples = self._prepare_samples(cap)
            self._audit_samples(cap, samples)
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(cap, samples)},
            ]
            result = self.llm_client.chat_json(
                messages=messages, model=os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o-mini"
            )
            return self._normalize_llm_result(cap, result)
        except Exception as e:
            return {
                "id": cap,
                "status": "partial",
                "confidence": 0.4,
                "docs_detected": [],
                "docs_missing": [],
                "always_included": False,
                "why": f"LLM call failed: {e}",
            }

    # -----------------------------
    # 6. Helper: privacy-safe sampling (stub for now)
    # -----------------------------
    def _prepare_samples(self, cap: str) -> list[dict]:
        """
        Extract small, sanitized snippets from supported files.
        This ensures no PII or sensitive info leaves the environment.
        """
        samples: list[dict] = []

        for f in self.docs:
            ext = f.suffix.lower()

            if ext in {".csv", ".tsv"}:
                text = self._sample_csv(f)
            elif ext in {".yaml", ".yml", ".json"}:
                text = self._sample_yaml_json(f)
            elif ext in {".md", ".txt"}:
                text = self._sample_text(f)
            else:
                # Skip binaries or large files
                continue

            # Apply redaction + truncation
            sanitized = self._sanitize_text(text)
            if sanitized.strip():
                samples.append({"filename": f.name, "sample_text": sanitized[:1500]})  # ≈400 tokens

        # Randomize order slightly to reduce positional bias
        random.shuffle(samples)
        return samples[:5]  # limit number of samples sent

    # ----------------------------------------------------------
    # CSV sampler
    # ----------------------------------------------------------
    def _sample_csv(self, path: Path) -> str:
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                rows = list(reader)
            header = ", ".join(rows[0][:6]) if rows else ""
            # Up to 2 random rows (for structure only)
            data_rows = (
                random.sample(rows[1:], min(2, max(0, len(rows) - 1))) if len(rows) > 1 else []
            )
            return f"Headers: {header}\nExamples:\n" + "\n".join(
                [", ".join(r[:6]) for r in data_rows]
            )
        except Exception:
            return ""

    # ----------------------------------------------------------
    # YAML / JSON sampler
    # ----------------------------------------------------------
    def _sample_yaml_json(self, path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                data = json.loads(text)
            else:
                data = yaml.safe_load(text)

            # Only top-level keys + 1 nested example if dict
            if isinstance(data, dict):
                keys = list(data.keys())[:6]
                snippet = {k: data[k] for k in keys}
                return json.dumps(snippet, indent=2)[:1500]
            elif isinstance(data, list):
                return json.dumps(data[:2], indent=2)[:1500]
            else:
                return str(data)[:1500]
        except Exception:
            return ""

    # ----------------------------------------------------------
    # Text / Markdown sampler
    # ----------------------------------------------------------
    def _sample_text(self, path: Path) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            non_empty = [line for line in lines if line.strip()]
            if not non_empty:
                return ""
            # Take first paragraph + one random mid-section paragraph
            para1 = non_empty[0]
            para2 = random.choice(non_empty[1:]) if len(non_empty) > 3 else ""
            return f"{para1}\n...\n{para2}"
        except Exception:
            return ""

    # ----------------------------------------------------------
    # Sanitization (PII masking, numeric bucketing)
    # ----------------------------------------------------------
    def _sanitize_text(self, text: str) -> str:
        # Mask common PII patterns
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
        text = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", "[PHONE]", text)
        text = re.sub(r"[A-Z]{2}\d{6,}", "[ACCOUNT_ID]", text)
        text = re.sub(r"\b\d{12,19}\b", "[CARD_NO]", text)
        text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD_NO]", text)
        text = re.sub(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?", "$X,XXX", text)
        text = re.sub(r"\b\d{5}(?:-\d{4})?\b", "[ZIP]", text)

        # General entity placeholders
        text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[PERSON]", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ----------------------------------------------------------
    # Audit
    # ----------------------------------------------------------
    def _audit_samples(self, cap: str, samples: list[dict]) -> None:
        """
        Save the sanitized samples that were used for LLM sufficiency check.
        Allows transparency and offline review.
        """
        if not samples:
            return

        audit_path = self.data_dir / "samples_audit.json"
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "vertical": self.vertical,
            "capability": cap,
            "num_samples": len(samples),
            "samples": samples,
        }

        try:
            if audit_path.exists():
                existing = json.loads(audit_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    existing.append(record)
                    audit_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                    return
            # otherwise new file
            audit_path.write_text(json.dumps([record], indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Could not write audit preview: {e}")

    # -----------------------------
    # 7. Helper: prompts
    # -----------------------------
    def _system_prompt(self) -> str:
        return (
            "You are a capability sufficiency evaluator for an AI Agent Factory. "
            "Return JSON with keys {capability, status, confidence, docs_detected, docs_missing, why}. "
            "Do not include or reproduce input text. Do not infer personal data."
        )

    def _user_prompt(self, cap: str, samples: List[Dict[str, str]]) -> str:
        doc_section = "\n\n".join([f"--- {s['filename']} ---\n{s['sample_text']}" for s in samples])
        return f"Vertical: {self.vertical}\nCapability: {cap}\nDocuments:\n{doc_section}"

    # -----------------------------
    # 8. Helper: normalize LLM output
    # -----------------------------
    def _normalize_llm_result(self, cap: str, result: dict) -> dict:
        """Ensure structure conforms to expected schema."""
        if not isinstance(result, dict) or not self._validate_llm_output(result):
            return self._heuristic_assess(cap)

        return {
            "id": cap,
            "status": result["status"],
            "confidence": float(result.get("confidence", 0.5)),
            "docs_detected": result.get("docs_detected", []),
            "docs_missing": result.get("docs_missing", []),
            "always_included": False,
            "why": result.get("why", "LLM-assessed sufficiency."),
        }

    # -----------------------------
    # 9. Helper: summary
    # -----------------------------
    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        summary = {"ready": 0, "partial": 0, "missing": 0, "total": len(results)}
        for r in results:
            s = r.get("status")
            if s == "ready":
                summary["ready"] += 1
            elif s == "partial":
                summary["partial"] += 1
            else:
                summary["missing"] += 1
        return summary

    # -----------------------------
    # 10. Validate LLM output
    # -----------------------------
    def _validate_llm_output(self, data: dict) -> bool:
        """Return True if LLM output passes schema validation."""
        schema = {
            "type": "object",
            "required": [
                "capability",
                "status",
                "confidence",
                "docs_detected",
                "docs_missing",
                "why",
            ],
            "properties": {
                "capability": {"type": "string"},
                "status": {"type": "string", "enum": ["ready", "partial", "missing_docs"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "docs_detected": {"type": "array", "items": {"type": "string"}},
                "docs_missing": {"type": "array", "items": {"type": "string"}},
                "why": {"type": "string"},
            },
        }
        try:
            Draft202012Validator(schema).validate(data)
            return True
        except ValidationError as e:
            print(f"[WARN] LLM output schema invalid: {e.message}")
            return False
        except Exception:
            return False
