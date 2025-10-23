from pathlib import Path
from app.infer_capabilities import InferCapabilities

FIXTURES = Path(__file__).parent / "data_fixtures"


def test_infer_no_files(monkeypatch, tmp_path):
    ic = InferCapabilities(vertical="retail", data_dir=tmp_path)
    result = ic.infer(use_llm=False)
    caps = {c["id"]: c for c in result["capabilities"]}

    # guardrails and qa always included
    assert "guardrails" in caps and caps["guardrails"]["status"] == "ready"
    assert "qa" in caps

    # faq and complaint should be partial or missing
    assert caps["faq"]["status"] in {"partial", "missing_docs"}
    assert caps["complaint"]["status"] in {"partial", "missing_docs"}

    # summary counts should match
    total = result["summary"]["total"]
    assert total == len(result["capabilities"])


def test_infer_detects_faq_csv(tmp_path):
    csv_path = tmp_path / "BankFAQs.csv"
    csv_path.write_text("question,answer\nHow?,Like this.")

    ic = InferCapabilities(vertical="retail", data_dir=tmp_path)
    result = ic.infer(use_llm=False)
    faq_cap = next(c for c in result["capabilities"] if c["id"] == "faq")

    assert faq_cap["status"] == "ready"
    assert "faq_csv" in faq_cap["docs_detected"]
    assert faq_cap["confidence"] >= 0.8


def test_infer_detects_complaint_policy(tmp_path):
    yaml_path = tmp_path / "refunds_policy.yaml"
    yaml_path.write_text("rules:\n  - id: refund\n    description: Refunds allowed")

    ic = InferCapabilities(vertical="retail", data_dir=tmp_path)
    result = ic.infer(use_llm=False)
    complaint = next(c for c in result["capabilities"] if c["id"] == "complaint")

    assert complaint["status"] in {"ready", "partial"}
    assert any("refund" in d for d in complaint["docs_detected"])


def test_infer_handles_broken_files(tmp_path):
    bad_path = tmp_path / "broken.csv"
    bad_path.write_bytes(b"\xff\xfe\x00\x00")  # invalid utf-8

    ic = InferCapabilities(vertical="retail", data_dir=tmp_path)
    result = ic.infer(use_llm=False)
    assert any(c["id"] == "faq" for c in result["capabilities"])


def test_infer_always_includes_spine(tmp_path):
    ic = InferCapabilities(vertical="fintech", data_dir=tmp_path)
    result = ic.infer(use_llm=False)
    caps = [c["id"] for c in result["capabilities"]]

    assert "guardrails" in caps
    assert "qa" in caps


class MockLLM:
    def chat_json(self, **kwargs):
        return {
            "capability": "faq",
            "status": "ready",
            "confidence": 0.9,
            "docs_detected": ["faq_csv"],
            "docs_missing": [],
            "why": "FAQ file found.",
        }


def test_llm_mode_returns_valid(tmp_path):
    csv_path = tmp_path / "BankFAQs.csv"
    csv_path.write_text("question,answer\nQ?,A?")
    ic = InferCapabilities(vertical="retail", data_dir=tmp_path, llm_client=MockLLM())
    result = ic.infer(use_llm=True)
    faq = next(c for c in result["capabilities"] if c["id"] == "faq")

    assert faq["status"] == "ready"
    assert faq["confidence"] == 0.9
    assert faq["docs_detected"] == ["faq_csv"]


class MockBadLLM:
    def chat_json(self, **kwargs):
        return {"bad": "data"}  # Missing required keys


def test_llm_invalid_falls_back(tmp_path):
    csv_path = tmp_path / "BankFAQs.csv"
    csv_path.write_text("question,answer\nQ?,A?")
    ic = InferCapabilities(vertical="retail", data_dir=tmp_path, llm_client=MockBadLLM())
    result = ic.infer(use_llm=True)
    faq = next(c for c in result["capabilities"] if c["id"] == "faq")

    # Fallback should set why to "Heuristic fallback evaluation."
    assert faq["why"].startswith("Heuristic")
