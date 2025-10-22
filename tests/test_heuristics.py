# tests/test_heuristics.py
from pathlib import Path
import importlib
import sys


def write(p: Path, content="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_heuristic_counts_basic(tmp_path):
    write(tmp_path / "data" / "KYC_policy.pdf")
    write(tmp_path / "data" / "BankFAQs.csv")
    write(tmp_path / "data" / "complaint_SOP.md")

    if "app.dua_v0" in sys.modules:
        del sys.modules["app.dua_v0"]
    dua = importlib.import_module("app.dua_v0")

    files = list((tmp_path / "data").iterdir())
    counts = dua._heuristic_counts(files)

    assert isinstance(counts, dict)
    assert counts["fintech"] >= 1
    assert counts["generic_cs"] >= 1
