# tests/test_dua_step1_cli.py
import sys
from pathlib import Path
import importlib


def write(p: Path, content="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_cli_with_llm_advisory(tmp_path, monkeypatch, capsys):
    # Create data files
    write(tmp_path / "data" / "BankFAQs.csv", "q,a\nx,y")
    write(tmp_path / "data" / "refunds_policy.yaml", "rules: []")

    # Fresh import and patch
    if "app.dua_v0" in sys.modules:
        del sys.modules["app.dua_v0"]
    dua = importlib.import_module("app.dua_v0")

    # Stub out detect_signals_llm to avoid real network
    monkeypatch.setattr(
        dua,
        "detect_signals_llm",
        lambda filenames: {
            "primary": "fintech",
            "scores": {"fintech": 0.9},
            "explanation": "banking keywords",
        },
    )

    # Prepare argv and run
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dua_v0",  # program name placeholder (anything without leading dash is fine)
            "--data",
            str(tmp_path / "data"),
            "--vertical",
            "fintech",
            "--model",
            "gpt-4o-mini",
        ],
    )

    dua.main()
    out = capsys.readouterr().out.lower()

    assert "user-selected vertical: fintech" in out
    assert "llm advisory" in out
    assert "primary: fintech" in out
    assert "heuristic counts" in out


def test_cli_without_llm_fallback(tmp_path, monkeypatch, capsys):
    write(tmp_path / "data" / "some_FAQ.csv", "q,a\nx,y")
    write(tmp_path / "data" / "tone_policy.yaml", "rules: []")

    if "app.dua_v0" in sys.modules:
        del sys.modules["app.dua_v0"]
    dua = importlib.import_module("app.dua_v0")

    # Simulate LLM unavailable
    monkeypatch.setattr(dua, "detect_signals_llm", lambda filenames: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dua_v0",
            "--data",
            str(tmp_path / "data"),
            "--vertical",
            "retail",
        ],
    )

    dua.main()
    out = capsys.readouterr().out.lower()

    assert "llm unavailable" in out
    assert "heuristic counts" in out
    assert "user-selected vertical: retail" in out
