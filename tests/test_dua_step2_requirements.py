# tests/test_dua_step2_requirements.py
import sys
import json
import importlib
from pathlib import Path


def write(p: Path, content="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_writes_requirements_and_validates(tmp_path, monkeypatch, capsys):
    # Arrange: data files + a minimal schema in spec/
    data_dir = tmp_path / "data"
    spec_dir = tmp_path / "spec"
    out_path = tmp_path / "data" / "requirements.json"

    write(data_dir / "BankFAQs.csv", "q,a\nx,y")
    write(data_dir / "refunds_policy.yaml", "rules: []")

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["vertical", "capabilities"],
    }
    write(spec_dir / "requirements.schema.json", json.dumps(schema))

    # Fresh import
    if "app.dua_v0" in sys.modules:
        del sys.modules["app.dua_v0"]
    dua = importlib.import_module("app.dua_v0")

    # Avoid network: stub LLM helper
    monkeypatch.setattr(dua, "detect_signals_llm", lambda filenames: {"primary": "fintech"})

    # Run CLI
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dua_v0",
            "--data",
            str(data_dir),
            "--vertical",
            "fintech",
            "--spec",
            str(spec_dir),
            "--out",
            str(out_path),
        ],
    )
    dua.main()
    capsys.readouterr().out.lower()

    # Assert file written & schema-ish content
    assert out_path.exists(), "requirements.json should be created"
    req = json.loads(out_path.read_text(encoding="utf-8"))
    assert req["vertical"] == "fintech"
    assert isinstance(req["capabilities"], list) and len(req["capabilities"]) >= 1
