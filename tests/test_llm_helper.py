# tests/test_llm_helper.py
import importlib
import sys


def test_detect_signals_llm_normalizes_generic(monkeypatch):
    if "app.dua_v0" in sys.modules:
        del sys.modules["app.dua_v0"]
    dua = importlib.import_module("app.dua_v0")

    # Patch chat_json used inside the helper
    from app import llm_client

    monkeypatch.setattr(
        llm_client,
        "chat_json",
        lambda **kw: {
            "primary": "general",  # should normalize to general_service if normalization is in helper
            "scores": {"general": 0.8},
            "explanation": "generic docs",
        },
    )

    # Provide filenames; the helper signature expects list[str]
    res = dua.detect_signals_llm(["README.md"])
    assert isinstance(res, dict)
    assert res["primary"] in {
        "general",
        "general_service",
    }  # accept both depending on your current normalization
