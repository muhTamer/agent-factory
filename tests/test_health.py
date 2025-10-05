from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "version" in body

def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body
