"""CAD API tests (no pythonOCC required for status)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_cad_status():
    r = client.get("/api/v1/cad/status")
    assert r.status_code == 200
    body = r.json()
    assert "pythonocc_available" in body
    assert body["api_version"] == "1.1"


def test_cad_upload_binary():
    data = b"ISO-10303-21; HEADER; ENDSEC; END-ISO-10303-21;"
    r = client.post(
        "/api/v1/cad/upload/binary",
        content=data,
        headers={"Content-Type": "application/octet-stream", "X-Filename": "part.stp"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "model_id" in body
    assert body["filename"] == "part.stp"
