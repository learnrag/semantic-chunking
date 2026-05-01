"""API tests with mock embedder."""

from fastapi.testclient import TestClient

from chunker.api import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_chunk_mock():
    res = client.post(
        "/chunk",
        json={
            "text": "One sentence. Two sentence. Three sentence.",
            "mock": True,
            "percentile": 90,
            "document_id": "api_doc",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["count"] >= 1
    assert body["chunks"][0]["chunk_id"].startswith("api_doc_chunk_")
    assert "breakpoints" in body["diagnostics"]
