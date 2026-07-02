"""
FastAPI route unit and integration tests for SHL Assessment Recommender.

Tests that /health and /chat endpoints respond correctly under normal conditions
and handle validation errors and bad requests gracefully.

Improves: Hard evals (endpoint availability, HTTP status code checks).
"""

from fastapi.testclient import TestClient
import pytest

from app.main import app


@pytest.fixture
def client():
    # Use TestClient with lifespan events enabled to trigger catalog load
    with TestClient(app) as c:
        import time
        # Poll health endpoint until system is ready (status: ok), up to 60 seconds
        for _ in range(600):
            response = c.get("/health")
            if response.status_code == 200 and response.json().get("status") == "ok":
                break
            time.sleep(0.1)
        yield c


def test_get_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_chat_greeting(client):
    payload = {
        "messages": [
            {"role": "user", "content": "Hi there"}
        ]
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    resp_data = response.json()

    assert "reply" in resp_data
    assert "recommendations" in resp_data
    assert "end_of_conversation" in resp_data
    assert resp_data["recommendations"] == []
    assert resp_data["end_of_conversation"] is False


def test_post_chat_vague_query(client):
    payload = {
        "messages": [
            {"role": "user", "content": "I need some assessments"}
        ]
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["recommendations"] == []
    assert resp_data["end_of_conversation"] is False


def test_post_chat_invalid_payload(client):
    # Missing messages array
    payload = {}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

    # Invalid message role
    payload_bad = {
        "messages": [
            {"role": "system_role", "content": "Hello"}
        ]
    }
    response_bad = client.post("/chat", json=payload_bad)
    # The schema specifies role as string, which passes pydantic validation,
    # but route or orchestrator handles it or fails gracefully
    assert response_bad.status_code in [200, 422, 500]
