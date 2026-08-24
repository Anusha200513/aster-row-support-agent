"""Tests for Aster & Row FastAPI server (app.server)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_health_check_returns_ok():
    """Verify GET /health returns 200 OK and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint_returns_ok_or_static():
    """Verify GET / returns 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_chat_endpoint_valid_request():
    """Verify POST /api/chat calls handle_turn with session_id and message, and returns structured result."""
    mock_agent_result = {
        "answer": "Your order ORD-1007 is in transit with UPS.",
        "sources": ["01-returns-policy-current.md — Standard return window"],
        "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "ORD-1007"}}],
        "handoff": False,
    }

    with patch("app.server.handle_turn", return_value=mock_agent_result) as mock_handle_turn:
        payload = {
            "session_id": "test-session-123",
            "message": "Where is ORD-1007?",
        }
        response = client.post("/api/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Your order ORD-1007 is in transit with UPS."
        assert data["sources"] == ["01-returns-policy-current.md — Standard return window"]
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["name"] == "lookup_order"
        assert data["handoff"] is False

        mock_handle_turn.assert_called_once_with(
            session_id="test-session-123",
            user_message="Where is ORD-1007?",
        )


def test_chat_endpoint_rejects_empty_session_id():
    """Verify POST /api/chat rejects empty or whitespace-only session_id with 422."""
    response_empty = client.post("/api/chat", json={"session_id": "", "message": "Hello"})
    assert response_empty.status_code == 422

    response_whitespace = client.post("/api/chat", json={"session_id": "   ", "message": "Hello"})
    assert response_whitespace.status_code == 422


def test_chat_endpoint_rejects_empty_message():
    """Verify POST /api/chat rejects empty or whitespace-only message with 422."""
    response_empty = client.post("/api/chat", json={"session_id": "session-1", "message": ""})
    assert response_empty.status_code == 422

    response_whitespace = client.post("/api/chat", json={"session_id": "session-1", "message": "   "})
    assert response_whitespace.status_code == 422


def test_chat_endpoint_rejects_oversized_message():
    """Verify POST /api/chat rejects excessively oversized messages with 422."""
    huge_message = "A" * 5000
    response = client.post("/api/chat", json={"session_id": "session-1", "message": huge_message})
    assert response.status_code == 422


def test_chat_endpoint_rejects_malformed_or_missing_fields():
    """Verify POST /api/chat rejects missing fields with 422."""
    # Missing message
    res1 = client.post("/api/chat", json={"session_id": "session-1"})
    assert res1.status_code == 422

    # Missing session_id
    res2 = client.post("/api/chat", json={"message": "Where is my order?"})
    assert res2.status_code == 422

    # Empty payload
    res3 = client.post("/api/chat", json={})
    assert res3.status_code == 422


def test_chat_endpoint_handles_internal_exception_gracefully():
    """Verify internal unhandled exception returns safe 500 error without exposing traceback."""
    with patch("app.server.handle_turn", side_effect=RuntimeError("Simulated database crash and internal failure")):
        payload = {
            "session_id": "test-session-err",
            "message": "Where is ORD-1007?",
        }
        response = client.post("/api/chat", json=payload)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        # Ensure raw traceback / internal error message is NOT exposed to client
        assert "Traceback" not in data["detail"]
        assert "Simulated database crash" not in data["detail"]
        assert "internal server error" in data["detail"].lower()


def test_chat_endpoint_never_exposes_api_key(monkeypatch: pytest.MonkeyPatch):
    """Verify server responses never contain GROQ_API_KEY even if present in environment."""
    secret_key = "gsk_super_secret_groq_key_999888777"
    monkeypatch.setenv("GROQ_API_KEY", secret_key)

    mock_result = {
        "answer": "Order ORD-1007 is on its way.",
        "sources": [],
        "tool_calls": [],
        "handoff": False,
    }

    with patch("app.server.handle_turn", return_value=mock_result):
        response = client.post("/api/chat", json={"session_id": "sec-test", "message": "Status?"})
        assert response.status_code == 200
        assert secret_key not in response.text


def test_cors_middleware_configured():
    """Verify CORS headers are present for cross-origin requests."""
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
