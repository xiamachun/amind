"""Unit tests for the amind Python SDK."""

from __future__ import annotations

import pytest
import httpx
from pytest_httpx import HTTPXMock

from amind import (
    AmindClient,
    Memory,
    ScoredMemory,
    StoreResponse,
    DeleteResponse,
    SessionResponse,
    SessionSummary,
    HealthResponse,
    MetricsResponse,
    CoverageStats,
    NotFoundError,
    ValidationError,
)


@pytest.fixture
def client():
    c = AmindClient(host="127.0.0.1", port=8080)
    yield c
    c.close()


class TestHealth:
    def test_health_ok(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/health",
            json={"status": "ok", "version": "0.1.0"},
        )
        resp = client.health()
        assert resp.status == "ok"
        assert resp.version == "0.1.0"

    def test_is_healthy_true(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/health",
            json={"status": "ok"},
        )
        assert client.is_healthy() is True

    def test_is_healthy_false_on_error(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_exception(httpx.ConnectError("refused"))
        assert client.is_healthy() is False

    def test_metrics(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/metrics",
            json={"total_memories": 42, "graph_edges": 10},
        )
        resp = client.metrics()
        assert resp.total_memories == 42
        assert resp.graph_edges == 10


class TestMemoryCRUD:
    def test_store(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/memories",
            json={"memory_ids": [123], "async_refinement_scheduled": True},
        )
        resp = client.store("user likes dark theme", agent_id="agent-1", user_id="user1", scope="private", memory_type="ephemeral")
        assert resp.memory_ids == [123]
        assert resp.async_refinement_scheduled is True

    def test_recall(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/memories/recall",
            json={"results": [
                {"memory_id": 1, "content": "dark theme", "score": 0.95, "semantic_score": 0.9},
                {"memory_id": 2, "content": "light mode", "score": 0.7, "semantic_score": 0.6},
            ]},
        )
        results = client.recall("what theme does user prefer?")
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].content == "dark theme"

    def test_get_memory(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/memories/123",
            json={"memory_id": 123, "content": "hello", "agent_id": "default", "user_id": "anonymous", "scope": "private", "memory_type": "ephemeral", "phase": "active"},
        )
        mem = client.get(123)
        assert mem.content == "hello"
        assert mem.agent_id == "default"

    def test_get_memory_not_found(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/memories/999",
            status_code=404,
            json={"error": "memory not found"},
        )
        with pytest.raises(NotFoundError):
            client.get(999)

    def test_delete(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/memories/123",
            json={"deleted": True, "invalidated_count": 2, "invalidated_ids": [456, 789]},
        )
        resp = client.delete(123)
        assert resp.deleted is True
        assert resp.invalidated_count == 2

    def test_get_history(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/memories/123/history",
            json={"versions": [
                {"memory_id": 123, "content": "v2", "version": 2},
                {"memory_id": 100, "content": "v1", "version": 1},
            ]},
        )
        history = client.get_history(123)
        assert len(history) == 2
        assert history[0].version == 2

    def test_list_memories(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            json={"total": 100, "page": 1, "per_page": 10, "memories": [
                {"memory_id": "1", "content": "a", "agent_id": "default", "user_id": "anonymous", "scope": "private", "memory_type": "ephemeral"},
            ]},
        )
        resp = client.list_memories(page=1, per_page=10, agent_id="default")
        assert resp.total == 100
        assert len(resp.memories) == 1


class TestSessions:
    def test_session_start(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/sessions/start",
            json={"session_id": 42},
        )
        resp = client.session_start(agent_id="agent-1", user_id="user1")
        assert resp.session_id == 42

    def test_session_turn(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            json={"ok": True, "current_intent": "question", "turn_count": 3, "fact_count": 1},
        )
        resp = client.session_turn(42, user_input="hello", agent_response="hi there")
        assert resp.ok is True
        assert resp.current_intent == "question"

    def test_session_close(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(json={"closed": True})
        resp = client.session_close(42)
        assert resp["closed"] is True

    def test_session_summary(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            json={"session_id": 42, "agent_id": "a", "user_id": "u1", "turn_count": 5, "active": True},
        )
        resp = client.session_summary(42)
        assert resp.turn_count == 5
        assert resp.active is True


class TestMetaCognition:
    def test_conflicts(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://127.0.0.1:8080/v1/metacognition/conflicts",
            json=[{"memory_a": "1", "memory_b": "2", "conflict_type": "Contradicts"}],
        )
        conflicts = client.conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "Contradicts"

    def test_coverage(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            json={"total": 50, "active": 40, "stale": 5, "conflicted": 3},
        )
        stats = client.coverage()
        assert stats.total == 50
        assert stats.active == 40


class TestVariables:
    def test_list_variables(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            json={"variables": [
                {"name": "decay_rate", "value": "0.01", "type": "float", "mode": "DYNAMIC"},
            ]},
        )
        vars = client.list_variables()
        assert len(vars) == 1
        assert vars[0].name == "decay_rate"

    def test_set_variable(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            json={"ok": True, "name": "decay_rate", "old_value": "0.01", "new_value": "0.02"},
        )
        resp = client.set_variable("decay_rate", "0.02")
        assert resp.ok is True
        assert resp.new_value == "0.02"


class TestErrorHandling:
    def test_validation_error(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            status_code=400,
            json={"error": "invalid JSON"},
        )
        with pytest.raises(ValidationError) as exc_info:
            client.store("")
        assert exc_info.value.status_code == 400

    def test_connection_error(self, client: AmindClient, httpx_mock: HTTPXMock):
        httpx_mock.add_exception(httpx.ConnectError("refused"))
        from amind.exceptions import ConnectionError
        with pytest.raises(ConnectionError):
            client.health()
