"""amind synchronous client."""

from __future__ import annotations

from typing import Any

import httpx

from .exceptions import (
    AmindError,
    ConnectionError,
    NotFoundError,
    ServerError,
    ValidationError,
)
from .models import (
    CoverageStats,
    Conflict,
    DeleteResponse,
    Edge,
    HealthResponse,
    InterceptResponse,
    ListEdgesResponse,
    ListMemoriesResponse,
    Memory,
    Message,
    MetricsResponse,
    RecallResponse,
    ScoredMemory,
    SessionResponse,
    SessionSummary,
    SetVarResponse,
    StoreResponse,
    TurnResponse,
    Variable,
)


class AmindClient:
    """Synchronous client for the amind REST API."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        timeout: float = 30.0,
        token: str | None = None,
    ) -> None:
        self._base_url = f"http://{host}:{port}"
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "AmindClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to amind at {self._base_url}") from e

        if resp.status_code == 404:
            body = resp.json() if resp.content else {}
            raise NotFoundError(body.get("error", "not found"), 404)
        if resp.status_code == 400:
            body = resp.json() if resp.content else {}
            raise ValidationError(body.get("error", "bad request"), 400)
        if resp.status_code >= 500:
            body = resp.json() if resp.content else {}
            raise ServerError(body.get("error", "server error"), resp.status_code)
        if resp.status_code >= 400:
            body = resp.json() if resp.content else {}
            raise AmindError(body.get("error", "unknown error"), resp.status_code)

        if not resp.content:
            return {}
        return resp.json()

    # ── Health & Metrics ─────────────────────────────────────────────────────

    def health(self) -> HealthResponse:
        data = self._request("GET", "/v1/health")
        return HealthResponse(**data)

    def is_healthy(self) -> bool:
        try:
            return self.health().status == "ok"
        except Exception:
            return False

    def metrics(self) -> MetricsResponse:
        data = self._request("GET", "/v1/metrics")
        return MetricsResponse(**data)

    # ── Memory CRUD ──────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        agent_id: str = "default",
        user_id: str = "anonymous",
        scope: str = "private",
        memory_type: str = "ephemeral",
        metadata: dict | None = None,
    ) -> StoreResponse:
        payload: dict[str, Any] = {
            "content": content,
            "agent_id": agent_id,
            "user_id": user_id,
            "scope": scope,
            "memory_type": memory_type,
        }
        if metadata:
            payload["metadata"] = metadata
        data = self._request("POST", "/v1/memories", json=payload)
        return StoreResponse(**data)

    def recall(
        self,
        query: str,
        agent_id: str = "default",
        user_id: str = "anonymous",
        scope: str = "private",
        memory_type: str = "ephemeral",
        top_k: int = 10,
        filters: dict | None = None,
        fast: bool = False,
    ) -> list[ScoredMemory]:
        payload: dict[str, Any] = {
            "query": query,
            "agent_id": agent_id,
            "user_id": user_id,
            "scope": scope,
            "memory_type": memory_type,
            "top_k": top_k,
        }
        if filters:
            payload["filters"] = filters
        if fast:
            payload["fast"] = True
        data = self._request("POST", "/v1/memories/recall", json=payload)
        return [ScoredMemory(**m) for m in data.get("results", [])]

    def get(self, memory_id: int) -> Memory:
        data = self._request("GET", f"/v1/memories/{memory_id}")
        return Memory(**data)

    def get_history(self, memory_id: int) -> list[Memory]:
        data = self._request("GET", f"/v1/memories/{memory_id}/history")
        return [Memory(**m) for m in data.get("versions", [])]

    def delete(self, memory_id: int) -> DeleteResponse:
        data = self._request("DELETE", f"/v1/memories/{memory_id}")
        return DeleteResponse(**data)

    def feedback(self, memory_id: int, action: str = "verify") -> dict:
        return self._request("POST", f"/v1/memories/{memory_id}/feedback", json={
            "action": action,
        })

    def list_memories(
        self,
        page: int = 1,
        per_page: int = 50,
        agent_id: str | None = None,
        user_id: str | None = None,
        scope: str | None = None,
        memory_type: str | None = None,
        phase: str | None = None,
        query: str | None = None,
    ) -> ListMemoriesResponse:
        params: dict[str, str] = {"page": str(page), "per_page": str(per_page)}
        if agent_id:
            params["agent_id"] = agent_id
        if user_id:
            params["user_id"] = user_id
        if scope:
            params["scope"] = scope
        if memory_type:
            params["memory_type"] = memory_type
        if phase:
            params["phase"] = phase
        if query:
            params["q"] = query
        data = self._request("GET", "/v1/memories/list", params=params)
        return ListMemoriesResponse(**data)

    # ── Intercept ────────────────────────────────────────────────────────────

    def intercept(
        self,
        messages: list[Message] | list[dict[str, str]],
        agent_id: str = "default",
        user_id: str = "anonymous",
    ) -> InterceptResponse:
        msg_dicts = [
            m.model_dump() if isinstance(m, Message) else m
            for m in messages
        ]
        data = self._request("POST", "/v1/intercept", json={
            "messages": msg_dicts,
            "agent_id": agent_id,
            "user_id": user_id,
        })
        return InterceptResponse(**data)

    # ── Sessions ─────────────────────────────────────────────────────────────

    def session_start(self, agent_id: str = "default", user_id: str = "anonymous") -> SessionResponse:
        data = self._request("POST", "/v1/sessions/start", json={
            "agent_id": agent_id,
            "user_id": user_id,
        })
        return SessionResponse(**data)

    def session_turn(
        self,
        session_id: int,
        user_input: str = "",
        agent_response: str = "",
    ) -> TurnResponse:
        payload: dict[str, Any] = {}
        if user_input and agent_response:
            payload["user_input"] = user_input
            payload["agent_response"] = agent_response
        data = self._request("POST", f"/v1/sessions/{session_id}/turn", json=payload)
        return TurnResponse(**data)

    def session_close(self, session_id: int) -> dict:
        return self._request("POST", f"/v1/sessions/{session_id}/close", json={})

    def session_summary(self, session_id: int) -> SessionSummary:
        data = self._request("GET", f"/v1/sessions/{session_id}/summary")
        return SessionSummary(**data)

    def list_sessions(self) -> list[SessionSummary]:
        data = self._request("GET", "/v1/sessions/list")
        return [SessionSummary(**s) for s in data] if isinstance(data, list) else []

    # ── MetaCognition ────────────────────────────────────────────────────────

    def conflicts(self) -> list[Conflict]:
        data = self._request("GET", "/v1/metacognition/conflicts")
        return [Conflict(**c) for c in data] if isinstance(data, list) else []

    def coverage(self, agent_id: str = "") -> CoverageStats:
        params = {"agent_id": agent_id} if agent_id else {}
        data = self._request("GET", "/v1/metacognition/coverage", params=params)
        return CoverageStats(**data)

    # ── Variables ────────────────────────────────────────────────────────────

    def list_variables(self, like: str = "%") -> list[Variable]:
        data = self._request("GET", "/v1/variables", params={"like": like})
        return [Variable(**v) for v in data.get("variables", [])]

    def set_variable(self, name: str, value: str) -> SetVarResponse:
        data = self._request("PUT", f"/v1/variables/{name}", json={"value": value})
        return SetVarResponse(**data)

    def reload_config(self) -> dict:
        return self._request("POST", "/v1/config/reload")

    # ── Backup ───────────────────────────────────────────────────────────────

    def export_backup(self, backup_type: str = "memories") -> str:
        data = self._request("GET", "/v1/backup/export", params={"type": backup_type})
        return data.get("data", "")

    def import_backup(self, data: str, backup_type: str = "memories") -> int:
        resp = self._request(
            "POST", "/v1/backup/import",
            params={"type": backup_type},
            json={"data": data},
        )
        return resp.get("imported", 0)

    # ── Graph ────────────────────────────────────────────────────────────────

    def graph_edges(self, page: int = 1, per_page: int = 100) -> ListEdgesResponse:
        data = self._request("GET", "/v1/graph/edges", params={
            "page": str(page), "per_page": str(per_page),
        })
        return ListEdgesResponse(**data)

    def graph_neighbors(self, memory_id: int) -> list[Edge]:
        data = self._request("GET", f"/v1/graph/neighbors/{memory_id}")
        return [Edge(**e) for e in data] if isinstance(data, list) else []
