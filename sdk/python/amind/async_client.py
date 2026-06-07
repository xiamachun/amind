"""amind asynchronous client."""

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
    GateLogEntry,
    GateLogResponse,
    GateLogStats,
    HealthResponse,
    InterceptResponse,
    ListEdgesResponse,
    ListMemoriesResponse,
    Memory,
    Message,
    MetricsResponse,
    RecallResponse,
    ResurrectResponse,
    ScoredMemory,
    SessionResponse,
    SessionSummary,
    SetVarResponse,
    StoreResponse,
    TurnResponse,
    Variable,
)


class AsyncAmindClient:
    """Asynchronous client for the amind REST API."""

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
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncAmindClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        try:
            resp = await self._http.request(method, path, **kwargs)
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

    async def health(self) -> HealthResponse:
        data = await self._request("GET", "/v1/health")
        return HealthResponse(**data)

    async def is_healthy(self) -> bool:
        try:
            return (await self.health()).status == "ok"
        except Exception:
            return False

    async def metrics(self) -> MetricsResponse:
        data = await self._request("GET", "/v1/metrics")
        return MetricsResponse(**data)

    # ── Memory CRUD ──────────────────────────────────────────────────────────

    async def store(
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
        data = await self._request("POST", "/v1/memories", json=payload)
        return StoreResponse(**data)

    async def recall(
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
        """Retrieve memories.

        Set fast=True to skip the LLM-based intent analysis preflight; recall
        then runs in pure embedding+keyword+graph mode (typically ~10x faster
        but loses query rewriting and entity boosting).
        """
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
        data = await self._request("POST", "/v1/memories/recall", json=payload)
        return [ScoredMemory(**m) for m in data.get("results", [])]

    async def get(self, memory_id: int) -> Memory:
        data = await self._request("GET", f"/v1/memories/{memory_id}")
        return Memory(**data)

    async def get_history(self, memory_id: int) -> list[Memory]:
        data = await self._request("GET", f"/v1/memories/{memory_id}/history")
        return [Memory(**m) for m in data.get("versions", [])]

    async def delete(self, memory_id: int) -> DeleteResponse:
        data = await self._request("DELETE", f"/v1/memories/{memory_id}")
        return DeleteResponse(**data)

    async def feedback(self, memory_id: int, action: str = "verify") -> dict:
        return await self._request("POST", f"/v1/memories/{memory_id}/feedback", json={
            "action": action,
        })

    async def list_memories(
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
        data = await self._request("GET", "/v1/memories/list", params=params)
        return ListMemoriesResponse(**data)

    # ── Intercept ────────────────────────────────────────────────────────────

    async def intercept(
        self,
        messages: list[Message] | list[dict[str, str]],
        agent_id: str = "default",
        user_id: str = "anonymous",
    ) -> InterceptResponse:
        msg_dicts = [
            m.model_dump() if isinstance(m, Message) else m
            for m in messages
        ]
        data = await self._request("POST", "/v1/intercept", json={
            "messages": msg_dicts,
            "agent_id": agent_id,
            "user_id": user_id,
        })
        return InterceptResponse(**data)

    # ── Sessions ─────────────────────────────────────────────────────────────

    async def session_start(self, agent_id: str = "default", user_id: str = "anonymous") -> SessionResponse:
        data = await self._request("POST", "/v1/sessions/start", json={
            "agent_id": agent_id,
            "user_id": user_id,
        })
        return SessionResponse(**data)

    async def session_turn(
        self,
        session_id: int,
        user_input: str = "",
        agent_response: str = "",
    ) -> TurnResponse:
        payload: dict[str, Any] = {}
        if user_input and agent_response:
            payload["user_input"] = user_input
            payload["agent_response"] = agent_response
        data = await self._request("POST", f"/v1/sessions/{session_id}/turn", json=payload)
        return TurnResponse(**data)

    async def session_close(self, session_id: int) -> dict:
        return await self._request("POST", f"/v1/sessions/{session_id}/close", json={})

    async def session_summary(self, session_id: int) -> SessionSummary:
        data = await self._request("GET", f"/v1/sessions/{session_id}/summary")
        return SessionSummary(**data)

    async def list_sessions(self) -> list[SessionSummary]:
        data = await self._request("GET", "/v1/sessions/list")
        return [SessionSummary(**s) for s in data] if isinstance(data, list) else []

    # ── MetaCognition ────────────────────────────────────────────────────────

    async def conflicts(self) -> list[Conflict]:
        data = await self._request("GET", "/v1/metacognition/conflicts")
        return [Conflict(**c) for c in data] if isinstance(data, list) else []

    async def coverage(self, agent_id: str = "") -> CoverageStats:
        params = {"agent_id": agent_id} if agent_id else {}
        data = await self._request("GET", "/v1/metacognition/coverage", params=params)
        return CoverageStats(**data)

    # ── Variables ────────────────────────────────────────────────────────────

    async def list_variables(self, like: str = "%") -> list[Variable]:
        data = await self._request("GET", "/v1/variables", params={"like": like})
        return [Variable(**v) for v in data.get("variables", [])]

    async def set_variable(self, name: str, value: str) -> SetVarResponse:
        data = await self._request("PUT", f"/v1/variables/{name}", json={"value": value})
        return SetVarResponse(**data)

    async def reload_config(self) -> dict:
        return await self._request("POST", "/v1/config/reload")

    # ── Backup ───────────────────────────────────────────────────────────────

    async def export_backup(self, backup_type: str = "memories") -> str:
        data = await self._request("GET", "/v1/backup/export", params={"type": backup_type})
        return data.get("data", "")

    async def import_backup(self, data: str, backup_type: str = "memories") -> int:
        resp = await self._request(
            "POST", "/v1/backup/import",
            params={"type": backup_type},
            json={"data": data},
        )
        return resp.get("imported", 0)

    # ── Graph ────────────────────────────────────────────────────────────────

    async def graph_edges(self, page: int = 1, per_page: int = 100) -> ListEdgesResponse:
        data = await self._request("GET", "/v1/graph/edges", params={
            "page": str(page), "per_page": str(per_page),
        })
        return ListEdgesResponse(**data)

    async def graph_neighbors(self, memory_id: int) -> list[Edge]:
        data = await self._request("GET", f"/v1/graph/neighbors/{memory_id}")
        return [Edge(**e) for e in data] if isinstance(data, list) else []

    # ── Gate Log (audit + resurrect) ─────────────────────────────────────────

    async def gate_log(
        self,
        limit: int = 100,
        decision: str | None = None,            # "Accepted" | "Rejected" | "Deferred"
        agent_id: str | None = None,
        since_ms: int | None = None,
        only_unresurrected: bool = False,
    ) -> GateLogResponse:
        params: dict[str, str] = {"limit": str(limit)}
        if decision: params["decision"] = decision
        if agent_id: params["agent_id"] = agent_id
        if since_ms is not None: params["since_ms"] = str(since_ms)
        if only_unresurrected: params["only_unresurrected"] = "1"
        data = await self._request("GET", "/v1/gate/log", params=params)
        return GateLogResponse(**data)

    async def gate_log_stats(
        self,
        agent_id: str | None = None,
        since_ms: int | None = None,
    ) -> GateLogStats:
        params: dict[str, str] = {}
        if agent_id: params["agent_id"] = agent_id
        if since_ms is not None: params["since_ms"] = str(since_ms)
        data = await self._request("GET", "/v1/gate/log/stats", params=params)
        return GateLogStats(**data)

    async def gate_resurrect(
        self,
        entry_id: int | str,
        strategy: str = "coexist",   # "coexist" | "replace_conflict" | "update_existing"
    ) -> ResurrectResponse:
        data = await self._request(
            "POST", f"/v1/gate/log/{entry_id}/resurrect",
            json={"strategy": strategy},
        )
        return ResurrectResponse(**data)
