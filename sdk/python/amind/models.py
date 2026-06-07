"""amind SDK data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str = ""


class MetricsResponse(BaseModel):
    total_memories: int = 0
    graph_edges: int = 0
    pool_connections: int = 0
    pool_total_acquired: int = 0
    pool_total_reused: int = 0
    pool_circuit_open: bool = False


class StoreResponse(BaseModel):
    memory_ids: list[int] = Field(default_factory=list)
    async_refinement_scheduled: bool = False


class Memory(BaseModel):
    memory_id: int | str = 0
    content: str = ""
    agent_id: str = ""
    user_id: str = ""
    scope: str = ""
    memory_type: str = ""
    phase: str = ""
    confidence: str = ""
    importance: float = 0.0
    version: int = 0
    parent_id: int = 0
    access_count: int = 0
    created_at: int = 0
    last_accessed: int = 0
    has_embedding: bool = False
    metadata: dict = Field(default_factory=dict)


class ScoredMemory(BaseModel):
    memory_id: int | str = 0
    content: str = ""
    agent_id: str = ""
    user_id: str = ""
    scope: str = ""
    memory_type: str = ""
    phase: str = ""
    confidence: str = ""
    score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    recency_score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class RecallResponse(BaseModel):
    results: list[ScoredMemory] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted: bool = False
    invalidated_count: int = 0
    invalidated_ids: list[int] = Field(default_factory=list)


class InterceptResponse(BaseModel):
    capture_scheduled: bool = False
    captured_ids: list[int] = Field(default_factory=list)


class Message(BaseModel):
    role: str
    content: str


class SessionResponse(BaseModel):
    session_id: int


class TurnResponse(BaseModel):
    ok: bool = False
    current_intent: str = ""
    turn_count: int = 0
    fact_count: int = 0


class SessionSummary(BaseModel):
    session_id: int | str = 0
    agent_id: str = ""
    user_id: str = ""
    turn_count: int = 0
    current_intent: str = ""
    memory_count: int = 0
    fact_count: int = 0
    started_at: int = 0
    last_turn_at: int = 0
    active: bool = True


class Conflict(BaseModel):
    memory_a: str = ""
    memory_b: str = ""
    conflict_type: str = ""
    explanation: str = ""


class CoverageStats(BaseModel):
    total: int = 0
    active: int = 0
    stale: int = 0
    conflicted: int = 0
    last_updated: int = 0
    agent_id_distribution: dict[str, int] = Field(default_factory=dict)
    scope_distribution: dict[str, int] = Field(default_factory=dict)
    memory_type_distribution: dict[str, int] = Field(default_factory=dict)
    phase_distribution: dict[str, int] = Field(default_factory=dict)
    confidence_distribution: dict[str, int] = Field(default_factory=dict)


class Variable(BaseModel):
    name: str
    value: str
    default: str = ""
    type: str = ""
    mode: str = ""
    category: str = ""
    description: str = ""


class SetVarResponse(BaseModel):
    ok: bool = False
    name: str = ""
    old_value: str = ""
    new_value: str = ""


class GateLogEntry(BaseModel):
    entry_id: str = ""
    timestamp_ms: int = 0
    agent_id: str = ""
    user_id: str = ""
    content: str = ""
    decision: str = ""           # "Accepted" / "Rejected" / "Deferred"
    reason: str = ""
    marginal_value: float = 0.0
    conflict_with_id: str = "0"
    scope: str = ""
    memory_type: str = ""
    layer: str = "Raw"           # "Raw" / "Derived"
    user_metadata: dict = Field(default_factory=dict)
    memory_id: str = "0"
    resurrected_to: str = "0"
    resurrected_at_ms: int = 0
    resurrect_strategy: str = ""


class GateLogStats(BaseModel):
    accepted: int = 0
    rejected: int = 0
    deferred: int = 0
    resurrected: int = 0
    total: int = 0


class GateLogResponse(BaseModel):
    entries: list[GateLogEntry] = Field(default_factory=list)
    stats: GateLogStats = Field(default_factory=GateLogStats)
    memory_size: int = 0


class ResurrectResponse(BaseModel):
    memory_id: str = ""
    strategy: str = ""
    replaced_id: str | None = None
    note: str | None = None


class Edge(BaseModel):
    from_id: str = ""
    to_id: str = ""
    type: str = ""
    weight: float = 0.0
    created_at: int = 0


class ListMemoriesResponse(BaseModel):
    total: int = 0
    page: int = 1
    per_page: int = 50
    memories: list[Memory] = Field(default_factory=list)


class ListEdgesResponse(BaseModel):
    total: int = 0
    page: int = 1
    per_page: int = 100
    edges: list[Edge] = Field(default_factory=list)
