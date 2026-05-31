"""amind — Python SDK for Agent's Mind memory engine."""

from .client import AmindClient
from .async_client import AsyncAmindClient
from .models import (
    Conflict,
    CoverageStats,
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
from .exceptions import (
    AmindError,
    ConnectionError,
    NotFoundError,
    ServerError,
    ValidationError,
)

__version__ = "0.1.0"
__all__ = [
    "AmindClient",
    "AsyncAmindClient",
    # Models
    "Conflict",
    "CoverageStats",
    "DeleteResponse",
    "Edge",
    "GateLogEntry",
    "GateLogResponse",
    "GateLogStats",
    "ResurrectResponse",
    "HealthResponse",
    "InterceptResponse",
    "ListEdgesResponse",
    "ListMemoriesResponse",
    "Memory",
    "Message",
    "MetricsResponse",
    "RecallResponse",
    "ScoredMemory",
    "SessionResponse",
    "SessionSummary",
    "SetVarResponse",
    "StoreResponse",
    "TurnResponse",
    "Variable",
    # Exceptions
    "AmindError",
    "ConnectionError",
    "NotFoundError",
    "ServerError",
    "ValidationError",
]
