"""amind SDK exceptions."""

from __future__ import annotations


class AmindError(Exception):
    """Base exception for amind SDK."""

    def __init__(self, message: str, status_code: int = 0):
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}" if status_code else message)


class ConnectionError(AmindError):
    """Failed to connect to amind server."""


class NotFoundError(AmindError):
    """Resource not found (404)."""


class ValidationError(AmindError):
    """Invalid request (400)."""


class ServerError(AmindError):
    """Server-side error (5xx)."""
