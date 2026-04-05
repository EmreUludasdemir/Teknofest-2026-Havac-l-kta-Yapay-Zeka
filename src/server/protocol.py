from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.frame_state import FrameEnvelope, FrameResult
from src.core.session_state import SessionState
from src.server.client import HttpResponse


class ProtocolError(RuntimeError):
    """Protokol seviyesinde beklenmeyen durum."""


class ProtocolAdapter(ABC):
    @abstractmethod
    def login(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fetch_session_state(self) -> SessionState:
        raise NotImplementedError

    @abstractmethod
    def download_image(self, frame: FrameEnvelope) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def build_wire_prediction(self, result: FrameResult) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def send_wire_prediction(self, payload: dict[str, Any]) -> HttpResponse:
        raise NotImplementedError
