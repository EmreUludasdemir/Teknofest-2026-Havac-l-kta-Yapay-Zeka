from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.core.utils import infer_modality


def _build_mock_pgm() -> bytes:
    width = 64
    height = 64
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            value = 220 if 12 <= x < 52 and 12 <= y < 52 and ((x // 4 + y // 4) % 2 == 0) else 20
            if x == y or x + y == width - 1:
                value = 255
            pixels[y * width + x] = value
    return f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(pixels)


MOCK_IMAGE_BYTES = _build_mock_pgm()


@dataclass(slots=True)
class MockFrameSpec:
    image_path: str
    video_name: str


@dataclass(slots=True)
class MockTranslationSpec:
    translation_x: float
    translation_y: float
    translation_z: float
    health_status: str


@dataclass(slots=True)
class MockSessionData:
    session_name: str
    frames: list[MockFrameSpec]
    translations: list[MockTranslationSpec]


def build_default_session(frame_count: int = 3, session_name: str = "mock_session") -> MockSessionData:
    frames = [
        MockFrameSpec(image_path=f"/{session_name}/frame_{index:06d}.jpg", video_name=session_name)
        for index in range(frame_count)
    ]
    translations = [
        MockTranslationSpec(
            translation_x=float(index),
            translation_y=float(index) * 0.5,
            translation_z=10.0 + float(index),
            health_status="1" if index < max(frame_count - 1, 1) else "0",
        )
        for index in range(frame_count)
    ]
    return MockSessionData(session_name=session_name, frames=frames, translations=translations)


@dataclass(slots=True)
class _RateWindow:
    limit_per_minute: int
    events: deque[float] = field(default_factory=deque)

    def allow(self) -> bool:
        now = time.time()
        while self.events and now - self.events[0] > 60.0:
            self.events.popleft()
        if len(self.events) >= self.limit_per_minute:
            return False
        self.events.append(now)
        return True


@dataclass(slots=True)
class MockServerState:
    username: str
    password: str
    session: MockSessionData
    mode: str = "batch"
    prediction_limit_per_minute: int = 80
    manifest_limit_per_minute: int = 5
    token: str = "mock-token"
    object_limit: int = 16
    accepted_predictions: dict[str, dict[str, Any]] = field(default_factory=dict)
    prediction_window: _RateWindow = field(init=False)
    manifest_window: _RateWindow = field(init=False)
    sequential_session_id: str | None = None
    sequential_next_index: int = 0
    sequential_waiting_frame_url: str | None = None
    sequential_open: bool = False
    sequential_ready_at: float = 0.0
    sequential_warmup_delay_s: float = 0.0
    next_frame_delay_s: float = 0.0
    prediction_delay_s: float = 0.0
    next_frame_retry_failures_remaining: int = 0
    prediction_retry_failures_remaining: int = 0
    next_frame_timeout_failures_remaining: int = 0
    next_frame_timeout_delay_s: float = 0.0

    def __post_init__(self) -> None:
        self.prediction_window = _RateWindow(limit_per_minute=self.prediction_limit_per_minute)
        self.manifest_window = _RateWindow(limit_per_minute=self.manifest_limit_per_minute)

    def reset_sequential(self) -> None:
        self.sequential_session_id = None
        self.sequential_next_index = 0
        self.sequential_waiting_frame_url = None
        self.sequential_open = False
        self.sequential_ready_at = 0.0


class OfficialRepoMockServer:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        username: str = "team",
        password: str = "password",
        session: MockSessionData | None = None,
        mode: str = "batch",
        prediction_limit_per_minute: int = 80,
        manifest_limit_per_minute: int = 5,
        sequential_warmup_delay_s: float = 0.0,
        next_frame_delay_s: float = 0.0,
        prediction_delay_s: float = 0.0,
        next_frame_retry_failures: int = 0,
        prediction_retry_failures: int = 0,
        next_frame_timeout_failures: int = 0,
        next_frame_timeout_delay_s: float = 0.0,
    ) -> None:
        self.state = MockServerState(
            username=username,
            password=password,
            session=session or build_default_session(),
            mode=mode,
            prediction_limit_per_minute=prediction_limit_per_minute,
            manifest_limit_per_minute=manifest_limit_per_minute,
            sequential_warmup_delay_s=sequential_warmup_delay_s,
            next_frame_delay_s=next_frame_delay_s,
            prediction_delay_s=prediction_delay_s,
            next_frame_retry_failures_remaining=next_frame_retry_failures,
            prediction_retry_failures_remaining=prediction_retry_failures,
            next_frame_timeout_failures_remaining=next_frame_timeout_failures,
            next_frame_timeout_delay_s=next_frame_timeout_delay_s,
        )
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        state = self.state
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, status_code: int, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_bytes(self, status_code: int, payload: bytes, content_type: str = "application/octet-stream") -> None:
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_empty(self, status_code: int) -> None:
                self.send_response(status_code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _is_authorized(self) -> bool:
                return self.headers.get("Authorization") == f"Token {state.token}"

            def _frame_payload(self) -> list[dict[str, Any]]:
                payload: list[dict[str, Any]] = []
                for index, frame in enumerate(state.session.frames, start=1):
                    payload.append(
                        {
                            "url": f"{server_ref.base_url}frames/{index}/",
                            "image_url": frame.image_path,
                            "video_name": frame.video_name,
                        }
                    )
                return payload

            def _translation_payload(self) -> list[dict[str, Any]]:
                payload: list[dict[str, Any]] = []
                for translation in state.session.translations:
                    payload.append(
                        {
                            "translation_x": translation.translation_x,
                            "translation_y": translation.translation_y,
                            "translation_z": translation.translation_z,
                            "health_status": translation.health_status,
                        }
                    )
                return payload

            def _sequential_frame_payload(self, index: int) -> dict[str, Any]:
                frame = state.session.frames[index]
                translation = state.session.translations[index]
                frame_url = f"{server_ref.base_url}frames/{index + 1}/"
                return {
                    "session_id": state.sequential_session_id,
                    "frame_id": index + 1,
                    "frame_url": frame_url,
                    "image_url": frame.image_path,
                    "video_name": frame.video_name,
                    "translation_x": translation.translation_x,
                    "translation_y": translation.translation_y,
                    "translation_z": translation.translation_z,
                    "health_status": translation.health_status,
                    "camera_mode": infer_modality(frame.video_name),
                    "deadline_ms": 1000,
                }

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/frames/":
                    if not self._is_authorized():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Unauthorized"})
                        return
                    if not state.manifest_window.allow():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "You do not have permission to perform this action."})
                        return
                    self._send_json(HTTPStatus.OK, self._frame_payload())
                    return

                if parsed.path == "/translation/":
                    if not self._is_authorized():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Unauthorized"})
                        return
                    self._send_json(HTTPStatus.OK, self._translation_payload())
                    return

                if parsed.path == "/session/":
                    if not self._is_authorized():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Unauthorized"})
                        return
                    self._send_json(HTTPStatus.OK, {"name": state.session.session_name})
                    return

                if parsed.path == "/session/next/":
                    if not self._is_authorized():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Unauthorized"})
                        return
                    if state.mode != "sequential":
                        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Sequential mode disabled"})
                        return
                    if not state.sequential_open or not state.sequential_session_id:
                        self._send_json(HTTPStatus.CONFLICT, {"detail": "Session is not open"})
                        return
                    if time.time() < state.sequential_ready_at:
                        self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "Warm-up in progress"})
                        return
                    if state.next_frame_retry_failures_remaining > 0:
                        state.next_frame_retry_failures_remaining -= 1
                        self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "Retry next_frame"})
                        return
                    if state.next_frame_timeout_failures_remaining > 0:
                        state.next_frame_timeout_failures_remaining -= 1
                        time.sleep(state.next_frame_timeout_delay_s)
                    if state.next_frame_delay_s > 0:
                        time.sleep(state.next_frame_delay_s)
                    if state.sequential_waiting_frame_url is not None:
                        self._send_json(HTTPStatus.CONFLICT, {"detail": "Prediction required before next frame"})
                        return
                    if state.sequential_next_index >= len(state.session.frames):
                        self._send_empty(HTTPStatus.NO_CONTENT)
                        return

                    payload = self._sequential_frame_payload(state.sequential_next_index)
                    state.sequential_waiting_frame_url = str(payload["frame_url"])
                    self._send_json(HTTPStatus.OK, payload)
                    return

                if parsed.path.startswith("/classes/"):
                    class_id = parsed.path.strip("/").split("/")[-1]
                    self._send_json(HTTPStatus.OK, {"id": class_id, "url": f"{server_ref.base_url}classes/{class_id}/"})
                    return

                if parsed.path.startswith("/media/"):
                    self._send_bytes(HTTPStatus.OK, MOCK_IMAGE_BYTES, content_type="image/x-portable-graymap")
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))

                if parsed.path == "/auth/":
                    form = parse_qs(raw_body.decode("utf-8"))
                    username = form.get("username", [""])[0]
                    password = form.get("password", [""])[0]
                    if username == state.username and password == state.password:
                        self._send_json(HTTPStatus.OK, {"token": state.token})
                    else:
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Login failed"})
                    return

                if parsed.path == "/prediction/":
                    if not self._is_authorized():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Unauthorized"})
                        return
                    if not state.prediction_window.allow():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "You do not have permission to perform this action."})
                        return
                    payload = json.loads(raw_body.decode("utf-8") or "{}")
                    if state.mode == "batch":
                        required_keys = {"frame", "detected_objects", "detected_translations"}
                        payload_keys = set(payload.keys())
                        if payload_keys != required_keys:
                            self._send_json(
                                HTTPStatus.BAD_REQUEST,
                                {
                                    "detail": "Batch payload field mismatch",
                                    "missing": sorted(required_keys - payload_keys),
                                    "unexpected": sorted(payload_keys - required_keys),
                                },
                            )
                            return
                        for item in payload.get("detected_objects", []):
                            if "motion_status" in item:
                                self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "motion_status not allowed in batch 2025 payload"})
                                return
                    frame_url = payload.get("frame")
                    if frame_url in state.accepted_predictions:
                        self._send_json(HTTPStatus.NOT_ACCEPTABLE, {"detail": "Prediction already exists."})
                        return
                    state.accepted_predictions[str(frame_url)] = payload
                    self._send_json(HTTPStatus.CREATED, payload)
                    return

                if parsed.path == "/session/open/":
                    if not self._is_authorized():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Unauthorized"})
                        return
                    if state.mode != "sequential":
                        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Sequential mode disabled"})
                        return
                    state.reset_sequential()
                    state.sequential_session_id = f"seq-{int(time.time() * 1000)}"
                    state.sequential_open = True
                    state.sequential_ready_at = time.time() + max(float(state.sequential_warmup_delay_s), 0.0)
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "session_id": state.sequential_session_id,
                            "session_name": state.session.session_name,
                            "object_limit": state.object_limit,
                            "reference_manifest": [],
                        },
                    )
                    return

                if parsed.path == "/session/prediction/":
                    if not self._is_authorized():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Unauthorized"})
                        return
                    if state.mode != "sequential":
                        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Sequential mode disabled"})
                        return
                    if state.prediction_retry_failures_remaining > 0:
                        state.prediction_retry_failures_remaining -= 1
                        self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "Retry prediction"})
                        return
                    if state.prediction_delay_s > 0:
                        time.sleep(state.prediction_delay_s)
                    payload = json.loads(raw_body.decode("utf-8") or "{}")
                    session_id = payload.get("session_id")
                    frame_url = payload.get("frame")
                    if not state.sequential_open or session_id != state.sequential_session_id:
                        self._send_json(HTTPStatus.CONFLICT, {"detail": "Session mismatch"})
                        return
                    if state.sequential_waiting_frame_url is None:
                        self._send_json(HTTPStatus.CONFLICT, {"detail": "No frame pending prediction"})
                        return
                    if frame_url != state.sequential_waiting_frame_url:
                        self._send_json(HTTPStatus.NOT_ACCEPTABLE, {"detail": "Unexpected frame order"})
                        return
                    if frame_url in state.accepted_predictions:
                        self._send_json(HTTPStatus.NOT_ACCEPTABLE, {"detail": "Prediction already exists."})
                        return
                    state.accepted_predictions[str(frame_url)] = payload
                    state.sequential_waiting_frame_url = None
                    state.sequential_next_index += 1
                    self._send_json(HTTPStatus.CREATED, payload)
                    return

                if parsed.path == "/session/close/":
                    if not self._is_authorized():
                        self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Unauthorized"})
                        return
                    payload = json.loads(raw_body.decode("utf-8") or "{}")
                    closed = bool(state.sequential_open and payload.get("session_id") == state.sequential_session_id)
                    state.reset_sequential()
                    self._send_json(HTTPStatus.OK, {"closed": closed})
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

        return Handler


if __name__ == "__main__":
    server = OfficialRepoMockServer()
    server.start()
    print(server.base_url)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()
