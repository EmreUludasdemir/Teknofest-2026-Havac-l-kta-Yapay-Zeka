from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from time import sleep
from typing import Any

from src.config.settings import SequentialProtocolSettings
from src.core.frame_state import FrameEnvelope, FrameResult
from src.core.logger import StructuredLogger
from src.server.client import HttpResponse, SimpleHttpClient
from src.server.protocol import ProtocolError


class SequentialProtocolAdapter(ABC):
    """Final yarisma icin kare sirali adapter arayuzu."""

    @abstractmethod
    def login(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def open_session(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def fetch_next_frame(self) -> FrameEnvelope | None:
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

    @abstractmethod
    def close_session(self) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class FinalSequentialAdapter(SequentialProtocolAdapter):
    """Config-driven sequential adapter; batch adapter'dan bagimsizdir."""

    settings: SequentialProtocolSettings
    client: SimpleHttpClient = field(default_factory=SimpleHttpClient)
    logger: StructuredLogger | None = None
    auth_token: str | None = None
    session_id: str | None = None
    session_name: str | None = None
    current_frame: FrameEnvelope | None = None
    warmup_completed: bool = False
    _warmup_wait_logged: bool = False

    def __post_init__(self) -> None:
        base_url = self.settings.normalized_base_url
        self.url_login = f"{base_url}{self.settings.resolve_path('auth')}"
        self.url_open_session = f"{base_url}{self.settings.resolve_path('open_session')}"
        self.url_next_frame = f"{base_url}{self.settings.resolve_path('next_frame')}"
        self.url_prediction = f"{base_url}{self.settings.resolve_path('prediction')}"
        self.url_close_session = f"{base_url}{self.settings.resolve_path('close_session')}"
        self.base_url = base_url
        self.logger = self.logger or StructuredLogger()

    def _auth_headers(self) -> dict[str, str]:
        if not self.auth_token:
            raise ProtocolError("Sequential adapter giris yapmadan kullanilamaz.")
        return {"Authorization": f"Token {self.auth_token}"}

    def login(self) -> str:
        t0 = perf_counter()
        response = self.client.post_form(
            self.url_login,
            {"username": self.settings.username, "password": self.settings.password},
            timeout=self.settings.effective_timeout("request_timeout_s", self.settings.request_timeout_s),
        )
        latency_ms = round((perf_counter() - t0) * 1000.0, 3)
        if response.status_code != 200:
            self.logger.log_error(
                event="sequential_login_failed",
                adapter=type(self).__name__,
                status_code=response.status_code,
                latency_ms=latency_ms,
                diagnostics={"response_text": response.text},
            )
            raise ProtocolError(f"Sequential login basarisiz: {response.status_code}")
        payload = response.json()
        token = payload.get("token")
        if not token:
            raise ProtocolError("Sequential login cevabinda token alani yok.")
        self.auth_token = str(token)
        self.logger.log_runtime(
            event="sequential_login_succeeded",
            adapter=type(self).__name__,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return self.auth_token

    def open_session(self) -> dict[str, Any]:
        if not self.auth_token:
            self.login()
        response = self.client.post_json(
            self.url_open_session,
            {},
            headers=self._auth_headers(),
            timeout=self.settings.effective_timeout("request_timeout_s", self.settings.request_timeout_s),
        )
        if response.status_code != 200:
            raise ProtocolError(f"Sequential session acilamadi: {response.status_code} {response.text}")
        payload = response.json()
        self.session_id = str(payload.get("session_id"))
        self.session_name = str(payload.get("session_name", self.session_id))
        self.warmup_completed = False
        self._warmup_wait_logged = False
        self.logger.log_runtime(
            event="sequential_session_opened",
            adapter=type(self).__name__,
            session_name=self.session_name,
            status_code=response.status_code,
            diagnostics={"object_limit": payload.get("object_limit"), "reference_manifest": payload.get("reference_manifest", [])},
        )
        return payload

    def fetch_next_frame(self) -> FrameEnvelope | None:
        if not self.auth_token:
            self.login()
        if not self.session_id:
            self.open_session()
        first_frame_wait = not self.warmup_completed
        if first_frame_wait and not self._warmup_wait_logged:
            self.logger.log_runtime(
                event="sequential_warmup_wait_started",
                adapter=type(self).__name__,
                session_name=self.session_name,
                diagnostics={"warmup_timeout_s": self.settings.warmup_timeout_s},
            )
            self._warmup_wait_logged = True
        response = self._retry_request(
            request_label="next_frame",
            request_fn=lambda timeout: self.client.get_json(
                self.url_next_frame,
                headers=self._auth_headers(),
                timeout=timeout,
            ),
            retryable_statuses=self.settings.next_frame_retryable_statuses,
            timeout=(
                self.settings.effective_timeout("first_frame_timeout_s", self.settings.first_frame_timeout_s)
                if first_frame_wait
                else self.settings.effective_timeout("request_timeout_s", self.settings.request_timeout_s)
            ),
        )
        if response.status_code == 204:
            self.current_frame = None
            return None
        if response.status_code != 200:
            raise ProtocolError(f"Sequential frame alinamadi: {response.status_code} {response.text}")
        payload = response.json()
        frame = FrameEnvelope(
            frame_url=str(payload["frame_url"]),
            image_url=str(payload["image_url"]),
            video_name=str(payload["video_name"]),
            translation_x=float(payload["translation_x"]),
            translation_y=float(payload["translation_y"]),
            translation_z=float(payload["translation_z"]),
            health_status=str(payload["health_status"]),
            metadata={
                "session_id": payload.get("session_id"),
                "frame_id": payload.get("frame_id"),
                "camera_mode": payload.get("camera_mode"),
                "deadline_ms": payload.get("deadline_ms"),
            },
        )
        self.current_frame = frame
        if not self.warmup_completed:
            self.warmup_completed = True
            self.logger.log_runtime(
                event="sequential_warmup_completed",
                adapter=type(self).__name__,
                session_name=self.session_name,
                frame_url=frame.frame_url,
                diagnostics={"frame_id": frame.metadata.get("frame_id")},
            )
        return frame

    def resolve_image_url(self, image_url: str) -> str:
        if image_url.startswith("http://") or image_url.startswith("https://"):
            return image_url
        normalized = image_url.lstrip("/")
        if normalized.startswith("media/"):
            return f"{self.base_url}{normalized}"
        return f"{self.base_url}media/{normalized}"

    def download_image(self, frame: FrameEnvelope) -> bytes:
        response = self.client.get_bytes(
            self.resolve_image_url(frame.image_url),
            timeout=self.settings.effective_timeout("image_timeout_s", self.settings.image_timeout_s),
        )
        if response.status_code != 200:
            raise ProtocolError(f"Sequential gorsel indirilemedi: {response.status_code}")
        return response.body

    def build_wire_prediction(self, result: FrameResult) -> dict[str, Any]:
        if not self.current_frame:
            raise ProtocolError("Prediction olusturmak icin aktif frame yok.")

        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "frame_id": self.current_frame.metadata.get("frame_id"),
            "frame": result.frame_url,
            "detected_objects": [],
            "detected_translations": [],
            "detected_undefined_objects": [],
        }
        for obj in result.detected_objects:
            wire_obj: dict[str, Any] = {
                "cls": f"{self.base_url}classes/{int(obj.class_id) + 1}/",
                "landing_status": str(obj.landing_status),
                "top_left_x": str(obj.top_left_x),
                "top_left_y": str(obj.top_left_y),
                "bottom_right_x": str(obj.bottom_right_x),
                "bottom_right_y": str(obj.bottom_right_y),
            }
            # motion_status is only valid for vehicles (class_id=0) with 2026 spec values 0/1
            if int(obj.class_id) == 0 and obj.motion_status in (0, 1):
                wire_obj["motion_status"] = str(obj.motion_status)
            payload["detected_objects"].append(wire_obj)

        for translation in result.detected_translations:
            payload["detected_translations"].append(
                {
                    "translation_x": str(translation.translation_x),
                    "translation_y": str(translation.translation_y),
                    "translation_z": str(translation.translation_z),
                }
            )

        # Task 3: detected_undefined_objects per 2026 spec
        for uobj in result.detected_undefined_objects:
            payload["detected_undefined_objects"].append(
                {
                    "object_id": str(uobj.object_id),
                    "top_left_x": str(uobj.top_left_x),
                    "top_left_y": str(uobj.top_left_y),
                    "bottom_right_x": str(uobj.bottom_right_x),
                    "bottom_right_y": str(uobj.bottom_right_y),
                }
            )

        return payload

    def send_wire_prediction(self, payload: dict[str, Any]) -> HttpResponse:
        if not self.auth_token:
            self.login()
        response = self._retry_request(
            request_label="prediction",
            request_fn=lambda timeout: self.client.post_json(
                self.url_prediction,
                payload,
                headers=self._auth_headers(),
                timeout=timeout,
            ),
            retryable_statuses=self.settings.prediction_retryable_statuses,
            timeout=self.settings.effective_timeout("request_timeout_s", self.settings.request_timeout_s),
        )
        if response.status_code != 201:
            raise ProtocolError(f"Sequential prediction gonderilemedi: {response.status_code} {response.text}")
        self.logger.log_runtime(
            event="sequential_prediction_sent",
            adapter=type(self).__name__,
            session_name=self.session_name,
            frame_url=str(payload.get("frame")),
            status_code=response.status_code,
            detected_objects=len(payload.get("detected_objects", [])),
            detected_translations=len(payload.get("detected_translations", [])),
        )
        self.current_frame = None
        return response

    def close_session(self) -> None:
        if not self.auth_token or not self.session_id:
            self.current_frame = None
            self.session_id = None
            return
        response = self.client.post_json(
            self.url_close_session,
            {"session_id": self.session_id},
            headers=self._auth_headers(),
            timeout=self.settings.effective_timeout("request_timeout_s", self.settings.request_timeout_s),
        )
        if response.status_code != 200:
            raise ProtocolError(f"Sequential session kapatilamadi: {response.status_code} {response.text}")
        self.logger.log_runtime(
            event="sequential_session_closed",
            adapter=type(self).__name__,
            session_name=self.session_name,
            status_code=response.status_code,
        )
        self.current_frame = None
        self.session_id = None
        self.session_name = None
        self.warmup_completed = False
        self._warmup_wait_logged = False

    def _retry_request(
        self,
        *,
        request_label: str,
        request_fn,
        retryable_statuses: list[int],
        timeout: float,
    ) -> HttpResponse:
        max_retries = int(self.settings.retry_policy.get("max_retries", 0))
        backoff_s = float(self.settings.retry_policy.get("backoff_s", 0.0))
        last_error: str | None = None
        for attempt in range(max_retries + 1):
            try:
                response = request_fn(timeout)
            except Exception as exc:
                last_error = str(exc)
                if attempt >= max_retries:
                    break
                self.logger.log_runtime(
                    event="sequential_request_retrying",
                    adapter=type(self).__name__,
                    session_name=self.session_name,
                    fallback_mode=request_label,
                    diagnostics={"attempt": attempt + 1, "error": last_error},
                )
                sleep(backoff_s)
                continue

            if response.status_code not in retryable_statuses or attempt >= max_retries:
                return response
            self.logger.log_runtime(
                event="sequential_request_retrying",
                adapter=type(self).__name__,
                session_name=self.session_name,
                fallback_mode=request_label,
                status_code=response.status_code,
                diagnostics={"attempt": attempt + 1, "response_text": response.text},
            )
            sleep(backoff_s)

        self.logger.log_error(
            event="sequential_request_retry_exhausted",
            adapter=type(self).__name__,
            session_name=self.session_name,
            fallback_mode=request_label,
            diagnostics={"error": last_error, "max_retries": max_retries},
        )
        raise ProtocolError(f"Sequential {request_label} retry limiti doldu: {last_error}")
