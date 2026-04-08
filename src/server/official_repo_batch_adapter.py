from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from src.config.settings import OfficialRepoSettings
from src.core.frame_state import FrameEnvelope, FrameResult
from src.core.logger import StructuredLogger
from src.core.session_state import SessionState
from src.core.utils import infer_modality
from src.server.client import HttpResponse, SimpleHttpClient
from src.server.protocol import ProtocolAdapter, ProtocolError


@dataclass(slots=True)
class OfficialRepoBatchAdapter(ProtocolAdapter):
    settings: OfficialRepoSettings
    client: SimpleHttpClient = field(default_factory=SimpleHttpClient)
    logger: StructuredLogger | None = None
    auth_token: str | None = None

    def __post_init__(self) -> None:
        self.base_url = self.settings.normalized_base_url
        self.url_login = f"{self.base_url}auth/"
        self.url_frames = f"{self.base_url}frames/"
        self.url_translations = f"{self.base_url}translation/"
        self.url_prediction = f"{self.base_url}prediction/"
        self.url_session = f"{self.base_url}session/"
        self.logger = self.logger or StructuredLogger()

    def _auth_headers(self) -> dict[str, str]:
        if not self.auth_token:
            raise ProtocolError("Adapter giris yapmadan kullanilamaz.")
        return {"Authorization": f"Token {self.auth_token}"}

    def login(self) -> str:
        t0 = perf_counter()
        response = self.client.post_form(
            self.url_login,
            {"username": self.settings.username, "password": self.settings.password},
            timeout=self.settings.request_timeout_s,
        )
        latency_ms = round((perf_counter() - t0) * 1000.0, 3)
        if response.status_code != 200:
            self.logger.log_error(
                event="adapter_login_failed",
                adapter=type(self).__name__,
                status_code=response.status_code,
                latency_ms=latency_ms,
                diagnostics={"response_text": response.text},
            )
            raise ProtocolError(f"Giris basarisiz: {response.status_code} {response.text}")
        payload = response.json()
        token = payload.get("token")
        if not token:
            self.logger.log_error(
                event="adapter_login_missing_token",
                adapter=type(self).__name__,
                status_code=response.status_code,
                latency_ms=latency_ms,
                diagnostics={"response_text": response.text},
            )
            raise ProtocolError("Giris cevabinda token alani yok.")
        self.auth_token = token
        self.logger.log_runtime(
            event="adapter_login_succeeded",
            adapter=type(self).__name__,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return token

    def fetch_session_state(self) -> SessionState:
        if not self.auth_token:
            self.login()

        t0 = perf_counter()
        frames_response = self.client.get_json(
            self.url_frames,
            headers=self._auth_headers(),
            timeout=self.settings.request_timeout_s,
        )
        translations_response = self.client.get_json(
            self.url_translations,
            headers=self._auth_headers(),
            timeout=self.settings.request_timeout_s,
        )
        latency_ms = round((perf_counter() - t0) * 1000.0, 3)

        if frames_response.status_code != 200:
            self.logger.log_error(
                event="batch_frames_fetch_failed",
                adapter=type(self).__name__,
                status_code=frames_response.status_code,
                latency_ms=latency_ms,
                diagnostics={"response_text": frames_response.text},
            )
            raise ProtocolError(f"frames/ beklenmeyen cevap: {frames_response.status_code}")
        if translations_response.status_code != 200:
            self.logger.log_error(
                event="batch_translations_fetch_failed",
                adapter=type(self).__name__,
                status_code=translations_response.status_code,
                latency_ms=latency_ms,
                diagnostics={"response_text": translations_response.text},
            )
            raise ProtocolError(f"translation/ beklenmeyen cevap: {translations_response.status_code}")

        raw_frames = frames_response.json()
        raw_translations = translations_response.json()

        if not isinstance(raw_frames, list) or not isinstance(raw_translations, list):
            self.logger.log_error(
                event="batch_manifest_invalid_shape",
                adapter=type(self).__name__,
                latency_ms=latency_ms,
                diagnostics={"frames_type": type(raw_frames).__name__, "translations_type": type(raw_translations).__name__},
            )
            raise ProtocolError("Batch manifest list formatinda donmedi.")
        if len(raw_frames) != len(raw_translations):
            self.logger.log_error(
                event="batch_manifest_length_mismatch",
                adapter=type(self).__name__,
                latency_ms=latency_ms,
                diagnostics={"frames_count": len(raw_frames), "translations_count": len(raw_translations)},
            )
            raise ProtocolError("Frames ve translations uzunluklari esit degil.")

        frames: list[FrameEnvelope] = []
        for index, (raw_frame, raw_translation) in enumerate(zip(raw_frames, raw_translations)):
            camera_mode = infer_modality(str(raw_frame.get("video_name", "")))
            frames.append(
                FrameEnvelope(
                    frame_url=str(raw_frame["url"]),
                    image_url=str(raw_frame["image_url"]),
                    video_name=str(raw_frame["video_name"]),
                    translation_x=float(raw_translation["translation_x"]),
                    translation_y=float(raw_translation["translation_y"]),
                    translation_z=float(raw_translation["translation_z"]),
                    health_status=str(raw_translation["health_status"]),
                    metadata={
                        "raw_frame": raw_frame,
                        "raw_translation": raw_translation,
                        "frame_index": index,
                        "camera_mode": camera_mode,
                    },
                )
            )

        if not frames:
            self.logger.log_error(
                event="batch_manifest_empty",
                adapter=type(self).__name__,
                latency_ms=latency_ms,
            )
            raise ProtocolError("Bos oturum manifesti alindi.")

        session = SessionState(
            session_name=frames[0].video_name,
            video_name=frames[0].video_name,
            frames=frames,
            raw_frames=raw_frames,
            raw_translations=raw_translations,
            metadata={"session_url": self.url_session},
        )
        self.logger.log_runtime(
            event="batch_manifest_fetched",
            adapter=type(self).__name__,
            session_name=session.session_name,
            video_name=session.video_name,
            status_code=200,
            latency_ms=latency_ms,
            diagnostics={"frame_count": len(frames), "translation_count": len(raw_translations)},
        )
        return session

    def resolve_image_url(self, image_url: str) -> str:
        if image_url.startswith("http://") or image_url.startswith("https://"):
            return image_url

        normalized = image_url.lstrip("/")
        if normalized.startswith("media/"):
            return f"{self.base_url}{normalized}"
        return f"{self.base_url}media/{normalized}"

    def download_image(self, frame: FrameEnvelope) -> bytes:
        t0 = perf_counter()
        response = self.client.get_bytes(
            self.resolve_image_url(frame.image_url),
            timeout=self.settings.image_timeout_s,
        )
        latency_ms = round((perf_counter() - t0) * 1000.0, 3)
        if response.status_code != 200:
            self.logger.log_error(
                event="image_download_failed",
                adapter=type(self).__name__,
                session_name=frame.video_name,
                frame_url=frame.frame_url,
                video_name=frame.video_name,
                health_status=frame.health_status,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
            raise ProtocolError(f"Gorsel indirilemedi: {response.status_code}")
        self.logger.log_runtime(
            event="image_downloaded",
            adapter=type(self).__name__,
            session_name=frame.video_name,
            frame_url=frame.frame_url,
            video_name=frame.video_name,
            health_status=frame.health_status,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return response.body

    def build_wire_prediction(self, result: FrameResult) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        t0 = perf_counter()
        response = self.client.post_json(
            self.url_prediction,
            payload,
            headers=self._auth_headers(),
            timeout=self.settings.request_timeout_s,
        )
        latency_ms = round((perf_counter() - t0) * 1000.0, 3)
        frame_url = str(payload.get("frame")) if payload.get("frame") is not None else None

        if response.status_code == 201:
            self.logger.log_runtime(
                event="prediction_sent",
                adapter=type(self).__name__,
                frame_url=frame_url,
                status_code=response.status_code,
                latency_ms=latency_ms,
                detected_objects=len(payload.get("detected_objects", [])),
                detected_translations=len(payload.get("detected_translations", [])),
            )
        else:
            self.logger.log_error(
                event="prediction_send_failed",
                adapter=type(self).__name__,
                frame_url=frame_url,
                status_code=response.status_code,
                latency_ms=latency_ms,
                detected_objects=len(payload.get("detected_objects", [])),
                detected_translations=len(payload.get("detected_translations", [])),
                diagnostics={"response_text": response.text},
            )

        return response
