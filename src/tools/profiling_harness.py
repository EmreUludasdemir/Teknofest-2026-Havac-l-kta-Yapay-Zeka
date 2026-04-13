from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.core.logger import StructuredLogger
from src.core.utils import percentile
from src.core.vision import decode_image_bytes, is_cv2_available
from src.task1.detector import LocalModelDetectorBackend
from src.tools.report_paths import PROFILING_REPORTS_DIR
from src.tools.vram_monitor import VramSnapshot, query_vram, recovery_mb

if is_cv2_available():  # pragma: no branch - ortama bagli
    from src.core.vision import cv2, np
else:  # pragma: no cover - cv2 yoksa
    cv2 = None
    np = None


@dataclass(slots=True)
class CandidateResult:
    task_name: str
    candidate_name: str
    available: bool
    install_risk: str
    export_readiness: str
    exception_rate: float
    cold_load_latency_ms: float | None
    warm_p50_latency_ms: float | None
    warm_p95_latency_ms: float | None
    peak_vram_mb: float | None
    unload_recovery_mb: float | None
    diagnostics: dict[str, Any]


class ProfilingCandidate(ABC):
    task_name: str
    candidate_name: str

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, str]:
        raise NotImplementedError


@dataclass(slots=True)
class OptionalDependencyCandidate(ProfilingCandidate):
    task_name: str
    candidate_name: str
    dependency_name: str
    export_readiness: str

    def available(self) -> tuple[bool, str]:
        return importlib.util.find_spec(self.dependency_name) is not None, "optional_dependency"

    def load(self) -> None:
        return

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {"status": "dependency_only_probe"}

    def unload(self) -> None:
        gc.collect()

    def describe(self) -> dict[str, str]:
        return {"install_risk": "medium", "export_readiness": self.export_readiness}


@dataclass(slots=True)
class LocalTask1ModelCandidate(ProfilingCandidate):
    task_name: str
    candidate_name: str
    runtime_settings: MvpRuntimeSettings
    backend: LocalModelDetectorBackend = field(init=False)

    def __post_init__(self) -> None:
        self.backend = LocalModelDetectorBackend(
            runtime_settings=self.runtime_settings,
            backend_name=self.candidate_name,
            model_path=self.runtime_settings.resolve_task1_model_path(self.candidate_name),
        )

    def available(self) -> tuple[bool, str]:
        return self.backend.is_available(), self.backend.availability_error or "local_model_ready"

    def load(self) -> None:
        self.backend.load()

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        detections = self.backend.detect(
            sample["frame"],
            sample["image_bytes"],
            decoded_frame=sample["decoded_frame"],
        )
        return {
            "detection_count": len(detections),
            "backend_name": self.candidate_name,
            "model_path": str(self.backend._resolve_model_path()),
            "runtime_device": str(self.backend.runtime_device or "auto"),
        }

    def unload(self) -> None:
        self.backend.unload()

    def describe(self) -> dict[str, str]:
        return {"install_risk": "medium", "export_readiness": "good"}


@dataclass(slots=True)
class OpenCvTask2Candidate(ProfilingCandidate):
    task_name: str
    candidate_name: str
    mode: str

    def available(self) -> tuple[bool, str]:
        return is_cv2_available(), "opencv_runtime"

    def load(self) -> None:
        return

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not is_cv2_available():
            raise RuntimeError("cv2 unavailable")
        previous = sample["previous_gray"]
        current = sample["current_gray"]
        if self.mode == "phase_correlation":
            shift, response = cv2.phaseCorrelate(previous.astype(np.float32), current.astype(np.float32))
            return {"shift_x": float(shift[0]), "shift_y": float(shift[1]), "confidence": float(response)}
        if self.mode == "lucas_kanade_flow":
            features = cv2.goodFeaturesToTrack(previous, maxCorners=32, qualityLevel=0.01, minDistance=6)
            if features is None:
                return {"point_count": 0, "confidence": 0.0}
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, features, None)
            valid = status.reshape(-1) == 1 if status is not None else []
            if next_points is None or not np.any(valid):
                return {"point_count": 0, "confidence": 0.0}
            deltas = next_points.reshape(-1, 2)[valid] - features.reshape(-1, 2)[valid]
            median = np.median(deltas, axis=0)
            return {"shift_x": float(median[0]), "shift_y": float(median[1]), "point_count": int(np.sum(valid)), "confidence": 0.6}
        orb = cv2.ORB_create(nfeatures=128)
        keypoints_a, descriptors_a = orb.detectAndCompute(previous, None)
        keypoints_b, descriptors_b = orb.detectAndCompute(current, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return {"match_count": 0, "confidence": 0.0}
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_a, descriptors_b)
        return {"match_count": len(matches), "confidence": min(len(matches) / 30.0, 1.0)}

    def unload(self) -> None:
        gc.collect()

    def describe(self) -> dict[str, str]:
        return {"install_risk": "low", "export_readiness": "n/a"}


@dataclass(slots=True)
class OpenCvTask3Candidate(ProfilingCandidate):
    task_name: str
    candidate_name: str
    mode: str

    def available(self) -> tuple[bool, str]:
        return is_cv2_available(), "opencv_runtime"

    def load(self) -> None:
        return

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not is_cv2_available():
            raise RuntimeError("cv2 unavailable")
        reference = sample["reference_gray"]
        frame = sample["current_gray"]
        if self.mode == "template_ncc_verifier":
            response = cv2.matchTemplate(frame, reference, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(response)
            return {"score": float(max_value), "location": list(max_location)}

        feature_factory = cv2.ORB_create if self.mode == "orb_bf_homography" else cv2.AKAZE_create
        detector = feature_factory()
        ref_keypoints, ref_descriptors = detector.detectAndCompute(reference, None)
        frame_keypoints, frame_descriptors = detector.detectAndCompute(frame, None)
        if ref_descriptors is None or frame_descriptors is None:
            return {"inlier_count": 0, "score": 0.0}
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(ref_descriptors, frame_descriptors)
        return {"inlier_count": len(matches), "score": min(len(matches) / 25.0, 1.0)}

    def unload(self) -> None:
        gc.collect()

    def describe(self) -> dict[str, str]:
        return {"install_risk": "low", "export_readiness": "n/a"}


def _base_runtime_settings(runtime_settings: MvpRuntimeSettings | None = None) -> MvpRuntimeSettings:
    settings = runtime_settings or MvpRuntimeSettings()
    return replace(
        settings,
        task1_candidate_paths=dict(settings.task1_candidate_paths),
    )


def _make_synthetic_samples() -> dict[str, Any]:
    frame = FrameEnvelope(
        frame_url="http://profiling/frames/1/",
        image_url="/profiling/frame.png",
        video_name="profiling_rgb",
        translation_x=0.0,
        translation_y=0.0,
        translation_z=10.0,
        health_status="1",
        metadata={"image_width": 640, "image_height": 512, "frame_index": 1},
    )
    sample: dict[str, Any] = {
        "frame": frame,
        "image_bytes": b"",
        "decoded_frame": DecodedFrame(
            bgr=None,
            gray=None,
            width=640,
            height=512,
            channel_count=0,
            modality="rgb",
            frame_index=1,
        ),
    }
    if not is_cv2_available():
        return sample

    image = np.zeros((512, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 120), (220, 240), (255, 255, 255), thickness=-1)
    cv2.circle(image, (420, 180), 40, (220, 220, 220), thickness=-1)
    cv2.line(image, (300, 300), (500, 360), (180, 180, 180), thickness=8)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return sample

    image_bytes = encoded.tobytes()
    decoded_frame = decode_image_bytes(frame, image_bytes)
    previous = np.zeros((128, 128), dtype=np.uint8)
    previous[36:72, 40:78] = 255
    current = np.zeros((128, 128), dtype=np.uint8)
    current[40:76, 46:84] = 255
    reference = previous[32:80, 32:80].copy()
    sample.update(
        {
            "frame": frame,
            "image_bytes": image_bytes,
            "decoded_frame": decoded_frame,
            "previous_gray": previous,
            "current_gray": current,
            "reference_gray": reference,
        }
    )
    return sample


def build_candidate_registry(
    *,
    runtime_settings: MvpRuntimeSettings | None = None,
    task: str | None = None,
    candidate: str | None = None,
    model_path: str | Path | None = None,
) -> list[ProfilingCandidate]:
    settings = _base_runtime_settings(runtime_settings)
    if model_path is not None and candidate is not None:
        settings.task1_candidate_paths[candidate] = str(model_path)

    registry: list[ProfilingCandidate] = [
        LocalTask1ModelCandidate("task1", "yolo11n", settings),
        LocalTask1ModelCandidate("task1", "yolo26n", settings),
        OptionalDependencyCandidate("task1", "small_onnx", "onnxruntime", "good"),
        OpenCvTask2Candidate("task2", "phase_correlation", "phase_correlation"),
        OpenCvTask2Candidate("task2", "lucas_kanade_flow", "lucas_kanade_flow"),
        OpenCvTask2Candidate("task2", "orb_shift_estimator", "orb_shift_estimator"),
        OpenCvTask3Candidate("task3", "orb_bf_homography", "orb_bf_homography"),
        OpenCvTask3Candidate("task3", "akaze_bf_homography", "akaze_bf_homography"),
        OpenCvTask3Candidate("task3", "template_ncc_verifier", "template_ncc_verifier"),
    ]
    filtered = [
        item
        for item in registry
        if (task is None or item.task_name == task) and (candidate is None or item.candidate_name == candidate)
    ]
    return filtered


def evaluate_candidates(
    *,
    runtime_settings: MvpRuntimeSettings | None = None,
    output_dir: str | Path = PROFILING_REPORTS_DIR,
    mode: str = "smoke",
    task: str | None = None,
    candidate: str | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    settings = _base_runtime_settings(runtime_settings)
    logger = StructuredLogger()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample = _make_synthetic_samples()
    results: list[CandidateResult] = []

    for registry_item in build_candidate_registry(
        runtime_settings=settings,
        task=task,
        candidate=candidate,
        model_path=model_path,
    ):
        available, availability_stage = registry_item.available()
        description = registry_item.describe()
        before_load = query_vram(settings.profiling_gpu_query_cmd)
        cold_load_latency_ms: float | None = None
        warm_latencies: list[float] = []
        peak_vram: float | None = before_load.used_mb
        exception_count = 0
        diagnostics: dict[str, Any] = {"availability_stage": availability_stage, "smoke_replay_success": False}

        if available:
            load_t0 = perf_counter()
            try:
                registry_item.load()
            except Exception as exc:
                available = False
                diagnostics["load_error"] = str(exc)
                exception_count += 1
            cold_load_latency_ms = round((perf_counter() - load_t0) * 1000.0, 3)

        after_load = query_vram(settings.profiling_gpu_query_cmd)
        peak_vram = max_known_vram(peak_vram, after_load)
        if available:
            iterations = 2 if mode == "smoke" else 5
            for _ in range(iterations):
                t0 = perf_counter()
                try:
                    infer_output = registry_item.infer(sample)
                    diagnostics["last_output"] = infer_output
                    diagnostics["smoke_replay_success"] = True
                except Exception as exc:
                    exception_count += 1
                    diagnostics["infer_error"] = str(exc)
                    break
                warm_latencies.append(round((perf_counter() - t0) * 1000.0, 3))
                peak_vram = max_known_vram(peak_vram, query_vram(settings.profiling_gpu_query_cmd))

        before_unload = query_vram(settings.profiling_gpu_query_cmd)
        try:
            registry_item.unload()
        except Exception as exc:
            diagnostics["unload_error"] = str(exc)
            exception_count += 1
        gc.collect()
        after_unload = query_vram(settings.profiling_gpu_query_cmd)

        exception_rate = round(exception_count / max(len(warm_latencies) + (1 if cold_load_latency_ms is not None else 0), 1), 4)
        result = CandidateResult(
            task_name=registry_item.task_name,
            candidate_name=registry_item.candidate_name,
            available=available,
            install_risk=description["install_risk"],
            export_readiness=description["export_readiness"],
            exception_rate=exception_rate,
            cold_load_latency_ms=cold_load_latency_ms,
            warm_p50_latency_ms=percentile(warm_latencies, 50.0) if warm_latencies else None,
            warm_p95_latency_ms=percentile(warm_latencies, 95.0) if warm_latencies else None,
            peak_vram_mb=peak_vram,
            unload_recovery_mb=recovery_mb(before_unload, after_unload),
            diagnostics={
                **diagnostics,
                "profiling_stage": mode,
                "gpu_metrics_unavailable": not after_unload.available,
                "mean_latency_ms": round(mean(warm_latencies), 3) if warm_latencies else None,
            },
        )
        logger.log_runtime(
            event="profiling_candidate_evaluated",
            adapter="ProfilingHarness",
            diagnostics={
                "candidate_name": result.candidate_name,
                "task_name": result.task_name,
                "profiling_stage": mode,
                "available": result.available,
                **result.diagnostics,
            },
        )
        results.append(result)

    payload = {
        "mode": mode,
        "task": task,
        "candidate": candidate,
        "env_name": settings.task1_env_name,
        "device": settings.task1_device or "auto",
        "results": [candidate_result_to_dict(result) for result in results],
        "task1_recommendation": recommend_task1_candidates(results),
    }
    (output_path / "profiling_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_path / "profiling_table.md").write_text(render_markdown_table(results), encoding="utf-8")
    _write_task1_profile(outputs_dir=output_path, results=results, payload=payload, env_name=settings.task1_env_name)
    return payload


def _write_task1_profile(*, outputs_dir: Path, results: list[CandidateResult], payload: dict[str, Any], env_name: str) -> None:
    task1_results = [item for item in results if item.task_name == "task1"]
    if not task1_results:
        return
    task1_payload = {
        "results": [candidate_result_to_dict(item) for item in task1_results],
        "recommendation": recommend_task1_candidates(task1_results),
        "env_name": env_name,
    }
    (outputs_dir / "task1_real_profile.json").write_text(json.dumps(task1_payload, indent=2), encoding="utf-8")
    (outputs_dir / "task1_real_profile.md").write_text(render_markdown_table(task1_results), encoding="utf-8")
    (outputs_dir / f"task1_real_profile_{env_name}.json").write_text(json.dumps(task1_payload, indent=2), encoding="utf-8")
    (outputs_dir / f"task1_real_profile_{env_name}.md").write_text(render_markdown_table(task1_results), encoding="utf-8")
    write_task1_comparison(outputs_dir)


def write_task1_comparison(outputs_dir: Path) -> None:
    cpu_path = outputs_dir / "task1_real_profile_cpu.json"
    gpu_path = outputs_dir / "task1_real_profile_gpu.json"
    if not cpu_path.exists() or not gpu_path.exists():
        return
    cpu_payload = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu_payload = json.loads(gpu_path.read_text(encoding="utf-8"))
    comparison_lines = [
        "| Candidate | Env | Available | Runtime | Cold Load ms | P50 ms | P95 ms | Peak VRAM MB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for env_name, payload in (("cpu", cpu_payload), ("gpu", gpu_payload)):
        for result in payload.get("results", []):
            diagnostics = result.get("diagnostics", {})
            comparison_lines.append(
                "| {candidate} | {env_name} | {available} | {runtime} | {cold} | {p50} | {p95} | {peak} |".format(
                    candidate=result.get("candidate_name"),
                    env_name=env_name,
                    available="yes" if result.get("available") else "no",
                    runtime=diagnostics.get("last_output", {}).get("runtime_device", "-"),
                    cold=result.get("cold_load_latency_ms", "-"),
                    p50=result.get("warm_p50_latency_ms", "-"),
                    p95=result.get("warm_p95_latency_ms", "-"),
                    peak=result.get("peak_vram_mb", "-"),
                )
            )
    (outputs_dir / "task1_real_profile_comparison.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")


def recommend_task1_candidates(results: list[CandidateResult]) -> dict[str, Any]:
    task1_results = [item for item in results if item.task_name == "task1"]
    viable = [
        item
        for item in task1_results
        if item.available and bool(item.diagnostics.get("smoke_replay_success"))
    ]
    viable.sort(
        key=lambda item: (
            item.peak_vram_mb if item.peak_vram_mb is not None else float("inf"),
            item.warm_p95_latency_ms if item.warm_p95_latency_ms is not None else float("inf"),
            item.install_risk,
        )
    )
    return {
        "primary": viable[0].candidate_name if viable else None,
        "model_fallback_candidate": viable[1].candidate_name if len(viable) > 1 else None,
        "operational_fallback": "synthetic",
    }


def max_known_vram(current_peak: float | None, snapshot: VramSnapshot) -> float | None:
    if snapshot.used_mb is None:
        return current_peak
    if current_peak is None:
        return snapshot.used_mb
    return max(current_peak, snapshot.used_mb)


def render_markdown_table(results: list[CandidateResult]) -> str:
    lines = [
        "| Task | Candidate | Available | Smoke Success | Install Risk | Cold Load ms | P50 ms | P95 ms | Peak VRAM MB | Recovery MB | Export |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            "| {task} | {candidate} | {available} | {smoke} | {risk} | {cold} | {p50} | {p95} | {peak} | {recovery} | {export} |".format(
                task=item.task_name,
                candidate=item.candidate_name,
                available="yes" if item.available else "no",
                smoke="yes" if item.diagnostics.get("smoke_replay_success") else "no",
                risk=item.install_risk,
                cold=item.cold_load_latency_ms if item.cold_load_latency_ms is not None else "-",
                p50=item.warm_p50_latency_ms if item.warm_p50_latency_ms is not None else "-",
                p95=item.warm_p95_latency_ms if item.warm_p95_latency_ms is not None else "-",
                peak=item.peak_vram_mb if item.peak_vram_mb is not None else "-",
                recovery=item.unload_recovery_mb if item.unload_recovery_mb is not None else "-",
                export=item.export_readiness,
            )
        )
    return "\n".join(lines) + "\n"


def candidate_result_to_dict(result: CandidateResult) -> dict[str, Any]:
    return {
        "task_name": result.task_name,
        "candidate_name": result.candidate_name,
        "available": result.available,
        "install_risk": result.install_risk,
        "export_readiness": result.export_readiness,
        "exception_rate": result.exception_rate,
        "cold_load_latency_ms": result.cold_load_latency_ms,
        "warm_p50_latency_ms": result.warm_p50_latency_ms,
        "warm_p95_latency_ms": result.warm_p95_latency_ms,
        "peak_vram_mb": result.peak_vram_mb,
        "unload_recovery_mb": result.unload_recovery_mb,
        "diagnostics": result.diagnostics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEKNOFEST Faz 5 profiling harness")
    parser.add_argument("--mode", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--output-dir", default=str(PROFILING_REPORTS_DIR))
    parser.add_argument("--task", choices=("task1", "task2", "task3"))
    parser.add_argument("--candidate")
    parser.add_argument("--model-path")
    parser.add_argument("--device")
    parser.add_argument("--env-name", default="cpu")
    args = parser.parse_args(argv)
    settings = MvpRuntimeSettings(task1_device=args.device, task1_env_name=args.env_name)
    evaluate_candidates(
        runtime_settings=settings,
        output_dir=args.output_dir,
        mode=args.mode,
        task=args.task,
        candidate=args.candidate,
        model_path=args.model_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
