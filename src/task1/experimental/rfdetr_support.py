from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.core.frame_state import CanonicalDetection


DEFAULT_MODEL_ROOT = Path(r"C:\Users\Emre\.teknofest_models")
DEFAULT_DATASET_ROOTS = (
    Path(r"C:\TEKNOFEST\data"),
    Path(r"C:\Users\Emre\Downloads"),
    Path(r"C:\Users\Emre\Desktop"),
    Path(r"C:\Users\Emre\Documents"),
)
DEFAULT_PYTHON_CANDIDATES = (
    Path(sys.executable),
    Path(r"C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe"),
    Path(r"C:\Users\Emre\AppData\Local\Programs\Python\Python312\python.exe"),
)
DEFAULT_OPTIONAL_DEPENDENCIES = (
    "cv2",
    "ultralytics",
    "torch",
    "timm",
    "transformers",
    "sahi",
    "onnxruntime",
    "tensorrt",
)

VISDRONE_TO_TASK1_CLASS: dict[int, int | None] = {
    0: None,
    1: 1,
    2: 1,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: None,
}
UAVDT_TO_TASK1_CLASS: dict[str, int | None] = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "vehicle": 0,
    "van": 0,
    "pedestrian": 1,
    "person": 1,
}


@dataclass(slots=True)
class PythonRuntimeProbe:
    selected_path: str | None
    version: str | None
    is_repo_compatible: bool
    checked: list[dict[str, str]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_path": self.selected_path,
            "version": self.version,
            "is_repo_compatible": self.is_repo_compatible,
            "checked": self.checked,
            "issues": self.issues,
        }


@dataclass(slots=True)
class ArtifactProbe:
    baseline_yolo26n_paths: list[str] = field(default_factory=list)
    control_yolo11n_paths: list[str] = field(default_factory=list)
    rtdetr_candidate_paths: list[str] = field(default_factory=list)
    rfdetr_candidate_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_yolo26n_paths": self.baseline_yolo26n_paths,
            "control_yolo11n_paths": self.control_yolo11n_paths,
            "rtdetr_candidate_paths": self.rtdetr_candidate_paths,
            "rfdetr_candidate_paths": self.rfdetr_candidate_paths,
        }


@dataclass(slots=True)
class DatasetConnection:
    name: str
    root: str
    adapter_name: str
    supports_label_parsing: bool
    canonical_class_coverage: list[int]
    notes: str
    labels_format: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "root": self.root,
            "adapter_name": self.adapter_name,
            "supports_label_parsing": self.supports_label_parsing,
            "canonical_class_coverage": self.canonical_class_coverage,
            "notes": self.notes,
            "labels_format": self.labels_format,
        }


@dataclass(frozen=True, slots=True)
class TileWindow:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


def discover_python_runtime(
    candidate_paths: tuple[Path, ...] = DEFAULT_PYTHON_CANDIDATES,
    *,
    min_version: tuple[int, int] = (3, 10),
) -> PythonRuntimeProbe:
    checked: list[dict[str, str]] = []
    compatible: list[tuple[Path, tuple[int, int, int], str]] = []
    issues: list[str] = []
    seen: set[str] = set()
    for candidate in candidate_paths:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if not candidate.exists():
            checked.append({"path": normalized, "status": "missing"})
            continue
        try:
            completed = subprocess.run(
                [
                    str(candidate),
                    "-c",
                    "import json,sys; print(json.dumps({'version': list(sys.version_info[:3]), 'executable': sys.executable}))",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10.0,
            )
            payload = json.loads(completed.stdout.strip())
            version = tuple(int(value) for value in payload["version"])
            checked.append(
                {
                    "path": normalized,
                    "status": "ok",
                    "version": ".".join(str(value) for value in version),
                    "reported_executable": str(payload.get("executable", normalized)),
                }
            )
            if version[:2] >= min_version:
                compatible.append((candidate, version, str(payload.get("executable", normalized))))
        except Exception as exc:  # pragma: no cover - environment dependent
            checked.append({"path": normalized, "status": f"failed:{exc.__class__.__name__}"})

    if compatible:
        selected = compatible[0]
        return PythonRuntimeProbe(
            selected_path=selected[2],
            version=".".join(str(value) for value in selected[1]),
            is_repo_compatible=True,
            checked=checked,
            issues=issues,
        )

    issues.append("no_python_3_10_plus_runtime_found")
    return PythonRuntimeProbe(
        selected_path=None,
        version=None,
        is_repo_compatible=False,
        checked=checked,
        issues=issues,
    )


def discover_local_task1_artifacts(model_root: Path = DEFAULT_MODEL_ROOT) -> ArtifactProbe:
    if not model_root.exists():
        return ArtifactProbe()
    yolo26n_paths = _gather_paths(model_root, ("*yolo26n*.pt", "*yolo26n*.onnx", "*yolo26n*.engine"))
    yolo11n_paths = _gather_paths(model_root, ("*yolo11n*.pt", "*yolo11n*.onnx", "*yolo11n*.engine"))
    rtdetr_paths = _gather_paths(model_root, ("*rtdetr*", "*rt-detr*"))
    rfdetr_paths = _gather_paths(model_root, ("*rfdetr*", "*rf-detr*"))
    return ArtifactProbe(
        baseline_yolo26n_paths=yolo26n_paths,
        control_yolo11n_paths=yolo11n_paths,
        rtdetr_candidate_paths=rtdetr_paths,
        rfdetr_candidate_paths=rfdetr_paths,
    )


def discover_task1_datasets(
    search_roots: tuple[Path, ...] = DEFAULT_DATASET_ROOTS,
    *,
    explicit_root: Path | None = None,
    explicit_labels_format: str | None = None,
) -> list[DatasetConnection]:
    connections: list[DatasetConnection] = []
    seen_roots: set[str] = set()
    if explicit_root is not None:
        explicit_connection = connect_explicit_dataset(explicit_root, explicit_labels_format)
        if explicit_connection is not None:
            normalized = explicit_connection.root
            seen_roots.add(normalized)
            connections.append(explicit_connection)

    for root in search_roots:
        if not root.exists():
            continue
        for current, dirs, _ in os.walk(root):
            current_path = Path(current)
            for directory_name in list(dirs):
                candidate_root = current_path / directory_name
                if directory_name == "VisDrone2019-DET":
                    normalized = str(candidate_root)
                    if normalized not in seen_roots:
                        seen_roots.add(normalized)
                        connections.append(
                            DatasetConnection(
                                name="VisDrone2019-DET",
                                root=normalized,
                                adapter_name="visdrone_det",
                                supports_label_parsing=True,
                                canonical_class_coverage=[0, 1],
                                notes="Vehicle/human coverage only; UAP/UAI remain uncovered.",
                                labels_format="visdrone",
                            )
                        )
                if directory_name == "UAVDT":
                    normalized = str(candidate_root)
                    if normalized not in seen_roots:
                        seen_roots.add(normalized)
                        connections.append(
                            DatasetConnection(
                                name="UAVDT",
                                root=normalized,
                                adapter_name="uavdt_discovery",
                                supports_label_parsing=False,
                                canonical_class_coverage=[0],
                                notes="Discovery-only for now; local structure must be verified before parsing.",
                                labels_format="uavdt",
                            )
                        )
            depth = len(current_path.parts) - len(root.parts)
            if depth >= 3:
                dirs[:] = []
    return sorted(connections, key=lambda item: (item.name, item.root))


def connect_explicit_dataset(dataset_root: str | Path, labels_format: str | None) -> DatasetConnection | None:
    root = Path(dataset_root)
    if not root.exists():
        return None
    normalized = str(root)
    if not labels_format:
        return DatasetConnection(
            name=root.name or "explicit_dataset",
            root=normalized,
            adapter_name="explicit_discovery_only",
            supports_label_parsing=False,
            canonical_class_coverage=[],
            notes="Root exists, but labels_format was not supplied.",
            labels_format=None,
        )
    normalized_format = labels_format.strip().lower()
    if normalized_format == "coco":
        return DatasetConnection(
            name=root.name or "explicit_coco",
            root=normalized,
            adapter_name="explicit_coco",
            supports_label_parsing=True,
            canonical_class_coverage=[0, 1, 2, 3],
            notes="Explicit COCO-style dataset root provided by CLI.",
            labels_format="coco",
        )
    if normalized_format == "yolo":
        return DatasetConnection(
            name=root.name or "explicit_yolo",
            root=normalized,
            adapter_name="explicit_yolo",
            supports_label_parsing=True,
            canonical_class_coverage=[0, 1, 2, 3],
            notes="Explicit YOLO-style dataset root provided by CLI.",
            labels_format="yolo",
        )
    return DatasetConnection(
        name=root.name or "explicit_dataset",
        root=normalized,
        adapter_name=f"explicit_{normalized_format}",
        supports_label_parsing=False,
        canonical_class_coverage=[],
        notes="Explicit dataset root exists, but this labels_format is not parsed yet.",
        labels_format=normalized_format,
    )


def probe_optional_dependencies(module_names: tuple[str, ...] = DEFAULT_OPTIONAL_DEPENDENCIES) -> dict[str, bool]:
    return {module_name: importlib.util.find_spec(module_name) is not None for module_name in module_names}


def build_tile_windows(
    frame_width: int,
    frame_height: int,
    *,
    tile_size: int,
    overlap: float,
) -> list[TileWindow]:
    if frame_width <= 0 or frame_height <= 0:
        return []
    effective_tile = max(1, int(tile_size))
    overlap_ratio = min(max(float(overlap), 0.0), 0.9)
    step = max(1, int(round(effective_tile * (1.0 - overlap_ratio))))
    horizontal = _axis_windows(frame_width, effective_tile, step)
    vertical = _axis_windows(frame_height, effective_tile, step)
    return [
        TileWindow(left=left, top=top, width=min(effective_tile, frame_width - left), height=min(effective_tile, frame_height - top))
        for top in vertical
        for left in horizontal
    ]


def offset_detection_to_frame(
    detection: CanonicalDetection,
    *,
    x_offset: int,
    y_offset: int,
    frame_width: int,
    frame_height: int,
) -> CanonicalDetection:
    metadata = dict(detection.metadata)
    metadata["tile_offset_x"] = int(x_offset)
    metadata["tile_offset_y"] = int(y_offset)
    return CanonicalDetection(
        class_id=int(detection.class_id),
        landing_status=int(detection.landing_status),
        motion_status=detection.motion_status,
        top_left_x=float(max(0.0, min(float(detection.top_left_x) + x_offset, frame_width - 1.0))),
        top_left_y=float(max(0.0, min(float(detection.top_left_y) + y_offset, frame_height - 1.0))),
        bottom_right_x=float(max(0.0, min(float(detection.bottom_right_x) + x_offset, frame_width - 1.0))),
        bottom_right_y=float(max(0.0, min(float(detection.bottom_right_y) + y_offset, frame_height - 1.0))),
        metadata=metadata,
    )


def canonical_task1_class_from_visdrone_category(category_id: int) -> int | None:
    return VISDRONE_TO_TASK1_CLASS.get(int(category_id))


def canonical_task1_class_from_uavdt_label(label: str) -> int | None:
    return UAVDT_TO_TASK1_CLASS.get(label.strip().lower())


def _gather_paths(model_root: Path, patterns: tuple[str, ...]) -> list[str]:
    results: list[str] = []
    for pattern in patterns:
        results.extend(str(path) for path in sorted(model_root.rglob(pattern)) if path.is_file())
    return _dedupe_paths(results)


def _axis_windows(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions: list[int] = []
    current = 0
    while current + tile_size < length:
        positions.append(current)
        current += step
    final_start = max(length - tile_size, 0)
    if not positions or positions[-1] != final_start:
        positions.append(final_start)
    return positions


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered
