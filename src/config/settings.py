from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_reference_dir() -> Path:
    return _project_root() / "data" / "THYZ_2026_Ornek_Veri_Seti-20260403T083511Z-3-001" / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_Referans_Nesneler"


def _default_task3_eval_reference_dir() -> Path:
    return _project_root() / "data" / "references" / "2026_baseline"


def _default_task3_eval_manifest_path() -> Path:
    return _project_root() / "data" / "task3_eval_manifest.json"


def _default_task3_yoloe_weight_path() -> Path:
    return _project_root() / "data" / "weights" / "task3" / "yoloe-11m-seg.pt"


def _default_task3_debug_dump_dir() -> Path:
    return _project_root() / "_logs" / "debug" / "task3_rejects"


@dataclass(slots=True)
class OfficialRepoSettings:
    """Resmi repo uyumlu batch adapter ayarlari."""

    base_url: str
    username: str
    password: str
    request_timeout_s: float = 5.0
    image_timeout_s: float = 10.0
    cache_dir: Path = field(default_factory=lambda: Path("_runtime_cache"))
    get_frames_limit_per_minute: int = 5
    prediction_limit_per_minute: int = 80

    @property
    def normalized_base_url(self) -> str:
        if self.base_url.endswith("/"):
            return self.base_url
        return f"{self.base_url}/"


@dataclass(slots=True)
class SequentialProtocolSettings:
    """Final yarisma sequential adapter ayarlari."""

    base_url: str
    username: str
    password: str
    request_timeout_s: float = 5.0
    image_timeout_s: float = 10.0
    profile_name: str = "default"
    wire_style: str = "teknofest_v1"
    path_overrides: dict[str, str] = field(default_factory=dict)
    timeout_policy: dict[str, float] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=lambda: {"max_retries": 2, "backoff_s": 0.25})
    warmup_timeout_s: float = 3.0
    first_frame_timeout_s: float = 5.0
    prediction_retryable_statuses: list[int] = field(default_factory=lambda: [408, 429, 500, 502, 503, 504])
    next_frame_retryable_statuses: list[int] = field(default_factory=lambda: [408, 429, 500, 502, 503, 504])
    auth_path: str = "auth/"
    open_session_path: str = "session/open/"
    next_frame_path: str = "session/next/"
    prediction_path: str = "session/prediction/"
    close_session_path: str = "session/close/"

    @property
    def normalized_base_url(self) -> str:
        if self.base_url.endswith("/"):
            return self.base_url
        return f"{self.base_url}/"

    def resolve_path(self, path_name: str) -> str:
        defaults = {
            "auth": self.auth_path,
            "open_session": self.open_session_path,
            "next_frame": self.next_frame_path,
            "prediction": self.prediction_path,
            "close_session": self.close_session_path,
        }
        path = str(self.path_overrides.get(path_name, defaults[path_name]))
        return path if path.endswith("/") else f"{path}/"

    def effective_timeout(self, timeout_name: str, default_value: float) -> float:
        return float(self.timeout_policy.get(timeout_name, default_value))


@dataclass(slots=True)
class MvpRuntimeSettings:
    """Task 3 odakli runtime davranis ayarlari."""

    max_objects_per_frame: int = 16
    task3_mode: str = "orb_template"
    task3_reference_dir: Path = field(default_factory=_default_reference_dir)
    task3_eval_reference_dir: Path = field(default_factory=_default_task3_eval_reference_dir)
    task3_eval_manifest_path: Path = field(default_factory=_default_task3_eval_manifest_path)
    task3_eval_frame_stride: int = 60
    task3_eval_frame_limit: int | None = 120
    task3_orb_features: int = 256
    task3_match_min_inliers: int = 4
    task3_match_ratio_threshold: float = 0.75
    task3_template_proposal_threshold: float = 0.55
    task3_learned_backbone: str = "resnet18.a1_in1k"
    task3_learned_input_size: int = 224
    task3_learned_min_similarity: float = 0.78
    task3_learned_ambiguity_gap: float = 0.03
    task3_learned_pretrained: bool = True
    task3_min_score: float = 0.70
    task3_ambiguity_margin: float = 0.05
    task3_yoloe_weight_path: Path = field(default_factory=_default_task3_yoloe_weight_path)
    task3_yoloe_device: str | None = None
    task3_yoloe_allow_cpu: bool = False
    task3_yoloe_conf: float = 0.10
    task3_yoloe_iou: float = 0.50
    task3_yoloe_imgsz: int = 1280
    task3_yoloe_max_det_per_class: int = 5
    task3_yoloe_verify_every_k: int = 1
    task3_lightglue_min_matches: int = 15
    task3_superpoint_max_kpts: int = 1024
    task3_min_crop_side: int = 24
    task3_resize_crop_to: int = 256
    task3_yoloe_match_normalization_scale: int = 50
    task3_yoloe_min_score: float = 0.4520
    task3_yoloe_thermal_min_score: float = 0.50
    task3_yoloe_score_confidence_weight: float = 0.25
    task3_yoloe_score_matches_weight: float = 0.61
    task3_yoloe_score_inlier_weight: float = 0.14
    task3_yoloe_homography_ransac_reproj_threshold: float = 5.0
    task3_debug_dump_rejects: bool = False
    task3_debug_dump_dir: Path = field(default_factory=_default_task3_debug_dump_dir)
    task3_debug_export_keypoints: bool = False
    profiling_gpu_query_cmd: tuple[str, ...] = (
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )

    def __post_init__(self) -> None:
        yoloe_weight_sum = (
            float(self.task3_yoloe_score_confidence_weight)
            + float(self.task3_yoloe_score_matches_weight)
            + float(self.task3_yoloe_score_inlier_weight)
        )
        if not math.isclose(yoloe_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "task3_yoloe_score_confidence_weight + task3_yoloe_score_matches_weight + "
                f"task3_yoloe_score_inlier_weight must equal 1.0, got {yoloe_weight_sum:.12f}"
            )
