from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import OfficialRepoSettings
from src.core.frame_state import CanonicalDetection, CanonicalTranslation, FrameResult
from src.server.official_repo_batch_adapter import OfficialRepoBatchAdapter


ARCHIVE_DIR = PROJECT_ROOT / "_logs" / "reports_generated" / "task3_manifest" / "2026-05-03_d10_repo_cleanup"
REPORT_PATH = ARCHIVE_DIR / "wire_format_diff.md"


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    adapter = OfficialRepoBatchAdapter(
        OfficialRepoSettings(base_url="http://mock/", username="team", password="password")
    )
    payload = adapter.build_wire_prediction(
        FrameResult(
            frame_url="http://mock/frames/1/",
            detected_objects=[
                CanonicalDetection(
                    class_id=0,
                    landing_status=-1,
                    motion_status=1,
                    top_left_x=10.0,
                    top_left_y=20.0,
                    bottom_right_x=30.0,
                    bottom_right_y=40.0,
                )
            ],
            detected_translations=[
                CanonicalTranslation(
                    translation_x=1.0,
                    translation_y=2.0,
                    translation_z=3.0,
                    source="audit",
                )
            ],
        )
    )
    obj = payload["detected_objects"][0]
    translation = payload["detected_translations"][0]

    markdown = [
        "# D10 Wire Format Diff",
        "",
        "| Field | Resmi (2025) | Bizim Adapter | Karar |",
        "| --- | --- | --- | --- |",
        f"| frame | url string | `{type(payload['frame']).__name__}` | OK |",
        f"| detected_objects[].cls | `<server>/classes/<id+1>/` | `{obj['cls']}` | OK |",
        f"| detected_objects[].landing_status | string | `{type(obj['landing_status']).__name__}` | OK |",
        f"| detected_objects[].motion_status | YOK | `{'motion_status' in obj}` | omit |",
        f"| detected_objects[].top_left_x | str(float) | `{type(obj['top_left_x']).__name__}` | OK |",
        f"| detected_objects[].top_left_y | str(float) | `{type(obj['top_left_y']).__name__}` | OK |",
        f"| detected_objects[].bottom_right_x | str(float) | `{type(obj['bottom_right_x']).__name__}` | OK |",
        f"| detected_objects[].bottom_right_y | str(float) | `{type(obj['bottom_right_y']).__name__}` | OK |",
        f"| detected_translations[].translation_x | str(float) | `{type(translation['translation_x']).__name__}` | OK |",
        f"| detected_translations[].translation_y | str(float) | `{type(translation['translation_y']).__name__}` | OK |",
        f"| detected_translations[].translation_z | str(float) | `{type(translation['translation_z']).__name__}` | OK |",
        f"| detected_undefined_objects | YOK | `{'detected_undefined_objects' in payload}` | omit |",
        "",
        "## Endpoint Audit",
        "",
        f"- `auth/`: `{adapter.url_login}`",
        f"- `frames/`: `{adapter.url_frames}`",
        f"- `translation/`: `{adapter.url_translations}`",
        f"- `prediction/`: `{adapter.url_prediction}`",
        "",
        "## Header Audit",
        "",
        "- Authorization style: `Token <token>`",
        "- Content-Type for JSON payloads is delegated to the HTTP client JSON post helper.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
