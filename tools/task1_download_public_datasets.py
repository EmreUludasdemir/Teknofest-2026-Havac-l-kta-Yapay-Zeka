from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.experimental import TASK1_PUBLIC_DOWNLOADS_ROOT, extract_zip_safe

try:  # pragma: no cover - dependency probe
    import requests
except Exception:  # pragma: no cover - dependency probe
    requests = None


DATASETS: list[dict[str, Any]] = [
    {
        "name": "visdrone",
        "similarity": "high",
        "modality": "rgb_aerial",
        "classes": [
            "pedestrian",
            "person",
            "car",
            "van",
            "bus",
            "truck",
            "motor",
            "bicycle",
            "tricycle",
            "awning_tricycle",
        ],
        "license_note": "Public research benchmark; follow original VisDrone usage terms.",
        "files": [
            {
                "archive_name": "VisDrone2019-DET-train.zip",
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
            },
            {
                "archive_name": "VisDrone2019-DET-val.zip",
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
            },
        ],
    },
    {
        "name": "uavdt",
        "similarity": "high",
        "modality": "rgb_aerial",
        "classes": ["car", "truck", "bus"],
        "license_note": "Official dataset exists, but direct anonymous archive URL is not wired in this branch yet.",
        "files": [],
        "blocked_reason": "blocked_manual_access",
    },
    {
        "name": "hit_uav",
        "similarity": "medium",
        "modality": "thermal_uav",
        "classes": ["person", "car", "bicycle", "other_vehicle"],
        "license_note": "Open thermal UAV dataset; keep modality mismatch explicit in reports.",
        "files": [
            {
                "archive_name": "HIT-UAV-Infrared-Thermal-Dataset-v1.2.zip",
                "url": "https://zenodo.org/records/7633120/files/suojiashun%2FHIT-UAV-Infrared-Thermal-Dataset-v1.2.zip?download=1",
            }
        ],
    },
    {
        "name": "seadronessee",
        "similarity": "low",
        "modality": "rgb_maritime",
        "classes": ["boat", "jetski", "lifesaving_appliance", "buoy", "swimmer"],
        "license_note": "Benchmark access can require registration; skip if anonymous direct archive is unavailable.",
        "files": [],
        "blocked_reason": "blocked_manual_access",
    },
]


def _download_file(url: str, destination: Path) -> dict[str, Any]:
    if requests is None:
        return {"status": "blocked", "reason": "missing_dependency:requests"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60.0) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            return {"status": "blocked", "reason": f"html_response:{content_type}"}
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return {"status": "downloaded", "size_bytes": destination.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download public Task 1 datasets when anonymous access works.")
    parser.add_argument("--output-root", default=str(TASK1_PUBLIC_DOWNLOADS_ROOT))
    parser.add_argument("--status-json", default=str(TASK1_PUBLIC_DOWNLOADS_ROOT / "download_status.json"))
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_root = output_root / dataset["name"]
        dataset_root.mkdir(parents=True, exist_ok=True)
        if dataset.get("blocked_reason"):
            status_rows.append(
                {
                    **dataset,
                    "status": dataset["blocked_reason"],
                    "downloaded_files": [],
                    "notes": [dataset["license_note"]],
                }
            )
            continue
        downloaded_files: list[dict[str, Any]] = []
        dataset_status = "downloaded"
        notes = [dataset["license_note"]]
        for file_spec in dataset["files"]:
            archive_path = dataset_root / file_spec["archive_name"]
            result = _download_file(file_spec["url"], archive_path)
            file_row = {"archive_name": file_spec["archive_name"], "url": file_spec["url"], **result}
            downloaded_files.append(file_row)
            if result["status"] != "downloaded":
                dataset_status = "blocked"
                notes.append(result.get("reason", "download_failed"))
                continue
            extraction = extract_zip_safe(archive_path, dataset_root / "extracted")
            file_row["extraction"] = extraction.to_dict()
        status_rows.append(
            {
                **dataset,
                "status": dataset_status,
                "downloaded_files": downloaded_files,
                "notes": notes,
            }
        )

    payload = {"datasets": status_rows}
    status_json = Path(args.status_json)
    status_json.parent.mkdir(parents=True, exist_ok=True)
    status_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
