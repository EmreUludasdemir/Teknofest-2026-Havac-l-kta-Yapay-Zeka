from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from src.config.settings import MvpRuntimeSettings
from src.core.frame_state import DecodedFrame, FrameEnvelope
from src.evaluation.task2_long_sequence import iter_video_frames
from src.task3.matcher import Task3Matcher
from src.task3.no_match_logic import filter_no_match_candidates
from src.task3.reference_cache import ReferenceCache
from src.task3.verifier import verify_matches


def discover_task3_videos(reference_dir: str | Path | None = None) -> list[Path]:
    base = Path(reference_dir) if reference_dir is not None else MvpRuntimeSettings().task3_eval_reference_dir
    parent = base.parent if base.is_dir() else base
    deduped: dict[str, Path] = {}
    for candidate in list(parent.glob("*.MP4")) + list(parent.glob("*.mp4")):
        deduped[str(candidate.resolve()).lower()] = candidate
    return sorted(deduped.values())


def evaluate_task3_frames(
    frames: Iterable[DecodedFrame],
    *,
    runtime_settings: MvpRuntimeSettings | None = None,
    reference_dir: str | Path | None = None,
    video_name: str = "task3_replay",
    mode: str = "orb_template",
) -> dict[str, object]:
    settings = runtime_settings or MvpRuntimeSettings()
    cache = ReferenceCache()
    cache.preload_from_directory(reference_dir or settings.task3_eval_reference_dir, orb_features=settings.task3_orb_features)
    matcher = Task3Matcher(reference_cache=cache, runtime_settings=settings)
    reference_ids = cache.list_ids()

    total_frames = 0
    raw_candidate_frames = 0
    accepted_match_count = 0
    rejected_verification_count = 0
    ambiguity_suppression_count = 0
    no_match_frames = 0
    descriptor_path_count = 0
    template_path_count = 0
    false_positive_proxy_count = 0
    match_scores: list[float] = []

    for frame_index, decoded in enumerate(frames):
        total_frames += 1
        frame = FrameEnvelope(
            frame_url=f"http://task3-eval/frames/{frame_index + 1}/",
            image_url=f"/task3/{frame_index + 1}.jpg",
            video_name=video_name,
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            health_status="1",
            metadata={"frame_index": decoded.frame_index, "image_width": decoded.width, "image_height": decoded.height},
        )
        if mode == "learned_descriptor":
            raw_matches = matcher.match(frame, b"", reference_ids, decoded_frame=decoded, mode="learned_descriptor")
        else:
            raw_matches = matcher.match(frame, b"", reference_ids, decoded_frame=decoded, mode="orb_template")
        if raw_matches:
            raw_candidate_frames += 1
        filtered = filter_no_match_candidates(
            raw_matches,
            min_score=settings.task3_min_score,
            ambiguity_margin=settings.task3_ambiguity_margin,
        )
        if raw_matches and not filtered and len(raw_matches) > 1:
            ambiguity_suppression_count += 1
        verified = verify_matches(
            frame,
            filtered,
            decoded_frame=decoded,
            min_inliers=settings.task3_match_min_inliers,
        )
        rejected_verification_count += max(len(filtered) - len(verified), 0)
        if not verified:
            no_match_frames += 1
            continue

        for match in verified:
            accepted_match_count += 1
            score = float(match.metadata.get("match_score", 0.0))
            match_scores.append(score)
            source = str(match.metadata.get("matcher_source", ""))
            if "template" in source:
                template_path_count += 1
                if score < 0.88:
                    false_positive_proxy_count += 1
            elif "learned" in source:
                descriptor_path_count += 1
                if float(match.metadata.get("similarity", 0.0)) < settings.task3_learned_min_similarity:
                    false_positive_proxy_count += 1
            else:
                descriptor_path_count += 1
                if float(match.metadata.get("inlier_ratio", 0.0)) < 0.45:
                    false_positive_proxy_count += 1

    decision = _decide_learned_descriptor_need(
        accepted_match_count=accepted_match_count,
        false_positive_proxy_count=false_positive_proxy_count,
        descriptor_path_count=descriptor_path_count,
        template_path_count=template_path_count,
    )
    return {
        "video_name": video_name,
        "status": "ok",
        "reference_count": len(reference_ids),
        "total_frames": total_frames,
        "raw_candidate_frames": raw_candidate_frames,
        "accepted_match_count": accepted_match_count,
        "rejected_verification_count": rejected_verification_count,
        "ambiguity_suppression_count": ambiguity_suppression_count,
        "no_match_suppression_rate": round(no_match_frames / max(total_frames, 1), 6),
        "false_positive_proxy_count": false_positive_proxy_count,
        "descriptor_path_count": descriptor_path_count,
        "template_path_count": template_path_count,
        "mean_match_score": round(mean(match_scores), 6) if match_scores else 0.0,
        "decision": decision,
        "mode": mode,
    }


def evaluate_task3_baseline(
    *,
    runtime_settings: MvpRuntimeSettings | None = None,
    output_dir: str | Path = "reports",
    mode: str = "orb_template",
) -> dict[str, object]:
    settings = runtime_settings or MvpRuntimeSettings()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for video_path in discover_task3_videos(settings.task3_eval_reference_dir):
        try:
            frames = iter_video_frames(
                video_path,
                frame_stride=settings.task3_eval_frame_stride,
                limit=settings.task3_eval_frame_limit,
                video_name=video_path.stem,
            )
            summary = evaluate_task3_frames(
                frames,
                runtime_settings=settings,
                reference_dir=settings.task3_eval_reference_dir,
                video_name=video_path.stem,
                mode=mode,
            )
        except Exception as exc:
            summary = {"video_name": video_path.stem, "status": "failed", "error": str(exc), "total_frames": 0}
        results.append(summary)

    aggregate = {
        "video_count": len(results),
        "ok_count": sum(1 for item in results if item.get("status") == "ok"),
        "no_match_suppression_rate": round(_safe_mean(item.get("no_match_suppression_rate", 0.0) for item in results if item.get("status") == "ok"), 6),
        "false_positive_proxy_count": int(sum(int(item.get("false_positive_proxy_count", 0)) for item in results if item.get("status") == "ok")),
        "accepted_match_count": int(sum(int(item.get("accepted_match_count", 0)) for item in results if item.get("status") == "ok")),
        "decision": _aggregate_decision(results),
    }
    payload = {"results": results, "aggregate": aggregate, "mode": mode}
    if mode == "orb_template":
        (output_path / "task3_baseline_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (output_path / "task3_baseline_table.md").write_text(render_task3_table(results), encoding="utf-8")
    suffix = "orb" if mode == "orb_template" else "learned"
    (output_path / f"task3_baseline_{suffix}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_path / f"task3_baseline_{suffix}_table.md").write_text(render_task3_table(results), encoding="utf-8")
    write_task3_comparison(output_path)
    return payload


def write_task3_comparison(output_dir: str | Path) -> dict[str, object] | None:
    output_path = Path(output_dir)
    orb_path = output_path / "task3_baseline_orb_summary.json"
    learned_path = output_path / "task3_baseline_learned_summary.json"
    if not orb_path.exists() or not learned_path.exists():
        return None
    orb_payload = json.loads(orb_path.read_text(encoding="utf-8"))
    learned_payload = json.loads(learned_path.read_text(encoding="utf-8"))
    orb_aggregate = orb_payload.get("aggregate", {})
    learned_aggregate = learned_payload.get("aggregate", {})
    comparison_decision = decide_learned_descriptor_gain(orb_aggregate, learned_aggregate)
    markdown = (
        "| Metric | ORB/Template | Learned | Delta |\n"
        "| --- | --- | --- | --- |\n"
        f"| Accepted Match Count | {orb_aggregate.get('accepted_match_count', '-')} | {learned_aggregate.get('accepted_match_count', '-')} | "
        f"{int(learned_aggregate.get('accepted_match_count', 0)) - int(orb_aggregate.get('accepted_match_count', 0))} |\n"
        f"| False Positive Proxy | {orb_aggregate.get('false_positive_proxy_count', '-')} | {learned_aggregate.get('false_positive_proxy_count', '-')} | "
        f"{int(learned_aggregate.get('false_positive_proxy_count', 0)) - int(orb_aggregate.get('false_positive_proxy_count', 0))} |\n"
        f"| No-match Suppression Rate | {orb_aggregate.get('no_match_suppression_rate', '-')} | {learned_aggregate.get('no_match_suppression_rate', '-')} | "
        f"{round(float(learned_aggregate.get('no_match_suppression_rate', 0.0)) - float(orb_aggregate.get('no_match_suppression_rate', 0.0)), 6)} |\n"
        f"| Decision | {orb_aggregate.get('decision', '-')} | {learned_aggregate.get('decision', '-')} | - |\n"
        f"| Faz 6 Learned Gain | - | {comparison_decision} | - |\n"
    )
    (output_path / "task3_baseline_comparison.md").write_text(markdown, encoding="utf-8")
    return {"orb": orb_aggregate, "learned": learned_aggregate, "decision": comparison_decision}


def render_task3_table(results: list[dict[str, object]]) -> str:
    lines = [
        "| Video | Status | Frames | Accepted | Rejected | No-match Rate | FP Proxy | Descriptor | Template | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            "| {video} | {status} | {frames} | {accepted} | {rejected} | {no_match} | {fp} | {descriptor} | {template} | {decision} |".format(
                video=item.get("video_name"),
                status=item.get("status"),
                frames=item.get("total_frames", 0),
                accepted=item.get("accepted_match_count", "-"),
                rejected=item.get("rejected_verification_count", "-"),
                no_match=item.get("no_match_suppression_rate", "-"),
                fp=item.get("false_positive_proxy_count", "-"),
                descriptor=item.get("descriptor_path_count", "-"),
                template=item.get("template_path_count", "-"),
                decision=item.get("decision", "-"),
            )
        )
    return "\n".join(lines) + "\n"


def _decide_learned_descriptor_need(
    *,
    accepted_match_count: int,
    false_positive_proxy_count: int,
    descriptor_path_count: int,
    template_path_count: int,
) -> str:
    if accepted_match_count == 0:
        return "belirsiz"
    if false_positive_proxy_count > max(2, accepted_match_count // 5):
        return "gerekli"
    if descriptor_path_count >= template_path_count and false_positive_proxy_count == 0:
        return "henuz_gereksiz"
    return "belirsiz"


def _aggregate_decision(results: list[dict[str, object]]) -> str:
    decisions = [str(item.get("decision", "belirsiz")) for item in results if item.get("status") == "ok"]
    if not decisions:
        return "belirsiz"
    if any(item == "gerekli" for item in decisions):
        return "gerekli"
    if all(item == "henuz_gereksiz" for item in decisions):
        return "henuz_gereksiz"
    return "belirsiz"


def decide_learned_descriptor_gain(orb_aggregate: dict[str, object], learned_aggregate: dict[str, object]) -> str:
    orb_fp = int(orb_aggregate.get("false_positive_proxy_count", 0))
    learned_fp = int(learned_aggregate.get("false_positive_proxy_count", 0))
    orb_no_match = float(orb_aggregate.get("no_match_suppression_rate", 0.0))
    learned_no_match = float(learned_aggregate.get("no_match_suppression_rate", 0.0))
    orb_accept = int(orb_aggregate.get("accepted_match_count", 0))
    learned_accept = int(learned_aggregate.get("accepted_match_count", 0))

    fp_improved = learned_fp <= int(round(orb_fp * 0.8))
    no_match_ok = (learned_no_match - orb_no_match) <= 0.10
    accept_ok = learned_accept >= int(round(orb_accept * 0.85))
    if fp_improved and no_match_ok and accept_ok:
        return "kazanc_var"
    if learned_fp >= orb_fp and learned_accept <= orb_accept:
        return "kazanc_yok"
    return "belirsiz"


def _safe_mean(values: Iterable[float]) -> float:
    filtered = [float(value) for value in values]
    return mean(filtered) if filtered else 0.0
