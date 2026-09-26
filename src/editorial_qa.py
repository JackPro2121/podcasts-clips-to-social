from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.composition_planner import CompositionPlan
from src.edit_director import EditPlan, validate_edit_plan


@dataclass
class QaIssue:
    code: str
    severity: str
    message: str
    start: Optional[float] = None
    end: Optional[float] = None

    @property
    def blocks_publish(self) -> bool:
        return self.severity in {"error", "fatal"}


@dataclass
class QaReport:
    report_id: str
    passed: bool
    issues: List[QaIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "0.1"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["blocks_publish"] = any(issue.blocks_publish for issue in self.issues)
        return payload


def _run(command: List[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def _parse_fraction(value: str, default: float = 0.0) -> float:
    if not value or "/" not in value:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    numerator, denominator = value.split("/", 1)
    try:
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else default
    except (TypeError, ValueError):
        return default


def _probe_output(path: Path) -> Dict[str, Any]:
    result = _run([
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Output probe failed: {result.stderr[-500:]}")
    return json.loads(result.stdout)


def _detect_intervals(path: Path, filter_name: str, pattern: str) -> List[Dict[str, float]]:
    result = _run([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-vf",
        filter_name,
        "-an",
        "-f",
        "null",
        "NUL",
    ])
    events: List[Dict[str, float]] = []
    active_start: Optional[float] = None
    for line in result.stderr.splitlines():
        start_match = re.search(pattern + r".*?start:\s*([0-9.]+)", line)
        end_match = re.search(pattern + r".*?end:\s*([0-9.]+)", line)
        duration_match = re.search(pattern + r".*?duration:\s*([0-9.]+)", line)
        if start_match:
            active_start = float(start_match.group(1))
        if end_match and active_start is not None:
            events.append({
                "start": active_start,
                "end": float(end_match.group(1)),
                "duration": float(duration_match.group(1)) if duration_match else float(end_match.group(1)) - active_start,
            })
            active_start = None
    return events


def _loudness_metrics(path: Path) -> Dict[str, float]:
    result = _run([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-filter_complex",
        "[0:a]ebur128=framelog=verbose:peak=true",
        "-f",
        "null",
        "NUL",
    ])
    metrics: Dict[str, float] = {}
    for line in result.stderr.splitlines():
        if "I:" in line:
            match = re.search(r"I:\s*(-?[0-9.]+)", line)
            if match:
                metrics["integrated_lufs"] = float(match.group(1))
        if "Peak:" in line:
            match = re.search(r"Peak:\s*(-?[0-9.]+)", line)
            if match:
                metrics["peak_dbfs"] = float(match.group(1))
    return metrics


def _output_issues(
    path: Path,
    expected_duration: float,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    max_black_duration: float,
    max_freeze_duration: float,
) -> tuple[List[QaIssue], Dict[str, Any]]:
    issues: List[QaIssue] = []
    metrics: Dict[str, Any] = {}
    try:
        payload = _probe_output(path)
    except Exception as error:
        return [QaIssue("output_probe", "fatal", str(error))], metrics
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if not video:
        issues.append(QaIssue("video_stream_missing", "fatal", "Rendered output has no video stream"))
    if not audio:
        issues.append(QaIssue("audio_stream_missing", "error", "Rendered output has no audio stream"))
    if video:
        fps = _parse_fraction(str(video.get("avg_frame_rate") or "0/1"))
        duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0)
        metrics["duration"] = duration
        metrics["fps"] = fps
        if video.get("codec_name") != "h264":
            issues.append(QaIssue("video_codec", "error", f"Unexpected video codec: {video.get('codec_name')}"))
        if int(video.get("width", 0)) != expected_width or int(video.get("height", 0)) != expected_height:
            issues.append(QaIssue("video_dimensions", "error", "Unexpected output dimensions"))
        if abs(fps - expected_fps) > 0.5:
            issues.append(QaIssue("video_fps", "error", "Unexpected output frame rate"))
        if abs(duration - expected_duration) > 0.5:
            issues.append(QaIssue("output_duration", "error", "Output duration does not match edit plan"))
    black_events = _detect_intervals(path, "blackdetect=d=0.12:pix_th=0.10", r"black")
    freeze_events = _detect_intervals(path, "freezedetect=n=0.003:d=0.5", r"freeze")
    metrics["black_intervals"] = black_events
    metrics["freeze_intervals"] = freeze_events
    for event in black_events:
        if event["duration"] > max_black_duration:
            issues.append(QaIssue("black_interval", "error", "Output contains a sustained black interval", event["start"], event["end"]))
    for event in freeze_events:
        if event["duration"] > max_freeze_duration:
            issues.append(QaIssue("freeze_interval", "error", "Output contains a sustained freeze interval", event["start"], event["end"]))
    if audio:
        sample_rate = int(audio.get("sample_rate", 0))
        metrics["sample_rate"] = sample_rate
        if audio.get("codec_name") != "aac" or sample_rate != 48000:
            issues.append(QaIssue("audio_stream", "error", "Output audio must be AAC 48kHz"))
    metrics.update(_loudness_metrics(path))
    if "integrated_lufs" in metrics and abs(metrics["integrated_lufs"] + 14.0) > 1.0:
        issues.append(QaIssue("loudness", "error", f"Integrated loudness is {metrics['integrated_lufs']:.1f} LUFS"))
    if "peak_dbfs" in metrics and metrics["peak_dbfs"] > -1.0:
        issues.append(QaIssue("peak", "error", f"Peak is {metrics['peak_dbfs']:.1f} dBFS"))
    return issues, metrics


def _plan_issues(
    edit_plan: EditPlan,
    composition: CompositionPlan,
    source_duration: Optional[float] = None,
) -> List[QaIssue]:
    validation_codes = validate_edit_plan(edit_plan, source_duration)
    issues = [
        QaIssue(code, "error", code)
        for code in validation_codes
        if code != "endpoint_not_proven_complete"
    ]
    if "endpoint_not_proven_complete" in validation_codes or "endpoint_not_proven_complete" in edit_plan.warnings:
        issues.append(QaIssue("incomplete_endpoint", "warning", "Edit endpoint is not proven complete"))
    if "endpoint_caption_fragment" in edit_plan.warnings:
        issues.append(QaIssue("fragmented_endpoint", "warning", "Edit endpoint is a weak caption fragment"))
    if not composition.shots:
        issues.append(QaIssue("composition_missing", "error", "Composition has no shots"))
    for shot in composition.shots:
        # Every protected region, not just the first. A shot with two text
        # blocks and a collision on the second one was passing QA, which is the
        # one case the check exists to catch.
        colliding = [
            region for region in shot.protected_regions
            if shot.caption_rect.intersects(region)
        ]
        if colliding:
            issues.append(QaIssue(
                "caption_source_collision",
                "error",
                f"Caption overlaps {len(colliding)} source text region(s) in {shot.shot_id}",
            ))
        if shot.max_static_hold > 2.5:
            issues.append(QaIssue("static_hold_limit", "error", f"Static hold exceeds limit in {shot.shot_id}"))
    return issues


def save_qa_report(report: QaReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f"{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def run_editorial_qa(
    path: Path,
    edit_plan: EditPlan,
    composition: CompositionPlan,
    expected_width: int = 1080,
    expected_height: int = 1920,
    expected_fps: float = 30.0,
    source_duration: Optional[float] = None,
    max_black_duration: float = 0.5,
    max_freeze_duration: float = 0.8,
    report_id: str = "qa_report",
) -> QaReport:
    issues, metrics = _output_issues(
        path,
        edit_plan.duration,
        expected_width,
        expected_height,
        expected_fps,
        max_black_duration,
        max_freeze_duration,
    )
    issues.extend(_plan_issues(edit_plan, composition, source_duration))
    return QaReport(
        report_id=report_id,
        passed=not any(issue.blocks_publish for issue in issues),
        issues=issues,
        metrics=metrics,
    )
