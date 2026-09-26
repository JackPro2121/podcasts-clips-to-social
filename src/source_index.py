from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.scene_classifier import detect_clip_shots, detect_slide_heuristics
from src.text_detection import TextBox, merge_text_boxes


@dataclass
class SourceMediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str = ""
    audio_codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    pix_fmt: str = ""
    sar: str = ""
    dar: str = ""
    time_base: str = ""


@dataclass
class TextRegion:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    source: str = "edge_heuristic"


@dataclass
class AudioEvent:
    start: float
    end: float
    kind: str
    confidence: float = 1.0


@dataclass
class IndexedShot:
    shot_id: str
    start: float
    end: float
    shot_type: str
    motion_score: float
    static_score: float
    presentation_ratio: float
    face_count: float
    text_regions: List[TextRegion] = field(default_factory=list)
    freeze_intervals: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class SourceIndex:
    media: SourceMediaInfo
    shots: List[IndexedShot]
    schema_version: str = "0.1"
    audio_events: List[AudioEvent] = field(default_factory=list)
    frame_sample_fps: float = 2.0
    analyzer_version: str = "source-index-0.1"
    quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "media": asdict(self.media),
            "shots": [asdict(shot) for shot in self.shots],
            "audio_events": [asdict(event) for event in self.audio_events],
            "frame_sample_fps": self.frame_sample_fps,
            "analyzer_version": self.analyzer_version,
            "quality": self.quality,
        }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ratio(value: str, default: float = 0.0) -> float:
    if not value or "/" not in value:
        return default
    numerator, denominator = value.split("/", 1)
    denominator_value = _float(denominator)
    return _float(numerator) / denominator_value if denominator_value else default


def probe_source_media(video_path: Path) -> SourceMediaInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path.name}: {result.stderr[-500:]}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video: Dict[str, Any] = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio: Dict[str, Any] = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    format_info = payload.get("format", {})
    return SourceMediaInfo(
        path=str(video_path),
        duration=_float(format_info.get("duration")),
        width=int(_float(video.get("width"))),
        height=int(_float(video.get("height"))),
        fps=_ratio(str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "")),
        video_codec=str(video.get("codec_name", "")),
        audio_codec=str(audio.get("codec_name", "")),
        sample_rate=int(_float(audio.get("sample_rate"))),
        channels=int(_float(audio.get("channels"))),
        pix_fmt=str(video.get("pix_fmt", "")),
        sar=str(video.get("sample_aspect_ratio", "")),
        dar=str(video.get("display_aspect_ratio", "")),
        time_base=str(video.get("time_base", "")),
    )


def _estimate_text_regions(frame: np.ndarray) -> List[TextRegion]:
    """Detect text regions via the configured tier and convert to index form.

    Detection and per-frame merging live in `src.text_detection`. The boxes are
    merged across the whole shot below, because a per-frame detector jitters and
    an unmerged list would produce a hundred near-identical protected rects.
    """
    from src.text_detection import detect_text_regions

    regions: List[TextRegion] = []
    for box in detect_text_regions(frame):
        regions.append(TextRegion(
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
            confidence=box.confidence,
            source=box.source,
        ))
    return regions


def _sample_frames(
    video_path: Path,
    start_time: float,
    end_time: float,
    sample_fps: float,
    max_samples: int,
) -> List[Tuple[float, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps / max(0.1, sample_fps))))
    start_frame = max(0, int(start_time * fps))
    end_frame = max(start_frame + 1, int(end_time * fps))
    samples: List[Tuple[float, np.ndarray]] = []
    try:
        frame_index = start_frame
        while frame_index < end_frame and len(samples) < max_samples:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                break
            samples.append((frame_index / max(1.0, fps), frame))
            frame_index += step
    finally:
        capture.release()
    return samples


def sample_frames_at(
    video_path: Path,
    start_time: float,
    end_time: float,
    count: int = 12,
) -> List[Tuple[float, np.ndarray]]:
    """
    Return roughly `count` evenly-spaced (timestamp, frame) samples across a window.

    Exposed so the director's contact sheet reuses the same decoding path the
    index already uses, rather than adding a second ffmpeg invocation with
    different seeking behaviour.
    """
    safe_count = max(1, int(count))
    duration = max(0.1, end_time - start_time)
    return _sample_frames(
        video_path,
        start_time,
        end_time,
        sample_fps=max(0.01, safe_count / duration),
        max_samples=safe_count,
    )


def _motion_score(previous: np.ndarray, current: np.ndarray) -> float:
    previous_small = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (160, 90))
    current_small = cv2.resize(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (160, 90))
    difference = cv2.absdiff(previous_small, current_small)
    return float(np.mean(difference) / 255.0)


def _freeze_intervals(
    samples: List[Tuple[float, np.ndarray]],
    threshold: float = 0.006,
) -> List[Tuple[float, float]]:
    if len(samples) < 2:
        return []
    intervals: List[Tuple[float, float]] = []
    interval_start: Optional[float] = None
    for previous, current in zip(samples, samples[1:]):
        previous_time, previous_frame = previous
        current_time, current_frame = current
        if _motion_score(previous_frame, current_frame) <= threshold:
            if interval_start is None:
                interval_start = previous_time
        elif interval_start is not None:
            intervals.append((round(interval_start, 2), round(previous_time, 2)))
            interval_start = None
    if interval_start is not None:
        intervals.append((round(interval_start, 2), round(samples[-1][0], 2)))
    return [(start, end) for start, end in intervals if end - start >= 0.5]


def _face_count(frame: np.ndarray, detect_faces: bool) -> float:
    if not detect_faces:
        return 0.0
    try:
        from src.face_tracker import detect_faces_in_frame, get_face_detector
        detector = get_face_detector()
        if detector is None:
            return 0.0
        height, width = frame.shape[:2]
        return float(len(detect_faces_in_frame(detector, frame, width, height)))
    except Exception:
        return 0.0


def _audio_events(video_path: Path, start_time: float, end_time: float) -> List[AudioEvent]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{max(0.0, end_time - start_time):.3f}",
            "-af",
            "silencedetect=n=-35dB:d=0.35",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    events: List[AudioEvent] = []
    pending_start: Optional[float] = None
    for line in result.stderr.splitlines():
        silence_start = re.search(r"silence_start:\s*([0-9.]+)", line)
        silence_end = re.search(r"silence_end:\s*([0-9.]+)", line)
        if silence_start:
            pending_start = _float(silence_start.group(1)) + start_time
        if silence_end and pending_start is not None:
            end = _float(silence_end.group(1)) + start_time
            events.append(AudioEvent(round(pending_start, 2), round(end, 2), "silence"))
            pending_start = None
    return events


def _merge_shot_text_regions(
    regions: List[TextRegion],
    iou_threshold: float = 0.3,
) -> List[TextRegion]:
    """
    Collapse per-frame text detections into one stable region per text block.

    Runs on a whole shot at once so the same line detected in twenty frames
    becomes a single protected rect rather than twenty overlapping ones.
    """
    if not regions:
        return []
    boxes = [
        TextBox(
            x=region.x, y=region.y, width=region.width, height=region.height,
            confidence=region.confidence, source=region.source,
        )
        for region in regions
    ]
    return [
        TextRegion(
            x=box.x, y=box.y, width=box.width, height=box.height,
            confidence=box.confidence, source=box.source,
        )
        for box in merge_text_boxes(boxes, iou_threshold=iou_threshold)
    ]


def build_source_index(
    video_path: Path,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    sample_fps: float = 2.0,
    max_samples: int = 240,
    detect_faces: bool = False,
) -> SourceIndex:
    media = probe_source_media(video_path)
    resolved_end = media.duration if end_time is None else min(media.duration, end_time)
    resolved_start = max(0.0, min(start_time, max(0.0, resolved_end - 0.1)))
    resolved_end = max(resolved_start + 0.1, resolved_end)
    samples = _sample_frames(video_path, resolved_start, resolved_end, sample_fps, max_samples)
    shot_boundaries = detect_clip_shots(video_path, resolved_start, resolved_end, min_shot_duration=0.4)
    shots: List[IndexedShot] = []
    all_freezes: List[Tuple[float, float]] = []
    for shot_index, (shot_start, shot_end) in enumerate(shot_boundaries, 1):
        shot_samples = [
            sample for sample in samples
            if shot_start <= sample[0] <= shot_end
        ]
        if not shot_samples:
            shot_samples = samples[:1]
        motion_values = [
            _motion_score(previous[1], current[1])
            for previous, current in zip(shot_samples, shot_samples[1:])
        ]
        motion_score = float(np.mean(motion_values)) if motion_values else 0.0
        presentation_ratio = float(np.mean([
            1.0 if detect_slide_heuristics(frame) else 0.0
            for _, frame in shot_samples
        ]))
        face_values = [_face_count(frame, detect_faces) for _, frame in shot_samples]
        face_count = float(np.mean(face_values)) if face_values else 0.0
        # Merge text regions across the whole shot by overlap. The previous
        # exact-coordinate de-duplication happened to work for the edge
        # heuristic's stable boxes, but a real detector jitters per frame and
        # would have produced a near-identical rect for every sampled frame --
        # which would make the caption collision check fire on everything.
        shot_text: List[TextRegion] = []
        for _, frame in shot_samples:
            shot_text.extend(_estimate_text_regions(frame))
        shot_text = _merge_shot_text_regions(shot_text)
        shot_freezes = _freeze_intervals(shot_samples)
        all_freezes.extend(shot_freezes)
        shot_type = "presentation" if presentation_ratio >= 0.5 else "human_or_scene"
        if face_count >= 1.5:
            shot_type = "split_screen"
        shots.append(IndexedShot(
            shot_id=f"shot_{shot_index:04d}",
            start=round(shot_start, 2),
            end=round(shot_end, 2),
            shot_type=shot_type,
            motion_score=round(motion_score, 4),
            static_score=round(1.0 - motion_score, 4),
            presentation_ratio=round(presentation_ratio, 4),
            face_count=round(face_count, 2),
            text_regions=shot_text,
            freeze_intervals=shot_freezes,
            confidence=0.65 if samples else 0.2,
        ))
    quality = {
        "sample_count": len(samples),
        "motion_mean": round(float(np.mean([
            shot.motion_score for shot in shots
        ])) if shots else 0.0, 4),
        "freeze_interval_count": len(all_freezes),
        "freeze_duration": round(sum(end - start for start, end in all_freezes), 2),
        "text_region_shot_count": sum(1 for shot in shots if shot.text_regions),
        "face_detector_enabled": detect_faces,
    }
    return SourceIndex(
        media=media,
        shots=shots,
        audio_events=_audio_events(video_path, resolved_start, resolved_end),
        frame_sample_fps=sample_fps,
        quality=quality,
    )


def save_source_index(index: SourceIndex, output_path: Path) -> Path:
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
            json.dump(index.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return output_path
