from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.composition_planner import CompositionPlan, SafeZone
from src.edit_director import EditPlan


@dataclass
class PlatformProfile:
    service: str
    width: int
    height: int
    fps: int
    max_duration: float
    min_duration: float
    safe_zone: SafeZone
    caption_max_words: int
    title_max_chars: int
    caption_max_chars: int
    hashtag_max: int
    audio_target_lufs: float
    audio_target_tp: float
    requires_clean_master: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlatformVariant:
    variant_id: str
    service: str
    start: float
    end: float
    width: int
    height: int
    fps: int
    safe_zone: SafeZone
    caption_max_words: int
    title_max_chars: int
    caption_max_chars: int
    hashtag_max: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "0.1"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceFeedback:
    run_id: str
    clip_id: str
    service: str
    views: int = 0
    average_watch_seconds: float = 0.0
    completion_rate: float = 0.0
    saves: int = 0
    shares: int = 0
    comments: int = 0
    retention_3s_rate: Optional[float] = None
    retention_30s_rate: Optional[float] = None
    source: str = "manual_import"
    recorded_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PROFILE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "master": {
        "service": "master",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration": 140.0,
        "min_duration": 30.0,
        "safe_zone": SafeZone(0.12, 0.22, 0.08, 0.16),
        "caption_max_words": 4,
        "title_max_chars": 100,
        "caption_max_chars": 2200,
        "hashtag_max": 5,
        "audio_target_lufs": -14.0,
        "audio_target_tp": -1.5,
    },
    "tiktok": {
        "service": "tiktok",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration": 180.0,
        "min_duration": 30.0,
        "safe_zone": SafeZone(0.12, 0.25, 0.08, 0.17),
        "caption_max_words": 4,
        "title_max_chars": 90,
        "caption_max_chars": 2200,
        "hashtag_max": 5,
        "audio_target_lufs": -14.0,
        "audio_target_tp": -1.5,
    },
    "instagram": {
        "service": "instagram",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration": 90.0,
        "min_duration": 30.0,
        "safe_zone": SafeZone(0.12, 0.27, 0.08, 0.16),
        "caption_max_words": 3,
        "title_max_chars": 220,
        "caption_max_chars": 2200,
        "hashtag_max": 5,
        "audio_target_lufs": -14.0,
        "audio_target_tp": -1.5,
    },
    "youtube": {
        "service": "youtube",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration": 180.0,
        "min_duration": 30.0,
        "safe_zone": SafeZone(0.10, 0.18, 0.08, 0.16),
        "caption_max_words": 4,
        "title_max_chars": 100,
        "caption_max_chars": 5000,
        "hashtag_max": 3,
        "audio_target_lufs": -14.0,
        "audio_target_tp": -1.5,
    },
}


def get_platform_profile(service: str) -> PlatformProfile:
    definition = PROFILE_DEFINITIONS.get(service.lower())
    if definition is None:
        raise ValueError(f"Unknown platform profile: {service}")
    return PlatformProfile(**definition)


def build_platform_variant(
    edit_plan: EditPlan,
    composition: CompositionPlan,
    service: str,
    variant_id: Optional[str] = None,
) -> PlatformVariant:
    profile = get_platform_profile(service)
    duration = min(edit_plan.duration, profile.max_duration)
    start = edit_plan.start
    end = min(edit_plan.end, start + duration)
    if end - start < profile.min_duration:
        raise ValueError(f"{service} variant cannot satisfy minimum duration")
    return PlatformVariant(
        variant_id=variant_id or f"{edit_plan.plan_id}_{profile.service}",
        service=profile.service,
        start=round(start, 2),
        end=round(end, 2),
        width=profile.width,
        height=profile.height,
        fps=profile.fps,
        safe_zone=profile.safe_zone,
        caption_max_words=profile.caption_max_words,
        title_max_chars=profile.title_max_chars,
        caption_max_chars=profile.caption_max_chars,
        hashtag_max=profile.hashtag_max,
        metadata={
            "master_plan_id": edit_plan.plan_id,
            "composition_plan_id": composition.plan_id,
            "source_duration": edit_plan.duration,
            "duration_strategy": "trim_to_platform_limit" if edit_plan.duration > profile.max_duration else "preserve_master",
        },
    )


def save_feedback(feedback: PerformanceFeedback, output_path: Path) -> Path:
    payload = feedback.to_dict()
    if not payload["recorded_at"]:
        from datetime import datetime, timezone
        payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
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
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return output_path
