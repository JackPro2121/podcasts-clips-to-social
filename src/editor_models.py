from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


SCHEMA_VERSION = "0.1"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RunManifest:
    run_id: str
    source_url: str
    source_id: str = ""
    schema_version: str = SCHEMA_VERSION
    status: str = "created"
    settings: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    qa: Dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""
    output_hash: str = ""
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunManifest":
        if not isinstance(payload, dict):
            raise ValueError("Run manifest must be a JSON object")
        if not payload.get("run_id") or not payload.get("source_url"):
            raise ValueError("Run manifest requires run_id and source_url")
        return cls(
            run_id=str(payload["run_id"]),
            source_url=str(payload["source_url"]),
            source_id=str(payload.get("source_id", "")),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            status=str(payload.get("status", "created")),
            settings=dict(payload.get("settings", {})),
            stages=dict(payload.get("stages", {})),
            artifacts=dict(payload.get("artifacts", {})),
            qa=dict(payload.get("qa", {})),
            input_hash=str(payload.get("input_hash", "")),
            output_hash=str(payload.get("output_hash", "")),
            error=str(payload.get("error", "")),
            created_at=str(payload.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=str(payload.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "num_clips",
        "niche",
        "framing",
        "subtitle_style",
        "subtitles_mode",
        "post_to_buffer",
        "watermark",
        "platform",
        "run_mode",
    }
    safe: Dict[str, Any] = {}
    for key in sorted(settings):
        if key not in allowed:
            continue
        value = settings[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    return safe
