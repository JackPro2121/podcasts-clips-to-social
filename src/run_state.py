from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from src.editor_models import RunManifest, SCHEMA_VERSION, StageStatus, sanitize_settings


class RunStateStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _safe_run_id(self, run_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
            raise ValueError(f"Invalid run_id: {run_id}")
        return run_id

    def manifest_path(self, run_id: str) -> Path:
        return self.root / self._safe_run_id(run_id) / "manifest.json"

    def create(
        self,
        run_id: str,
        source_url: str,
        source_id: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ) -> RunManifest:
        manifest = RunManifest(
            run_id=self._safe_run_id(run_id),
            source_url=source_url,
            source_id=source_id,
            settings=sanitize_settings(settings or {}),
        )
        self.save(manifest)
        return manifest

    def save(self, manifest: RunManifest) -> Path:
        path = self.manifest_path(manifest.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix="manifest.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(manifest.to_dict(), handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
        return path

    def load(self, run_id: str) -> RunManifest:
        path = self.manifest_path(run_id)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        manifest = RunManifest.from_dict(payload)
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema: {manifest.schema_version}")
        return manifest

    def set_stage(self, run_id: str, stage: str, status: StageStatus) -> RunManifest:
        manifest = self.load(run_id)
        manifest.stages[stage] = status.value
        manifest.touch()
        self.save(manifest)
        return manifest

    def record_artifact(self, run_id: str, name: str, path: Path | str) -> RunManifest:
        manifest = self.load(run_id)
        manifest.artifacts[name] = str(path)
        manifest.touch()
        self.save(manifest)
        return manifest

    def set_status(self, run_id: str, status: str) -> RunManifest:
        manifest = self.load(run_id)
        manifest.status = status
        manifest.touch()
        self.save(manifest)
        return manifest

    def mark_failed(self, run_id: str, error: str) -> RunManifest:
        manifest = self.load(run_id)
        manifest.status = "failed"
        manifest.error = error
        manifest.touch()
        self.save(manifest)
        return manifest
