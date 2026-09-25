from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.composition_planner import CompositionPlan, build_composition_plan
from src.edit_director import EditPlan, HookCandidate, StoryBeat, build_semantic_edit_plans
from src.source_index import SourceIndex, build_source_index, save_source_index
from src.transcriber import TranscriptSegment, WordTimestamp


@dataclass
class ClipEditorArtifacts:
    source_index: SourceIndex
    edit_plan: EditPlan
    composition_plan: CompositionPlan
    semantic_suggestions: List[EditPlan]


def _relative_segments(
    segments: Sequence[TranscriptSegment],
    clip_start: float,
    clip_end: float,
) -> List[TranscriptSegment]:
    relative: List[TranscriptSegment] = []
    for segment in segments:
        start = max(0.0, segment.start - clip_start)
        end = min(clip_end - clip_start, segment.end - clip_start)
        if end <= start:
            continue
        words: List[WordTimestamp] = []
        for word in segment.words:
            word_start = max(0.0, word.start - clip_start)
            word_end = min(clip_end - clip_start, word.end - clip_start)
            if word_end <= word_start:
                continue
            words.append(WordTimestamp(
                word=word.word,
                start=word_start,
                end=word_end,
                is_estimated=word.is_estimated,
            ))
        relative.append(TranscriptSegment(
            start=start,
            end=end,
            text=segment.text,
            words=words,
        ))
    return sorted(relative, key=lambda segment: (segment.start, segment.end))


def _hook(segments: Sequence[TranscriptSegment], duration: float) -> HookCandidate:
    words: List[str] = []
    for segment in segments:
        for word in segment.words:
            if word.start < min(3.0, duration):
                words.append(word.word.strip(".,!?;:'\"()[]{}-"))
    text = " ".join(word for word in words if word)[:160]
    return HookCandidate(
        hook_type="story",
        text=text,
        start=0.0,
        end=min(3.0, duration),
        score=0.7 if text else 0.0,
        evidence=["clip_opening"],
    )


def _beats(segments: Sequence[TranscriptSegment]) -> List[StoryBeat]:
    return [
        StoryBeat(
            start=segment.start,
            end=segment.end,
            kind="context_or_method",
            text=segment.text.strip(),
            source_segment_indexes=[index],
        )
        for index, segment in enumerate(segments)
        if segment.text.strip()
    ]


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return path


def build_clip_editor_artifacts(
    video_path: Path,
    segments: Sequence[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    artifact_dir: Path,
    run_id: str,
    clip_index: int,
    sample_fps: float = 1.0,
    detect_faces: bool = False,
) -> ClipEditorArtifacts:
    duration = max(0.1, clip_end - clip_start)
    relative_segments = _relative_segments(segments, clip_start, clip_end)
    source_index = build_source_index(
        video_path,
        sample_fps=sample_fps,
        max_samples=120,
        detect_faces=detect_faces,
    )
    source_path = artifact_dir / f"clip_{clip_index}_source_index.json"
    save_source_index(source_index, source_path)
    suggestions = build_semantic_edit_plans(relative_segments, source_index, num_clips=1)
    warnings: List[str] = []
    if relative_segments and not relative_segments[-1].text.rstrip().endswith((".", "?", "!")):
        warnings.append("endpoint_not_proven_complete")
    plan = EditPlan(
        plan_id=f"{run_id}_clip_{clip_index}",
        start=0.0,
        end=round(duration, 2),
        hook=_hook(relative_segments, duration),
        beats=_beats(relative_segments),
        selected_shot_ids=[shot.shot_id for shot in source_index.shots],
        confidence=0.6 if relative_segments else 0.2,
        warnings=warnings,
    )
    composition = build_composition_plan(plan, source_index)
    edit_path = artifact_dir / f"clip_{clip_index}_edit_plan.json"
    composition_path = artifact_dir / f"clip_{clip_index}_composition_plan.json"
    suggestions_path = artifact_dir / f"clip_{clip_index}_semantic_suggestions.json"
    _write_json(edit_path, plan.to_dict())
    _write_json(composition_path, composition.to_dict())
    _write_json(suggestions_path, {"suggestions": [item.to_dict() for item in suggestions]})
    return ClipEditorArtifacts(
        source_index=source_index,
        edit_plan=plan,
        composition_plan=composition,
        semantic_suggestions=suggestions,
    )
