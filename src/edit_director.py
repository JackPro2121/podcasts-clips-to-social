from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import MIN_CLIP_DURATION, SHORT_FORM_MAX_DURATION
from src.engagement import analyze_window
from src.source_index import SourceIndex
from src.transcriber import TranscriptSegment, WordTimestamp


@dataclass
class HookCandidate:
    hook_type: str
    text: str
    start: float
    end: float
    score: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class StoryBeat:
    start: float
    end: float
    kind: str
    text: str
    source_segment_indexes: List[int] = field(default_factory=list)


@dataclass
class EditPlan:
    plan_id: str
    start: float
    end: float
    hook: HookCandidate
    beats: List[StoryBeat]
    selected_shot_ids: List[str]
    confidence: float
    warnings: List[str] = field(default_factory=list)
    schema_version: str = "0.1"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


RESULT_WORDS = {"paid", "house", "million", "free", "income", "dream", "bought", "won"}
CONTRAST_WORDS = {"but", "never", "instead", "wrong", "cut", "because", "nobody"}
CURIOSITY_WORDS = {"how", "why", "what", "secret", "nobody", "everyone"}
FILLER_WORDS = {"um", "uh", "yeah", "you", "know", "like", "really"}


def _clean_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _is_complete_word(word: str) -> bool:
    return word.rstrip("\"')]} ").endswith((".", "?", "!"))


def _segment_words(segment: TranscriptSegment) -> List[WordTimestamp]:
    return list(segment.words)


def _segment_complete(segment: TranscriptSegment) -> bool:
    words = _segment_words(segment)
    if words:
        return _is_complete_word(words[-1].word) or _is_complete_word(segment.text)
    return _is_complete_word(segment.text)


def _boundaries(segments: Sequence[TranscriptSegment]) -> List[Tuple[float, float, bool, str, int]]:
    boundaries: List[Tuple[float, float, bool, str, int]] = []
    for index, segment in enumerate(segments):
        if not boundaries or abs(boundaries[-1][1] - segment.start) > 0.05:
            boundaries.append((round(segment.start, 2), round(segment.start, 2), False, "", index))
        previous_end = boundaries[-1][1] if boundaries else segment.start
        if segment.end > previous_end:
            boundaries.append((
                round(segment.start, 2),
                round(segment.end, 2),
                _segment_complete(segment),
                segment.text.strip(),
                index,
            ))
    return boundaries


def _words_in_window(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
) -> List[Tuple[float, str]]:
    words: List[Tuple[float, str]] = []
    for segment in segments:
        for word in segment.words:
            if word.start < end and word.end > start:
                words.append((word.start, _clean_word(word.word)))
    return sorted(words)


def _hook_for_window(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
) -> HookCandidate:
    words = _words_in_window(segments, start, min(end, start + 3.0))
    tokens = [token for _, token in words if token]
    text = " ".join(tokens)
    lowered = set(tokens)
    if lowered & RESULT_WORDS and any(token.isdigit() for token in tokens):
        hook_type = "result_first"
    elif lowered & CONTRAST_WORDS:
        hook_type = "contrarian"
    elif lowered & CURIOSITY_WORDS:
        hook_type = "curiosity"
    else:
        hook_type = "story"
    filler_count = sum(1 for token in tokens if token in FILLER_WORDS)
    score = max(0.0, min(1.0, 0.45 + len(tokens) * 0.03 - filler_count * 0.04))
    return HookCandidate(
        hook_type=hook_type,
        text=text[:160],
        start=round(start, 2),
        end=round(min(end, start + 3.0), 2),
        score=round(score, 4),
        evidence=["spoken_opening", f"words={len(tokens)}"],
    )


def _visual_score(index: Optional[SourceIndex], start: float, end: float) -> Tuple[float, List[str], List[str]]:
    if index is None:
        return 0.5, [], []
    overlapping = [
        shot for shot in index.shots
        if shot.end > start and shot.start < end
    ]
    if not overlapping:
        return 0.35, [], []
    shot_ids = [shot.shot_id for shot in overlapping]
    motion = sum(shot.motion_score for shot in overlapping) / len(overlapping)
    static = sum(shot.static_score for shot in overlapping) / len(overlapping)
    text_penalty = min(0.3, sum(len(shot.text_regions) for shot in overlapping) * 0.015)
    freeze_penalty = min(0.4, sum(
        max(0.0, min(shot.end, end) - max(shot.start, start))
        for shot in overlapping
        for _ in shot.freeze_intervals
    ) * 0.08)
    score = max(0.0, min(1.0, 0.55 + motion * 0.35 - static * 0.15 - text_penalty - freeze_penalty))
    return score, shot_ids, [f"visual_score={score:.3f}"]


def _classify_beat(text: str) -> str:
    tokens = {_clean_word(token) for token in text.split()}
    if tokens & RESULT_WORDS:
        return "payoff_or_proof"
    if tokens & CONTRAST_WORDS:
        return "tension_or_contrast"
    if any(token.isdigit() for token in tokens):
        return "specific_evidence"
    return "context_or_method"


def _build_beats(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
) -> List[StoryBeat]:
    beats: List[StoryBeat] = []
    for index, segment in enumerate(segments):
        overlap_start = max(start, segment.start)
        overlap_end = min(end, segment.end)
        if overlap_end <= overlap_start:
            continue
        text = segment.text.strip()
        if text:
            beats.append(StoryBeat(
                start=round(overlap_start, 2),
                end=round(overlap_end, 2),
                kind=_classify_beat(text),
                text=text,
                source_segment_indexes=[index],
            ))
    return beats


def _score_plan(
    hook: HookCandidate,
    start: float,
    end: float,
    visual_score: float,
    segments: Sequence[TranscriptSegment],
) -> float:
    duration = end - start
    duration_score = 1.0 - min(1.0, abs(duration - 60.0) / 100.0)
    words = _words_in_window(segments, start, end)
    payoff_words = sum(1 for _, token in words if token in RESULT_WORDS)
    payoff_score = min(1.0, payoff_words / 5.0)
    filler_count = sum(1 for _, token in words if token in FILLER_WORDS)
    filler_score = max(0.0, 1.0 - filler_count / max(20, len(words)))
    energy_score = analyze_window(segments, start, end).score
    score = (
        hook.score * 0.30
        + visual_score * 0.20
        + energy_score * 0.20
        + duration_score * 0.15
        + payoff_score * 0.10
        + filler_score * 0.05
    )
    return round(max(0.0, min(1.0, score)), 4)


def duration_warnings(start: float, end: float) -> List[str]:
    """
    Flag windows that sit outside the documented short-form sweet spot.

    Reported rather than clamped: narrowing MIN/MAX_CLIP_DURATION risks the
    detector finding no window at all, which aborts the whole run. Making the
    deviation visible lets an operator tighten the gate deliberately instead.
    """
    warnings: List[str] = []
    duration = end - start
    if duration > SHORT_FORM_MAX_DURATION:
        warnings.append(
            f"exceeds_short_form_window:{duration:.1f}s>{SHORT_FORM_MAX_DURATION:.0f}s"
        )
    if duration < MIN_CLIP_DURATION:
        warnings.append(f"below_min_clip_duration:{duration:.1f}s")
    return warnings


def build_semantic_edit_plans(
    segments: Sequence[TranscriptSegment],
    source_index: Optional[SourceIndex] = None,
    num_clips: int = 1,
    min_duration: float = 30.0,
    max_duration: float = 140.0,
) -> List[EditPlan]:
    if not segments or num_clips <= 0:
        return []
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end))
    boundaries = _boundaries(ordered)
    starts = list(dict.fromkeys(item[0] for item in boundaries))
    ends = sorted({item[1] for item in boundaries})
    candidates: List[EditPlan] = []
    for start in starts:
        minimum_end = start + min_duration
        maximum_end = start + max_duration
        possible_ends = [end for end in ends if minimum_end <= end <= maximum_end]
        if not possible_ends:
            continue
        end = possible_ends[0]
        for candidate_end in possible_ends:
            candidate_boundary = next((item for item in boundaries if abs(item[1] - candidate_end) < 0.05), None)
            if candidate_boundary and candidate_boundary[2]:
                end = candidate_end
                break
        hook = _hook_for_window(ordered, start, end)
        visual_score, shot_ids, visual_evidence = _visual_score(source_index, start, end)
        confidence = _score_plan(hook, start, end, visual_score, ordered)
        warnings: List[str] = []
        end_segment = next((segment for segment in ordered if abs(segment.end - end) < 0.05), None)
        if end_segment is not None and not _segment_complete(end_segment):
            warnings.append("endpoint_requires_review")
        if source_index is not None and source_index.quality.get("freeze_duration", 0) > 0.8:
            warnings.append("source_contains_freeze_intervals")
        candidates.append(EditPlan(
            plan_id=f"plan_{len(candidates) + 1:04d}_{int(start)}",
            start=round(start, 2),
            end=round(end, 2),
            hook=hook,
            beats=_build_beats(ordered, start, end),
            selected_shot_ids=shot_ids,
            confidence=confidence,
            warnings=warnings + duration_warnings(start, end),
        ))
    candidates.sort(key=lambda plan: plan.confidence, reverse=True)
    selected: List[EditPlan] = []
    for candidate in candidates:
        if any(
            not (candidate.end <= existing.start or candidate.start >= existing.end)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= num_clips:
            break
    return selected


def validate_edit_plan(plan: EditPlan, source_duration: Optional[float] = None) -> List[str]:
    errors: List[str] = []
    if plan.end <= plan.start:
        errors.append("end_must_be_after_start")
    duration = plan.duration
    if duration < 30.0 or duration > 140.0:
        errors.append("duration_out_of_contract")
    if not plan.hook.text.strip():
        errors.append("hook_text_missing")
    if not plan.beats:
        errors.append("story_beats_missing")
    if source_duration is not None and plan.end > source_duration + 0.01:
        errors.append("plan_exceeds_source")
    if "endpoint_requires_review" in plan.warnings:
        errors.append("endpoint_not_proven_complete")
    return errors


def build_director_prompt(
    segments: Sequence[TranscriptSegment],
    source_index: Optional[SourceIndex],
    num_clips: int = 1,
) -> str:
    transcript = "\n".join(
        f"{segment.start:.2f}-{segment.end:.2f} {segment.text.strip()}"
        for segment in segments
    )
    visual_evidence: List[Dict[str, Any]] = []
    if source_index is not None:
        visual_evidence = [
            {
                "shot_id": shot.shot_id,
                "start": shot.start,
                "end": shot.end,
                "type": shot.shot_type,
                "motion": shot.motion_score,
                "static": shot.static_score,
                "presentation_ratio": shot.presentation_ratio,
                "text_regions": len(shot.text_regions),
                "freeze_intervals": shot.freeze_intervals,
            }
            for shot in source_index.shots
        ]
    evidence_json = json.dumps(visual_evidence, ensure_ascii=False)
    return f"""You are the senior video edit director. Plan complete short-form edits using transcript and visual evidence.
Return {num_clips} strongest self-contained edit plans. Every plan must have a complete start and end boundary, a spoken or visual hook in the first 1-3 seconds, story beats, visual shot evidence, and a complete payoff. Reject partial words, static holds, text collisions, and unsupported timestamps.

TRANSCRIPT:
{transcript}

VISUAL_EVIDENCE_JSON:
{evidence_json}
"""
