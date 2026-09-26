"""
Director v2: a multimodal shot-level edit director.

Before this module the "director" only ever produced text. It picked clip
windows and wrote a plan, then the renderer ignored every layout decision in
that plan and ran whatever `face_tracker` decided. Brain B could veto a render
but could never tell the renderer what to change, which is why failures read
`fix -> fail -> fix -> fail`.

This module closes that loop in four steps:

1. Sample keyframes from the clip and compose a labelled contact sheet, so the
   model actually *sees* the video instead of guessing from statistics.
2. Send that sheet plus the transcript plus the per-shot visual evidence to
   Gemini, which accepts images natively. No local vision model, no GPU.
3. Parse the reply into `ShotDirective` objects under strict validation. An
   untrusted model must never be able to select a layout the renderer cannot
   execute, a time range that does not exist, or a value that silently corrupts
   the filtergraph.
4. Apply the directives to the `FramingDecision` the renderer consumes, with
   per-directive guards and a full audit trail.

Every failure mode returns an empty directive list, leaving the existing
`face_tracker` behaviour untouched. Director v2 is additive by construction.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.engagement import analyze_window
from src.source_index import SourceIndex, sample_frames_at
from src.transcriber import TranscriptSegment

# The layouts the renderer can actually execute. A directive naming anything
# else is dropped rather than passed through: the renderer's filtergraph has a
# fixed set of branches, and an unknown mode would fall through to a branch the
# shot has no geometry for.
DIRECTOR_LAYOUTS: Tuple[str, ...] = (
    "presentation_slide",
    "portrait_face",
    "split_screen",
    "pip_slide",
    "blur_stack",
)

# Layouts that depend on face geometry being present on the shot. The director
# sees pixels and cannot see `FaceBox`, so it will confidently ask for a
# split_screen on a shot where only one face was ever detected. Honouring that
# would emit a split-screen filter chain with a missing box.
LAYOUT_GEOMETRY_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "split_screen": ("speaker1_box", "speaker2_box"),
    "pip_slide": ("speaker1_box",),
}

DIRECTOR_CROP_TARGETS: Tuple[str, ...] = ("auto", "speaker1", "speaker2", "center")
DIRECTOR_MOTIONS: Tuple[str, ...] = ("drift", "push_in", "pull_out", "hold")
# Anchors the composition planner can resolve. Mirrors composition_planner.
DIRECTOR_CAPTION_ANCHORS: Tuple[str, ...] = (
    "upper_center",
    "center",
    "lower_center",
    "left_safe",
    "right_safe",
    "split_divider",
)

# A directive must claim at least this much of the shot it is matched to.
# Below it we treat the reply as imprecise and leave the shot alone.
MIN_DIRECTIVE_OVERLAP = 0.5

# Contact sheet geometry.
CONTACT_TILE_WIDTH = 320
CONTACT_MAX_TILES = 16
CONTACT_COLUMNS = 4


@dataclass
class ShotDirective:
    """
    One shot-level edit instruction from the director.

    `start`/`end` are clip-relative seconds and identify the shot by time rather
    than by id. The two brains segment shots independently -- `face_tracker`
    drives the renderer while `SourceIndex` drives the plan -- so their shot ids
    do not correspond, and time overlap is the only reliable join key.
    """

    start: float
    end: float
    layout: Optional[str] = None
    crop_target: str = "auto"
    motion: str = "drift"
    caption_anchor: Optional[str] = None
    keep: bool = True
    reason: str = ""
    confidence: float = 0.5

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["duration"] = round(self.duration, 3)
        return payload


@dataclass
class DirectorReview:
    """Everything the director produced, including why it was or was not used."""

    directives: List[ShotDirective] = field(default_factory=list)
    audit: List[str] = field(default_factory=list)
    contact_sheet_path: Optional[Path] = None
    model: Optional[str] = None
    enabled: bool = True
    failure: Optional[str] = None

    @property
    def usable(self) -> bool:
        return bool(self.directives)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "failure": self.failure,
            "contact_sheet": str(self.contact_sheet_path) if self.contact_sheet_path else None,
            "directives": [directive.to_dict() for directive in self.directives],
            "audit": list(self.audit),
        }


@dataclass
class ContactSheet:
    path: Path
    tiles: List[float]
    columns: int
    rows: int
    tile_width: int
    tile_height: int

    def legend(self) -> str:
        """Map tile position to timestamp for the prompt text."""
        parts = []
        for index, timestamp in enumerate(self.tiles):
            row, column = divmod(index, self.columns)
            parts.append(f"r{row}c{column}={timestamp:.1f}s")
        return " ".join(parts)


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):
        return default
    return result


def _coerce_bool(value: Any, default: bool = True) -> bool:
    """
    Parse an LLM boolean without the `bool("false") is True` trap.

    Models routinely emit the JSON string `"false"`, which is truthy in Python.
    A single dropped shot on the strength of that is a silent content loss, so
    the string cases are handled explicitly.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("false", "no", "0", "cut", "drop", "off"):
            return False
        if token in ("true", "yes", "1", "keep", "on"):
            return True
        return default
    if value is None:
        return default
    return bool(value)


def _strip_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _letterbox_tile(frame: np.ndarray, width: int) -> np.ndarray:
    height, frame_width = frame.shape[:2]
    if frame_width <= 0 or height <= 0:
        return np.zeros((max(2, width * 9 // 16), width, 3), dtype=np.uint8)
    target_height = max(2, int(round(width * height / frame_width)))
    return cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)


def build_contact_sheet(
    video_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    tile_count: int = 12,
    columns: int = CONTACT_COLUMNS,
) -> Optional[ContactSheet]:
    """
    Compose a labelled grid of keyframes the model can reason about visually.

    Built with OpenCV rather than ffmpeg's `tile` + `drawtext` because that path
    needs a specific font file to be present on the runner, and a missing font
    would silently drop every timestamp label. OpenCV always has a built-in
    Hershey font, and it keeps the sheet testable without invoking ffmpeg.

    Returns None when the clip cannot be decoded or yields no frames.
    """
    safe_tiles = max(1, min(int(tile_count), CONTACT_MAX_TILES))
    try:
        samples = sample_frames_at(video_path, start_time, end_time, safe_tiles)
    except Exception as error:
        print(f"[-] Contact sheet sampling failed: {error}")
        return None
    frames = [(timestamp, frame) for timestamp, frame in samples if frame is not None]
    if not frames:
        return None

    tiles = [_letterbox_tile(frame, CONTACT_TILE_WIDTH) for _, frame in frames]
    tile_height = max(tile.shape[0] for tile in tiles)
    label_band = 22
    pad = 4
    cell_height = tile_height + label_band

    safe_columns = max(1, min(int(columns), len(tiles)))
    rows = int(np.ceil(len(tiles) / safe_columns))
    sheet_width = safe_columns * CONTACT_TILE_WIDTH + (safe_columns + 1) * pad
    sheet_height = rows * cell_height + (rows + 1) * pad
    sheet = np.full((sheet_height, sheet_width, 3), 24, dtype=np.uint8)

    for index, (timestamp, _) in enumerate(frames):
        row, column = divmod(index, safe_columns)
        tile = tiles[index]
        top = pad + row * cell_height
        left = pad + column * (CONTACT_TILE_WIDTH + pad)
        sheet[top:top + tile.shape[0], left:left + CONTACT_TILE_WIDTH] = tile
        cv2.putText(
            sheet,
            f"{index + 1}  t={timestamp:.1f}s",
            (left + 4, top + tile.shape[0] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 82]):
        print("[-] Contact sheet could not be encoded to JPEG.")
        return None
    return ContactSheet(
        path=output_path,
        tiles=[timestamp for timestamp, _ in frames],
        columns=safe_columns,
        rows=rows,
        tile_width=CONTACT_TILE_WIDTH,
        tile_height=tile_height,
    )


def _shot_evidence(index: Optional[SourceIndex], clip_start: float, clip_end: float) -> List[Dict[str, Any]]:
    if index is None:
        return []
    rows: List[Dict[str, Any]] = []
    for shot in index.shots:
        overlap_start = max(clip_start, shot.start)
        overlap_end = min(clip_end, shot.end)
        if overlap_end <= overlap_start:
            continue
        rows.append({
            "start": round(overlap_start, 2),
            "end": round(overlap_end, 2),
            "type": shot.shot_type,
            "motion": round(shot.motion_score, 3),
            "static": round(shot.static_score, 3),
            "presentation_ratio": round(shot.presentation_ratio, 3),
            "face_count": round(shot.face_count, 2),
            "text_regions": len(shot.text_regions),
            "freeze_intervals": [
                [round(start, 2), round(end, 2)] for start, end in shot.freeze_intervals
            ],
        })
    return rows


def build_director_prompt(
    contact_sheet: ContactSheet,
    source_index: Optional[SourceIndex],
    segments: Sequence[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    max_chars: int = 12000,
) -> str:
    """
    Build the shot-directive request.

    The contact sheet, the transcript, and the per-shot statistics are supplied
    together on purpose: the image tells the model what the shot *looks like*,
    the statistics tell it what the cheap analysers already measured, and the
    transcript tells it what is being said. Any one of the three alone produces
    noticeably worse layout decisions.
    """
    transcript_lines: List[str] = []
    used = 0
    for segment in segments:
        overlap_start = max(clip_start, segment.start)
        overlap_end = min(clip_end, segment.end)
        if overlap_end <= overlap_start:
            continue
        line = f"[{overlap_start - clip_start:.1f}s-{overlap_end - clip_start:.1f}s] {segment.text.strip()}"
        used += len(line)
        if used > max_chars:
            transcript_lines.append("... [truncated] ...")
            break
        transcript_lines.append(line)

    energy_rows = []
    for row in _shot_evidence(source_index, clip_start, clip_end):
        window = analyze_window(segments, row["start"], row["end"])
        row = dict(row)
        row["energy"] = round(window.score, 3)
        energy_rows.append(row)
    evidence_json = json.dumps(energy_rows, ensure_ascii=False)

    return f"""You are the senior edit director for a vertical short-form video (1080x1920).
You are shown a CONTACT SHEET of {len(contact_sheet.tiles)} keyframes sampled across the clip,
in reading order, each labelled with its clip-relative timestamp.

CONTACT SHEET LEGEND (row/column -> timestamp):
{contact_sheet.legend()}

TRANSCRIPT (clip-relative seconds):
{chr(10).join(transcript_lines) if transcript_lines else "(no speech recognised in this window)"}

DETECTED SHOTS (JSON, clip-relative seconds):
{evidence_json}

### YOUR TASK
Return one directive per detected shot, deciding how each shot should be edited.

Layout meanings (choose exactly one per shot):
- "presentation_slide": a slide or screen-share dominates. Face small or absent.
- "portrait_face": one speaker fills the frame. Tight crop on a talking head.
- "split_screen": two speakers visible at comparable size in one frame.
- "pip_slide": a slide fills the frame with the speaker as a small corner inset.
- "blur_stack": no usable subject; a blurred background stack is the only safe option.

HARD RULES:
1. Use ONLY the five layouts above. Any other value is discarded.
2. "split_screen" is only correct when the image really shows two comparable faces.
   "pip_slide" is only correct when the image shows a slide with one small face inset.
3. Every "start"/"end" must come from the DETECTED SHOTS list. Do not invent ranges.
4. Set "keep": false only for a shot that is genuinely unusable (black, a long
   frozen hold, a camera pointed at nothing). Keep the strong shots.
5. Never return fewer than one directive with "keep": true. A clip must not be emptied.
6. Reason from the image. The transcript explains the edit's purpose, not the layout.

### OUTPUT FORMAT -- valid JSON only, no markdown fence, no commentary:
{{
  "directives": [
    {{
      "start": 0.0,
      "end": 4.2,
      "layout": "presentation_slide",
      "crop_target": "auto",
      "motion": "drift",
      "caption_anchor": "center",
      "keep": true,
      "reason": "Full-screen slide with the host in a small corner inset",
      "confidence": 0.82
    }}
  ]
}}
"""


def parse_shot_directives(
    raw_text: str,
    clip_duration: float,
    max_directives: int = 32,
) -> Tuple[List[ShotDirective], List[str]]:
    """
    Parse a director reply into validated directives.

    Validation is deliberately per-field rather than all-or-nothing: a reply
    with one bad layout should still yield a usable `keep`/`reason` for that
    shot, because dropping the whole plan over one stray enum value would make
    the director far less useful than the tracker it supplements. Anything that
    cannot be validated is replaced with a safe default, never trusted.

    Returns (directives, audit).
    """
    audit: List[str] = []
    if not raw_text or not raw_text.strip():
        audit.append("empty_response")
        return [], audit
    try:
        payload = json.loads(_strip_json_fence(raw_text))
    except (json.JSONDecodeError, ValueError) as error:
        audit.append(f"json_unparseable:{error}")
        return [], audit
    if not isinstance(payload, dict):
        audit.append("response_not_an_object")
        return [], audit
    raw_items = payload.get("directives")
    if not isinstance(raw_items, list) or not raw_items:
        audit.append("no_directives_key")
        return [], audit

    duration = max(0.0, clip_duration)
    directives: List[ShotDirective] = []
    for index, item in enumerate(raw_items[:max_directives]):
        if not isinstance(item, dict):
            audit.append(f"[{index}] not_an_object")
            continue
        start = _coerce_float(item.get("start"))
        end = _coerce_float(item.get("end"))
        if start is None or end is None:
            audit.append(f"[{index}] missing_or_bad_range")
            continue
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start < 0.1:
            audit.append(f"[{index}] range_too_short")
            continue

        layout_raw = item.get("layout")
        layout: Optional[str] = None
        if isinstance(layout_raw, str):
            candidate = layout_raw.strip().lower()
            if candidate in DIRECTOR_LAYOUTS:
                layout = candidate
            else:
                audit.append(f"[{index}] unknown_layout:{candidate[:40]!r}")

        crop_target = str(item.get("crop_target", "auto")).strip().lower()
        if crop_target not in DIRECTOR_CROP_TARGETS:
            audit.append(f"[{index}] unknown_crop_target:{crop_target[:24]!r}")
            crop_target = "auto"

        motion = str(item.get("motion", "drift")).strip().lower()
        if motion not in DIRECTOR_MOTIONS:
            audit.append(f"[{index}] unknown_motion:{motion[:24]!r}")
            motion = "drift"

        caption_anchor: Optional[str] = None
        anchor_raw = item.get("caption_anchor")
        if isinstance(anchor_raw, str):
            candidate = anchor_raw.strip().lower()
            if candidate in DIRECTOR_CAPTION_ANCHORS:
                caption_anchor = candidate
            elif candidate:
                audit.append(f"[{index}] unknown_caption_anchor:{candidate[:32]!r}")

        confidence = _coerce_float(item.get("confidence"), default=0.5)
        directives.append(ShotDirective(
            start=round(start, 2),
            end=round(end, 2),
            layout=layout,
            crop_target=crop_target,
            motion=motion,
            caption_anchor=caption_anchor,
            keep=_coerce_bool(item.get("keep"), default=True),
            reason=str(item.get("reason", "")).strip()[:280],
            confidence=round(max(0.0, min(1.0, confidence or 0.5)), 3),
        ))

    if not directives:
        audit.append("all_directives_rejected")
    return directives, audit


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_directive_for_shot(
    shot_start: float,
    shot_end: float,
    directives: Sequence[ShotDirective],
) -> Optional[ShotDirective]:
    """Pick the directive with the greatest overlap with this shot."""
    best: Optional[ShotDirective] = None
    best_overlap = 0.0
    for directive in directives:
        overlap = _overlap(shot_start, shot_end, directive.start, directive.end)
        if overlap > best_overlap:
            best, best_overlap = directive, overlap
    if best is None or best_overlap < MIN_DIRECTIVE_OVERLAP:
        return None
    return best


def apply_shot_directives(
    framing: Any,
    directives: Sequence[ShotDirective],
) -> Tuple[Any, List[str]]:
    """
    Apply directives to a `FramingDecision`, returning a new decision and an audit.

    The input is never mutated: the renderer may be invoked more than once by the
    repair loop, and a directive that had already been applied must not be
    applied twice.

    Guards, in order:
    - A shot's layout can only change to a layout whose geometry the shot has.
    - A dropped shot is restored if it would leave the clip with no shots.
    - A directive that does not meaningfully overlap its shot is ignored.
    """
    audit: List[str] = []
    shots = list(getattr(framing, "shots", []) or [])
    if not shots or not directives:
        return framing, audit

    updated = copy.copy(framing)
    updated.shots = [copy.copy(shot) for shot in shots]

    kept: List[int] = []
    for position, shot in enumerate(updated.shots):
        directive = _best_directive_for_shot(shot.start, shot.end, directives)
        if directive is None:
            audit.append(f"{position}:{shot.start:.2f}-{shot.end:.2f}:no_directive")
            kept.append(position)
            continue
        if not directive.keep:
            audit.append(
                f"{position}:{shot.start:.2f}-{shot.end:.2f}:dropped_by_director"
            )
            continue
        kept.append(position)

        label = f"{position}:{shot.start:.2f}-{shot.end:.2f}"
        if directive.layout and directive.layout != shot.mode:
            required = LAYOUT_GEOMETRY_REQUIREMENTS.get(directive.layout, ())
            missing = [name for name in required if not getattr(shot, name, None)]
            if missing:
                audit.append(
                    f"{label}:layout_{directive.layout}_rejected_missing_{'_'.join(missing)}"
                )
            else:
                audit.append(f"{label}:layout_{shot.mode}->{directive.layout}")
                shot.mode = directive.layout

        if directive.motion != "drift":
            if directive.motion not in DIRECTOR_MOTIONS:
                audit.append(f"{label}:motion_rejected")
            else:
                # 'hold' does not disable the drift. The motion guarantee is
                # unconditional; a hold only means "add no extra zoom", because
                # honouring it as a freeze would reintroduce the exact defect
                # Phase 0 was written to eliminate.
                audit.append(f"{label}:motion->{directive.motion}")
                shot.motion = directive.motion

        if directive.crop_target in ("speaker1", "speaker2"):
            box_name = f"{directive.crop_target}_box"
            if not getattr(shot, box_name, None):
                audit.append(f"{label}:crop_target_{directive.crop_target}_rejected_no_box")
            elif shot.mode == "split_screen":
                # A split screen already uses both boxes; picking one would
                # silently break the other half.
                audit.append(f"{label}:crop_target_ignored_in_split_screen")
            else:
                source_box = getattr(shot, box_name)
                if directive.crop_target == "speaker2":
                    shot.speaker1_box = source_box
                audit.append(f"{label}:crop_target->{directive.crop_target}")
        elif directive.crop_target == "center":
            audit.append(f"{label}:crop_target->center")

    if not kept:
        audit.append("all_shots_dropped_restored_first")
        kept = [0]

    updated.shots = [updated.shots[position] for position in kept]
    return updated, audit


def apply_caption_directives(
    composition_plan: Any,
    directives: Sequence[ShotDirective],
) -> Tuple[Any, List[str]]:
    """
    Apply director caption anchors to a `CompositionPlan`, returning a new plan.

    Guarded the same way as layouts: an anchor the composition planner cannot
    resolve, or a directive that does not overlap a shot, is ignored rather than
    propagated into ASS output the subtitle generator has no style for.

    `motion` is intentionally NOT executed. The renderer's per-frame zoom path
    has no regression test yet, so a model asking for "push_in" is recorded in
    the decision log and kept out of the filtergraph until that path is proven.
    """
    from src.composition_planner import _caption_rect

    audit: List[str] = []
    shots = list(getattr(composition_plan, "shots", []) or [])
    if not shots or not directives:
        return composition_plan, audit

    updated = copy.copy(composition_plan)
    updated.shots = [copy.copy(shot) for shot in shots]
    for position, shot in enumerate(updated.shots):
        directive = _best_directive_for_shot(shot.start, shot.end, directives)
        if directive is None or not directive.keep or not directive.caption_anchor:
            continue
        if directive.caption_anchor not in DIRECTOR_CAPTION_ANCHORS:
            audit.append(f"{position}:caption_anchor_rejected")
            continue
        if directive.caption_anchor == shot.caption_anchor:
            continue
        audit.append(
            f"{position}:caption_{shot.caption_anchor}->{directive.caption_anchor}"
        )
        shot.caption_anchor = directive.caption_anchor
        shot.caption_rect = _caption_rect(
            directive.caption_anchor, composition_plan.safe_zone
        )
    return updated, audit


def plan_shot_directives(
    video_path: Path,
    source_index: Optional[SourceIndex],
    segments: Sequence[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    output_path: Path,
    api_key: Optional[str] = None,
    tile_count: int = 12,
    max_characters: int = 12000,
) -> DirectorReview:
    """
    Run the full director pass. Never raises; every failure yields an empty review.

    The caller is the render path, so an unavailable API, a missing key, an
    undecodable clip, or a malformed reply must all degrade to "no directives"
    rather than abort a render that the existing tracker can already produce.
    """
    from src.config import DIRECTOR_V2_ENABLED, GEMINI_API_KEY
    from src.viral_detector import query_gemini_models

    if not DIRECTOR_V2_ENABLED:
        return DirectorReview(enabled=False, failure="disabled_by_config")

    duration = max(0.1, clip_end - clip_start)
    review = DirectorReview(enabled=True)

    sheet = build_contact_sheet(
        video_path=video_path,
        start_time=clip_start,
        end_time=clip_end,
        output_path=output_path,
        tile_count=tile_count,
    )
    if sheet is None:
        review.failure = "contact_sheet_unavailable"
        return review
    review.contact_sheet_path = sheet.path

    key = api_key or GEMINI_API_KEY
    if not key:
        review.failure = "no_api_key"
        return review

    prompt = build_director_prompt(
        contact_sheet=sheet,
        source_index=source_index,
        segments=segments,
        clip_start=clip_start,
        clip_end=clip_end,
        max_chars=max_characters,
    )
    try:
        raw = query_gemini_models(
            prompt,
            key,
            image_path=sheet.path,
            context_note=f"contact sheet {sheet.path.name} with {len(sheet.tiles)} keyframes",
        )
    except Exception as error:
        review.failure = f"api_error:{error}"
        return review

    if not raw:
        review.failure = "no_api_response"
        return review

    directives, parse_audit = parse_shot_directives(raw, duration)
    review.audit.extend(parse_audit)
    review.directives = directives
    if not directives:
        review.failure = "no_valid_directives"
    return review
