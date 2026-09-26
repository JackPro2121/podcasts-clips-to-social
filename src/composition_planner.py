from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from src.edit_director import EditPlan
from src.source_index import IndexedShot, SourceIndex, TextRegion


@dataclass
class NormalizedRect:
    x: float
    y: float
    width: float
    height: float

    def intersects(self, other: "NormalizedRect", tolerance: float = 0.01) -> bool:
        return not (
            self.x + self.width <= other.x + tolerance
            or other.x + other.width <= self.x + tolerance
            or self.y + self.height <= other.y + tolerance
            or other.y + other.height <= self.y + tolerance
        )


@dataclass
class SafeZone:
    top: float
    bottom: float
    left: float
    right: float


@dataclass
class OverlayPlan:
    name: str
    start: float
    end: float
    anchor: str
    rect: NormalizedRect
    z_index: int = 10
    reason: str = ""
    allow_dynamic_motion: bool = False


@dataclass
class ShotComposition:
    shot_id: str
    start: float
    end: float
    source_type: str
    crop: NormalizedRect
    caption_anchor: str
    caption_rect: NormalizedRect
    protected_regions: List[NormalizedRect] = field(default_factory=list)
    broll_mode: str = "none"
    transition: str = "cut"
    max_static_hold: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class CompositionPlan:
    plan_id: str
    width: int
    height: int
    safe_zone: SafeZone
    shots: List[ShotComposition]
    overlays: List[OverlayPlan] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    schema_version: str = "0.1"

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "width": self.width,
            "height": self.height,
            "safe_zone": asdict(self.safe_zone),
            "shots": [asdict(shot) for shot in self.shots],
            "overlays": [asdict(overlay) for overlay in self.overlays],
            "decisions": self.decisions,
        }


DEFAULT_SAFE_ZONE = SafeZone(top=0.12, bottom=0.22, left=0.08, right=0.16)


def _rect_from_region(region: TextRegion) -> NormalizedRect:
    return NormalizedRect(region.x, region.y, region.width, region.height)


def _caption_rect(anchor: str, safe_zone: SafeZone) -> NormalizedRect:
    content_width = 1.0 - safe_zone.left - safe_zone.right
    if anchor == "split_divider":
        return NormalizedRect(0.48, 0.42, 0.04, 0.16)
    if anchor == "upper_center":
        return NormalizedRect(safe_zone.left, 0.24, content_width, 0.14)
    if anchor == "center":
        return NormalizedRect(safe_zone.left, 0.42, content_width, 0.14)
    if anchor == "left_safe":
        return NormalizedRect(safe_zone.left, 0.58, content_width * 0.48, 0.14)
    if anchor == "right_safe":
        return NormalizedRect(1.0 - safe_zone.right - content_width * 0.48, 0.58, content_width * 0.48, 0.14)
    return NormalizedRect(safe_zone.left, 0.62, content_width, 0.14)


def _protected_regions(shot: IndexedShot) -> List[NormalizedRect]:
    return [_rect_from_region(region) for region in shot.text_regions]


def _bottom_text_collision(regions: List[NormalizedRect], safe_zone: SafeZone) -> bool:
    bottom_limit = 1.0 - safe_zone.bottom
    return any(region.y + region.height >= bottom_limit - 0.08 for region in regions)


def _caption_anchor(shot: IndexedShot, protected_regions: List[NormalizedRect], safe_zone: SafeZone) -> str:
    if shot.shot_type == "split_screen" or shot.face_count >= 1.5:
        return "split_divider"
    if protected_regions and _bottom_text_collision(protected_regions, safe_zone):
        return "upper_center"
    if shot.shot_type == "presentation":
        return "center"
    if shot.face_count >= 1.0:
        return "lower_center"
    return "center"


def _shot_crop(shot: IndexedShot) -> NormalizedRect:
    if shot.shot_type == "split_screen":
        return NormalizedRect(0.0, 0.0, 1.0, 1.0)
    if shot.shot_type == "presentation":
        return NormalizedRect(0.0, 0.0, 1.0, 1.0)
    return NormalizedRect(0.0, 0.0, 1.0, 1.0)


def _freeze_hold(shot: IndexedShot) -> float:
    return max((end - start for start, end in shot.freeze_intervals), default=0.0)


@dataclass
class CaptionPlacement:
    """A concrete, pixel-based caption position resolved from a plan anchor.

    The composition planner reasons in normalised coordinates so it can be tested
    without a renderer, while ASS subtitles are positioned in output pixels. This
    is the single conversion point, so the director's decision about *where*
    text goes can actually reach the renderer.
    """

    start: float
    end: float
    anchor: str
    shot_id: str
    # ASS alignment codes: 2 = bottom-centre, 5 = middle-centre, 8 = top-centre
    alignment: int
    # Distance from the frame edge named by `alignment`, in output pixels.
    margin_v: int
    # True when the anchor exists to dodge source text rather than by preference.
    collision_avoidance: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _anchor_to_ass(anchor: str, rect: NormalizedRect, width: int, height: int) -> Tuple[int, int, bool]:
    """
    Map a normalised caption anchor onto an ASS alignment and vertical margin.

    Returns (alignment, margin_v, collision_avoidance).
    """
    # A bottom-anchored caption sits `margin_v` pixels up from the bottom edge,
    # so the safe zone's bottom padding is the minimum legal value.
    from_bottom_px = int(round((1.0 - (rect.y + rect.height)) * height))
    from_top_px = int(round(rect.y * height))

    if anchor == "upper_center":
        # Explicitly chosen because the lower band is occupied by source text.
        return 8, from_top_px, True
    if anchor in ("center", "split_divider"):
        return 5, from_top_px, False
    if anchor in ("left_safe", "right_safe"):
        # ASS cannot horizontally offset a centre-aligned line via margins, so
        # these keep bottom-centre placement; the anchor still documents intent.
        return 2, from_bottom_px, False
    return 2, from_bottom_px, False


def _force_collision_avoidance(
    shot: ShotComposition,
    rect: NormalizedRect,
) -> Tuple[str, NormalizedRect]:
    """
    Relocate a caption that overlaps the shot's protected source text.

    Used on a repair attempt: QA reported a caption_source_collision, so the
    caption is moved out of the way rather than re-rendering the same overlapping
    layout. Returns the anchor name alongside the rect so the reported anchor can
    never disagree with the geometry that was actually used.
    """
    if not shot.protected_regions:
        return shot.caption_anchor, rect
    if not any(rect.intersects(region) for region in shot.protected_regions):
        return shot.caption_anchor, rect
    for candidate in ("upper_center", "center"):
        alternative = _caption_rect(candidate, DEFAULT_SAFE_ZONE)
        if not any(alternative.intersects(region) for region in shot.protected_regions):
            return candidate, alternative
    # Every safe band is blocked; keep the original so the QA error still reports.
    return shot.caption_anchor, rect


def build_caption_placements(
    plan: CompositionPlan,
    width: int = 1080,
    height: int = 1920,
    avoid_collisions: bool = False,
) -> List[CaptionPlacement]:
    """Resolve a CompositionPlan into time-ranged caption placements.

    avoid_collisions relocates any caption that still overlaps a shot's protected
    source text, which is the render-side remedy for a caption_source_collision
    QA error.
    """
    placements: List[CaptionPlacement] = []
    for shot in plan.shots:
        anchor = shot.caption_anchor
        rect = shot.caption_rect
        if avoid_collisions:
            anchor, rect = _force_collision_avoidance(shot, rect)
        alignment, margin_v, avoidance = _anchor_to_ass(anchor, rect, width, height)
        placements.append(CaptionPlacement(
            start=shot.start,
            end=shot.end,
            anchor=anchor,
            shot_id=shot.shot_id,
            alignment=alignment,
            margin_v=max(0, margin_v),
            collision_avoidance=avoidance,
        ))
    placements.sort(key=lambda item: item.start)
    return placements


def build_composition_plan(
    edit_plan: EditPlan,
    source_index: SourceIndex,
    width: int = 1080,
    height: int = 1920,
    safe_zone: Optional[SafeZone] = None,
) -> CompositionPlan:
    resolved_safe_zone = safe_zone or DEFAULT_SAFE_ZONE
    overlapping_shots = [
        shot for shot in source_index.shots
        if shot.end > edit_plan.start and shot.start < edit_plan.end
    ]
    shot_plans: List[ShotComposition] = []
    overlays: List[OverlayPlan] = []
    decisions: List[str] = []
    for shot in overlapping_shots:
        shot_start = max(edit_plan.start, shot.start)
        shot_end = min(edit_plan.end, shot.end)
        protected = _protected_regions(shot)
        anchor = _caption_anchor(shot, protected, resolved_safe_zone)
        caption_rect = _caption_rect(anchor, resolved_safe_zone)
        static_hold = _freeze_hold(shot)
        broll_mode = "ken_burns_motion" if static_hold >= 0.8 else "none"
        transition = "motion_cut" if static_hold >= 0.8 else "cut"
        if static_hold >= 0.8:
            decisions.append(f"{shot.shot_id}:static_hold={static_hold:.2f}s->motion")
        if protected:
            decisions.append(f"{shot.shot_id}:protected_text_regions={len(protected)}")
        if shot.shot_type == "split_screen":
            decisions.append(f"{shot.shot_id}:split_layout")
        shot_plans.append(ShotComposition(
            shot_id=shot.shot_id,
            start=round(shot_start, 2),
            end=round(shot_end, 2),
            source_type=shot.shot_type,
            crop=_shot_crop(shot),
            caption_anchor=anchor,
            caption_rect=caption_rect,
            protected_regions=protected,
            broll_mode=broll_mode,
            transition=transition,
            max_static_hold=round(min(static_hold, 2.5), 2),
            warnings=["source_text_requires_protection"] if protected else [],
        ))
        overlays.append(OverlayPlan(
            name="captions",
            start=round(shot_start, 2),
            end=round(shot_end, 2),
            anchor=anchor,
            rect=caption_rect,
            z_index=30,
            reason="dynamic_visual_placement",
            allow_dynamic_motion=static_hold >= 0.8,
        ))
    if not shot_plans:
        decisions.append("no_overlapping_source_shots")
    return CompositionPlan(
        plan_id=f"composition_{edit_plan.plan_id}",
        width=width,
        height=height,
        safe_zone=resolved_safe_zone,
        shots=shot_plans,
        overlays=overlays,
        decisions=decisions,
    )
