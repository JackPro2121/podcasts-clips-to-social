from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
    # Upper-face bands from the shot's detected faces. Kept separate from
    # protected_regions so QA messages can still say "source text" for text, but
    # the repair path treats both as forbidden.
    face_regions: List[NormalizedRect] = field(default_factory=list)
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
    collision_avoidance_applied: bool = False
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
            "collision_avoidance_applied": self.collision_avoidance_applied,
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
    lower_rect = _caption_rect("lower_center", safe_zone)
    bottom_limit = 1.0 - safe_zone.bottom
    return any(
        lower_rect.intersects(region) or (region.y + region.height >= bottom_limit - 0.08)
        for region in regions
    )


def _caption_anchor(shot: IndexedShot, protected_regions: List[NormalizedRect], safe_zone: SafeZone) -> str:
    """Choose where the caption goes for this shot.

    Two changes here, both from measured defects.

    **Faces are protected regions.** Previously the only protected regions came
    from OCR and the edge heuristic, i.e. *text*. A face is not text, so moving a
    caption to avoid on-screen text moved it straight onto the presenter's face.
    That is what produced the golden-master measurements of **100% of burned-in
    caption pixels on the subject's brow and eyes** in three of six published
    clips. `_face_avoidance_regions` adds the upper face -- brow through eyes --
    to the protected set, which is the band that makes a caption unreadable.

    **The split-screen early return is gone.** It returned "split_divider" before
    any collision checking ran, so AGENTS.md 3.4's requirement that captions
    "never sit raw on split cuts" was violated on every split-screen shot. Split
    shots now take part in the same avoidance search as everything else.
    """
    avoid = list(protected_regions) + list(_face_avoidance_regions(shot, safe_zone))

    if avoid:
        for candidate in _anchor_search_order(shot):
            rect = _caption_rect(candidate, safe_zone)
            if not _conflicts(rect, avoid):
                return candidate

    if shot.shot_type == "split_screen" or shot.face_count >= 1.5:
        # Nothing is free. Between the panes is at least not on a face, and the
        # panes are where a two-speaker shot is least likely to be unreadable.
        return "split_divider"
    if shot.shot_type == "presentation":
        return "lower_center"
    if shot.face_count >= 1.0:
        return "lower_center"
    return "center"


# Ordered by how much of the subject they obscure. `lower_center` first because
# for a talking head the lower chest area is the least destructive place for text,
# and it is also where Hormozi-style captions conventionally sit.
#
# `center` is last on purpose for a human shot: at 0.42-0.56 of frame height it
# sits over the mouth and chin, which is worse than the alternatives.
_CENTRE_LAST_ORDER = ("lower_center", "upper_center", "center")

# A split-screen shot gets the divider band first, because the gap between the two
# panes is the one place that is over neither speaker. The divider anchor is a
# narrow central sliver (x 0.48-0.52), which is nonsense for a single-subject
# shot -- it would put the caption in a thin column over the subject's chest -- so
# it is only ever a candidate when there really are two panes.
_SPLIT_ORDER = ("split_divider", "lower_center", "center", "upper_center")

# Human shots never get `upper_center`. Measured: every on-face caption in the
# corpus -- including the 100% case that the pixel gate blocked on a real run --
# was an `upper_center` band landing on the subject's brow and eyes. The face
# regions that should have vetoed it come from the source index's face detector,
# and one missed detection per shot is enough to make the band look free. The
# chest-level bands are a weaker placement but never unreadable, so they are the
# only candidates a person shot may use.
_FACE_SAFE_ORDER = ("lower_center", "left_safe", "right_safe", "center")


def _anchor_search_order(shot: IndexedShot) -> Sequence[str]:
    if shot.shot_type == "split_screen" or shot.face_count >= 1.5:
        return _SPLIT_ORDER
    if shot.face_count >= 1.0:
        return _FACE_SAFE_ORDER
    return _CENTRE_LAST_ORDER


# A candidate band is treated as blocked when a protected region comes within this
# fraction of the frame height, not only when it strictly overlaps.
#
# The previous code had this tolerance in `_bottom_text_collision`
# (`region.y + region.height >= bottom_limit - 0.08`) and dropping it was a real
# regression: a source lower-third spanning 0.80-0.92 barely misses a
# `lower_center` band that ends at 0.802, and strict intersection accepted it, so
# the caption landed on the text. A near miss is still a collision.
_PROXIMITY_MARGIN = 0.08


def _conflicts(
    rect: NormalizedRect, regions: Sequence[NormalizedRect], margin: float = _PROXIMITY_MARGIN
) -> bool:
    """True when ``rect`` overlaps, or comes within ``margin`` of, any region."""
    if not regions:
        return False
    if any(rect.intersects(region) for region in regions):
        return True
    grown = NormalizedRect(
        x=max(0.0, rect.x - margin),
        y=max(0.0, rect.y - margin),
        width=min(1.0, rect.width + 2 * margin),
        height=min(1.0, rect.height + 2 * margin),
    )
    return any(grown.intersects(region) for region in regions)

# Fraction of a face box treated as the unreadable band. Eyes sit at roughly 40% of
# face height from the top, and the brow is just above, so 0.55 covers both while
# leaving the chin and neck free.
_FACE_AVOIDANCE_TOP_FRACTION = 0.55
_FACE_AVOIDANCE_SIDE_FRACTION = 0.30


def _face_avoidance_regions(shot: IndexedShot, safe_zone: SafeZone) -> List[NormalizedRect]:
    """Normalised rectangles covering the unreadable part of each detected face.

    Pure in the sense that it takes plain numbers, so the geometry is unit-testable
    without a detector. ``shot.face_boxes`` is populated by face_tracker's
    ``ShotPlan.face_boxes``, which holds source-pixel boxes for the largest faces
    in the shot; an empty list means no face was detected, and this returns
    nothing rather than guessing.
    """
    source_width = float(getattr(shot, "source_width", 0) or 0)
    source_height = float(getattr(shot, "source_height", 0) or 0)
    if source_width <= 0 or source_height <= 0:
        return []

    regions: List[NormalizedRect] = []
    for box in list(getattr(shot, "face_boxes", []) or [])[:3]:
        try:
            x, y, width, height = (float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        # Only the upper face is protected. Protecting the whole face would make
        # every candidate anchor collide in a tight portrait crop, where the face
        # fills the frame, and the search would have nowhere left to go.
        protected_height = height * _FACE_AVOIDANCE_TOP_FRACTION
        pad = width * _FACE_AVOIDANCE_SIDE_FRACTION
        regions.append(
            NormalizedRect(
                x=max(0.0, (x - pad) / source_width),
                y=max(0.0, y / source_height),
                width=min(1.0, (width + 2 * pad) / source_width),
                height=min(1.0, protected_height / source_height),
            )
        )
    return regions


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

    The `\an5` branch was wrong. ASS alignment 5 is *middle-centre*, and for
    alignment 5 `MarginV` is measured **from the vertical centre of the frame**,
    not from the top. The old code fed it a top-anchored offset:

        from_top_px = int(round(rect.y * height))       # e.g. 806 for rect.y=0.42
        return 5, from_top_px                            # interpreted as centre+806

    so a caption intended for 42% frame height rendered at
    960 + 806 = 1766px of 1920 -- inside the bottom platform UI band, on top of
    the TikTok/Reels handle and caption block. Every centre-anchored and
    split-divider caption was landing in the worst possible place, and
    `tests/test_caption_placement_bridge.py` only asserted the alignment codes,
    never the margin, so the suite stayed green.
    """
    from_bottom_px = int(round((1.0 - (rect.y + rect.height)) * height))

    if anchor == "upper_center":
        # Alignment 8 is top-centre, so MarginV is measured from the top here.
        # This is the only anchor whose margin needs a top-relative value.
        return 8, int(round(rect.y * height)), True
    if anchor in ("center", "split_divider"):
        # Alignment 5 is middle-centre: MarginV is an offset FROM the middle.
        centre_offset = int(round((rect.y + rect.height / 2.0 - 0.5) * height))
        return 5, centre_offset, False
    if anchor in ("left_safe", "right_safe"):
        # ASS cannot horizontally offset a centre-aligned line via margins, so
        # these keep bottom-centre placement; the anchor still documents intent.
        return 2, from_bottom_px, False
    return 2, from_bottom_px, False


def _force_collision_avoidance(
    shot: ShotComposition,
    rect: NormalizedRect,
    safe_zone: SafeZone = DEFAULT_SAFE_ZONE,
) -> Tuple[str, NormalizedRect]:
    """
    Relocate a caption that overlaps the shot's protected source text.

    Used on a repair attempt: QA reported a caption_source_collision, so the
    caption is moved out of the way rather than re-rendering the same overlapping
    layout. Returns the anchor name alongside the rect so the reported anchor can
    never disagree with the geometry that was actually used.

    The first version checked only ``protected_regions`` (source text), so the
    repair that fixed a text collision moved the caption straight onto the
    presenter's face -- the measured 100% caption-on-face defect. When both text
    and a face are in the way the face wins: a caption on clothing is readable, a
    caption across the eyes is not.
    """
    avoid = list(shot.protected_regions) + list(shot.face_regions)
    if not avoid:
        return shot.caption_anchor, rect

    def face_conflicts(candidate: NormalizedRect) -> int:
        return sum(1 for region in shot.face_regions if _conflicts(candidate, [region]))

    def text_conflicts(candidate: NormalizedRect) -> int:
        return sum(
            1 for region in shot.protected_regions if _conflicts(candidate, [region])
        )

    def score(candidate: NormalizedRect) -> Tuple[int, int]:
        return (face_conflicts(candidate), text_conflicts(candidate))

    if score(rect) == (0, 0):
        return shot.caption_anchor, rect

    candidates = [shot.caption_anchor] + [
        name
        for name in ("lower_center", "center", "left_safe", "right_safe")
        if name != shot.caption_anchor
    ]
    if not shot.face_regions and getattr(shot, "face_count", 0.0) < 1.0 and "upper_center" not in candidates:
        candidates.append("upper_center")
    best_anchor = shot.caption_anchor
    best_rect = rect
    best_score = score(rect)
    for candidate in candidates:
        alternative = _caption_rect(candidate, safe_zone)
        candidate_score = score(alternative)
        if candidate_score < best_score:
            best_anchor = candidate
            best_rect = alternative
            best_score = candidate_score
        if best_score == (0, 0):
            break
    return best_anchor, best_rect


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
    if avoid_collisions:
        plan.collision_avoidance_applied = True
    placements: List[CaptionPlacement] = []
    for shot in plan.shots:
        anchor = shot.caption_anchor
        rect = shot.caption_rect
        if avoid_collisions:
            anchor, rect = _force_collision_avoidance(shot, rect, plan.safe_zone)
            shot.caption_anchor = anchor
            shot.caption_rect = rect
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
        face_regions = _face_avoidance_regions(shot, resolved_safe_zone)
        anchor = _caption_anchor(shot, protected, resolved_safe_zone)
        caption_rect = _caption_rect(anchor, resolved_safe_zone)
        static_hold = _freeze_hold(shot)
        broll_mode = "ken_burns_motion" if static_hold >= 0.8 else "none"
        transition = "motion_cut" if static_hold >= 0.8 else "cut"
        if static_hold >= 0.8:
            decisions.append(f"{shot.shot_id}:static_hold={static_hold:.2f}s->motion")
        if protected:
            decisions.append(f"{shot.shot_id}:protected_text_regions={len(protected)}")
        if face_regions:
            decisions.append(f"{shot.shot_id}:protected_face_regions={len(face_regions)}")
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
            face_regions=face_regions,
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
