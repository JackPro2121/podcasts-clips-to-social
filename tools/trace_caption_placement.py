"""Integration check: do face boxes actually reach the caption anchor chooser?

Unit tests can pass while an integration is unwired. This walks the real chain on
real footage and prints what each stage produces, so a broken link is visible
rather than inferred.

Chain under test:

    source_index.build_source_index(detect_faces=True)
        -> IndexedShot.face_boxes            (source pixels)
        -> composition_planner.build_composition_plan
            -> _caption_anchor(shot, ...)
                -> _face_avoidance_regions(shot)   (normalised)
        -> composition.caption_anchor

Run:  python tools/trace_caption_placement.py [fixture_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.composition_planner import (  # noqa: E402
    _anchor_to_ass,
    _caption_anchor,
    _caption_rect,
    _face_avoidance_regions,
    DEFAULT_SAFE_ZONE,
    NormalizedRect,
)
from src.source_index import build_source_index  # noqa: E402
from src.verification import fixtures, probe  # noqa: E402


def trace(master_id: str) -> int:
    master = fixtures.by_id(master_id)
    if master is None:
        print(f"fixture '{master_id}' not found; run tools/fetch_fixtures.py")
        return 1

    duration = probe.duration_seconds(master.path)
    width, height = probe.dimensions(master.path)
    print(f"=== {master_id}  {master.file}")
    print(f"    {duration:.2f}s  {width}x{height}")

    index = build_source_index(
        master.path,
        start_time=0.0,
        end_time=duration,
        sample_fps=1.0,
        max_samples=60,
        detect_faces=True,
    )
    print(f"    source_index: {len(index.shots)} shot(s), media {index.media.width}x{index.media.height}")

    problems = 0
    for shot in index.shots:
        boxes = list(getattr(shot, "face_boxes", []) or [])
        regions = _face_avoidance_regions(shot, DEFAULT_SAFE_ZONE)
        text_rects = [NormalizedRect(r.x, r.y, r.width, r.height) for r in shot.text_regions]
        anchor = _caption_anchor(shot, text_rects, DEFAULT_SAFE_ZONE)
        rect = _caption_rect(anchor, DEFAULT_SAFE_ZONE)
        alignment, margin_v, _ = _anchor_to_ass(anchor, rect, width, height)
        print(
            f"    shot {shot.shot_id} t={shot.start:.1f}-{shot.end:.1f} "
            f"type={shot.shot_type} faces={shot.face_count:.2f} text_regions={len(text_rects)}"
        )
        print(f"        face_boxes  : {boxes[:2] if boxes else 'none'}")
        print(
            f"        anchor={anchor} rect y={rect.y:.2f}..{rect.y + rect.height:.2f} "
            f"an={alignment} marginV={margin_v}"
        )
        if not boxes:
            print("        NOTE: no face detected; the anchor is not face-aware here")
            problems += 1
        for region in regions:
            overlap = not (
                rect.y + rect.height <= region.y or region.y + region.height <= rect.y
            )
            if overlap:
                problems += 1
            print(
                f"        face band y={region.y:.2f}..{region.y + region.height:.2f}"
                f" -> {'OVERLAP' if overlap else 'clear'}"
            )
    print(f"    problems: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "broke_1"
    raise SystemExit(trace(target))
