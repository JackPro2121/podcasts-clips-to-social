"""Caption placement must clear the subject's face.

The defect
----------
On three of the six published golden-master clips, **100% of burned-in caption
pixels sit on the subject's brow and eyes**; on a fourth, 30%. The captions are
unreadable.

Two independent causes, both fixed here:

1. **Faces were invisible to collision avoidance.** ``_caption_anchor`` only ever
   received *text* regions, from OCR or the edge heuristic. A face is not text, so
   the "move the caption up, away from the on-screen text" path moved the caption
   straight onto the face: ``upper_center`` resolves to roughly 25% of frame
   height, which clears the 240px platform UI band and lands on the eyes.

2. **The split-screen branch skipped avoidance entirely.** It returned
   ``"split_divider"`` before any collision check ran, so AGENTS.md 3.4's
   requirement that captions "never sit raw on split cuts" was violated on every
   split-screen shot.

Plus a third, unrelated defect found on the way: the ``\\an5`` branch fed a
top-anchored offset into a slot ASS measures **from the frame's vertical centre**,
so every centre-anchored caption rendered about 800px lower than intended -- into
the bottom platform UI band. The existing bridge test only asserted alignment
codes and never the margin, which is how it survived.
"""

from __future__ import annotations

import unittest

from src.composition_planner import (
    _anchor_to_ass,
    _caption_anchor,
    _conflicts,
    _face_avoidance_regions,
    _force_collision_avoidance,
    DEFAULT_SAFE_ZONE,
    NormalizedRect,
    ShotComposition,
)
from src.source_index import IndexedShot, TextRegion

WIDTH, HEIGHT = 1080, 1920


def _shot(**kwargs) -> IndexedShot:
    base = dict(
        shot_id="shot_1",
        start=0.0,
        end=30.0,
        shot_type="human_or_scene",
        motion_score=0.3,
        static_score=0.7,
        presentation_ratio=0.0,
        face_count=1.0,
    )
    base.update(kwargs)
    return IndexedShot(**base)


class TestFaceAvoidanceRegions(unittest.TestCase):
    """L1: only the upper face is protected, and only when a size is known."""

    def test_upper_face_only(self) -> None:
        """Protecting the whole face would leave nowhere to put a caption.

        In a tight portrait crop the face fills the frame, so a whole-face region
        collides with every candidate anchor and the search degenerates. The
        golden-master measurement that justifies this: whole-face overlap reads
        100% on five of six clips, upper-face overlap reads 0-100% and actually
        discriminates.
        """
        shot = _shot(
            face_boxes=[(300, 100, 400, 600)],
            source_width=1920,
            source_height=1080,
        )
        regions = _face_avoidance_regions(shot, DEFAULT_SAFE_ZONE)
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertAlmostEqual(region.y, 100 / 1080, places=3)
        self.assertAlmostEqual(region.height, (600 * 0.55) / 1080, places=3)

    def test_no_source_size_means_no_regions(self) -> None:
        """Guessing a size would put the region in the wrong place."""
        shot = _shot(face_boxes=[(10, 10, 50, 50)])
        self.assertEqual(_face_avoidance_regions(shot, DEFAULT_SAFE_ZONE), [])

    def test_no_faces_means_no_regions(self) -> None:
        shot = _shot(source_width=1920, source_height=1080)
        self.assertEqual(_face_avoidance_regions(shot, DEFAULT_SAFE_ZONE), [])

    def test_malformed_boxes_are_skipped_not_crashed(self) -> None:
        shot = _shot(
            face_boxes=[(0, 0, 0, 0), (10, 10, 40, 40), "junk"],
            source_width=1920,
            source_height=1080,
        )
        regions = _face_avoidance_regions(shot, DEFAULT_SAFE_ZONE)
        self.assertEqual(len(regions), 1, "only the well-formed, non-degenerate box")

    def test_at_most_three_faces(self) -> None:
        shot = _shot(
            face_boxes=[(i * 100, 0, 80, 80) for i in range(9)],
            source_width=1920,
            source_height=1080,
        )
        self.assertLessEqual(len(_face_avoidance_regions(shot, DEFAULT_SAFE_ZONE)), 3)


class TestAnchorAvoidsFaces(unittest.TestCase):
    """L1: the chosen anchor must not intersect a face region."""

    # A face filling the middle of a 1080x1920 portrait crop, which is what
    # portrait_face produces and what the corpus measured.
    BIG_FACE = [(180, 60, 720, 900)]
    SIZE = dict(source_width=1080, source_height=1920)

    def test_face_pushes_caption_out_of_the_centre(self) -> None:
        shot = _shot(face_boxes=self.BIG_FACE, **self.SIZE)
        anchor = _caption_anchor(shot, [], DEFAULT_SAFE_ZONE)
        avoid = _face_avoidance_regions(shot, DEFAULT_SAFE_ZONE)
        rect = _caption_rect_for(anchor)
        self.assertFalse(
            _conflicts(rect, avoid),
            msg=(
                f"anchor {anchor!r} at y={rect.y:.2f}..{rect.y + rect.height:.2f} "
                f"overlaps the face at y={avoid[0].y:.2f}..{avoid[0].y + avoid[0].height:.2f}"
            ),
        )

    def test_split_screen_no_longer_skips_avoidance(self) -> None:
        """The old early return made AGENTS.md 3.4 unsatisfiable on split shots.

        Mutation guard: restoring
        `if shot.shot_type == 'split_screen': return 'split_divider'` at the top
        of `_caption_anchor` fails this test.
        """
        shot = _shot(
            shot_type="split_screen",
            face_count=2.0,
            face_boxes=[(40, 60, 480, 620), (560, 60, 480, 620)],
            **self.SIZE,
        )
        blocked = _face_avoidance_regions(shot, DEFAULT_SAFE_ZONE)
        self.assertTrue(blocked)
        anchor = _caption_anchor(shot, [], DEFAULT_SAFE_ZONE)
        rect = _caption_rect_for(anchor)
        self.assertFalse(
            _conflicts(rect, blocked),
            f"split-screen anchor {anchor!r} still lands on a speaker",
        )

    def test_divider_anchor_is_not_used_for_a_single_subject(self) -> None:
        """`split_divider` is a narrow central sliver; on one subject it is nonsense."""
        shot = _shot(face_count=1.0, source_width=1080, source_height=1920)
        anchor = _caption_anchor(shot, [], DEFAULT_SAFE_ZONE)
        self.assertNotEqual(anchor, "split_divider")

    def test_bottom_text_still_wins_over_the_face(self) -> None:
        """Text avoidance and face avoidance must compose, not fight."""
        shot = _shot(face_boxes=self.BIG_FACE, **self.SIZE)
        bottom_text = [TextRegion(0.05, 0.86, 0.9, 0.10, 0.9)]
        avoid_regions = _face_avoidance_regions(shot, DEFAULT_SAFE_ZONE)
        anchor = _caption_anchor(
            shot,
            [NormalizedRect(r.x, r.y, r.width, r.height) for r in bottom_text],
            DEFAULT_SAFE_ZONE,
        )
        rect = _caption_rect_for(anchor)
        self.assertFalse(_conflicts(rect, avoid_regions))
        for region in (NormalizedRect(r.x, r.y, r.width, r.height) for r in bottom_text):
            self.assertFalse(
                _conflicts(rect, [region]),
                f"anchor {anchor!r} landed on the source text instead",
            )


class TestAssMarginSemantics(unittest.TestCase):
    """L1: `\an5` MarginV is measured from the centre, not the top.

    This is the bug `test_caption_placement_bridge.py` could not see: it asserted
    alignment codes (5 and 8) and never the margin, so a margin that was off by
    the height of the frame passed unnoticed.
    """

    def _rendered_y(self, anchor: str, rect: NormalizedRect) -> float:
        alignment, margin_v, _ = _anchor_to_ass(anchor, rect, WIDTH, HEIGHT)
        if alignment == 2:  # bottom-centre
            return HEIGHT - margin_v - rect.height * HEIGHT
        if alignment == 5:  # middle-centre
            return HEIGHT / 2.0 + margin_v - rect.height * HEIGHT / 2.0
        if alignment == 8:  # top-centre
            return margin_v
        raise AssertionError(f"unexpected alignment {alignment}")

    def test_every_anchor_renders_where_it_asked_to(self) -> None:
        for anchor in ("lower_center", "center", "split_divider", "upper_center"):
            for y in (0.15, 0.30, 0.45, 0.60, 0.70):
                with self.subTest(anchor=anchor, y=y):
                    rect = NormalizedRect(x=0.1, y=y, width=0.8, height=0.14)
                    rendered_top = self._rendered_y(anchor, rect)
                    intended_top = rect.y * HEIGHT
                    self.assertAlmostEqual(
                        rendered_top,
                        intended_top,
                        delta=HEIGHT * 0.09,
                        msg=(
                            f"{anchor} intended {intended_top:.0f}px but renders at "
                            f"{rendered_top:.0f}px"
                        ),
                    )

    def test_centre_anchored_caption_never_lands_in_the_bottom_ui(self) -> None:
        """The failure that shipped: a top offset pushed `\an5` into the handle band."""
        bottom_ui_top = HEIGHT - 380
        for y in (0.15, 0.30, 0.45, 0.60, 0.70):
            rect = NormalizedRect(x=0.1, y=y, width=0.8, height=0.14)
            rendered_top = self._rendered_y("center", rect)
            self.assertLess(
                rendered_top,
                bottom_ui_top,
                msg=f"centre caption at y={y} renders at {rendered_top:.0f}px, "
                    f"inside the bottom platform UI band starting {bottom_ui_top}px",
            )


class TestProximityTolerance(unittest.TestCase):
    """A near miss is still a collision; dropping the tolerance was a regression."""

    def test_lower_third_just_missing_the_band_still_blocks(self) -> None:
        # A source lower-third spanning 0.80-0.92 barely misses a lower_center
        # band ending at 0.76. Strict intersection accepts it; the caption then
        # lands on the text.
        rect = NormalizedRect(x=0.08, y=0.62, width=0.76, height=0.14)
        near_miss = [NormalizedRect(0.05, 0.80, 0.9, 0.12)]
        self.assertFalse(rect.intersects(near_miss[0]), "precondition: not a strict overlap")
        self.assertTrue(
            _conflicts(rect, near_miss),
            "the proximity margin must still treat this as a collision",
        )

    def test_distant_region_does_not_block(self) -> None:
        rect = NormalizedRect(x=0.08, y=0.62, width=0.76, height=0.14)
        far_away = [NormalizedRect(0.05, 0.05, 0.9, 0.08)]
        self.assertFalse(_conflicts(rect, far_away))

    def test_empty_regions_never_block(self) -> None:
        rect = NormalizedRect(x=0.08, y=0.62, width=0.76, height=0.14)
        self.assertFalse(_conflicts(rect, []))


class TestRepairPathAvoidsFaces(unittest.TestCase):
    """The repair relocation must treat faces as forbidden, not only source text.

    This is the path that actually shipped the defect: QA raised
    ``caption_source_collision`` on shots 0001/0003, the pipeline re-ran with
    ``avoid_collisions=True``, and ``_force_collision_avoidance`` -- which only
    checked ``protected_regions`` (text) -- moved the caption to ``upper_center``,
    straight onto the presenter's face.

    Mutation guard: restricting the avoid set back to ``shot.protected_regions``
    selects ``upper_center`` here, which is inside the face band, and the first
    assertion fails.
    """

    def _shot(self) -> ShotComposition:
        return ShotComposition(
            shot_id="shot_1",
            start=0.0,
            end=10.0,
            source_type="human_or_scene",
            crop=NormalizedRect(0.0, 0.0, 1.0, 1.0),
            caption_anchor="lower_center",
            caption_rect=NormalizedRect(0.08, 0.62, 0.76, 0.14),
            # Source lower-third text the caption currently overlaps.
            protected_regions=[NormalizedRect(0.05, 0.70, 0.90, 0.16)],
            # Upper-face band: center (0.42..0.56) is the one band clear of both.
            face_regions=[NormalizedRect(0.10, 0.08, 0.80, 0.24)],
        )

    def test_repair_never_lands_on_a_face(self) -> None:
        shot = self._shot()
        anchor, rect = _force_collision_avoidance(
            shot, shot.caption_rect, DEFAULT_SAFE_ZONE
        )
        self.assertFalse(
            _conflicts(rect, shot.face_regions),
            msg=(
                f"repair anchor {anchor!r} at y={rect.y:.2f}.."
                f"{rect.y + rect.height:.2f} landed on the face band"
            ),
        )

    def test_repair_still_clears_the_text_when_a_band_is_free(self) -> None:
        shot = self._shot()
        anchor, rect = _force_collision_avoidance(
            shot, shot.caption_rect, DEFAULT_SAFE_ZONE
        )
        self.assertFalse(
            _conflicts(rect, shot.protected_regions),
            f"repair anchor {anchor!r} still overlaps the source text",
        )

    def test_plain_relocation_is_a_no_op_when_nothing_collides(self) -> None:
        shot = self._shot()
        shot.face_regions = []
        shot.protected_regions = []
        anchor, rect = _force_collision_avoidance(
            shot, shot.caption_rect, DEFAULT_SAFE_ZONE
        )
        self.assertEqual(anchor, "lower_center")
        self.assertEqual(rect, shot.caption_rect)


def _caption_rect_for(anchor: str) -> NormalizedRect:
    from src.composition_planner import _caption_rect

    return _caption_rect(anchor, DEFAULT_SAFE_ZONE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
