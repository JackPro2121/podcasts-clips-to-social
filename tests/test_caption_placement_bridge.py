"""Tests for the director -> renderer caption bridge.

This is the wiring that stops the two-brain split: the composition planner's
text-collision analysis now produces real ASS caption positions instead of only
being judged against the render after the fact.
"""
import tempfile
import unittest
from pathlib import Path

from src.composition_planner import (
    CaptionPlacement,
    NormalizedRect,
    ShotComposition,
    build_caption_placements,
)
from src.subtitle_generator import create_styled_ass_subtitles
from src.transcriber import TranscriptSegment, WordTimestamp


def _plan_with_shots(shots):
    from src.composition_planner import CompositionPlan, DEFAULT_SAFE_ZONE
    return CompositionPlan(
        plan_id="composition_test",
        width=1080,
        height=1920,
        safe_zone=DEFAULT_SAFE_ZONE,
        shots=shots,
    )


def _shot(shot_id, start, end, anchor, rect):
    return ShotComposition(
        shot_id=shot_id,
        start=start,
        end=end,
        source_type="presentation",
        crop=NormalizedRect(0.0, 0.0, 1.0, 1.0),
        caption_anchor=anchor,
        caption_rect=rect,
    )


class TestBuildCaptionPlacements(unittest.TestCase):
    def test_lower_anchor_becomes_bottom_aligned(self):
        plan = _plan_with_shots([_shot("s1", 0.0, 10.0, "lower_center", NormalizedRect(0.08, 0.62, 0.76, 0.14))])
        placement = build_caption_placements(plan)[0]
        self.assertEqual(placement.alignment, 2)
        self.assertFalse(placement.collision_avoidance)
        # rect bottom is 0.76 -> 24% of 1920 = ~460px above the bottom edge
        self.assertAlmostEqual(placement.margin_v, int(round(0.24 * 1920)), delta=1)

    def test_upper_anchor_becomes_top_aligned_and_is_flagged_as_avoidance(self):
        plan = _plan_with_shots([_shot("s1", 0.0, 10.0, "upper_center", NormalizedRect(0.08, 0.24, 0.76, 0.14))])
        placement = build_caption_placements(plan)[0]
        self.assertEqual(placement.alignment, 8)
        self.assertTrue(placement.collision_avoidance)
        self.assertAlmostEqual(placement.margin_v, int(round(0.24 * 1920)), delta=1)

    def test_center_and_divider_anchors_are_middle_aligned(self):
        plan = _plan_with_shots([
            _shot("s1", 0.0, 5.0, "center", NormalizedRect(0.08, 0.42, 0.76, 0.14)),
            _shot("s2", 5.0, 10.0, "split_divider", NormalizedRect(0.48, 0.42, 0.04, 0.16)),
        ])
        placements = build_caption_placements(plan)
        self.assertEqual([p.alignment for p in placements], [5, 5])
        self.assertFalse(any(p.collision_avoidance for p in placements))

    def test_safe_side_anchors_stay_bottom_aligned(self):
        plan = _plan_with_shots([
            _shot("s1", 0.0, 5.0, "left_safe", NormalizedRect(0.08, 0.58, 0.36, 0.14)),
            _shot("s2", 5.0, 10.0, "right_safe", NormalizedRect(0.56, 0.58, 0.36, 0.14)),
        ])
        self.assertEqual([p.alignment for p in build_caption_placements(plan)], [2, 2])

    def test_unknown_anchor_falls_back_to_bottom_centre(self):
        plan = _plan_with_shots([_shot("s1", 0.0, 5.0, "nonsense", NormalizedRect(0.08, 0.62, 0.76, 0.14))])
        self.assertEqual(build_caption_placements(plan)[0].alignment, 2)

    def test_placements_are_time_sorted_and_never_negative(self):
        plan = _plan_with_shots([
            _shot("late", 20.0, 30.0, "upper_center", NormalizedRect(0.08, 0.24, 0.76, 0.14)),
            _shot("early", 0.0, 10.0, "center", NormalizedRect(0.08, 0.42, 0.76, 0.14)),
            _shot("off", 0.0, 10.0, "upper_center", NormalizedRect(0.08, 0.0, 0.76, 0.14)),
        ])
        placements = build_caption_placements(plan)
        self.assertEqual([p.start for p in placements], [0.0, 0.0, 20.0])
        for placement in placements:
            self.assertGreaterEqual(placement.margin_v, 0)

    def test_placements_serialise(self):
        plan = _plan_with_shots([_shot("s1", 0.0, 5.0, "center", NormalizedRect(0.08, 0.42, 0.76, 0.14))])
        payload = build_caption_placements(plan)[0].to_dict()
        self.assertEqual(payload["anchor"], "center")
        self.assertIn("alignment", payload)


class TestSubtitlesHonourPlacements(unittest.TestCase):
    def _write_ass(self, placements, out_path):
        words = [
            WordTimestamp(word="alpha", start=2.0, end=2.4),
            WordTimestamp(word="omega", start=7.0, end=7.4),
        ]
        segments = [TranscriptSegment(start=1.5, end=8.0, text="alpha omega", words=words)]
        return create_styled_ass_subtitles(
            segments=segments,
            clip_start=0.0,
            clip_end=10.0,
            output_ass_path=out_path,
            shots=None,
            caption_placements=placements,
        )

    @staticmethod
    def _dialogue_line(content, token):
        # The hormozi theme uppercases captions, so match case-insensitively.
        for line in content.splitlines():
            if "Dialogue:" in line and token.upper() in line.upper():
                return line
        raise AssertionError(f"no dialogue line containing {token!r}")

    def test_lower_half_uses_bottom_alignment_and_upper_half_top(self):
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "captions.ass"
        placements = [
            CaptionPlacement(start=0.0, end=5.0, anchor="lower_center", shot_id="s1",
                             alignment=2, margin_v=460, collision_avoidance=False),
            CaptionPlacement(start=5.0, end=10.0, anchor="upper_center", shot_id="s2",
                             alignment=8, margin_v=288, collision_avoidance=True),
        ]
        path = self._write_ass(placements, out)
        content = path.read_text(encoding="utf-8")
        lower = self._dialogue_line(content, "alpha")
        upper = self._dialogue_line(content, "omega")
        self.assertIn(",Default,", lower)
        self.assertIn("460", lower)
        self.assertIn(",DefaultTop,", upper)
        self.assertIn("288", upper)

    def test_absent_placements_fall_back_to_legacy_layout(self):
        tmp = Path(tempfile.mkdtemp())
        path = self._write_ass(None, tmp / "fallback.ass")
        content = path.read_text(encoding="utf-8")
        self.assertTrue(self._dialogue_line(content, "alpha"))
        self.assertTrue(self._dialogue_line(content, "omega"))

    def test_placements_outside_the_window_are_ignored(self):
        tmp = Path(tempfile.mkdtemp())
        placements = [
            CaptionPlacement(start=500.0, end=600.0, anchor="upper_center", shot_id="s1",
                             alignment=8, margin_v=10, collision_avoidance=True),
        ]
        path = self._write_ass(placements, tmp / "ignored.ass")
        content = path.read_text(encoding="utf-8")
        self.assertNotIn(",8,", content.split("Dialogue:")[1][:20])


if __name__ == "__main__":
    unittest.main()
