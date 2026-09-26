"""
Tests for Director v2.

The theme throughout: the director is an untrusted input. A model reply naming
a layout the renderer cannot execute, or asking to cut every shot, must be
neutralised rather than obeyed. These tests pin that behaviour down, plus the
happy path that proves a directive actually reaches the filtergraph.
"""
import contextlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import numpy as np

from src import director_v2
from src.composition_planner import (
    CaptionPlacement,
    NormalizedRect,
    build_caption_placements,
)
from src.director_v2 import (
    DIRECTOR_CAPTION_ANCHORS,
    DIRECTOR_LAYOUTS,
    ShotDirective,
    apply_caption_directives,
    apply_shot_directives,
    build_contact_sheet,
    build_director_prompt,
    parse_shot_directives,
    plan_shot_directives,
)
from src.face_tracker import FramingDecision, ShotPlan
from src.source_index import SourceIndex, SourceMediaInfo
from src.transcriber import TranscriptSegment

BOX_A = (100, 100, 200, 200)
BOX_B = (1400, 100, 200, 200)


def _shot(start: float, end: float, mode: str = "portrait_face", **kwargs) -> ShotPlan:
    return ShotPlan(start=start, end=end, mode=mode, **kwargs)


def _framing(shots: List[ShotPlan], mode: str = "multi_shot_dynamic") -> FramingDecision:
    return FramingDecision(
        mode=mode,
        face_count=len(shots),
        shots=shots,
        video_width=1920,
        video_height=1080,
        active_w=1920,
        active_h=1080,
    )


def _index() -> SourceIndex:
    return SourceIndex(
        media=SourceMediaInfo(
            path="x.mp4", duration=48.0, width=1920, height=1080, fps=30.0
        ),
        shots=[],
    )


def _segments() -> List[TranscriptSegment]:
    return [
        TranscriptSegment(start=0.0, end=4.0, text="The first thing nobody tells you about money", words=[]),
        TranscriptSegment(start=4.0, end=9.0, text="is that the first dollar is the hardest", words=[]),
    ]


def _reply(*directives: dict) -> str:
    return json.dumps({"directives": list(directives)})


class TestContactSheet(unittest.TestCase):
    def _synthetic(self, count: int) -> List[Tuple[float, np.ndarray]]:
        samples = []
        for index in range(count):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :, index % 3] = 40 + index * 5
            samples.append((index * 1.5, frame))
        return samples

    def test_builds_a_labelled_grid(self):
        with tempfile_dir() as tmp:
            with mock.patch.object(
                director_v2, "sample_frames_at", return_value=self._synthetic(8)
            ):
                sheet = build_contact_sheet(
                    Path("fake.mp4"), 0.0, 12.0, tmp / "sheet.jpg", tile_count=8
                )
            self.assertIsNotNone(sheet)
            assert sheet is not None
            self.assertTrue(sheet.path.exists())
            self.assertEqual(len(sheet.tiles), 8)
            self.assertEqual(sheet.columns, 4)
            self.assertEqual(sheet.rows, 2)
            self.assertEqual(sheet.tiles, [i * 1.5 for i in range(8)])
            legend = sheet.legend()
            self.assertIn("r0c0=0.0s", legend)
            self.assertIn("r1c3=10.5s", legend)

    def test_tile_count_is_capped(self):
        """The cap protects the decode request, so that is what must be bounded."""
        with tempfile_dir() as tmp:
            with mock.patch.object(
                director_v2, "sample_frames_at", return_value=self._synthetic(8)
            ) as sampler:
                build_contact_sheet(
                    Path("fake.mp4"), 0.0, 60.0, tmp / "sheet.jpg", tile_count=40
                )
        self.assertEqual(
            sampler.call_args.args[3], director_v2.CONTACT_MAX_TILES
        )

    def test_no_frames_returns_none(self):
        with tempfile_dir() as tmp:
            with mock.patch.object(director_v2, "sample_frames_at", return_value=[]):
                self.assertIsNone(
                    build_contact_sheet(Path("f.mp4"), 0, 5, tmp / "s.jpg")
                )

    def test_decoder_failure_returns_none_rather_than_raising(self):
        with tempfile_dir() as tmp:
            with mock.patch.object(
                director_v2, "sample_frames_at", side_effect=RuntimeError("codec boom")
            ):
                self.assertIsNone(
                    build_contact_sheet(Path("f.mp4"), 0, 5, tmp / "s.jpg")
                )


class TestParseShotDirectives(unittest.TestCase):
    def test_happy_path(self):
        raw = _reply({
            "start": 0.0, "end": 4.2, "layout": "presentation_slide",
            "crop_target": "auto", "motion": "drift", "caption_anchor": "center",
            "keep": True, "reason": "full screen slide", "confidence": 0.82,
        })
        directives, audit = parse_shot_directives(raw, 48.0)
        self.assertEqual(len(directives), 1)
        directive = directives[0]
        self.assertEqual(directive.layout, "presentation_slide")
        self.assertEqual(directive.caption_anchor, "center")
        self.assertTrue(directive.keep)
        self.assertEqual(directive.confidence, 0.82)
        self.assertEqual(audit, [])

    def test_strips_markdown_fence(self):
        raw = "```json\n" + _reply({"start": 0, "end": 4, "layout": "portrait_face"}) + "\n```"
        directives, _ = parse_shot_directives(raw, 48.0)
        self.assertEqual(len(directives), 1)

    def test_unknown_layout_drops_only_that_field(self):
        """A stray enum must not cost us the rest of an otherwise good plan."""
        raw = _reply({
            "start": 0.0, "end": 4.0, "layout": "holographic_cube",
            "keep": False, "reason": "unusable", "confidence": 0.4,
        })
        directives, audit = parse_shot_directives(raw, 48.0)
        self.assertEqual(len(directives), 1)
        self.assertIsNone(directives[0].layout)
        self.assertFalse(directives[0].keep)
        self.assertTrue(any("unknown_layout" in line for line in audit))

    def test_string_false_is_not_truthy(self):
        """`bool("false") is True` in Python; a dropped shot would be silent loss."""
        for value in ("false", "False", "no", "0", "cut", "drop"):
            raw = _reply({"start": 0.0, "end": 4.0, "keep": value})
            directives, _ = parse_shot_directives(raw, 48.0)
            self.assertFalse(directives[0].keep, f"keep={value!r} was treated as true")
        for value in ("true", "yes", "1", "keep"):
            raw = _reply({"start": 0.0, "end": 4.0, "keep": value})
            directives, _ = parse_shot_directives(raw, 48.0)
            self.assertTrue(directives[0].keep, f"keep={value!r} was treated as false")

    def test_missing_keep_defaults_to_true(self):
        raw = _reply({"start": 0.0, "end": 4.0})
        directives, _ = parse_shot_directives(raw, 48.0)
        self.assertTrue(directives[0].keep)

    def test_ranges_are_clamped_to_the_clip(self):
        raw = _reply({"start": -12.0, "end": 999.0, "layout": "portrait_face"})
        directives, _ = parse_shot_directives(raw, 48.0)
        self.assertEqual(directives[0].start, 0.0)
        self.assertEqual(directives[0].end, 48.0)

    def test_nan_and_infinity_are_rejected(self):
        raw = _reply(
            {"start": 0.0, "end": 4.0},
            {"start": "NaN", "end": 4.0},
            {"start": 0.0, "end": "Infinity"},
        )
        directives, audit = parse_shot_directives(raw, 48.0)
        self.assertEqual(len(directives), 1)
        self.assertEqual(len([a for a in audit if "bad_range" in a]), 2)

    def test_degenerate_range_dropped(self):
        raw = _reply({"start": 4.0, "end": 4.0}, {"start": 5.0, "end": 4.0})
        directives, audit = parse_shot_directives(raw, 48.0)
        self.assertEqual(directives, [])
        self.assertEqual(len([a for a in audit if "too_short" in a]), 2)

    def test_confidence_is_clamped(self):
        raw = _reply(
            {"start": 0, "end": 4, "confidence": 5.5},
            {"start": 5, "end": 9, "confidence": -3},
            {"start": 10, "end": 14, "confidence": "not a number"},
        )
        directives, _ = parse_shot_directives(raw, 48.0)
        self.assertEqual([d.confidence for d in directives], [1.0, 0.0, 0.5])

    def test_unknown_enums_fall_back_to_defaults(self):
        raw = _reply({
            "start": 0.0, "end": 4.0, "crop_target": "the_moon",
            "motion": "spin_out", "caption_anchor": "somewhere_over_there",
        })
        directives, audit = parse_shot_directives(raw, 48.0)
        self.assertEqual(len(directives), 1)
        directive = directives[0]
        self.assertEqual(directive.crop_target, "auto")
        self.assertEqual(directive.motion, "drift")
        self.assertIsNone(directive.caption_anchor)
        self.assertEqual(len(audit), 3)

    def test_layout_is_case_insensitive(self):
        raw = _reply({"start": 0.0, "end": 4.0, "layout": "  PIP_Slide "})
        directives, _ = parse_shot_directives(raw, 48.0)
        self.assertEqual(directives[0].layout, "pip_slide")

    def test_bad_responses_yield_nothing(self):
        for raw, expected in [
            ("", "empty_response"),
            ("   ", "empty_response"),
            ("not json at all", "json_unparseable"),
            ("[1,2,3]", "response_not_an_object"),
            ('{"other": 1}', "no_directives_key"),
            ('{"directives": []}', "no_directives_key"),
            ('{"directives": "nope"}', "no_directives_key"),
        ]:
            directives, audit = parse_shot_directives(raw, 48.0)
            self.assertEqual(directives, [], raw)
            self.assertTrue(
                any(expected in line for line in audit), f"{raw!r} -> {audit}"
            )

    def test_non_object_items_are_skipped(self):
        raw = json.dumps({"directives": ["nope", 7, None, {"start": 1, "end": 5}]})
        directives, audit = parse_shot_directives(raw, 48.0)
        self.assertEqual(len(directives), 1)
        self.assertEqual(len([a for a in audit if "not_an_object" in a]), 3)

    def test_directive_count_is_capped(self):
        raw = _reply(*[{"start": i, "end": i + 1} for i in range(60)])
        directives, _ = parse_shot_directives(raw, 120.0, max_directives=8)
        self.assertEqual(len(directives), 8)

    def test_every_advertised_layout_survives_parsing(self):
        for layout in DIRECTOR_LAYOUTS:
            raw = _reply({"start": 0.0, "end": 4.0, "layout": layout})
            self.assertEqual(parse_shot_directives(raw, 48.0)[0][0].layout, layout)

    def test_every_advertised_anchor_survives_parsing(self):
        for anchor in DIRECTOR_CAPTION_ANCHORS:
            raw = _reply({"start": 0.0, "end": 4.0, "caption_anchor": anchor})
            self.assertEqual(
                parse_shot_directives(raw, 48.0)[0][0].caption_anchor, anchor
            )


class TestApplyShotDirectives(unittest.TestCase):
    def test_layout_change_applied_when_geometry_exists(self):
        framing = _framing([_shot(0, 5, "portrait_face", speaker1_box=BOX_A)])
        directives = [ShotDirective(start=0.0, end=5.0, layout="pip_slide")]
        updated, audit = apply_shot_directives(framing, directives)
        self.assertEqual(updated.shots[0].mode, "pip_slide")
        self.assertTrue(any("portrait_face->pip_slide" in line for line in audit))

    def test_pip_slide_rejected_without_a_face_box(self):
        """The model sees pixels and cannot see FaceBox; it will get this wrong."""
        framing = _framing([_shot(0, 5, "portrait_face")])
        updated, audit = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=5.0, layout="pip_slide")]
        )
        self.assertEqual(updated.shots[0].mode, "portrait_face")
        self.assertTrue(any("rejected_missing_speaker1_box" in line for line in audit))

    def test_split_screen_rejected_with_only_one_box(self):
        framing = _framing([_shot(0, 5, "portrait_face", speaker1_box=BOX_A)])
        updated, audit = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=5.0, layout="split_screen")]
        )
        self.assertEqual(updated.shots[0].mode, "portrait_face")
        self.assertTrue(any("rejected_missing_speaker2_box" in line for line in audit))

    def test_split_screen_accepted_with_both_boxes(self):
        framing = _framing(
            [_shot(0, 5, "portrait_face", speaker1_box=BOX_A, speaker2_box=BOX_B)]
        )
        updated, _ = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=5.0, layout="split_screen")]
        )
        self.assertEqual(updated.shots[0].mode, "split_screen")

    def test_input_framing_is_never_mutated(self):
        """The repair loop re-renders; a mutated input would apply twice."""
        framing = _framing([_shot(0, 5, "portrait_face", speaker1_box=BOX_A)])
        apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=5.0, layout="pip_slide")]
        )
        self.assertEqual(framing.shots[0].mode, "portrait_face")
        self.assertIsNotNone(framing.shots[0].speaker1_box)

    def test_applying_twice_is_idempotent(self):
        framing = _framing([_shot(0, 5, "portrait_face", speaker1_box=BOX_A)])
        directives = [ShotDirective(start=0.0, end=5.0, layout="pip_slide")]
        once, _ = apply_shot_directives(framing, directives)
        twice, _ = apply_shot_directives(once, directives)
        self.assertEqual(twice.shots[0].mode, "pip_slide")
        self.assertEqual(len(twice.shots), len(once.shots))

    def test_dropping_every_shot_restores_the_first(self):
        framing = _framing([_shot(0, 5), _shot(5, 10)])
        directives = [
            ShotDirective(start=0.0, end=5.0, keep=False),
            ShotDirective(start=5.0, end=10.0, keep=False),
        ]
        updated, audit = apply_shot_directives(framing, directives)
        self.assertEqual(len(updated.shots), 1)
        self.assertTrue(any("all_shots_dropped_restored_first" in line for line in audit))

    def test_one_of_two_shots_can_be_dropped(self):
        framing = _framing([_shot(0, 5), _shot(5, 10)])
        updated, _ = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=5.0, keep=False)]
        )
        self.assertEqual(len(updated.shots), 1)
        self.assertEqual(updated.shots[0].start, 5.0)

    def test_non_overlapping_directive_is_ignored(self):
        framing = _framing([_shot(20, 25, "portrait_face", speaker1_box=BOX_A)])
        updated, audit = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=4.0, layout="pip_slide")]
        )
        self.assertEqual(updated.shots[0].mode, "portrait_face")
        self.assertTrue(any("no_directive" in line for line in audit))

    def test_greatest_overlap_wins(self):
        framing = _framing([_shot(0, 10, "portrait_face", speaker1_box=BOX_A)])
        directives = [
            ShotDirective(start=0.0, end=2.0, layout="blur_stack"),
            ShotDirective(start=6.0, end=10.0, layout="pip_slide"),
        ]
        updated, _ = apply_shot_directives(framing, directives)
        self.assertEqual(updated.shots[0].mode, "pip_slide")

    def test_crop_target_speaker2_moves_the_framing_box(self):
        framing = _framing(
            [_shot(0, 5, "portrait_face", speaker1_box=BOX_A, speaker2_box=BOX_B)]
        )
        updated, audit = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=5.0, crop_target="speaker2")]
        )
        self.assertEqual(updated.shots[0].speaker1_box, BOX_B)
        self.assertTrue(any("crop_target->speaker2" in line for line in audit))

    def test_crop_target_rejected_when_the_box_is_absent(self):
        framing = _framing([_shot(0, 5, "portrait_face", speaker1_box=BOX_A)])
        updated, audit = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=5.0, crop_target="speaker2")]
        )
        self.assertEqual(updated.shots[0].speaker1_box, BOX_A)
        self.assertTrue(any("rejected_no_box" in line for line in audit))

    def test_crop_target_ignored_in_split_screen(self):
        framing = _framing(
            [_shot(0, 5, "split_screen", speaker1_box=BOX_A, speaker2_box=BOX_B)]
        )
        updated, audit = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=5.0, crop_target="speaker2")]
        )
        self.assertEqual(updated.shots[0].speaker1_box, BOX_A)
        self.assertTrue(any("ignored_in_split_screen" in line for line in audit))

    def test_no_directives_returns_the_same_object(self):
        framing = _framing([_shot(0, 5)])
        updated, audit = apply_shot_directives(framing, [])
        self.assertIs(updated, framing)
        self.assertEqual(audit, [])

    def test_no_shots_returns_the_same_object(self):
        framing = _framing([])
        updated, _ = apply_shot_directives(
            framing, [ShotDirective(start=0.0, end=4.0, layout="pip_slide")]
        )
        self.assertIs(updated, framing)


class TestApplyCaptionDirectives(unittest.TestCase):
    def _plan(self, anchor: str = "center"):
        from src.composition_planner import (
            CompositionPlan,
            DEFAULT_SAFE_ZONE,
            ShotComposition,
        )
        return CompositionPlan(
            plan_id="p",
            width=1080,
            height=1920,
            safe_zone=DEFAULT_SAFE_ZONE,
            shots=[
                ShotComposition(
                    shot_id="shot_0001",
                    start=0.0,
                    end=5.0,
                    source_type="presentation",
                    crop=NormalizedRect(0, 0, 1, 1),
                    caption_anchor=anchor,
                    caption_rect=NormalizedRect(0.08, 0.62, 0.76, 0.14),
                )
            ],
        )

    def test_anchor_is_applied_and_the_rect_follows(self):
        plan = self._plan("center")
        updated, audit = apply_caption_directives(
            plan, [ShotDirective(start=0.0, end=5.0, caption_anchor="upper_center")]
        )
        self.assertEqual(updated.shots[0].caption_anchor, "upper_center")
        self.assertNotEqual(updated.shots[0].caption_rect.y, 0.62)
        self.assertTrue(any("caption_center->upper_center" in line for line in audit))

    def test_applied_anchor_flows_into_ass_placements(self):
        plan = self._plan("center")
        updated, _ = apply_caption_directives(
            plan, [ShotDirective(start=0.0, end=5.0, caption_anchor="upper_center")]
        )
        placements = build_caption_placements(updated, 1080, 1920)
        self.assertEqual(len(placements), 1)
        assert isinstance(placements[0], CaptionPlacement)
        self.assertEqual(placements[0].alignment, 8)
        self.assertEqual(placements[0].anchor, "upper_center")

    def test_input_plan_is_not_mutated(self):
        plan = self._plan("center")
        apply_caption_directives(
            plan, [ShotDirective(start=0.0, end=5.0, caption_anchor="upper_center")]
        )
        self.assertEqual(plan.shots[0].caption_anchor, "center")

    def test_directive_for_a_dropped_shot_is_not_applied(self):
        plan = self._plan("center")
        updated, _ = apply_caption_directives(
            plan,
            [ShotDirective(start=0.0, end=5.0, keep=False, caption_anchor="upper_center")],
        )
        self.assertEqual(updated.shots[0].caption_anchor, "center")

    def test_same_anchor_is_a_no_op(self):
        plan = self._plan("center")
        updated, audit = apply_caption_directives(
            plan, [ShotDirective(start=0.0, end=5.0, caption_anchor="center")]
        )
        self.assertEqual(audit, [])


class TestDirectorPrompt(unittest.TestCase):
    def _sheet(self, count: int = 8):
        from src.director_v2 import ContactSheet
        return ContactSheet(
            path=Path("sheet.jpg"),
            tiles=[i * 1.5 for i in range(count)],
            columns=4,
            rows=2,
            tile_width=320,
            tile_height=180,
        )

    def test_contains_legend_transcript_and_schema(self):
        prompt = build_director_prompt(
            contact_sheet=self._sheet(),
            source_index=_index(),
            segments=_segments(),
            clip_start=0.0,
            clip_end=48.0,
        )
        self.assertIn("r0c0=0.0s", prompt)
        self.assertIn("The first thing nobody tells you", prompt)
        self.assertIn("presentation_slide", prompt)
        self.assertIn("pip_slide", prompt)
        for layout in DIRECTOR_LAYOUTS:
            self.assertIn(layout, prompt)

    def test_survives_a_missing_source_index(self):
        prompt = build_director_prompt(
            contact_sheet=self._sheet(),
            source_index=None,
            segments=[],
            clip_start=0.0,
            clip_end=48.0,
        )
        self.assertIn("no speech recognised", prompt)

    def test_transcript_is_truncated_not_unbounded(self):
        long_segments = [
            TranscriptSegment(start=float(i), end=float(i) + 1, text="word " * 60, words=[])
            for i in range(200)
        ]
        prompt = build_director_prompt(
            contact_sheet=self._sheet(),
            source_index=_index(),
            segments=long_segments,
            clip_start=0.0,
            clip_end=200.0,
            max_chars=2000,
        )
        self.assertIn("[truncated]", prompt)
        self.assertLess(len(prompt), 8000)

    def test_segments_outside_the_window_are_excluded(self):
        prompt = build_director_prompt(
            contact_sheet=self._sheet(),
            source_index=_index(),
            segments=[TranscriptSegment(start=900.0, end=999.0, text="unrelated chatter", words=[])],
            clip_start=0.0,
            clip_end=48.0,
        )
        self.assertNotIn("unrelated chatter", prompt)


class TestPlanShotDirectives(unittest.TestCase):
    def test_disabled_by_config_returns_nothing(self):
        with mock.patch.object(director_v2, "build_contact_sheet") as sheet:
            review = plan_shot_directives(
                video_path=Path("v.mp4"),
                source_index=None,
                segments=[],
                clip_start=0.0,
                clip_end=10.0,
                output_path=Path("s.jpg"),
            )
        sheet.assert_not_called()
        self.assertFalse(review.enabled)
        self.assertEqual(review.directives, [])
        self.assertEqual(review.failure, "disabled_by_config")

    def _enable(self):
        return mock.patch("src.config.DIRECTOR_V2_ENABLED", True)

    def test_no_api_key_is_reported_not_raised(self):
        with self._enable(), mock.patch(
            "src.config.GEMINI_API_KEY", ""
        ), mock.patch.object(
            director_v2, "build_contact_sheet",
            return_value=director_v2.ContactSheet(
                path=Path("s.jpg"), tiles=[0.0], columns=1, rows=1,
                tile_width=320, tile_height=180,
            ),
        ):
            review = plan_shot_directives(
                video_path=Path("v.mp4"), source_index=None, segments=[],
                clip_start=0.0, clip_end=10.0, output_path=Path("s.jpg"),
            )
        self.assertEqual(review.failure, "no_api_key")
        self.assertEqual(review.directives, [])

    def test_api_exception_is_contained(self):
        sheet = director_v2.ContactSheet(
            path=Path("s.jpg"), tiles=[0.0, 5.0], columns=2, rows=1,
            tile_width=320, tile_height=180,
        )
        with self._enable(), mock.patch(
            "src.config.GEMINI_API_KEY", "k"
        ), mock.patch.object(
            director_v2, "build_contact_sheet", return_value=sheet
        ), mock.patch(
            "src.viral_detector.query_gemini_models", side_effect=RuntimeError("503")
        ):
            review = plan_shot_directives(
                video_path=Path("v.mp4"), source_index=None, segments=[],
                clip_start=0.0, clip_end=10.0, output_path=Path("s.jpg"),
            )
        self.assertIn("api_error", review.failure or "")
        self.assertEqual(review.directives, [])

    def test_happy_path_returns_directives(self):
        sheet = director_v2.ContactSheet(
            path=Path("s.jpg"), tiles=[0.0, 5.0], columns=2, rows=1,
            tile_width=320, tile_height=180,
        )
        reply = _reply({"start": 0.0, "end": 5.0, "layout": "pip_slide", "keep": True})
        with self._enable(), mock.patch(
            "src.config.GEMINI_API_KEY", "k"
        ), mock.patch.object(
            director_v2, "build_contact_sheet", return_value=sheet
        ), mock.patch(
            "src.viral_detector.query_gemini_models", return_value=reply
        ) as call:
            review = plan_shot_directives(
                video_path=Path("v.mp4"), source_index=_index(), segments=_segments(),
                clip_start=0.0, clip_end=10.0, output_path=Path("s.jpg"),
            )
        self.assertTrue(review.usable)
        self.assertEqual(review.directives[0].layout, "pip_slide")
        # The contact sheet must actually be attached, or the model is guessing.
        self.assertEqual(call.call_args.kwargs["image_path"], Path("s.jpg"))

    def test_malformed_reply_yields_no_directives(self):
        sheet = director_v2.ContactSheet(
            path=Path("s.jpg"), tiles=[0.0], columns=1, rows=1,
            tile_width=320, tile_height=180,
        )
        with self._enable(), mock.patch(
            "src.config.GEMINI_API_KEY", "k"
        ), mock.patch.object(
            director_v2, "build_contact_sheet", return_value=sheet
        ), mock.patch(
            "src.viral_detector.query_gemini_models", return_value="I think it's fine"
        ):
            review = plan_shot_directives(
                video_path=Path("v.mp4"), source_index=None, segments=[],
                clip_start=0.0, clip_end=10.0, output_path=Path("s.jpg"),
            )
        self.assertEqual(review.directives, [])
        self.assertEqual(review.failure, "no_valid_directives")

    def test_review_serialises(self):
        review = director_v2.DirectorReview(
            directives=[ShotDirective(start=0.0, end=4.0, layout="pip_slide")],
            audit=["ok"],
        )
        payload = review.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["directives"][0]["duration"], 4.0)
        self.assertTrue(payload["enabled"])


class TestDirectiveReachesTheRenderer(unittest.TestCase):
    """The point of the phase: a directive must change real filtergraph output."""

    def _graph(self, shot: ShotPlan) -> str:
        from src.video_editor import build_video_filtergraph
        return build_video_filtergraph(_framing([shot]))

    def test_pip_slide_directive_emits_a_real_pip_chain(self):
        from src.video_editor import (
            PIP_INSET_WIDTH,
            PIP_INSET_X,
            PIP_SLIDE_CANVAS_RATIO,
            build_video_filtergraph,
        )
        base = _shot(0, 5, "portrait_face", speaker1_box=BOX_A)
        before = build_video_filtergraph(_framing([base]))
        self.assertNotIn("boxblur", before)

        updated, _ = apply_shot_directives(
            _framing([base]), [ShotDirective(start=0.0, end=5.0, layout="pip_slide")]
        )
        graph = build_video_filtergraph(updated)
        # blurred slide canvas behind...
        self.assertIn("boxblur=30:5", graph)
        # ...plus the bordered host inset composited on top of it.
        self.assertIn(f"scale={PIP_INSET_WIDTH}", graph)
        self.assertIn(f"overlay={PIP_INSET_X}:", graph)
        self.assertIn("drawbox=x=", graph)
        # The canvas must stay under PIP_SLIDE_CANVAS_RATIO or the motion
        # guarantee has no slack and the slide renders frozen.
        canvas = f"crop={int(1920 * PIP_SLIDE_CANVAS_RATIO)}"
        self.assertIn(canvas, graph)

    def test_presentation_slide_directive_changes_the_chain(self):
        from src.video_editor import build_video_filtergraph
        base = _shot(0, 5, "portrait_face", speaker1_box=BOX_A)
        updated, _ = apply_shot_directives(
            _framing([base]),
            [ShotDirective(start=0.0, end=5.0, layout="presentation_slide")],
        )
        self.assertIn("overlay=(W-w)/2:(H-h)/2", build_video_filtergraph(updated))

    def test_rejected_directive_leaves_the_graph_unchanged(self):
        from src.video_editor import build_video_filtergraph
        base = _shot(0, 5, "portrait_face")
        before = build_video_filtergraph(_framing([base]))
        updated, _ = apply_shot_directives(
            _framing([base]), [ShotDirective(start=0.0, end=5.0, layout="pip_slide")]
        )
        self.assertEqual(build_video_filtergraph(updated), before)


class TestContactSheetAgainstRealVideo(unittest.TestCase):
    """
    The mocked tests cover the montage maths; this one covers the decode path.

    Everything else in this file can pass while `sample_frames_at` returns
    nothing from a real file, which would make the director silently blind.
    """

    def test_sheet_is_built_from_an_actual_encoded_file(self):
        from src.source_index import probe_source_media
        with tempfile_dir() as tmp:
            video = tmp / "clip.mp4"
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i",
                    "testsrc=size=640x360:rate=15:duration=8",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
                ],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0 or not video.exists():
                self.skipTest(f"ffmpeg could not build a fixture: {result.stderr[-200:]}")

            media = probe_source_media(video)
            self.assertGreater(media.duration, 5.0)

            sheet = build_contact_sheet(
                video, 0.0, media.duration, tmp / "sheet.jpg", tile_count=6
            )
            self.assertIsNotNone(sheet)
            assert sheet is not None
            self.assertTrue(sheet.path.exists())
            self.assertGreater(sheet.path.stat().st_size, 1024)

            # The legend must name every tile, or the model cannot cite a shot.
            legend = sheet.legend()
            for index, timestamp in enumerate(sheet.tiles):
                row, column = divmod(index, sheet.columns)
                self.assertIn(f"r{row}c{column}={timestamp:.1f}s", legend)
            self.assertLessEqual(sheet.tiles[0], sheet.tiles[-1])

            # A readable JPEG header, so a blank/garbage sheet is detectable.
            header = sheet.path.read_bytes()[:2]
            self.assertEqual(header, b"\xff\xd8")

    def test_sampler_returns_increasing_timestamps(self):
        from src.source_index import probe_source_media, sample_frames_at
        with tempfile_dir() as tmp:
            video = tmp / "clip.mp4"
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=10:duration=6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
                ],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0 or not video.exists():
                self.skipTest("ffmpeg fixture unavailable")
            duration = probe_source_media(video).duration
            samples = sample_frames_at(video, 0.0, duration, count=5)
        self.assertGreaterEqual(len(samples), 2)
        stamps = [timestamp for timestamp, _ in samples]
        self.assertEqual(stamps, sorted(stamps))
        for timestamp in stamps:
            self.assertGreaterEqual(timestamp, 0.0)
            self.assertLessEqual(timestamp, duration + 1.0)


class TestSourceIndexSampler(unittest.TestCase):
    def test_sample_frames_at_derives_a_sane_rate(self):
        from src.source_index import sample_frames_at
        with mock.patch("src.source_index._sample_frames") as inner:
            sample_frames_at(Path("v.mp4"), 0.0, 48.0, count=12)
        self.assertEqual(inner.call_args.kwargs["sample_fps"], 0.25)  # 12 / 48 s
        self.assertEqual(inner.call_args.kwargs["max_samples"], 12)

    def test_zero_count_cannot_divide_by_zero(self):
        from src.source_index import sample_frames_at
        with mock.patch("src.source_index._sample_frames") as inner:
            sample_frames_at(Path("v.mp4"), 5.0, 5.0, count=0)
        self.assertEqual(inner.call_args.kwargs["max_samples"], 1)


@contextlib.contextmanager
def tempfile_dir():
    path = Path(tempfile.mkdtemp())
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
