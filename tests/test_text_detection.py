"""
Tests for text-region detection (Phase 1b).

The check these unblock is `caption_source_collision`, a *blocking* QA code.
Before this phase nothing populated real `protected_regions`, so it could not
fire. Two properties matter more than raw detection quality:

1. A detector that runs per frame must not flood the protected-region list, or
   the collision check fires on everything and QA becomes a coin flip.
2. The collision check must consider every protected region, not just the first.
"""
import unittest
from typing import List
from unittest import mock

import cv2
import numpy as np

from src import text_detection as td
from src.composition_planner import (
    DEFAULT_SAFE_ZONE,
    NormalizedRect,
    ShotComposition,
    _bottom_text_collision,
)
from src.config import OCR_CONFIDENCE_MIN, OCR_MAX_REGIONS_PER_FRAME
from src.text_detection import (
    EdgeTextDetector,
    OcrTextDetector,
    TextBox,
    detect_text_regions,
    get_text_detector,
    merge_text_boxes,
    reset_text_detector_cache,
)

W, H = 1280, 720


def _box(x, y, w, h, confidence=0.5, source="test") -> TextBox:
    return TextBox(x=x, y=y, width=w, height=h, confidence=confidence, source=source)


def _slide_frame() -> np.ndarray:
    frame = np.full((H, W, 3), 245, np.uint8)
    cv2.rectangle(frame, (0, 0), (W, 90), (40, 40, 120), -1)
    cv2.putText(frame, "THE COMPOUND INTEREST TRAP", (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.putText(frame, "Most people misunderstand this", (60, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    return frame


def _quiet_frame() -> np.ndarray:
    return np.full((H, W, 3), 128, np.uint8)


class TestTextBox(unittest.TestCase):
    def test_edges(self):
        box = _box(0.1, 0.2, 0.3, 0.4)
        self.assertAlmostEqual(box.right, 0.4)
        self.assertAlmostEqual(box.bottom, 0.6)

    def test_iou_identical_is_one(self):
        self.assertAlmostEqual(_box(0.1, 0.1, 0.2, 0.2).iou(_box(0.1, 0.1, 0.2, 0.2)), 1.0)

    def test_iou_disjoint_is_zero(self):
        self.assertEqual(_box(0.0, 0.0, 0.1, 0.1).iou(_box(0.8, 0.8, 0.1, 0.1)), 0.0)

    def test_iou_half_overlap(self):
        value = _box(0.0, 0.0, 0.2, 0.2).iou(_box(0.1, 0.0, 0.2, 0.2))
        self.assertAlmostEqual(value, 1.0 / 3.0, places=4)

    def test_iou_touching_edges_do_not_count(self):
        self.assertEqual(_box(0.0, 0.0, 0.1, 0.1).iou(_box(0.1, 0.0, 0.1, 0.1)), 0.0)

    def test_iou_is_symmetric(self):
        a, b = _box(0.0, 0.0, 0.3, 0.2), _box(0.15, 0.05, 0.3, 0.4)
        self.assertAlmostEqual(a.iou(b), b.iou(a))


class TestMergeTextBoxes(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge_text_boxes([]), [])

    def test_identical_boxes_collapse_to_one(self):
        merged = merge_text_boxes([_box(0.1, 0.5, 0.4, 0.06) for _ in range(12)])
        self.assertEqual(len(merged), 1)

    def test_frame_to_frame_jitter_collapses(self):
        """The whole reason this function exists: a per-frame detector jitters."""
        jittered: List[TextBox] = []
        for index in range(20):
            jittered.append(_box(
                0.100 + (index % 5) * 0.002,
                0.500 + (index % 4) * 0.002,
                0.400, 0.060,
            ))
        self.assertEqual(len(merge_text_boxes(jittered)), 1)

    def test_distant_blocks_stay_separate(self):
        merged = merge_text_boxes([
            _box(0.05, 0.10, 0.40, 0.06),
            _box(0.05, 0.70, 0.40, 0.06),
        ])
        self.assertEqual(len(merged), 2)

    def test_cluster_keeps_the_highest_confidence(self):
        merged = merge_text_boxes([
            _box(0.1, 0.5, 0.4, 0.06, confidence=0.31, source="ocr"),
            _box(0.1, 0.5, 0.4, 0.06, confidence=0.93, source="ocr"),
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 0.93)
        self.assertEqual(merged[0].source, "ocr")

    def test_cluster_averages_position(self):
        merged = merge_text_boxes([
            _box(0.10, 0.50, 0.40, 0.06, confidence=0.9),
            _box(0.14, 0.50, 0.40, 0.06, confidence=0.8),
        ])
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].x, 0.12, places=4)

    def test_output_is_stably_ordered(self):
        merged = merge_text_boxes([
            _box(0.05, 0.70, 0.2, 0.05),
            _box(0.05, 0.10, 0.2, 0.05),
            _box(0.60, 0.10, 0.2, 0.05),
        ])
        self.assertEqual([(b.x, b.y) for b in merged], [(0.05, 0.10), (0.60, 0.10), (0.05, 0.70)])

    def test_threshold_is_respected(self):
        # IoU of this pair is ~0.33, so the threshold decides the outcome.
        a, b = _box(0.0, 0.0, 0.2, 0.2), _box(0.10, 0.0, 0.2, 0.2)
        self.assertAlmostEqual(a.iou(b), 1.0 / 3.0, places=3)
        self.assertEqual(len(merge_text_boxes([a, b], iou_threshold=0.3)), 1)
        self.assertEqual(len(merge_text_boxes([a, b], iou_threshold=0.4)), 2)


class TestEdgeTextDetector(unittest.TestCase):
    def test_finds_real_rendered_text(self):
        boxes = EdgeTextDetector().detect(_slide_frame())
        self.assertGreaterEqual(len(boxes), 2)
        for box in boxes:
            self.assertEqual(box.source, "edge")
            self.assertTrue(0.0 <= box.x <= 1.0)
            self.assertTrue(0.0 <= box.y <= 1.0)
            self.assertTrue(0.0 < box.width <= 1.0)
            self.assertTrue(0.0 < box.height <= 1.0)

    def test_finds_nothing_on_a_blank_frame(self):
        self.assertEqual(EdgeTextDetector().detect(_quiet_frame()), [])

    def test_is_deterministic(self):
        detector = EdgeTextDetector()
        frame = _slide_frame()
        first = [(b.x, b.y, b.width, b.height) for b in detector.detect(frame)]
        second = [(b.x, b.y, b.width, b.height) for b in detector.detect(frame)]
        self.assertEqual(first, second)

    def test_confidence_is_below_a_real_ocr_tier(self):
        """An OCR detection must be able to outrank the heuristic."""
        self.assertLess(EdgeTextDetector().detect(_slide_frame())[0].confidence, 0.9)


class TestOcrTextDetector(unittest.TestCase):
    def _engine_returning(self, payload):
        engine = mock.Mock()
        engine.return_value = payload
        return engine

    def _quad(self, x, y, w=100, h=30):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    def test_unavailable_package_yields_none(self):
        """This machine has no RapidOCR, which is the current real state."""
        with mock.patch.dict("sys.modules", {"rapidocr_onnxruntime": None}):
            self.assertIsNone(OcrTextDetector.try_create())

    def test_engine_that_fails_to_construct_yields_none(self):
        module = mock.Mock()
        module.RapidOCR.side_effect = RuntimeError("no weights")
        with mock.patch.dict("sys.modules", {"rapidocr_onnxruntime": module}):
            self.assertIsNone(OcrTextDetector.try_create())

    def test_parses_the_results_elapse_tuple(self):
        engine = self._engine_returning((
            [(self._quad(100, 200), "MOST PEOPLE", 0.94)],
            [0.1, 0.2, 0.3],
        ))
        boxes = OcrTextDetector(engine).detect(_slide_frame())
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].source, "ocr")
        self.assertGreaterEqual(boxes[0].confidence, OCR_CONFIDENCE_MIN)
        self.assertAlmostEqual(boxes[0].x, 100 / W, places=3)
        self.assertAlmostEqual(boxes[0].y, 200 / H, places=3)

    def test_parses_a_bare_list_result(self):
        engine = self._engine_returning([(self._quad(10, 20), "hi", 0.8)])
        self.assertEqual(len(OcrTextDetector(engine).detect(_slide_frame())), 1)

    def test_low_confidence_is_dropped(self):
        below = OCR_CONFIDENCE_MIN - 0.1
        engine = self._engine_returning((
            [(self._quad(10, 20), "noisy", below)],
            [0.1],
        ))
        self.assertEqual(OcrTextDetector(engine).detect(_slide_frame()), [])

    def test_engine_exception_returns_empty_not_raises(self):
        engine = mock.Mock(side_effect=RuntimeError("onnx blew up"))
        self.assertEqual(OcrTextDetector(engine).detect(_slide_frame()), [])

    def test_two_element_entries_are_rejected(self):
        """
        Regression: confidence is entry[2], not entry[1].

        Reading index 1 picks up the recognised text, float() fails, and the box
        is dropped silently -- a good response would produce zero regions with
        nothing logged. Pin the three-element contract.
        """
        engine = self._engine_returning(([(self._quad(10, 20), "0.95")], [0.1]))
        self.assertEqual(OcrTextDetector(engine).detect(_slide_frame()), [])

    def test_confidence_comes_from_the_third_element(self):
        engine = self._engine_returning((
            [(self._quad(100, 200), "MOST PEOPLE", 0.94)],
            [0.1],
        ))
        boxes = OcrTextDetector(engine).detect(_slide_frame())
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].confidence, 0.94, places=3)

    def test_text_that_looks_numeric_does_not_become_confidence(self):
        """A recognised string like "0.9" must not be read as the score."""
        engine = self._engine_returning((
            [(self._quad(100, 200), "0.9", 0.31)],
            [0.1],
        ))
        boxes = OcrTextDetector(engine).detect(_slide_frame())
        self.assertEqual(boxes, [])

    def test_malformed_shapes_are_skipped_not_fatal(self):
        engine = self._engine_returning((
            [
                "not a tuple",
                (self._quad(10, 20),),
                (self._quad(30, 40), "text", "not a number"),
                (self._quad(50, 60), "ok", 0.9),
            ],
            [0.1],
        ))
        boxes = OcrTextDetector(engine).detect(_slide_frame())
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].x, 50 / W, places=3)

    def test_too_few_corners_rejected(self):
        engine = self._engine_returning(([[[10, 20], [30, 40]], "x", 0.9], [0.1]))
        self.assertEqual(OcrTextDetector(engine).detect(_slide_frame()), [])

    def test_garbage_polygon_rejected(self):
        engine = self._engine_returning(([["a", "b"], "x", 0.9], [0.1]))
        self.assertEqual(OcrTextDetector(engine).detect(_slide_frame()), [])

    def test_none_result_returns_empty(self):
        engine = self._engine_returning(None)
        self.assertEqual(OcrTextDetector(engine).detect(_slide_frame()), [])

    def test_empty_result_returns_empty(self):
        self.assertEqual(OcrTextDetector(self._engine_returning(([], []))).detect(
            _slide_frame()), [])

    def test_regions_are_bounded(self):
        payload = (
            [(self._quad(10 + i * 5, 20), "x", 0.9) for i in range(80)],
            [0.1],
        )
        boxes = OcrTextDetector(self._engine_returning(payload)).detect(_slide_frame())
        self.assertLessEqual(len(boxes), OCR_MAX_REGIONS_PER_FRAME)


class TestDetectorResolution(unittest.TestCase):
    def setUp(self):
        reset_text_detector_cache()
        self.addCleanup(reset_text_detector_cache)

    def test_auto_falls_back_to_edge_here(self):
        detector = get_text_detector(force="auto")
        self.assertTrue(detector.name.endswith("edge"))

    def test_explicit_edge_is_the_edge_detector_alone(self):
        self.assertEqual(get_text_detector(force="edge").name, "edge")

    def test_explicit_ocr_still_degrades_to_something_usable(self):
        with mock.patch.object(OcrTextDetector, "try_create", return_value=None):
            detector = get_text_detector(force="ocr")
        self.assertTrue(detector.name.endswith("edge"))

    def test_auto_prefers_ocr_when_available(self):
        fake = mock.Mock()
        fake.name = "ocr"
        fake.detect.return_value = [_box(0.1, 0.1, 0.2, 0.05, source="ocr")]
        with mock.patch.object(OcrTextDetector, "try_create", return_value=fake):
            detector = get_text_detector(force="auto")
        self.assertEqual(detector.name, "ocr+edge")
        self.assertEqual(detector.detect(_slide_frame())[0].source, "ocr")

    def test_chained_detector_short_circuits_on_first_hit(self):
        first = mock.Mock()
        first.name = "ocr"
        first.detect.return_value = [_box(0.1, 0.1, 0.2, 0.05)]
        second = mock.Mock()
        second.name = "edge"
        second.detect.return_value = [_box(0.5, 0.5, 0.2, 0.05)]
        detector = td._ChainedDetector([first, second])
        self.assertEqual(len(detector.detect(_slide_frame())), 1)
        second.detect.assert_not_called()

    def test_chained_detector_survives_a_raising_tier(self):
        broken = mock.Mock()
        broken.name = "ocr"
        broken.detect.side_effect = RuntimeError("boom")
        healthy = mock.Mock()
        healthy.name = "edge"
        healthy.detect.return_value = [_box(0.5, 0.5, 0.2, 0.05)]
        detector = td._ChainedDetector([broken, healthy])
        self.assertEqual(len(detector.detect(_slide_frame())), 1)

    def test_force_does_not_poison_the_cache(self):
        first = get_text_detector(force="edge")
        second = get_text_detector(force="auto")
        self.assertEqual(first.name, "edge")
        self.assertIsNot(first, second)

    def test_unresolvable_tier_still_returns_a_working_detector(self):
        self.assertIsNotNone(get_text_detector(force="nonsense").detect(_slide_frame()) is not None)


class TestDetectTextRegions(unittest.TestCase):
    def setUp(self):
        reset_text_detector_cache()
        self.addCleanup(reset_text_detector_cache)

    def test_finds_text_on_a_real_frame(self):
        self.assertTrue(detect_text_regions(_slide_frame(), force="edge"))

    def test_blank_frame_yields_nothing(self):
        self.assertEqual(detect_text_regions(_quiet_frame(), force="edge"), [])

    def test_degenerate_frames_are_safe(self):
        self.assertEqual(detect_text_regions(np.zeros((0, 0, 3), np.uint8)), [])
        self.assertEqual(detect_text_regions(np.zeros((4, 4, 3), np.uint8)), [])

    def test_result_is_bounded_and_merged(self):
        boxes = detect_text_regions(_slide_frame(), force="edge")
        self.assertLessEqual(len(boxes), OCR_MAX_REGIONS_PER_FRAME)
        for box in boxes:
            self.assertTrue(0.0 <= box.x and box.x + box.width <= 1.001)


class TestSourceIndexRegionMerging(unittest.TestCase):
    """The anti-flooding guarantee at the place it actually matters."""

    def _region(self, x, y, w=0.4, h=0.06, confidence=0.5, source="edge"):
        from src.source_index import TextRegion
        return TextRegion(x=x, y=y, width=w, height=h, confidence=confidence, source=source)

    def test_jittering_regions_collapse_to_one(self):
        from src.source_index import _merge_shot_text_regions
        regions = [
            self._region(0.100 + (i % 4) * 0.002, 0.500 + (i % 3) * 0.002)
            for i in range(30)
        ]
        self.assertEqual(len(_merge_shot_text_regions(regions)), 1)

    def test_separate_blocks_stay_separate(self):
        from src.source_index import _merge_shot_text_regions
        regions = [self._region(0.05, 0.10), self._region(0.05, 0.80)]
        self.assertEqual(len(_merge_shot_text_regions(regions)), 2)

    def test_empty_input(self):
        from src.source_index import _merge_shot_text_regions
        self.assertEqual(_merge_shot_text_regions([]), [])

    def test_source_label_survives(self):
        from src.source_index import _merge_shot_text_regions
        merged = _merge_shot_text_regions([self._region(0.1, 0.5, source="ocr")])
        self.assertEqual(merged[0].source, "ocr")

    def test_estimate_uses_the_configured_detector(self):
        from src.source_index import _estimate_text_regions
        fake = mock.Mock()
        fake.name = "ocr"
        fake.detect.return_value = [_box(0.2, 0.4, 0.3, 0.05, source="ocr")]
        with mock.patch.object(td, "get_text_detector", return_value=fake):
            regions = _estimate_text_regions(_slide_frame())
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].source, "ocr")
        self.assertAlmostEqual(regions[0].x, 0.2, places=4)


class TestCollisionCheckNowFires(unittest.TestCase):
    """
    The bug this phase exists to fix: the check only looked at
    `protected_regions[0]`, so a collision on any later region passed QA.
    """

    def _plan(self, regions, caption_y=0.62):
        from src.composition_planner import CompositionPlan, ShotComposition
        return CompositionPlan(
            plan_id="p", width=1080, height=1920, safe_zone=DEFAULT_SAFE_ZONE,
            shots=[ShotComposition(
                shot_id="shot_0001", start=0.0, end=10.0, source_type="presentation",
                crop=NormalizedRect(0, 0, 1, 1),
                caption_anchor="lower_center",
                caption_rect=NormalizedRect(0.08, caption_y, 0.76, 0.14),
                protected_regions=regions,
            )],
        )

    def _edit_plan(self):
        from src.edit_director import EditPlan, HookCandidate
        return EditPlan(
            plan_id="p", start=0.0, end=10.0,
            hook=HookCandidate("story", "a hook", 0.0, 3.0, 0.8),
            beats=[], selected_shot_ids=["shot_0001"], confidence=0.8,
        )

    def _codes(self, plan):
        from src.editorial_qa import _plan_issues
        return [
            issue.code
            for issue in _plan_issues(self._edit_plan(), plan)
            if issue.code == "caption_source_collision"
        ]

    def test_no_regions_no_collision(self):
        self.assertEqual(self._codes(self._plan([])), [])

    def test_collision_on_the_first_region_fires(self):
        self.assertEqual(
            self._codes(self._plan([NormalizedRect(0.08, 0.60, 0.5, 0.10)])),
            ["caption_source_collision"],
        )

    def test_collision_on_the_second_region_fires(self):
        """This is the case that previously passed QA."""
        regions = [
            NormalizedRect(0.05, 0.05, 0.4, 0.08),
            NormalizedRect(0.08, 0.63, 0.5, 0.10),
        ]
        self.assertEqual(self._codes(self._plan(regions)), ["caption_source_collision"])

    def test_collision_on_the_last_of_many_fires(self):
        regions = [NormalizedRect(0.05, 0.02 + i * 0.05, 0.3, 0.04) for i in range(6)]
        regions.append(NormalizedRect(0.08, 0.64, 0.5, 0.10))
        self.assertEqual(self._codes(self._plan(regions)), ["caption_source_collision"])

    def test_non_overlapping_regions_pass(self):
        regions = [
            NormalizedRect(0.05, 0.05, 0.4, 0.08),
            NormalizedRect(0.05, 0.20, 0.4, 0.08),
        ]
        self.assertEqual(self._codes(self._plan(regions)), [])

    def test_message_reports_the_count(self):
        from src.editorial_qa import _plan_issues
        regions = [
            NormalizedRect(0.08, 0.60, 0.5, 0.10),
            NormalizedRect(0.30, 0.66, 0.4, 0.10),
        ]
        issues = [i for i in _plan_issues(self._edit_plan(), self._plan(regions))
                  if i.code == "caption_source_collision"]
        self.assertIn("2 source text region", issues[0].message)

    def test_bottom_collision_helper_agrees(self):
        self.assertTrue(_bottom_text_collision(
            [NormalizedRect(0.1, 0.85, 0.5, 0.1)], DEFAULT_SAFE_ZONE
        ))
        self.assertFalse(_bottom_text_collision(
            [NormalizedRect(0.1, 0.10, 0.5, 0.1)], DEFAULT_SAFE_ZONE
        ))


class TestProtectedRegionsReachTheCheck(unittest.TestCase):
    """End to end: a detected region must become a rect the check can fail on."""

    def test_detected_region_flows_into_a_collision(self):
        from src.composition_planner import CompositionPlan, _rect_from_region
        from src.editorial_qa import _plan_issues
        from src.edit_director import EditPlan, HookCandidate
        from src.source_index import TextRegion, _merge_shot_text_regions

        frame = _slide_frame()
        detected = [TextRegion(**box.to_dict()) for box in detect_text_regions(frame, force="edge")]
        self.assertTrue(detected, "the edge tier should see the rendered text")
        merged = _merge_shot_text_regions(detected)
        protected = [_rect_from_region(region) for region in merged]

        # Put the caption exactly where a detected line sits.
        target = protected[0]
        plan = CompositionPlan(
            plan_id="p", width=1080, height=1920, safe_zone=DEFAULT_SAFE_ZONE,
            shots=[ShotComposition(
                shot_id="shot_0001", start=0.0, end=10.0, source_type="presentation",
                crop=NormalizedRect(0, 0, 1, 1),
                caption_anchor="lower_center",
                caption_rect=target,
                protected_regions=protected,
            )],
        )
        edit_plan = EditPlan(
            plan_id="p", start=0.0, end=10.0,
            hook=HookCandidate("story", "a hook", 0.0, 3.0, 0.8),
            beats=[], selected_shot_ids=["shot_0001"], confidence=0.8,
        )
        codes = [i.code for i in _plan_issues(edit_plan, plan)]
        self.assertIn("caption_source_collision", codes)


class TestBuildSourceIndexMergesRegions(unittest.TestCase):
    """
    Integration: the merge must actually be wired into the shot assembly.

    Mutation testing found this gap. `test_jittering_regions_collapse_to_one`
    exercised `_merge_shot_text_regions` directly, so replacing the *call* in
    `build_source_index` with `list(shot_text)` left the whole suite green while
    the anti-flooding guarantee was gone. These tests go through the real entry
    point on a real encoded video.
    """

    def setUp(self):
        reset_text_detector_cache()
        self.addCleanup(reset_text_detector_cache)

    def _video(self) -> "object":
        import shutil
        import tempfile
        from pathlib import Path

        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        video = directory / "text.mp4"

        # Rendered with OpenCV rather than ffmpeg's drawtext on purpose:
        # drawtext needs fontconfig, which is not present on every runner, and a
        # missing font makes the fixture silently unbuildable. This is the same
        # reason the director's contact sheet is drawn with OpenCV.
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
        if not writer.isOpened():
            self.skipTest("OpenCV could not open an mp4v writer")
        try:
            for _ in range(90):
                frame = np.full((H, W, 3), 245, np.uint8)
                cv2.putText(frame, "THE COMPOUND INTEREST TRAP", (60, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
                cv2.putText(frame, "most people misunderstand this", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
                cv2.putText(frame, "and it costs them a decade", (60, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
                writer.write(frame)
        finally:
            writer.release()

        if not video.exists() or video.stat().st_size == 0:
            self.skipTest("OpenCV produced no fixture video")
        return video

    def test_index_regions_are_bounded_not_one_per_frame(self):
        from src.source_index import build_source_index
        video = self._video()
        index = build_source_index(video, sample_fps=2.0, max_samples=40, detect_faces=False)
        total = sum(len(shot.text_regions) for shot in index.shots)
        self.assertGreater(total, 0, "the fixture has text; nothing was detected")
        # ~12 sampled frames per shot. Without cross-frame merging this is ~12
        # overlapping regions per line; with it, a handful of stable regions.
        self.assertLessEqual(
            total, OCR_MAX_REGIONS_PER_FRAME,
            "regions were not merged across frames; QA would fire on everything",
        )

    def test_index_regions_are_normalised_and_labelled(self):
        from src.source_index import build_source_index
        video = self._video()
        index = build_source_index(video, sample_fps=2.0, max_samples=40, detect_faces=False)
        regions = [region for shot in index.shots for region in shot.text_regions]
        self.assertTrue(regions)
        for region in regions:
            self.assertGreaterEqual(region.x, 0.0)
            self.assertGreaterEqual(region.y, 0.0)
            self.assertLessEqual(region.x + region.width, 1.001)
            self.assertLessEqual(region.y + region.height, 1.001)
            self.assertTrue(region.source)

    def test_repeated_runs_are_stable(self):
        from src.source_index import build_source_index
        video = self._video()
        counts = []
        for _ in range(2):
            reset_text_detector_cache()
            index = build_source_index(
                video, sample_fps=2.0, max_samples=40, detect_faces=False
            )
            counts.append(sum(len(shot.text_regions) for shot in index.shots))
        self.assertEqual(counts[0], counts[1])


if __name__ == "__main__":
    unittest.main()
