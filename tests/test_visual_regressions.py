"""Reproduction tests for the visual defects found in the AGENTS.md audit.

Every test here is written to **fail against the artifacts that were actually
published**, not against a mock. That distinction is the whole point. The
original defect class escaped because the suite asserted on filtergraph *strings*
and plan geometry, never on pixels -- so a test that re-asserts the same strings
would have passed while the video was broken.

Two layers:

* **L1 (pure).** The metrics are computed from synthetic numpy arrays. No ffmpeg,
  fast, and they prove the *detector* works: a synthetic frame with known bars
  must be reported with those bars.
* **L3 (golden master).** The six real published clips, each carrying a
  ``human_assessment`` in the manifest. The clip-level assertions state what a
  person saw; the test states what the verifier computes. When those disagree the
  verifier is wrong, not the human.

Fixture clips are skipped when absent. Run ``python tools/fetch_fixtures.py``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.verification import fixtures, geometry, motion, probe


def _frame(width: int, height: int, box: tuple[int, int, int, int], fill: int = 180) -> list:
    """Greyscale frame that is black outside ``box`` and ``fill`` inside it."""
    import numpy as np

    array = np.zeros((height, width), dtype=np.uint8)
    x, y, box_width, box_height = box
    array[y : y + box_height, x : x + box_width] = fill
    return array


class TestGeometryDetectorL1(unittest.TestCase):
    """L1: the pillarbox detector must measure synthetic bars exactly.

    Mutation guard: forcing `ContentBox.x` to 0 and `width` to `frame_width` in
    `content_box` fails `test_detects_symmetric_pillarbox`,
    `test_detects_asymmetric_pillarbox` and `test_detects_only_left_bar`.
    """

    def test_detects_symmetric_pillarbox(self) -> None:
        # 1080-wide frame whose content is 540 wide, centred: 25% bar each side.
        frame = _frame(1080, 1920, (270, 0, 540, 1920))
        box = geometry.content_box(frame)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertEqual(box.width, 540)
        self.assertAlmostEqual(box.bar_left_pct, 25.0, places=1)
        self.assertAlmostEqual(box.bar_right_pct, 25.0, places=1)
        self.assertAlmostEqual(box.worst_vertical_bar_pct, 25.0, places=1)
        self.assertAlmostEqual(box.aspect, 540 / 1920, places=4)

    def test_detects_asymmetric_pillarbox(self) -> None:
        # The 400k clip's frozen segment: 540px content inside 1080 at x=270,
        # but measured worst-bar on one side in other segments.
        frame = _frame(1080, 1920, (60, 0, 846, 1920))
        box = geometry.content_box(frame)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertEqual(box.width, 846)
        self.assertAlmostEqual(box.bar_left_pct, 100 * 60 / 1080, places=1)
        self.assertAlmostEqual(box.bar_right_pct, 100 * (1080 - 60 - 846) / 1080, places=1)

    def test_detects_only_left_bar(self) -> None:
        frame = _frame(1080, 1920, (270, 0, 810, 1920))
        box = geometry.content_box(frame)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertAlmostEqual(box.bar_left_pct, 25.0, places=1)
        self.assertAlmostEqual(box.bar_right_pct, 0.0, places=1)

    def test_detects_letterbox(self) -> None:
        frame = _frame(1080, 1920, (0, 480, 1080, 960))
        box = geometry.content_box(frame)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertAlmostEqual(box.worst_horizontal_bar_pct, 25.0, places=1)
        self.assertAlmostEqual(box.worst_vertical_bar_pct, 0.0, places=1)

    def test_full_bleed_frame_reports_no_bars(self) -> None:
        frame = _frame(1080, 1920, (0, 0, 1080, 1920))
        box = geometry.content_box(frame)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertEqual(box.worst_vertical_bar_pct, 0.0)
        self.assertEqual(box.worst_horizontal_bar_pct, 0.0)
        self.assertAlmostEqual(box.aspect, 1080 / 1920, places=4)

    def test_all_black_frame_has_no_box(self) -> None:
        import numpy as np

        self.assertIsNotNone(np)
        frame = np.zeros((1920, 1080), dtype=np.uint8)
        self.assertIsNone(geometry.content_box(frame))

    def test_stray_highlight_is_not_treated_as_content(self) -> None:
        # A single bright pixel must not be read as a full-bleed subject, which
        # would report 0% bars and hide a genuinely black frame.
        import numpy as np

        array = np.zeros((1920, 1080), dtype=np.uint8)
        array[960, 540] = 255
        self.assertIsNone(geometry.content_box(array))

    def test_summarise_reports_worst_not_mean(self) -> None:
        """A clip correct for 9 of 10 samples and broken in 1 is broken.

        This is the property that makes the metric useful at all. A mean over
        samples would dilute a 25% bar to 2.5% and pass it.
        """
        boxes = []
        for index in range(10):
            box = (0, 0, 1080, 1920) if index != 5 else (270, 0, 540, 1920)
            found = geometry.content_box(_frame(1080, 1920, box))
            self.assertIsNotNone(found)
            assert found is not None
            boxes.append(found)
        summary = geometry.summarise(boxes, frame_count=10)
        self.assertAlmostEqual(summary["worst_vertical_bar_pct"], 25.0, places=1)
        self.assertAlmostEqual(
            summary["mean_aspect"], (9 * (1080 / 1920) + (540 / 1920)) / 10, places=4
        )
        self.assertLess(summary["worst_aspect"], summary["mean_aspect"])


class TestBarVersusFadeL1(unittest.TestCase):
    """L1: padding must be told apart from a faded frame edge.

    Both shrink the fitted content box, and the first version of `analyse`
    conflated them. It reported 4.1% bars on `clip_2_WHY_DISNEY_PAY_IS_A_TRAP.mp4`,
    where the only inset is a `blur_stack` blurred background that is designed to
    be near-black. A verifier that cries wolf is worse than none, because people
    learn to ignore it.

    Mutation guard: forcing `classify_bars` to always return
    `is_real_bar=True` fails `test_pure_black_padding_is_a_real_bar`'s sibling
    `test_faded_edge_is_not_a_bar` and `test_dark_content_is_not_judged`.
    """

    def test_pure_black_padding_is_a_real_bar(self) -> None:
        frame = _frame(1080, 1920, (270, 0, 540, 1920), fill=180)
        box = geometry.content_box(frame)
        self.assertIsNotNone(box)
        assert box is not None
        verdict = geometry.classify_bars(frame, box)
        self.assertTrue(verdict.is_real_bar, verdict.reason)
        self.assertLess(verdict.bar_to_content_luma, geometry.BAR_TO_CONTENT_LUMA_MAX)
        self.assertAlmostEqual(verdict.vertical_pct, 25.0, places=1)

    def test_faded_edge_is_not_a_bar(self) -> None:
        """A uniformly dimmed frame has no padding, only dark edges."""
        import numpy as np

        base = np.full((1920, 1080), 200, dtype=np.uint8)
        base[:, :540] = 0  # a real subject/background boundary, not padding
        faded = (base.astype(np.float32) * 0.08).astype(np.uint8)  # heavy fade
        box = geometry.content_box(faded)
        # At 8% exposure the whole frame is under threshold, so there is no box
        # at all -- which is the correct answer and must not become a bar.
        if box is None:
            self.assertIsNone(box)
            return
        verdict = geometry.classify_bars(faded, box)
        self.assertFalse(verdict.is_real_bar, verdict.reason)

    def test_dark_content_is_not_judged(self) -> None:
        """Content below the exposure floor is unjudgeable, never a bar."""
        import numpy as np

        array = np.zeros((1920, 1080), dtype=np.uint8)
        array[200:1700, 200:900] = 30  # above the 24 content threshold...
        # ...but well below MIN_CONTENT_MEAN_LUMA, so it is a dim frame, not a
        # bright subject sitting inside genuine black padding.
        box = geometry.content_box(array)
        self.assertIsNotNone(box)
        assert box is not None
        verdict = geometry.classify_bars(array, box)
        self.assertFalse(verdict.is_real_bar, verdict.reason)
        self.assertIn("too dark", verdict.reason)

    def test_full_bleed_frame_is_not_a_bar(self) -> None:
        frame = _frame(1080, 1920, (0, 0, 1080, 1920))
        box = geometry.content_box(frame)
        assert box is not None
        verdict = geometry.classify_bars(frame, box)
        self.assertFalse(verdict.is_real_bar)
        self.assertEqual(verdict.vertical_pct, 0.0)
        self.assertEqual(verdict.horizontal_pct, 0.0)

    def test_letterbox_strips_do_not_look_like_pillarbox(self) -> None:
        """An empty left/right strip must not drag the ratio down to 'padding'.

        A letterboxed frame has zero-area side strips. Including them in the mean
        would halve it and make a genuine letterbox read as a pillarbox.
        """
        frame = _frame(1080, 1920, (0, 480, 1080, 960), fill=180)
        box = geometry.content_box(frame)
        assert box is not None
        self.assertAlmostEqual(box.worst_horizontal_bar_pct, 25.0, places=1)
        verdict = geometry.classify_bars(frame, box)
        self.assertTrue(verdict.is_real_bar, verdict.reason)


class TestGradedSeverityL1(unittest.TestCase):
    """L1: the blocking threshold must separate the corpus's real cases.

    Measured on the six golden masters:

    * 24.8% -- `clip_1_HOW_I_MADE_400000_IN_A_MONTH.mp4`, 268px of dead frame on
      each side of a split-screen pane. A real defect.
    * 4.1% -- `clip_2_WHY_DISNEY_PAY_IS_A_TRAP.mp4`, the `blur_stack` layout's
      blurred background, which is designed to be near-black.

    `BLOCKING_BAR_PCT = 8.0` sits between them with roughly 2x margin each side.
    """

    def _report(self, vertical_pct: float) -> geometry.GeometryReport:
        report = geometry.GeometryReport()
        report.worst_vertical_bar_pct = vertical_pct
        return report

    def test_real_dead_frame_blocks(self) -> None:
        self.assertTrue(self._report(24.8).blocking)
        self.assertEqual(self._report(24.8).severity, "error")

    def test_designed_blur_stack_background_does_not_block(self) -> None:
        report = self._report(4.1)
        self.assertFalse(report.blocking)
        self.assertTrue(report.warning, "should still surface as a warning")
        self.assertEqual(report.severity, "warning")

    def test_negligible_inset_is_clean(self) -> None:
        report = self._report(0.4)
        self.assertFalse(report.blocking)
        self.assertFalse(report.warning)
        self.assertEqual(report.severity, "ok")

    def test_thresholds_are_ordered(self) -> None:
        self.assertLess(geometry.WARNING_BAR_PCT, geometry.BLOCKING_BAR_PCT)
        # The corpus's two real cases must fall on opposite sides.
        self.assertGreater(24.8, geometry.BLOCKING_BAR_PCT)
        self.assertLess(4.1, geometry.BLOCKING_BAR_PCT)
        self.assertGreater(4.1, geometry.WARNING_BAR_PCT)


class TestIntervalParserL1(unittest.TestCase):
    """L1: freeze/black interval parsing, including the freeze-to-EOF case.

    HANDOFF section 4 item 8 records a parser that only accepted events with a
    closing `freeze_duration`, which made a fully frozen clip look like a pass.
    That bug is re-introduced easily, so it is pinned here.
    """

    def test_parses_terminated_freezes(self) -> None:
        stderr = (
            "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_start: 41.1\n"
            "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_duration: 0.533\n"
            "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_end: 41.633\n"
            "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_start: 41.633\n"
            "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_duration: 0.6\n"
            "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_end: 42.233\n"
        )
        intervals = probe._parse_intervals(stderr, "freeze", clip_duration=43.8)
        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0].duration, 0.533, places=3)
        self.assertAlmostEqual(intervals[1].duration, 0.6, places=3)
        self.assertTrue(all(i.terminated for i in intervals))

    def test_unterminated_freeze_runs_to_end_of_clip(self) -> None:
        """A freeze that never closes must still be reported, and as unterminated.

        Mutation guard: returning an empty list when `terminated` is False fails
        this test, which is exactly the bug that let a frozen clip pass QA.
        """
        stderr = "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_start: 12.0\n"
        intervals = probe._parse_intervals(stderr, "freeze", clip_duration=20.0)
        self.assertEqual(len(intervals), 1)
        self.assertFalse(intervals[0].terminated)
        self.assertAlmostEqual(intervals[0].end, 20.0, places=3)
        self.assertAlmostEqual(intervals[0].duration, 8.0, places=3)

    def test_black_and_freeze_do_not_cross_contaminate(self) -> None:
        stderr = (
            "[Parsed_blackdetect_0 @ 0x1] lavfi.blackdetect.black_start: 1.0\n"
            "[Parsed_blackdetect_0 @ 0x1] lavfi.blackdetect.black_duration: 0.4\n"
            "[Parsed_freezedetect_0 @ 0x2] lavfi.freezedetect.freeze_start: 5.0\n"
            "[Parsed_freezedetect_0 @ 0x2] lavfi.freezedetect.freeze_duration: 1.0\n"
        )
        blacks = probe._parse_intervals(stderr, "black", clip_duration=10.0)
        freezes = probe._parse_intervals(stderr, "freeze", clip_duration=10.0)
        self.assertEqual(len(blacks), 1)
        self.assertEqual(len(freezes), 1)
        self.assertAlmostEqual(blacks[0].start, 1.0, places=3)
        self.assertAlmostEqual(freezes[0].start, 5.0, places=3)

    def test_empty_stderr_yields_no_intervals(self) -> None:
        self.assertEqual(probe._parse_intervals("", "freeze", 10.0), [])


class TestMotionMetricsL1(unittest.TestCase):
    """L1: cumulative freeze accounting, which is what the QA gate got wrong.

    `editorial_qa` gated on `event.duration > 0.8`. A published clip carried
    three events of 0.533s, 0.6s and 0.5s -- 1.63s of frozen video in total --
    and passed, because each event was individually under the bar. The number
    that matters is the sum.
    """

    def test_cumulative_freeze_exceeds_threshold_despite_small_events(self) -> None:
        events = [
            probe.Interval("freeze", 41.1, 41.633, 0.533, True),
            probe.Interval("freeze", 41.633, 42.233, 0.6, True),
            probe.Interval("freeze", 42.867, 43.367, 0.5, True),
        ]
        report = motion.summarise(events, clip_duration=43.8)
        self.assertAlmostEqual(report.cumulative_freeze_s, 1.633, places=3)
        self.assertAlmostEqual(report.longest_freeze_s, 0.6, places=3)
        self.assertEqual(report.event_count, 3)
        # Every single event is under the old 0.8s gate...
        self.assertTrue(all(e.duration <= 0.8 for e in events))
        # ...yet the clip is 3.7% frozen and must be judged as such.
        self.assertAlmostEqual(report.freeze_fraction, 1.633 / 43.8, places=4)

    def test_unterminated_freeze_counts_fully(self) -> None:
        events = [probe.Interval("freeze", 10.0, 20.0, 10.0, False)]
        report = motion.summarise(events, clip_duration=20.0)
        self.assertAlmostEqual(report.cumulative_freeze_s, 10.0, places=3)
        self.assertTrue(report.has_unterminated_freeze)

    def test_clean_clip_reports_zero(self) -> None:
        report = motion.summarise([], clip_duration=45.0)
        self.assertEqual(report.cumulative_freeze_s, 0.0)
        self.assertEqual(report.event_count, 0)
        self.assertEqual(report.freeze_fraction, 0.0)
        self.assertFalse(report.has_unterminated_freeze)


class TestGoldenMasterGeometry(unittest.TestCase):
    """L3: the pillarbox defect, measured on the clip that shipped with it.

    REPRODUCTION -- this fails against the current artifact, and that is correct.
    Measured 24.8% dead frame on each side (268px of 1080) across 43 of 300
    sampled frames, and `block_run` of 18 consecutive samples. The CI run
    reported ``success`` and editorial QA reported ``passed=True``, because no
    black-bar check existed.

    When Phase 2a fixes the pane geometry this flips to passing with no edit. If
    it needs editing to pass, the fix did not work.
    """

    def setUp(self) -> None:
        if not fixtures.all_present():
            self.skipTest(fixtures.fetch_hint())

    def test_400k_clip_must_not_be_pillarboxed(self) -> None:
        master = fixtures.by_id("four_hundred_k_1")
        self.assertIsNotNone(master, "fixture four_hundred_k_1 missing")
        assert master is not None

        report = geometry.analyse(master.path)
        self.assertTrue(report.ok, report.reason)

        # Sanity: the detector must still be seeing the defect, or this test has
        # stopped testing anything and would pass for the wrong reason.
        self.assertGreater(
            report.bar_samples,
            5,
            msg="expected many barred samples; if this drops, the detector broke",
        )

        # AGENTS.md 3.3: "each speaker pane scales to 1080x960 with zero black
        # bars on the sides".
        self.assertLessEqual(
            report.worst_vertical_bar_pct,
            geometry.WARNING_BAR_PCT,
            msg=(
                f"{master.file}: {report.worst_vertical_bar_pct:.1f}% pillarbox "
                f"in {report.bar_samples}/{report.samples} samples "
                f"(longest run {report.bar_run}), worst box {report.worst_box}. "
                f"AGENTS.md 3.3 requires zero side bars on split-screen panes; "
                f"blocking threshold is {geometry.BLOCKING_BAR_PCT}%."
            ),
        )
        self.assertFalse(
            report.blocking,
            msg=f"{master.file}: {report.worst_vertical_bar_pct:.1f}% dead frame is a blocking defect",
        )


class TestGoldenMasterMotion(unittest.TestCase):
    """L3: the freeze defect, measured on the clip that shipped with it.

    REPRODUCTION -- fails against the published artifact by design. Three events
    of 0.533s, 0.6s and 0.5s, 1.63s cumulative. The old gate was
    ``event.duration > 0.8``, so every individual event passed and the clip
    shipped 3.7% frozen.
    """

    FREEZE_FILTER = "freezedetect=n=0.003:d=0.5"

    def setUp(self) -> None:
        if not fixtures.all_present():
            self.skipTest(fixtures.fetch_hint())

    def test_400k_clip_must_not_freeze(self) -> None:
        master = fixtures.by_id("four_hundred_k_1")
        self.assertIsNotNone(master)
        assert master is not None

        events = probe.detect_intervals(master.path, self.FREEZE_FILTER, "freeze")
        total = probe.duration_seconds(master.path)
        report = motion.summarise(events, clip_duration=total)

        # Sanity: the three known events must still be detectable, or this test
        # has stopped testing anything.
        self.assertGreaterEqual(
            report.event_count,
            3,
            msg="expected the three known freeze events; if this drops, the detector broke",
        )
        self.assertAlmostEqual(
            report.cumulative_freeze_s,
            1.633,
            places=2,
            msg="measured freeze total changed; the corpus or the parser moved",
        )

        self.assertLessEqual(
            report.cumulative_freeze_s,
            0.3,
            msg=(
                f"{master.file}: {report.cumulative_freeze_s:.2f}s of frozen video "
                f"({report.freeze_fraction:.1%} of the clip) across "
                f"{report.event_count} events. AGENTS.md 3.1 requires rock-solid "
                f"framing with no bobbing or hold."
            ),
        )
        self.assertFalse(
            report.has_unterminated_freeze,
            msg=f"{master.file}: a freeze runs to end-of-file and was not detected",
        )

    def test_400k_clip_ends_with_the_freeze(self) -> None:
        """Pin *where* the freeze is, so a fix cannot move it rather than remove it.

        The freeze sits in the final 2.3 seconds, which is also where the layout
        collapses. A change that shifts the freeze earlier while leaving its total
        duration intact would still be a defect, just a differently shaped one.
        """
        master = fixtures.by_id("four_hundred_k_1")
        assert master is not None
        total = probe.duration_seconds(master.path)
        events = probe.detect_intervals(
            master.path, "freezedetect=n=0.003:d=0.5", "freeze", total
        )
        self.assertTrue(events, "no freeze events to locate")
        self.assertGreater(
            min(event.start for event in events),
            total * 0.9,
            msg="the freeze is no longer in the tail of the clip; re-inspect the fixture",
        )


class TestGoldenMasterCleanBaseline(unittest.TestCase):
    """L3: a clip that is visually clean must verify clean **for geometry**.

    This is the other half of the gate. A verifier that flags everything is as
    useless as one that flags nothing, so at least one golden master has to come
    back with no blocking geometry finding -- otherwise the suite is measuring its
    own thresholds rather than the renderer.

    Note the scope. `disney_2` is *visually* clean and that is exactly why it is
    the right control: its only defect is +117ms of A/V drift, which no amount of
    scrubbing reveals. Its verdict in the manifest is FAIL for that reason, so
    this test deliberately does **not** assert `expects_pass`. Asserting that would
    couple a geometry check to a non-visual finding and make the control
    useless the moment an unrelated defect appears.
    """

    def setUp(self) -> None:
        if not fixtures.all_present():
            self.skipTest(fixtures.fetch_hint())

    def test_disney_clip_has_no_blocking_geometry_finding(self) -> None:
        master = fixtures.by_id("disney_2")
        self.assertIsNotNone(master)
        assert master is not None
        # The manifest records the *visual* assessment as clean for geometry
        # purposes, even though the overall verdict is FAIL on A/V drift.
        self.assertFalse(
            [d for d in master.human_defects if d.startswith("B1")],
            "disney_2 is meant to be the geometry-clean control",
        )

        report = geometry.analyse(master.path)
        self.assertTrue(report.ok, report.reason)
        self.assertFalse(
            report.blocking,
            msg=(
                f"{master.file}: {report.worst_vertical_bar_pct:.1f}% inset is a "
                f"blocking finding. Note this clip's only inset is the blur_stack "
                f"layout's designed near-black blurred background, which measures "
                f"~4%; if the renderer changed, re-inspect the frame by hand."
            ),
        )
        self.assertAlmostEqual(
            report.declared_aspect, 1080 / 1920, places=3, msg="wrong output resolution"
        )
        # Recorded rather than asserted away: a known, non-blocking observation
        # that must not be quietly forgotten.
        if report.warning:
            self.assertLessEqual(report.worst_vertical_bar_pct, 6.0)

    def test_disney_clip_is_freeze_clean(self) -> None:
        master = fixtures.by_id("disney_2")
        self.assertIsNotNone(master)
        assert master is not None

        events = probe.detect_intervals(
            master.path, "freezedetect=n=0.003:d=0.5", "freeze"
        )
        report = motion.summarise(events, clip_duration=probe.duration_seconds(master.path))
        self.assertEqual(
            report.cumulative_freeze_s,
            0.0,
            msg=f"{master.file}: unexpected freeze {report.as_dict()}",
        )

    def test_every_fixture_is_accounted_for(self) -> None:
        """The manifest must cover the corpus, and the corpus must be present.

        A golden-master suite that silently shrinks to zero fixtures still passes,
        which is the same class of failure as a QA gate that cannot fire.
        """
        masters = fixtures.available()
        self.assertGreaterEqual(
            len(masters), 6, f"expected 6 fixtures, found {len(masters)}"
        )
        for master in masters:
            self.assertTrue(master.path.exists())
            self.assertGreater(master.path.stat().st_size, 0)
            self.assertIn(master.human_verdict.upper(), {"PASS", "FAIL"})
            self.assertTrue(
                master.sha256, f"{master.file} has no recorded sha256"
            )


class TestCaptionOnFace(unittest.TestCase):
    """L3: captions must not sit on the subject's brow and eyes.

    REPRODUCTION -- fails against the published artifacts by design.

    Measured on the corpus with the upper-face band (brow through eyes):

    ==================  =========
    clip                 on-face
    ==================  =========
    broke_1              100.0%
    grinding_2           100.0%
    hardwork_3           100.0%
    four_hundred_k_1      37.3%
    disney_2               0.0%
    sixty_k_1              0.0%
    ==================  =========

    This is the defect `editorial_qa` structurally cannot catch. It compares
    `shot.caption_rect` against `shot.protected_regions`, and both are written by
    the same code minutes apart, so it confirms the plan rather than the video.
    The caption that lands on the face is produced by the collision-avoidance
    *relocation* that those same regions triggered: `upper_center` resolves to
    roughly 25% of frame height, which clears the 240px platform UI band and lands
    squarely on the eyes.
    """

    def setUp(self) -> None:
        if not fixtures.all_present():
            self.skipTest(fixtures.fetch_hint())

    def test_broke_clip_captions_must_clear_the_face(self) -> None:
        from src.verification import captions, faces, verdict

        master = fixtures.by_id("broke_1")
        self.assertIsNotNone(master)
        assert master is not None

        face_report = faces.analyse(master.path, samples=12)
        self.assertTrue(face_report.any_faces, face_report.reason)
        avoid = faces.avoid_rects(face_report, upper_only=True)
        on_face, at, measured = captions.measure_on_face(
            master.path, avoid, max_samples=16
        )
        self.assertGreater(
            measured,
            5,
            msg="expected caption pixels to be found in several frames",
        )
        self.assertLessEqual(
            on_face,
            verdict.CAPTION_ON_FACE_BLOCK_PCT,
            msg=(
                f"{master.file}: {on_face:.0f}% of burned-in caption pixels sit on "
                f"the subject's brow/eyes (worst at {at:.0f}% through the clip). "
                f"AGENTS.md 2.A requires readable captions."
            ),
        )

    def test_disney_clip_captions_are_clear_of_the_face(self) -> None:
        """The negative control: a low number must be achievable, or the gate is noise."""
        from src.verification import captions, faces

        master = fixtures.by_id("disney_2")
        self.assertIsNotNone(master)
        assert master is not None

        face_report = faces.analyse(master.path, samples=12)
        if not face_report.any_faces:
            self.skipTest(face_report.reason)
        on_face, _at, _measured = captions.measure_on_face(
            master.path, faces.avoid_rects(face_report, upper_only=True), max_samples=16
        )
        self.assertLessEqual(
            on_face,
            15.0,
            msg=f"{master.file}: {on_face:.0f}% of caption pixels on the upper face",
        )


class TestUpperFaceBandL1(unittest.TestCase):
    """L1: the avoidance band must be the upper face, not the whole face.

    Mutation guard: switching `avoid_rects` to `padded()` fails
    `test_upper_band_is_much_smaller_than_the_whole_face` and would make five of
    six golden masters report 100% regardless of where the caption actually is.
    """

    def test_upper_band_is_much_smaller_than_the_whole_face(self) -> None:
        from src.verification import faces

        box = faces.FaceBox(x=182.8, y=9.8, width=407.7, height=303.5, score=0.9)
        upper = box.upper_face()
        whole = box.padded()
        self.assertLess(upper[3], whole[3] * 0.6)
        self.assertAlmostEqual(upper[1], box.y, places=3, msg="band must start at the face top")
        self.assertAlmostEqual(upper[3], box.height * 0.55, places=3)

    def test_upper_band_keeps_the_eye_line(self) -> None:
        from src.verification import faces

        # Eyes sit at roughly 40% of face height from the top.
        box = faces.FaceBox(x=0.0, y=100.0, width=200.0, height=300.0, score=0.9)
        _x, y, _w, h = box.upper_face()
        eye_line = 100.0 + 0.40 * 300.0
        self.assertLessEqual(y, eye_line)
        self.assertGreaterEqual(y + h, eye_line)

    def test_avoid_rects_defaults_to_upper_only(self) -> None:
        from src.verification import faces

        report = faces.FaceReport(detector="test", samples=1, frames_with_faces=1)
        report.boxes = [faces.FaceBox(x=100.0, y=50.0, width=200.0, height=200.0, score=0.9)]
        upper = faces.avoid_rects(report)
        whole = faces.avoid_rects(report, upper_only=False)
        self.assertEqual(len(upper), 1)
        self.assertLess(upper[0].height, whole[0].height)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
