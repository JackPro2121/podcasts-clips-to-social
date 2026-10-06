"""P1 Jump-Cut Engine: dead-air cuts from word timestamps, purely.

Research basis: dead-air removal is the industry-standard editing upgrade
(TimeBolt/SavvyCut/Jump Cutter) and pacing is retention (50-60% drop-off in
the first 3s). The engine must never ship a clip below the 30s contract floor,
so that guard has its own test.
"""

from __future__ import annotations

import unittest

from src.jump_cutter import build_cut_plan


class TestCutPlan(unittest.TestCase):
    def test_long_gap_is_cut_with_padding(self):
        words = [(0.0, 1.0), (3.0, 4.0)]
        plan = build_cut_plan(words, 4.0, min_duration_s=1.0)
        self.assertTrue(plan.applied)
        self.assertEqual(len(plan.keep_segments), 2)
        self.assertAlmostEqual(plan.keep_segments[0][0], 0.0)
        self.assertAlmostEqual(plan.keep_segments[0][1], 1.08, places=2)
        self.assertAlmostEqual(plan.keep_segments[1][0], 2.92, places=2)
        self.assertAlmostEqual(plan.keep_segments[1][1], 4.0, places=2)
        self.assertAlmostEqual(plan.removed_s, 1.84, places=2)
        self.assertAlmostEqual(plan.duration_s, 2.16, places=2)

    def test_small_gaps_are_speech_rhythm_not_cuts(self):
        words = [(0.0, 1.0), (1.3, 2.0), (2.2, 3.0)]
        plan = build_cut_plan(words, 3.0, min_duration_s=1.0)
        self.assertFalse(plan.applied)
        self.assertEqual(plan.keep_segments, [(0.0, 3.0)])
        self.assertEqual(plan.removed_s, 0.0)

    def test_floor_guard_refuses_a_short_result(self):
        words = [(0.0, 1.0), (3.0, 4.0), (10.0, 11.0)]
        plan = build_cut_plan(words, 31.0)
        self.assertFalse(plan.applied, "cutting to 3.4s must be refused (floor 30s)")
        self.assertEqual(plan.keep_segments, [(0.0, 31.0)])

    def test_no_words_is_identity(self):
        plan = build_cut_plan([], 45.0)
        self.assertFalse(plan.applied)
        self.assertEqual(plan.keep_segments, [(0.0, 45.0)])

    def test_head_and_tail_dead_air_are_removed(self):
        words = [(2.0, 40.0)]
        plan = build_cut_plan(words, 45.0, min_duration_s=1.0)
        self.assertTrue(plan.applied)
        self.assertAlmostEqual(plan.keep_segments[0][0], 1.92, places=2)
        self.assertAlmostEqual(plan.keep_segments[0][1], 40.08, places=2)
        self.assertAlmostEqual(plan.removed_s, 6.84, places=2)

    def test_head_and_tail_shorter_than_the_threshold_are_kept(self):
        # 0.32s of silence at each edge is rhythm, not dead air: identity.
        words = [(0.4, 44.6)]
        plan = build_cut_plan(words, 45.0, min_duration_s=1.0)
        self.assertFalse(plan.applied, "0.4s edges are rhythm, not dead air")
        self.assertEqual(plan.keep_segments, [(0.0, 45.0)])


class TestRemap(unittest.TestCase):
    def test_remap_shifts_after_the_cut(self):
        plan = build_cut_plan([(0.0, 1.0), (3.0, 4.0)], 4.0, min_duration_s=1.0)
        self.assertAlmostEqual(plan.remap(0.5), 0.5, places=3)
        self.assertAlmostEqual(plan.remap(3.5), 1.66, places=2)
        self.assertAlmostEqual(plan.remap(4.0), 2.16, places=2)

    def test_remap_snaps_removed_gaps_forward(self):
        plan = build_cut_plan([(0.0, 1.0), (3.0, 4.0)], 4.0, min_duration_s=1.0)
        # 2.0s sits inside the removed gap; it snaps to the next segment start.
        self.assertAlmostEqual(plan.remap(2.0), 1.08, places=2)

    def test_remap_of_identity_is_a_no_op(self):
        plan = build_cut_plan([(0.0, 1.0), (1.3, 2.9)], 3.0, min_duration_s=1.0)
        self.assertFalse(plan.applied)
        for moment in (0.0, 1.15, 2.9):
            self.assertAlmostEqual(plan.remap(moment), moment, places=6)


if __name__ == "__main__":
    unittest.main()
