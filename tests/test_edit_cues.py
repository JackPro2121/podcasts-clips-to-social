"""P4 edit cues: b-roll and cut-transition sound, planned from the edit.

The renderer accepted broll_cues/sfx_cues but nothing generated them; these
tests pin the planner's rules (hook guard, spacing, caps, cut arithmetic).
"""

from __future__ import annotations

import unittest

from src.edit_cues import plan_broll_cues, plan_cut_whooshes
from src.jump_cutter import build_cut_plan


class TestPlanBrollCues(unittest.TestCase):
    def test_hook_is_never_covered(self):
        cues = plan_broll_cues([(0.0, 12.0)], 40.0)
        self.assertTrue(cues)
        self.assertGreaterEqual(cues[0][0], 3.0)

    def test_cues_are_spaced_and_capped(self):
        spans = [(0.0, 6.0), (7.0, 12.0), (13.0, 18.0), (19.0, 25.0)]
        cues = plan_broll_cues(spans, 40.0)
        self.assertLessEqual(len(cues), 2)
        if len(cues) == 2:
            self.assertGreaterEqual(cues[1][0] - cues[0][0], 8.0)

    def test_window_length_is_capped(self):
        cues = plan_broll_cues([(4.0, 20.0)], 40.0)
        self.assertEqual(len(cues), 1)
        self.assertLessEqual(cues[0][1] - cues[0][0], 4.0)

    def test_short_spans_are_rejected(self):
        self.assertEqual(plan_broll_cues([(5.0, 6.5)], 40.0), [])

    def test_results_are_time_ordered(self):
        cues = plan_broll_cues([(2.0, 9.0), (15.0, 24.0)], 40.0)
        self.assertEqual(cues, sorted(cues))


class TestPlanCutWhooshes(unittest.TestCase):
    def test_identity_plan_has_no_cues(self):
        plan = build_cut_plan([(0.0, 3.0)], 40.0)
        self.assertEqual(plan_cut_whooshes(plan), [])

    def test_whoosh_sits_at_the_sum_of_kept_durations(self):
        plan = build_cut_plan([(0.0, 1.0), (3.0, 4.0), (10.0, 11.0)], 31.0, min_duration_s=1.0)
        self.assertTrue(plan.applied)
        times = plan_cut_whooshes(plan)
        self.assertEqual(len(times), len(plan.keep_segments) - 1)
        self.assertAlmostEqual(times[0], 1.08, places=2)
        expected_second = sum(end - start for start, end in plan.keep_segments[:2])
        self.assertAlmostEqual(times[1], expected_second, places=2)

    def test_cues_are_capped(self):
        words = [(index * 3.0, index * 3.0 + 1.0) for index in range(15)]
        plan = build_cut_plan(words, 45.0, min_duration_s=1.0)
        self.assertTrue(plan.applied)
        self.assertLessEqual(len(plan_cut_whooshes(plan)), 6)


if __name__ == "__main__":
    unittest.main()
