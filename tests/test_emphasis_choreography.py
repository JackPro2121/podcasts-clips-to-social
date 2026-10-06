"""P5 choreography: emphasis words drive impact SFX and punch shots together."""

from __future__ import annotations

import unittest

from src.emphasis_choreography import plan_choreography


WORDS = ["I", "lost", "$40,000", "because", "nobody", "told", "me", "the", "truth."]
TIMES = [
    (0.4, 0.6), (1.2, 1.5), (2.0, 2.6), (2.8, 3.1), (3.4, 3.9),
    (4.1, 4.4), (4.6, 4.8), (5.0, 5.2), (5.4, 5.9),
]
SHOTS = [(0.0, 3.0), (3.0, 6.0)]


class TestPlanChoreography(unittest.TestCase):
    def test_numbers_beat_power_words_when_capped(self):
        # $40,000 (2.0s) and lost (1.2s), nobody (3.4s), truth (5.4s); with a
        # single cue the number must win even though "lost" comes first.
        plan = plan_choreography(WORDS, TIMES, SHOTS, max_sfx=1)
        self.assertEqual(len(plan.sfx_cues), 1)
        self.assertAlmostEqual(plan.sfx_cues[0][0], 2.0, places=2)

    def test_sfx_spacing_and_opening_guard(self):
        plan = plan_choreography(WORDS, TIMES, SHOTS, min_gap_s=4.0, min_start_s=1.0)
        times = [t for t, _ in plan.sfx_cues]
        self.assertNotIn(0.4, times)
        for earlier, later in zip(times, times[1:]):
            self.assertGreaterEqual(later - earlier, 4.0)

    def test_push_in_follows_the_cue_shot_and_is_unique_and_capped(self):
        plan = plan_choreography(WORDS, TIMES, SHOTS)
        self.assertTrue(plan.sfx_cues)
        for time, _kind in plan.sfx_cues:
            containing = [
                index for index, (start, end) in enumerate(SHOTS) if start <= time < end
            ]
            if containing:
                self.assertIn(containing[0], plan.push_in_shots)
        self.assertEqual(len(plan.push_in_shots), len(set(plan.push_in_shots)))
        self.assertLessEqual(len(plan.push_in_shots), 2)

    def test_no_emphasis_means_silent_and_still(self):
        plan = plan_choreography(
            ["we", "were", "just", "talking"], [(0.5, 0.8), (1.0, 1.3), (1.5, 1.8), (2.0, 2.3)], SHOTS
        )
        self.assertEqual(plan.sfx_cues, [])
        self.assertEqual(plan.push_in_shots, [])

    def test_cues_come_back_in_time_order(self):
        plan = plan_choreography(WORDS, TIMES, SHOTS, min_gap_s=1.0)
        times = [t for t, _ in plan.sfx_cues]
        self.assertEqual(times, sorted(times))


if __name__ == "__main__":
    unittest.main()
