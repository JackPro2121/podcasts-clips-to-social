"""P2 hook engine: the opening 3 seconds are deliberate, not incidental.

Research: 50-60% of viewers drop off in the first three seconds; the hook is
the most important edit. The start must land on a spoken word (no silence, no
mid-breath cut) within a researched 0.5-1.6s lead-in, and the first caption
line must read as one breath.
"""

from __future__ import annotations

import unittest

from src.hook_engine import (
    HOOK_MAX_LEAD_IN_S,
    build_hook_plan,
    first_line,
    hook_start,
)


def _w(word: str, start: float, end: float) -> tuple[str, float, float]:
    return (word, start, end)


class TestHookStart(unittest.TestCase):
    def test_earliest_word_inside_the_window_wins(self):
        preceding = [_w("so", 48.6, 48.9), _w("I", 49.1, 49.3), _w("told", 49.4, 49.8)]
        # Moment starts at 50.0; all three are inside the 1.6s window.
        self.assertAlmostEqual(hook_start(preceding, 50.0), 48.6, places=3)

    def test_words_outside_the_window_are_ignored(self):
        preceding = [_w("anyway", 47.0, 47.4)]  # 3s back: too far
        self.assertAlmostEqual(hook_start(preceding, 50.0), 50.0, places=3)

    def test_no_preceding_words_keeps_the_moment_start(self):
        self.assertAlmostEqual(hook_start([], 50.0), 50.0, places=3)

    def test_a_word_that_starts_after_the_moment_is_ignored(self):
        preceding = [_w("next", 50.2, 50.5)]
        self.assertAlmostEqual(hook_start(preceding, 50.0), 50.0, places=3)

    def test_window_boundary_is_respected(self):
        preceding = [_w("edge", 50.0 - HOOK_MAX_LEAD_IN_S + 0.01, 50.0 - 1.0)]
        self.assertAlmostEqual(
            hook_start(preceding, 50.0), 50.0 - HOOK_MAX_LEAD_IN_S + 0.01, places=3
        )


class TestFirstLine(unittest.TestCase):
    def test_prefers_to_end_on_punctuation_inside_the_cap(self):
        words = [
            _w("So", 1.0, 1.2),
            _w("I", 1.3, 1.4),
            _w("quit,", 1.5, 1.9),
            _w("and", 2.0, 2.2),
            _w("never", 2.3, 2.7),
            _w("looked", 2.8, 3.2),
            _w("back.", 3.3, 3.7),
        ]
        line = first_line(words, 1.0, max_chars=20)
        self.assertEqual(line, "So I quit,")

    def test_falls_back_to_the_last_whole_word_that_fits(self):
        words = [_w("unbelievable", 1.0, 1.5), _w("story", 1.6, 2.0)]
        line = first_line(words, 1.0, max_chars=14)
        self.assertEqual(line, "unbelievable")

    def test_words_before_the_start_are_skipped(self):
        words = [_w("context", 0.2, 0.6), _w("The", 1.0, 1.2), _w("hook", 1.3, 1.7)]
        line = first_line(words, 1.0, max_chars=40)
        self.assertEqual(line, "The hook")

    def test_empty_when_nothing_at_or_after_start(self):
        self.assertEqual(first_line([_w("old", 0.1, 0.4)], 1.0), "")


class TestBuildHookPlan(unittest.TestCase):
    def test_lead_in_and_first_line_together(self):
        preceding = [_w("so", 48.6, 48.9), _w("I", 49.1, 49.3)]
        moment_words = [_w("quit", 50.0, 50.4), _w("cold.", 50.5, 50.9)]
        plan = build_hook_plan(preceding, moment_words, 50.0)
        self.assertAlmostEqual(plan.start_s, 48.6, places=3)
        self.assertAlmostEqual(plan.lead_in_s, 1.4, places=3)
        self.assertEqual(plan.first_line, "so I quit cold.")

    def test_no_lead_in_when_there_is_a_pause(self):
        preceding = [_w("long", 45.0, 45.3)]
        moment_words = [_w("Listen.", 50.0, 50.5)]
        plan = build_hook_plan(preceding, moment_words, 50.0)
        self.assertAlmostEqual(plan.start_s, 50.0, places=3)
        self.assertAlmostEqual(plan.lead_in_s, 0.0, places=3)
        self.assertEqual(plan.first_line, "Listen.")


if __name__ == "__main__":
    unittest.main()
