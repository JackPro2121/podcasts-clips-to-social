"""Clip-count backfill: the render loops get more candidates than num_clips.

AGENTS.md 3.5 requires exactly num_clips delivered. Individual moments are lost
to audio-language rejections, QA vetoes and pixel-verdict vetoes, and every loss
is a bare ``continue`` - a run delivered 1/3 with a success exit. The pool
guarantees the loops can replace a lost moment with the next-best non-overlapping
candidate instead of ending in a shortfall.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import main


def _moment(start, end):
    return SimpleNamespace(start_time=start, end_time=end, duration=end - start)


class TestDetectionPool(unittest.TestCase):
    def _run_pool(self, primary, extras, num_clips=3, extras_error=None):
        fallback_mock = mock.Mock(
            return_value=extras,
            side_effect=extras_error,
        )
        with mock.patch.object(main, "detect_viral_moments", return_value=primary), mock.patch.object(
            main, "fallback_rule_based_detector", fallback_mock
        ):
            pool = main._detection_pool([], num_clips=num_clips, niche="finance")
        return pool, fallback_mock

    def test_pool_tops_up_with_non_overlapping_extras(self):
        primary = [_moment(0, 40), _moment(100, 140), _moment(200, 240)]
        extras = [
            _moment(0, 40),      # overlaps primary -> skipped
            _moment(300, 340),   # appended
            _moment(400, 440),   # appended
            _moment(600, 640),   # beyond the target -> not needed
        ]
        pool, fallback_mock = self._run_pool(primary, extras)
        self.assertEqual([m.start_time for m in pool], [0, 100, 200, 300, 400])
        fallback_mock.assert_called_once()
        self.assertEqual(fallback_mock.call_args.kwargs["num_clips"], 5)

    def test_pool_is_capped_at_num_clips_plus_two(self):
        primary = [_moment(0, 40)]
        extras = [_moment(start, start + 40) for start in (100, 200, 300, 400, 500)]
        pool, _ = self._run_pool(primary, extras)
        self.assertEqual(len(pool), 5)
        self.assertEqual([m.start_time for m in pool], [0, 100, 200, 300, 400])

    def test_primary_only_pool_when_every_extra_overlaps(self):
        primary = [_moment(0, 40), _moment(100, 140), _moment(200, 240)]
        extras = [_moment(10, 50), _moment(110, 150), _moment(210, 250)]
        pool, _ = self._run_pool(primary, extras)
        self.assertEqual([m.start_time for m in pool], [0, 100, 200])

    def test_extras_form_the_pool_when_primary_detection_is_empty(self):
        extras = [_moment(start, start + 40) for start in (0, 100, 200, 300, 400)]
        pool, _ = self._run_pool([], extras)
        self.assertEqual(len(pool), 5)

    def test_primary_pool_skips_backfill_entirely_when_already_full(self):
        primary = [_moment(start, start + 40) for start in (0, 100, 200, 300, 400)]
        pool, fallback_mock = self._run_pool(primary, [_moment(600, 640)])
        self.assertEqual(len(pool), 5)
        fallback_mock.assert_not_called()

    def test_backfill_failure_is_tolerated(self):
        primary = [_moment(0, 40), _moment(100, 140)]
        pool, _ = self._run_pool(primary, [], extras_error=RuntimeError("detector exploded"))
        self.assertEqual([m.start_time for m in pool], [0, 100])

    def test_touching_windows_are_not_overlaps(self):
        primary = [_moment(0, 40)]
        extras = [_moment(40, 80)]
        pool, _ = self._run_pool(primary, extras)
        self.assertEqual(len(pool), 2)


if __name__ == "__main__":
    unittest.main()
