"""ASD-Lite decision logic: pick the face whose mouth moves with the voice.

The defect this feeds (run 37347481817, clip_2): the crop showed a face whose
mouth motion correlated with the voice at +0.09 with no lag alignment -- the
visible face was not the speaker. Composition must consult who is talking; the
decision core is pure so the talker is known by construction in these tests.

Synthetic series are non-negative on purpose: real mouth motion and audio RMS
envelopes are magnitudes.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.active_speaker import (
    pick_speaker,
    scores_from_series,
    windowed_speakers,
)


def _positive(size: int, seed: int, scale: float = 1.0) -> list[float]:
    return list(np.abs(np.random.default_rng(seed).standard_normal(size)) * scale)


def _follower(envelope: list[float], seed: int, gain: float = 3.0) -> list[float]:
    noise = np.random.default_rng(seed).standard_normal(len(envelope)) * 0.1
    return list(np.asarray(envelope) * gain + noise)


class TestScoresFromSeries(unittest.TestCase):
    def test_follower_scores_above_noise(self):
        envelope = _positive(40, 1)
        follower = _follower(envelope, 2, gain=1.5)
        stranger = _positive(40, 3, scale=0.5)
        scores = scores_from_series([follower, stranger], envelope)
        self.assertGreater(scores[0], 0.8)
        self.assertLess(abs(scores[1]), 0.5)

    def test_short_series_returns_zero(self):
        self.assertEqual(scores_from_series([[1.0, 2.0]], [1.0, 2.0]), [0.0])


class TestPickSpeaker(unittest.TestCase):
    def test_picks_the_follower(self):
        envelope = _positive(40, 4)
        follower = _follower(envelope, 5)
        stranger = _positive(40, 6, scale=0.5)
        self.assertEqual(pick_speaker([stranger, follower], envelope), 1)

    def test_returns_minus_one_when_nobody_follows(self):
        envelope = _positive(40, 7)
        self.assertEqual(
            pick_speaker([_positive(40, 8, scale=0.5), _positive(40, 9, scale=0.5)], envelope),
            -1,
        )

    def test_a_still_face_cannot_win(self):
        # A frozen "face" (photo/poster) with no mouth movement is not a
        # speaker even when its constant series would otherwise score.
        envelope = _positive(40, 10)
        still = [0.01] * 40
        follower = _follower(envelope, 11)
        self.assertEqual(pick_speaker([still, follower], envelope), 1)


class TestWindowedSpeakers(unittest.TestCase):
    def test_speaker_switch_between_windows(self):
        fps = 10.0
        window = 1.0
        first = _positive(50, 12)
        second = _positive(50, 13)
        # Face 0 talks in the first 2.5s, face 1 in the last 2.5s.
        envelope = first[:25] + second[25:]
        face0 = _follower(first, 14)[:25] + _positive(25, 15, scale=0.5)
        face1 = _positive(25, 16, scale=0.5) + _follower(second, 17)[25:]
        windows = windowed_speakers([face0, face1], envelope, fps, window)
        self.assertGreaterEqual(len(windows), 4)
        self.assertEqual(windows[0][1], 0)
        self.assertEqual(windows[-1][1], 1)

    def test_no_confidence_windows_report_minus_one(self):
        envelope = _positive(30, 18)
        windows = windowed_speakers([_positive(30, 19, scale=0.5)], envelope, 10.0, 1.0)
        self.assertTrue(windows)
        self.assertTrue(all(index == -1 for _start, index, _score in windows))


if __name__ == "__main__":
    unittest.main()
