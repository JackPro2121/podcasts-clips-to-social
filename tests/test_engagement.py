"""Tests for the editorial engagement (pacing energy) analysis."""
import unittest

from src.engagement import (
    DEAD_AIR_SECONDS,
    EngagementSample,
    analyze_window,
    best_energy_peak,
    engagement_curve,
    rank_candidates,
    summarise,
)
from src.transcriber import TranscriptSegment, WordTimestamp


def _segment(start, end, text, words=None):
    return TranscriptSegment(start=start, end=end, text=text, words=words or [])


def _words(start, end, tokens, step=None):
    step = step if step is not None else (end - start) / max(1, len(tokens))
    return [
        WordTimestamp(word=token, start=start + i * step, end=start + (i + 1) * step)
        for i, token in enumerate(tokens)
    ]


class TestAnalyzeWindow(unittest.TestCase):
    def test_empty_window_is_flagged(self):
        sample = analyze_window([], 0.0, 0.0)
        self.assertIn("empty_window", sample.warnings)
        self.assertEqual(sample.score, 0.0)

    def test_window_without_speech_scores_zero(self):
        sample = analyze_window([_segment(100.0, 110.0, "elsewhere")], 0.0, 30.0)
        self.assertIn("no_speech", sample.warnings)
        self.assertEqual(sample.score, 0.0)
        self.assertEqual(sample.word_count, 0)

    def test_counts_only_words_inside_the_window(self):
        segments = [_segment(0.0, 60.0, "", _words(0.0, 60.0, ["a"] * 60))]
        sample = analyze_window(segments, 10.0, 20.0)
        self.assertGreater(sample.word_count, 0)
        self.assertLess(sample.word_count, 60)

    def test_fast_confident_delivery_beats_slow_hesitant_delivery(self):
        fast = analyze_window(
            [_segment(0.0, 30.0, "we hit the number and it worked", _words(0.0, 30.0, ["we"] * 75))],
            0.0, 30.0,
        )
        slow = analyze_window(
            [_segment(0.0, 30.0, "um uh er", _words(0.0, 30.0, ["um", "uh", "er"]))],
            0.0, 30.0,
        )
        self.assertGreater(fast.score, slow.score)
        self.assertIn("filler_heavy", slow.warnings)
        self.assertIn("slow_delivery", slow.warnings)

    def test_dead_air_is_penalised_and_flagged(self):
        tokens = ["yes"] * 10
        words = []
        for i, token in enumerate(tokens):
            words.append(WordTimestamp(word=token, start=i * 0.3, end=i * 0.3 + 0.25))
        # long trailing silence inside the window
        words = words + [WordTimestamp(word="ok", start=20.0, end=20.2)]
        sample = analyze_window([_segment(0.0, 25.0, "yes yes ok", words)], 0.0, 25.0)
        self.assertIn("dead_air", sample.warnings)
        self.assertGreaterEqual(sample.longest_pause, DEAD_AIR_SECONDS)
        self.assertGreater(sample.dead_air, 0.0)

    def test_deliberate_pause_is_not_dead_air(self):
        words = [WordTimestamp(word="a", start=0.0, end=0.3),
                 WordTimestamp(word="b", start=1.0, end=1.3),
                 WordTimestamp(word="c", start=2.0, end=2.3),
                 WordTimestamp(word="d", start=3.0, end=3.3)]
        sample = analyze_window([_segment(0.0, 4.0, "a b c d", words)], 0.0, 4.0)
        self.assertNotIn("dead_air", sample.warnings)
        self.assertEqual(sample.dead_air, 0.0)

    def test_emphasis_and_questions_are_measured(self):
        segments = [
            _segment(0.0, 10.0, "This is the answer!"),
            _segment(10.0, 20.0, "But why did it work?"),
            _segment(20.0, 30.0, "plain narration here"),
        ]
        sample = analyze_window(segments, 0.0, 30.0)
        self.assertGreaterEqual(sample.emphasis_rate, 0.0)
        self.assertGreaterEqual(sample.question_rate, 0.0)

    def test_falls_back_to_text_when_no_word_timestamps(self):
        segments = [_segment(0.0, 30.0, "one two three four five six seven eight nine ten")]
        sample = analyze_window(segments, 0.0, 30.0)
        self.assertEqual(sample.word_count, 10)
        self.assertGreater(sample.words_per_second, 0.0)

    def test_score_is_bounded(self):
        segments = [_segment(0.0, 30.0, "Wow! Amazing! Incredible! Yes!", _words(0.0, 30.0, ["wow"] * 90))]
        sample = analyze_window(segments, 0.0, 30.0)
        self.assertGreaterEqual(sample.score, 0.0)
        self.assertLessEqual(sample.score, 1.0)

    def test_sample_serialises(self):
        sample = analyze_window([_segment(0.0, 5.0, "hello there", _words(0.0, 5.0, ["hello", "there"]))], 0.0, 5.0)
        payload = sample.to_dict()
        self.assertEqual(payload["start"], 0.0)
        self.assertIn("words_per_second", payload)


class TestEngagementCurve(unittest.TestCase):
    def test_curve_covers_the_range(self):
        segments = [_segment(0.0, 10.0, "a b c", _words(0.0, 10.0, ["a"] * 30))]
        curve = engagement_curve(segments, 0.0, 5.0, bucket_seconds=1.0)
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve[0].start, 0.0)
        self.assertAlmostEqual(curve[-1].end, 5.0, places=2)

    def test_curve_handles_degenerate_input(self):
        self.assertEqual(engagement_curve([], 5.0, 5.0), [])
        self.assertEqual(engagement_curve([], 0.0, 10.0, bucket_seconds=0.0), [])

    def test_summarise_reports_aggregates(self):
        segments = [_segment(0.0, 6.0, "a b c", _words(0.0, 6.0, ["a"] * 18))]
        summary = summarise(engagement_curve(segments, 0.0, 6.0, bucket_seconds=1.0))
        self.assertEqual(summary["buckets"], 6.0)
        self.assertLessEqual(summary["max_score"], 1.0)
        self.assertEqual(summarise([]), {})


class TestRankCandidates(unittest.TestCase):
    def test_out_of_range_windows_sort_last(self):
        segments = [
            _segment(0.0, 40.0, "tight energetic delivery", _words(0.0, 40.0, ["go"] * 100)),
            _segment(100.0, 125.0, "too short", _words(100.0, 125.0, ["go"] * 20)),
            _segment(200.0, 280.0, "too long", _words(200.0, 280.0, ["go"] * 300)),
        ]
        ranked = rank_candidates(
            [(0.0, 40.0, "good", 0.9), (100.0, 125.0, "short", 0.95), (200.0, 280.0, "long", 0.99)],
            segments,
        )
        self.assertEqual(ranked[0].label, "good")
        # A higher base score must not rescue a window outside the platform
        # duration window: the duration gate is the primary sort key. Among the
        # rejected windows the better combined score still wins.
        self.assertEqual({c.label for c in ranked[1:]}, {"short", "long"})
        self.assertGreaterEqual(ranked[1].combined_score, ranked[2].combined_score)
        out_of_range = {c.label: c.engagement.warnings for c in ranked[1:]}
        self.assertIn("too_short", out_of_range["short"])
        self.assertIn("too_long", out_of_range["long"])

    def test_too_long_is_flagged(self):
        segments = [_segment(0.0, 200.0, "long", _words(0.0, 200.0, ["a"] * 300))]
        ranked = rank_candidates([(0.0, 200.0, "long", 0.5)], segments)
        self.assertIn("too_long", ranked[0].engagement.warnings)

    def test_invalid_windows_are_dropped(self):
        self.assertEqual(rank_candidates([(10.0, 10.0, "empty", 1.0)], []), [])

    def test_combined_score_blends_base_and_energy(self):
        candidate = rank_candidates([], [])
        self.assertEqual(candidate, [])
        from src.engagement import WindowCandidate
        high_base_low_energy = WindowCandidate(
            start=0.0, end=30.0, base_score=1.0,
            engagement=EngagementSample(0.0, 30.0, word_count=10, score=0.0),
        )
        low_base_high_energy = WindowCandidate(
            start=0.0, end=30.0, base_score=0.0,
            engagement=EngagementSample(0.0, 30.0, word_count=10, score=1.0),
        )
        self.assertGreater(high_base_low_energy.combined_score, low_base_high_energy.combined_score)


class TestBestEnergyPeak(unittest.TestCase):
    def test_finds_the_busiest_window(self):
        words = []
        # quiet stretch then a burst
        for i in range(20):
            words.append(WordTimestamp(word="a", start=i * 0.4, end=i * 0.4 + 0.3))
        for i in range(120):
            words.append(WordTimestamp(word="go", start=10.0 + i * 0.12, end=10.0 + i * 0.12 + 0.1))
        segments = [_segment(0.0, 30.0, "mixed delivery", words)]
        peak = best_energy_peak(segments, 0.0, 30.0, min_duration=10.0, max_duration=15.0)
        self.assertIsNotNone(peak)
        start, end, score = peak
        self.assertGreaterEqual(start, 0.0)
        self.assertLessEqual(end, 30.0)
        self.assertGreaterEqual(end - start, 10.0 - 1e-6)
        self.assertGreaterEqual(score, 0.0)

    def test_returns_none_when_range_too_short(self):
        self.assertIsNone(best_energy_peak([], 0.0, 5.0, min_duration=30.0, max_duration=60.0))


if __name__ == "__main__":
    unittest.main()
