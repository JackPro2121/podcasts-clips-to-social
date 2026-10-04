"""Endpoint completion: one boundary policy for normalizer, repair and warning.

The run that motivated this file:
    [+] Extended clip endpoint to complete transcript sentence at 125.76s.
    [!] Editorial QA warnings for clip #1: ['incomplete_endpoint', ...]

125.76 was exactly ``start + 54.0`` and mid-sentence; the sentence ended at
126.2, which is inside the real 55-second contract. The old function clamped to
a magic 54-second ceiling and printed success without verifying. These tests
pin the boundary policy (terminal word or spoken pause), the contract ceiling,
the trim/incomplete fallbacks, and the honest reporting.
"""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from src.creative_spec import (
    DURATION_QUANTIZATION_MARGIN_S,
    MAX_CLIP_DURATION_S,
    MIN_CLIP_DURATION_S,
)
from src.endpoint import (
    EndpointResolution,
    endpoint_is_complete,
    resolve_endpoint,
    word_boundary_ends,
)
from src.transcriber import TranscriptSegment, WordTimestamp
from src.universal_editor import _endpoint_warnings


def _seg(start, end, text, spec):
    return TranscriptSegment(
        start,
        end,
        text,
        [WordTimestamp(word, word_start, word_end) for word, word_start, word_end in spec],
    )


def _run_case_segments():
    """The run's clip #1 shape: a run-on segment whose sentence ends at 126.2."""
    return [
        _seg(
            120.0,
            126.2,
            "The answer is simple you just watch.",
            [
                ("The", 124.0, 124.2),
                ("answer", 124.2, 124.6),
                ("is", 124.6, 124.8),
                ("simple", 124.8, 125.2),
                ("you", 125.2, 125.4),
                ("just", 125.4, 125.7),
                ("watch.", 125.9, 126.2),
            ],
        )
    ]


class TestBoundaryDetection(unittest.TestCase):
    def test_terminal_punctuation_is_a_boundary(self):
        segments = [
            _seg(0.0, 1.0, "Done.", [("Done.", 0.6, 1.0)]),
        ]
        self.assertEqual(word_boundary_ends(segments), [1.0])

    def test_trailing_quote_after_terminal_punctuation_is_a_boundary(self):
        segments = [
            _seg(0.0, 1.0, 'The problem."', [('problem."', 0.6, 1.0)]),
        ]
        self.assertEqual(word_boundary_ends(segments), [1.0])

    def test_speech_pause_is_a_boundary_without_punctuation(self):
        segments = [
            _seg(
                0.0,
                1.2,
                "What happens next",
                [("What", 0.0, 0.4), ("happens", 0.4, 0.8), ("next", 0.8, 1.2)],
            ),
            _seg(2.5, 3.0, "Then we see.", [("Then", 2.5, 2.8), ("we", 2.8, 2.9), ("see.", 2.9, 3.0)]),
        ]
        self.assertEqual(word_boundary_ends(segments), [1.2, 3.0])

    def test_no_boundary_mid_sentence(self):
        segments = [
            _seg(0.0, 2.0, "and then the", [("and", 0.0, 0.2), ("then", 0.4, 0.6), ("the", 0.8, 1.0)]),
        ]
        self.assertEqual(word_boundary_ends(segments), [])

    def test_rounded_moment_end_still_matches_the_boundary(self):
        # The normalizer rounds window ends to two decimals; raw word ends are
        # floats. Small rounding jitter must not turn a complete endpoint
        # incomplete.
        segments = [
            _seg(0.0, 499.75, "We close.", [("close.", 499.25, 499.75)]),
        ]
        self.assertTrue(endpoint_is_complete(segments, 499.79))

    def test_endpoint_inside_sentence_is_not_complete(self):
        segments = _run_case_segments()
        self.assertFalse(endpoint_is_complete(segments, 123.5))


class TestResolveEndpoint(unittest.TestCase):
    def test_complete_endpoint_is_untouched(self):
        segments = _run_case_segments()
        resolution = resolve_endpoint(
            segments, start=71.76, requested_end=126.2, ceiling=126.61, min_duration=30.0
        )
        self.assertEqual(resolution, EndpointResolution(126.2, "complete"))

    def test_extends_to_next_word_boundary_inside_a_run_on_segment(self):
        segments = _run_case_segments()
        resolution = resolve_endpoint(
            segments, start=71.76, requested_end=123.5, ceiling=126.61, min_duration=30.0
        )
        self.assertEqual(resolution.status, "extended")
        self.assertEqual(resolution.end, 126.2)

    def test_uses_the_full_contract_not_the_old_54_second_clamp(self):
        # Old code clamped to start + 54.0 = 125.76, which is not a boundary.
        # The 55-second contract (minus one frame of margin) reaches 126.61.
        segments = _run_case_segments()
        ceiling = 71.76 + MAX_CLIP_DURATION_S - DURATION_QUANTIZATION_MARGIN_S
        resolution = resolve_endpoint(
            segments, start=71.76, requested_end=123.5, ceiling=ceiling, min_duration=30.0
        )
        self.assertEqual(resolution.end, 126.2)
        self.assertLessEqual(resolution.end - 71.76, MAX_CLIP_DURATION_S - DURATION_QUANTIZATION_MARGIN_S)
        self.assertNotEqual(resolution.end, 125.76)

    def test_out_of_contract_sentence_trims_to_the_previous_boundary(self):
        segments = [
            _seg(
                0.0,
                60.0,
                "Win. You can do it!",
                [
                    ("Win.", 52.9, 53.2),
                    ("You", 53.4, 53.7),
                    ("can", 53.7, 54.0),
                    ("do", 54.0, 54.3),
                    ("it!", 55.0, 55.4),
                ],
            )
        ]
        resolution = resolve_endpoint(
            segments, start=0.0, requested_end=53.8, ceiling=54.85, min_duration=30.0
        )
        self.assertEqual(resolution.status, "trimmed")
        self.assertEqual(resolution.end, 53.2)

    def test_previous_boundary_too_far_back_is_left_incomplete(self):
        segments = [
            _seg(
                0.0,
                60.0,
                "Done. And now the long second sentence runs on and on and",
                [
                    ("Done.", 39.9, 40.2),
                    ("And", 41.0, 41.2),
                    ("now", 41.2, 41.4),
                    ("the", 41.4, 41.6),
                    ("long", 41.6, 42.0),
                    ("second", 42.0, 42.4),
                    ("sentence", 42.4, 42.8),
                    ("runs", 42.8, 43.1),
                    ("on", 43.1, 43.3),
                    ("and", 43.3, 43.5),
                    ("on", 43.5, 43.7),
                    ("and", 43.7, 43.9),
                    ("the", 56.0, 56.2),
                    ("end!", 58.0, 58.3),
                ],
            )
        ]
        resolution = resolve_endpoint(
            segments, start=0.0, requested_end=54.5, ceiling=54.85, min_duration=30.0
        )
        self.assertEqual(resolution.status, "incomplete")
        self.assertEqual(resolution.end, 54.5)

    def test_trim_never_violates_the_minimum_duration(self):
        segments = [
            _seg(
                0.0,
                40.5,
                "Only boundary. Then a long run-on stretch and the end.",
                [
                    ("boundary.", 29.1, 29.4),
                    ("far", 39.7, 40.0),
                    ("end.", 40.0, 40.3),
                ],
            )
        ]
        resolution = resolve_endpoint(
            segments, start=0.0, requested_end=31.0, ceiling=54.85, min_duration=MIN_CLIP_DURATION_S
        )
        # The previous boundary at 29.4 is within ENDPOINT_TRIM_MAX_S of the
        # request but below the 30s floor, so it may not be used as the end.
        self.assertEqual(resolution.status, "incomplete")
        self.assertEqual(resolution.end, 31.0)


class TestMomentRepair(unittest.TestCase):
    def test_run_reproduction_ends_at_the_sentence_not_the_magic_clamp(self):
        import main

        segments = _run_case_segments()
        moment = SimpleNamespace(start_time=71.76, end_time=123.5, duration=51.74)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main._extend_moment_to_complete_transcript(segments, moment)
        self.assertEqual(moment.end_time, 126.2)
        self.assertAlmostEqual(moment.duration, 54.44, places=2)
        self.assertNotEqual(moment.end_time, 125.76)
        self.assertIn("Extended", buffer.getvalue())

    def test_gap_boundary_endpoint_is_not_moved_and_not_claimed(self):
        import main

        segments = [
            _seg(
                10.0,
                11.0,
                "What happens next",
                [("What", 10.0, 10.4), ("happens", 10.4, 10.7), ("next", 10.7, 11.0)],
            ),
            _seg(12.5, 13.5, "This is the next sentence.", [("This", 12.5, 12.8), ("is", 12.8, 12.9), ("the", 12.9, 13.0), ("next", 13.0, 13.2), ("sentence.", 13.2, 13.5)]),
        ]
        moment = SimpleNamespace(start_time=10.0, end_time=11.0, duration=1.0)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main._extend_moment_to_complete_transcript(segments, moment)
        # The old function dragged this endpoint to 13.5 - a valid boundary,
        # but past the end of the detected moment, changing the story.
        self.assertEqual(moment.end_time, 11.0)
        self.assertNotIn("Extended", buffer.getvalue())

    def test_incomplete_endpoint_is_reported_honestly(self):
        import main

        segments = [
            _seg(
                0.0,
                60.0,
                "Done. A long second sentence that runs past the contract",
                [
                    ("Done.", 39.9, 40.2),
                    ("A", 41.0, 41.2),
                    ("long", 41.2, 41.6),
                    ("second", 41.6, 42.0),
                    ("sentence", 42.0, 42.4),
                    ("that", 42.4, 42.6),
                    ("runs", 42.6, 42.9),
                    ("past", 42.9, 43.2),
                    ("the", 43.2, 43.4),
                    ("contract", 43.4, 43.8),
                    ("end!", 56.0, 56.3),
                ],
            )
        ]
        moment = SimpleNamespace(start_time=0.0, end_time=54.5, duration=54.5)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main._extend_moment_to_complete_transcript(segments, moment)
        self.assertEqual(moment.end_time, 54.5)
        output = buffer.getvalue()
        self.assertNotIn("Extended", output)
        self.assertIn("keeping the detected window", output)


class TestEndpointWarnings(unittest.TestCase):
    def test_gap_terminated_window_is_not_flagged(self):
        segments = [
            _seg(
                0.0,
                1.2,
                "What happens next",
                [("What", 0.0, 0.4), ("happens", 0.4, 0.8), ("next", 0.8, 1.2)],
            ),
            _seg(2.5, 3.0, "Then we see.", [("Then", 2.5, 2.8), ("we", 2.8, 2.9), ("see.", 2.9, 3.0)]),
        ]
        self.assertEqual(_endpoint_warnings(segments, 0.0, 1.2), [])

    def test_mid_sentence_cut_inside_a_run_on_segment_is_flagged(self):
        segments = [
            _seg(
                0.0,
                20.0,
                "We start and the payoff lands.",
                [
                    ("We", 0.0, 0.5),
                    ("start", 0.5, 0.9),
                    ("and", 0.9, 1.1),
                    ("the", 1.1, 1.3),
                    ("payoff", 1.3, 1.6),
                    ("lands.", 19.5, 20.0),
                ],
            )
        ]
        # The segment text ends with a period (20.0), but the window stops at
        # 18.0, mid-sentence. The old text-only check missed this.
        self.assertEqual(_endpoint_warnings(segments, 0.0, 18.0), ["endpoint_not_proven_complete"])

    def test_window_ending_on_a_terminal_word_clears_the_warning(self):
        segments = _run_case_segments()
        self.assertEqual(_endpoint_warnings(segments, 71.76, 126.2), [])

    def test_dangling_fragment_word_is_still_flagged(self):
        segments = [
            _seg(
                0.0,
                40.0,
                "You know what I mean yeah.",
                [
                    ("You", 38.0, 38.2),
                    ("know", 38.2, 38.4),
                    ("what", 38.4, 38.6),
                    ("I", 38.6, 38.7),
                    ("mean", 38.7, 39.0),
                    ("yeah.", 39.5, 40.0),
                ],
            )
        ]
        self.assertEqual(_endpoint_warnings(segments, 0.0, 40.0), ["endpoint_caption_fragment"])

    def test_empty_window_produces_no_warnings(self):
        self.assertEqual(_endpoint_warnings([], 0.0, 10.0), [])


if __name__ == "__main__":
    unittest.main()
