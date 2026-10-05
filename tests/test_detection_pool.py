"""Clip-count backfill: the render loops get more candidates than num_clips.

AGENTS.md 3.5 requires exactly num_clips delivered. Individual moments are lost
to audio-language rejections, QA vetoes and pixel-verdict vetoes, and every loss
is a bare ``continue`` - a run delivered 1/3 with a success exit. The pool
guarantees the loops can replace a lost moment with the next-best non-overlapping
candidate instead of ending in a shortfall.
"""

import json
import tempfile
import unittest
from pathlib import Path
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


class TestRenumberRenderedClips(unittest.TestCase):
    """Published file names must be sequential even when a candidate was lost."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _clip(self, name: str, thumb: str | None = None):
        path = self.dir / name
        path.write_bytes(b"x")
        item: dict = {"path": str(path), "moment": "m"}
        if thumb:
            thumb_path = self.dir / thumb
            thumb_path.write_bytes(b"x")
            item["thumbnail"] = str(thumb_path)
        return item

    def test_gapped_indices_are_renumbered_to_publish_order(self) -> None:
        clips = [
            self._clip("clip_1_A.mp4", "clip_1_A_thumb.jpg"),
            self._clip("clip_3_B.mp4"),
            self._clip("clip_4_C.mp4"),
        ]
        out = main._renumber_rendered_clips(clips)
        self.assertEqual(
            [Path(item["path"]).name for item in out],
            ["clip_1_A.mp4", "clip_2_B.mp4", "clip_3_C.mp4"],
        )
        for item in out:
            self.assertTrue(Path(item["path"]).exists())
        self.assertEqual(Path(out[0]["thumbnail"]).name, "clip_1_A_thumb.jpg")
        self.assertFalse((self.dir / "clip_3_B.mp4").exists())

    def test_existing_target_is_not_overwritten(self) -> None:
        # A failed candidate already wrote clip_1_B.mp4; it is diagnosis evidence.
        (self.dir / "clip_1_B.mp4").write_bytes(b"failed")
        out = main._renumber_rendered_clips([self._clip("clip_3_B.mp4")])
        self.assertEqual(Path(out[0]["path"]).name, "clip_3_B.mp4")
        self.assertEqual((self.dir / "clip_1_B.mp4").read_bytes(), b"failed")

    def test_non_clip_names_are_left_alone(self) -> None:
        item = {"path": str(self.dir / "custom_name.mp4")}
        (self.dir / "custom_name.mp4").write_bytes(b"x")
        out = main._renumber_rendered_clips([item])
        self.assertEqual(Path(out[0]["path"]).name, "custom_name.mp4")

    def test_missing_file_is_tolerated(self) -> None:
        out = main._renumber_rendered_clips([{"path": str(self.dir / "clip_2_X.mp4")}])
        self.assertEqual(Path(out[0]["path"]).name, "clip_2_X.mp4")


class TestBackfillMetadataEnrichment(unittest.TestCase):
    """Backfill candidates get LLM-written metadata before publish.

    Run 37287944990 published "DONT THINK ITS ONE MEAN" and an earlier
    scheduled run shipped "SAM NEW HOUSE WAS BORN" to three Buffer channels;
    both were raw-word titles from the semantic fallback.
    """

    def _segments(self):
        from src.transcriber import TranscriptSegment

        return [
            TranscriptSegment(
                start=100.0,
                end=140.0,
                text="The real claim of this clip is stated in a full sentence here.",
                words=[],
            )
        ]

    def _backfill(self, start=100.0, end=140.0):
        return SimpleNamespace(
            start_time=start,
            end_time=end,
            duration=end - start,
            origin="backfill",
            title="RAW WORDS HERE",
            social_caption="old generic caption",
            hashtags=["#old"],
        )

    def _run(self, llm_return, key="key"):
        llm = mock.Mock(return_value=llm_return)
        primary = [_moment(0, 40)]
        extras = [self._backfill()]
        with mock.patch.object(main, "detect_viral_moments", return_value=primary), mock.patch.object(
            main, "fallback_rule_based_detector", return_value=extras
        ), mock.patch("src.config.GEMINI_API_KEY", key), mock.patch(
            "src.viral_detector.query_gemini_models", llm
        ):
            pool = main._detection_pool(self._segments(), num_clips=1, niche="finance")
        return pool, llm

    def _payload(self):
        return json.dumps(
            {
                "title": "THE REAL FIX",
                "social_caption": "One real sentence. Two real sentences.",
                "hashtags": ["#finance", "#money"],
            }
        )

    def test_backfill_candidate_gets_llm_metadata(self) -> None:
        pool, llm = self._run(self._payload())
        backfill = [item for item in pool if getattr(item, "origin", "") == "backfill"]
        self.assertEqual(len(backfill), 1)
        self.assertEqual(backfill[0].title, "THE REAL FIX")
        self.assertEqual(backfill[0].social_caption, "One real sentence. Two real sentences.")
        self.assertEqual(backfill[0].hashtags, ["#finance", "#money"])
        llm.assert_called_once()

    def test_failed_call_leaves_the_fallback_text(self) -> None:
        pool, llm = self._run(None)
        backfill = [item for item in pool if getattr(item, "origin", "") == "backfill"]
        self.assertEqual(backfill[0].title, "RAW WORDS HERE")
        self.assertEqual(backfill[0].social_caption, "old generic caption")
        llm.assert_called_once()

    def test_no_key_skips_the_call(self) -> None:
        pool, llm = self._run(self._payload(), key="")
        llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
