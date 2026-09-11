"""Regression tests for the core-loop rebuild.

Covers the pure logic behind the highest-impact fixes:
  * probe-first downloader error classification (no more 7x re-download of dead videos)
  * discovery candidate filtering (the old `duration > 600` bug made it inert)
  * viral-score coercion + emoji stripping
  * ASS timecode carry + caption safe-zone margins
  * filtergraph validity (only when ffmpeg is available)
"""
import shutil
import subprocess
import unittest

from src.downloader import (
    classify_ytdlp_error,
    best_available_height,
    VideoUnavailableError,
)
from src.channel_discovery import classify_candidate_entry
from src.viral_detector import _safe_int, strip_emojis
from src.subtitle_generator import format_ass_timestamp, default_margin_v


class TestDownloaderClassification(unittest.TestCase):
    def test_permanent_errors_abort(self):
        for msg in [
            "ERROR: Private video. Sign in if you've been granted access.",
            "Video unavailable",
            "This video has been removed by the uploader",
            "The uploader has not made this video available in your country",
        ]:
            self.assertEqual(classify_ytdlp_error(Exception(msg)), "permanent", msg)

    def test_transient_errors_rotate(self):
        for msg in [
            "Sign in to confirm you're not a bot",
            "HTTP Error 429: Too Many Requests",
            "Requested format is not available",
            "Unable to download webpage: read timed out",
        ]:
            self.assertEqual(classify_ytdlp_error(Exception(msg)), "transient", msg)

    def test_unknown_errors_default(self):
        self.assertEqual(classify_ytdlp_error(Exception("kaboom")), "unknown")

    def test_best_available_height(self):
        info = {"formats": [
            {"vcodec": "vp9", "height": 720},
            {"vcodec": "none", "height": 0},       # audio-only ignored
            {"vcodec": "avc1", "height": 1080},
            {"vcodec": "avc1", "height": None},     # tolerated
        ]}
        self.assertEqual(best_available_height(info), 1080)
        self.assertEqual(best_available_height({"formats": [], "height": 480}), 480)
        self.assertEqual(best_available_height({"formats": []}), 0)

    def test_unavailable_exception_type(self):
        self.assertTrue(issubclass(VideoUnavailableError, Exception))


class TestDiscoveryFilter(unittest.TestCase):
    def test_unknown_duration_is_kept(self):
        # The core bug: flat entries omit duration; old code discarded them all.
        got = classify_candidate_entry({"id": "abcdefghij1", "title": "Full Ep"}, set())
        self.assertIsNotNone(got)
        self.assertFalse(got[3])  # not confirmed-long, but still a candidate

    def test_confirmed_short_rejected(self):
        self.assertIsNone(classify_candidate_entry({"id": "abcdefghij2", "duration": 120}, set()))

    def test_confirmed_long_flagged(self):
        got = classify_candidate_entry({"id": "abcdefghij3", "duration": 3600, "title": "Pod"}, set())
        self.assertIsNotNone(got)
        self.assertTrue(got[3])

    def test_already_processed_rejected(self):
        self.assertIsNone(classify_candidate_entry({"id": "dQw4w9WgXcQ"}, {"dQw4w9WgXcQ"}))

    def test_missing_url_rejected(self):
        self.assertIsNone(classify_candidate_entry({}, set()))


class TestViralDetectorHelpers(unittest.TestCase):
    def test_safe_int_coercion(self):
        self.assertEqual(_safe_int("92.5"), 92)
        self.assertEqual(_safe_int(88), 88)
        self.assertEqual(_safe_int(None, default=85), 85)
        self.assertEqual(_safe_int("garbage", default=85), 85)

    def test_safe_int_clamped(self):
        self.assertEqual(_safe_int(150), 100)
        self.assertEqual(_safe_int(-10), 1)

    def test_emoji_stripped(self):
        self.assertNotIn("\U0001F447", strip_emojis("look below \U0001F447"))


class TestSubtitleFormatting(unittest.TestCase):
    def test_timecode_carry(self):
        # 59.999s must roll into the next minute, never emit SS=60.
        self.assertEqual(format_ass_timestamp(59.999), "0:01:00.00")
        self.assertEqual(format_ass_timestamp(3599.999), "1:00:00.00")
        self.assertEqual(format_ass_timestamp(0.0), "0:00:00.00")
        self.assertEqual(format_ass_timestamp(12.34), "0:00:12.34")

    def test_safe_zone_margins(self):
        # single_smooth must NOT be 0 (that pinned captions under the UI).
        self.assertEqual(default_margin_v("single_smooth"), 460)
        self.assertEqual(default_margin_v("blur_stack"), 400)
        self.assertEqual(default_margin_v("split_screen"), 0)
        self.assertEqual(default_margin_v("multi_shot_dynamic"), 460)  # fallback default


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class TestFiltergraphValidity(unittest.TestCase):
    """Prove the edited filtergraphs actually build (catches the split-screen
    aspect-fill edit and any pad-wiring mistakes)."""

    def _render(self, framing):
        from src.video_editor import build_video_filtergraph
        graph = build_video_filtergraph(framing, ass_subtitle_path=None, burn_subtitles=False)
        cmd = [
            "ffmpeg", "-hide_banner", "-f", "lavfi",
            "-i", "testsrc=s=1920x1080:d=1:r=6",
            "-filter_complex", graph, "-map", "[outv]",
            "-frames:v", "3", "-f", "null", "-",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-600:])

    def test_single_smooth(self):
        from src.face_tracker import FramingDecision
        self._render(FramingDecision(mode="single_smooth", face_count=1,
                                     smoothed_center_x=960, video_width=1920, video_height=1080))

    def test_blur_stack(self):
        from src.face_tracker import FramingDecision
        self._render(FramingDecision(mode="blur_stack", face_count=0,
                                     video_width=1920, video_height=1080))

    def test_split_screen(self):
        from src.face_tracker import FramingDecision
        self._render(FramingDecision(mode="split_screen", face_count=2,
                                     speaker1_box=(100, 0, 900, 1080),
                                     speaker2_box=(920, 0, 900, 1080),
                                     video_width=1920, video_height=1080))

    def test_multi_shot_dynamic(self):
        from src.face_tracker import FramingDecision, ShotPlan
        s1 = ShotPlan(start=0.0, end=0.5, mode="single_smooth", crop_x=480)
        s2 = ShotPlan(start=0.5, end=1.0, mode="split_screen", crop_x=0,
                      speaker1_box=(100, 0, 900, 1080), speaker2_box=(920, 0, 900, 1080))
        self._render(FramingDecision(mode="multi_shot_dynamic", face_count=2, shots=[s1, s2],
                                     video_width=1920, video_height=1080))


if __name__ == "__main__":
    unittest.main()
