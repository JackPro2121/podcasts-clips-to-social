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
from pathlib import Path
from unittest.mock import patch

from src.downloader import (
    classify_ytdlp_error,
    best_available_height,
    VideoUnavailableError,
)
from src.channel_discovery import classify_candidate_entry
from src.viral_detector import _safe_int, strip_emojis
from src.subtitle_generator import format_ass_timestamp, default_margin_v, get_subtitle_margin_v
from src.config import SUBTITLE_MARGIN_MIN_V, SUBTITLE_MARGIN_MAX_V


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

    def test_targeted_apify_download_uses_segment_actor(self):
        from src.downloader import download_via_apify
        expected = {"video_path": Path("clip.mp4"), "duration": 30.0}
        with patch("src.downloader.download_segment_via_apify", return_value=expected) as segment:
            result = download_via_apify(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                Path("downloads"),
                start_time=60.0,
                end_time=90.0,
                api_token="test-token",
            )
        self.assertEqual(result, expected)
        segment.assert_called_once()

    def test_download_segment_via_apify_integer_timestamps_and_offset(self):
        from unittest.mock import MagicMock
        from src.downloader import download_segment_via_apify
        import tempfile

        fake_start_response = MagicMock()
        fake_start_response.status_code = 201
        fake_start_response.json.return_value = [{"output": {"pollUrl": "https://api.apify.com/poll/123"}}]

        fake_poll_response = MagicMock()
        fake_poll_response.status_code = 200
        fake_poll_response.json.return_value = {"status": "COMPLETED", "downloadUrl": "https://cdn.example.com/clip.mp4"}

        fake_stream_response = MagicMock()
        fake_stream_response.status_code = 200
        fake_stream_response.iter_content.return_value = [b"x" * 1024]
        fake_stream_response.__enter__.return_value = fake_stream_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch("requests.post", return_value=fake_start_response) as mock_post, \
                 patch("requests.get", side_effect=[fake_poll_response, fake_stream_response]), \
                 patch("src.downloader.get_video_height", return_value=1080), \
                 patch("src.downloader.get_video_duration", return_value=30.6):
                result = download_segment_via_apify(
                    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    output_dir=out_dir,
                    start_time=74.4,
                    end_time=105.0,
                    api_token="dummy-token",
                )

                self.assertIsNotNone(result)
                # Verify Apify payload uses integers (not floats)
                mock_post.assert_called_once()
                sent_payload = mock_post.call_args[1]["json"]
                self.assertIsInstance(sent_payload["startTime"], int)
                self.assertIsInstance(sent_payload["endTime"], int)
                self.assertEqual(sent_payload["startTime"], 74)
                self.assertEqual(sent_payload["endTime"], 105)

                # Verify segment_start offset is calculated
                self.assertAlmostEqual(result["segment_start"], 0.4, places=2)
                self.assertEqual(result["height"], 1080)

    def test_download_clip_segment_prioritizes_apify(self):
        from unittest.mock import patch
        from src.downloader import download_clip_segment
        import tempfile

        fake_apify_res = {
            "video_path": Path("clip_apify.mp4"),
            "segment_start": 0.25,
            "duration": 30.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch("src.downloader.APIFY_API_TOKEN", "dummy-apify-token"), \
                 patch("src.downloader.download_via_apify", return_value=fake_apify_res) as mock_apify, \
                 patch("src.downloader.get_video_height", return_value=1080), \
                 patch("src.downloader.get_video_duration", return_value=30.0), \
                 patch("yt_dlp.YoutubeDL") as mock_ydl:
                # Provide a dummy file so get_video_height / stat don't fail
                dummy_file = out_dir / "clip_apify.mp4"
                dummy_file.write_bytes(b"\x00" * 1024)
                fake_apify_res["video_path"] = dummy_file

                res = download_clip_segment(
                    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    start_time=10.25,
                    end_time=40.25,
                    clip_index=1,
                    output_dir=out_dir,
                )
                self.assertIsNotNone(res)
                # Apify must have been called first
                mock_apify.assert_called_once()
                # yt-dlp should not be invoked at all when Apify succeeds
                mock_ydl.assert_not_called()
                self.assertAlmostEqual(res["segment_start"], 0.25)


class TestReleaseCleanupSafety(unittest.TestCase):
    def test_buffer_lookup_failure_preserves_releases(self):
        from src.release_cleaner import clean_old_releases

        class BrokenBuffer:
            last_error = "Buffer API timeout"

            def get_active_video_urls(self):
                return {}

        result = clean_old_releases(
            repo="owner/repo",
            token="test-token",
            buffer_client=BrokenBuffer(),
        )
        self.assertTrue(result["skipped"])
        self.assertEqual(result["deleted"], 0)


class TestDiscoveryFilter(unittest.TestCase):
    def test_unknown_duration_is_kept(self):
        # The core bug: flat entries omit duration; old code discarded them all.
        got = classify_candidate_entry({"id": "abcdefghij1", "title": "Full Ep"}, set())
        self.assertIsNotNone(got)
        self.assertFalse(got[3])  # not confirmed-long, but still a candidate

    def test_confirmed_short_rejected(self):
        self.assertIsNone(classify_candidate_entry({"id": "abcdefghij2", "duration": 120}, set()))

    def test_youtube_short_url_rejected_without_duration(self):
        self.assertIsNone(classify_candidate_entry(
            {"url": "https://www.youtube.com/shorts/jG-3AB56zLg", "title": "Already short"}, set()
        ))

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
        self.assertEqual(default_margin_v("split_screen"), 880)   # \an2 bottom-center at Y≈1000 (center divider)
        self.assertEqual(default_margin_v("multi_shot_dynamic"), 460)  # fallback default

    def test_safe_zone_margin_overrides_are_clamped(self):
        self.assertEqual(
            get_subtitle_margin_v("multi_shot_dynamic", 0),
            SUBTITLE_MARGIN_MIN_V,
        )
        self.assertEqual(
            get_subtitle_margin_v("multi_shot_dynamic", 9999),
            SUBTITLE_MARGIN_MAX_V,
        )


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

    def test_broll_render_terminates_at_clip_duration(self):
        import tempfile
        from src.face_tracker import FramingDecision
        from src.video_editor import render_viral_clip

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.mp4"
            broll_path = tmp_path / "broll.mp4"
            output_path = tmp_path / "output.mp4"

            for path, pattern, media_duration in (
                (source_path, "testsrc=size=320x240:rate=10", 3.0),
                (broll_path, "testsrc2=size=320x240:rate=10", 1.0),
            ):
                create = subprocess.run(
                    [
                        "ffmpeg", "-y", "-f", "lavfi", "-i", pattern,
                        "-t", f"{media_duration:.2f}", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(create.returncode, 0, create.stderr[-600:])

            render_viral_clip(
                source_video_path=source_path,
                start_time=0.0,
                end_time=2.0,
                output_clip_path=output_path,
                framing=FramingDecision(
                    mode="single_smooth",
                    face_count=0,
                    video_width=320,
                    video_height=240,
                    active_w=320,
                    active_h=240,
                    smoothed_center_x=160,
                ),
                burn_subtitles=False,
                broll_cues=[(0.5, 1.5, broll_path, "money")],
            )

            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr[-600:])
            self.assertLess(abs(float(probe.stdout.strip()) - 2.0), 0.3)

    def test_no_audio_render_produces_audio_stream(self):
        import json
        import tempfile
        from src.face_tracker import FramingDecision
        from src.video_editor import render_viral_clip

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.mp4"
            output_path = tmp_path / "output.mp4"
            create = subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x240:rate=10", "-t", "1.00",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(create.returncode, 0, create.stderr[-600:])

            render_viral_clip(
                source_video_path=source_path,
                start_time=0.0,
                end_time=1.0,
                output_clip_path=output_path,
                framing=FramingDecision(
                    mode="single_smooth",
                    face_count=0,
                    video_width=320,
                    video_height=240,
                    active_w=320,
                    active_h=240,
                    smoothed_center_x=160,
                ),
                burn_subtitles=False,
            )

            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "stream=codec_type,duration", "-of", "json", str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr[-600:])
            streams = json.loads(probe.stdout).get("streams", [])
            video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
            audio_stream = next(stream for stream in streams if stream.get("codec_type") == "audio")
            self.assertLess(
                abs(float(audio_stream["duration"]) - float(video_stream["duration"])),
                0.1,
            )

    def test_real_ass_burn_in_produces_subtitle_pixels(self):
        import tempfile
        from PIL import Image
        from src.face_tracker import FramingDecision
        from src.video_editor import build_video_filtergraph
        from src.subtitle_generator import create_styled_ass_subtitles
        from src.transcriber import TranscriptSegment, WordTimestamp

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ass_path = tmp_path / "burn.ass"
            frame_path = tmp_path / "frame.png"
            segment = TranscriptSegment(
                start=0.0,
                end=0.8,
                text="Visible subtitle",
                words=[WordTimestamp(word="Visible", start=0.0, end=0.8)],
            )
            create_styled_ass_subtitles(
                segments=[segment],
                clip_start=0.0,
                clip_end=0.8,
                output_ass_path=ass_path,
                watermark="",
            )
            graph = build_video_filtergraph(
                FramingDecision(
                    mode="single_smooth",
                    face_count=0,
                    video_width=320,
                    video_height=240,
                    active_w=320,
                    active_h=240,
                    smoothed_center_x=160,
                ),
                ass_subtitle_path=ass_path,
                burn_subtitles=True,
            )
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "color=c=black:s=320x240:r=10", "-filter_complex", graph,
                    "-map", "[outv]", "-frames:v", "1", str(frame_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-600:])
            image = Image.open(frame_path).convert("RGB")
            pixel_data = getattr(image, "get_flattened_data", image.getdata)()
            bright_pixels = sum(1 for pixel in pixel_data if max(pixel) > 180)
            self.assertGreater(bright_pixels, 10)

    def test_requested_missing_ass_file_fails_before_render(self):
        from src.face_tracker import FramingDecision
        from src.video_editor import build_video_filtergraph

        with self.assertRaises(FileNotFoundError):
            build_video_filtergraph(
                FramingDecision(mode="single_smooth", face_count=0),
                ass_subtitle_path=Path("missing-subtitle.ass"),
                burn_subtitles=True,
            )

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
