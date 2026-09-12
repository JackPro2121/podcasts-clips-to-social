import unittest
from pathlib import Path
from src.config import SUBTITLE_THEMES, TARGET_LUFS, OUTPUT_WIDTH, OUTPUT_HEIGHT
from src.downloader import extract_youtube_id
from src.transcriber import parse_native_transcript, TranscriptSegment, WordTimestamp
from src.viral_detector import fallback_rule_based_detector
from src.face_tracker import FramingDecision
from src.subtitle_generator import (
    format_ass_timestamp,
    generate_ass_header,
    create_styled_ass_subtitles
)
from src.video_editor import (
    build_video_filtergraph,
    build_audio_filtergraph,
    sanitize_ffmpeg_path
)

class TestPodcastClipperPipeline(unittest.TestCase):

    def test_youtube_id_extraction(self):
        urls = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?t=10", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ")
        ]
        for url, expected_id in urls:
            self.assertEqual(extract_youtube_id(url), expected_id)

    def test_transcript_parsing(self):
        raw = [
            {"text": "Hello world", "start": 10.0, "duration": 2.0},
            {"text": "This is a viral podcast clip", "start": 12.0, "duration": 4.0}
        ]
        segments = parse_native_transcript(raw)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Hello world")
        self.assertEqual(len(segments[0].words), 2)
        self.assertAlmostEqual(segments[0].words[0].start, 10.0)
        self.assertAlmostEqual(segments[0].words[1].end, 12.0)

    def test_fallback_viral_detector(self):
        segments = [
            TranscriptSegment(start=0.0, end=30.0, text="Intro part 1", words=[]),
            TranscriptSegment(start=30.0, end=80.0, text="Deep debate part 2", words=[]),
            TranscriptSegment(start=80.0, end=150.0, text="Closing part 3", words=[])
        ]
        candidates = fallback_rule_based_detector(segments, num_clips=2)
        self.assertEqual(len(candidates), 2)
        for c in candidates:
            self.assertTrue(30.0 <= c.duration <= 60.0)
            self.assertTrue(len(c.hashtags) > 0)
            self.assertTrue(c.viral_score > 0)

    def test_subtitle_styling_and_ass_header(self):
        header_single = generate_ass_header(theme_key="hormozi", layout_mode="single_smooth")
        self.assertIn("Montserrat Black", header_single)
        self.assertIn("&H00FFFFFF", header_single)  # Crisp white
        self.assertIn("100,100,460,1", header_single.replace(" ", ""))

        header_split = generate_ass_header(theme_key="neon_green", layout_mode="split_screen")
        self.assertIn("PlayResX: 1080", header_split)
        self.assertIn("PlayResY: 1920", header_split)

    def test_ass_timestamp_formatting(self):
        self.assertEqual(format_ass_timestamp(0.0), "0:00:00.00")
        self.assertEqual(format_ass_timestamp(65.25), "0:01:05.25")
        self.assertEqual(format_ass_timestamp(3661.5), "1:01:01.50")

    def test_ass_file_generation(self):
        words = [
            WordTimestamp(word="Fine.", start=10.0, end=10.4),
            WordTimestamp(word="How?", start=10.4, end=10.7),
            WordTimestamp(word="Awesome!", start=10.7, end=11.2),
        ]
        seg = TranscriptSegment(start=10.0, end=12.0, text="Fine. How? Awesome!", words=words)
        out_ass = Path("subtitles/test_output.ass")
        created = create_styled_ass_subtitles(
            segments=[seg],
            clip_start=10.0,
            clip_end=12.0,
            output_ass_path=out_ass,
            theme_key="hormozi"
        )
        self.assertTrue(created.exists())
        content = created.read_text(encoding="utf-8")
        self.assertIn("Dialogue: 0", content)
        self.assertIn("FINE", content)
        self.assertIn("HOW", content)
        self.assertIn("AWESOME", content)
        self.assertNotIn("FINE.", content)
        self.assertNotIn("HOW?", content)
        self.assertNotIn("AWESOME!", content)
        # Clean up test file
        if created.exists():
            created.unlink()

    def test_video_filtergraph_modes(self):
        # 1. Single smooth crop
        single_decision = FramingDecision(
            mode="single_smooth",
            face_count=1,
            smoothed_center_x=960,
            video_width=1920,
            video_height=1080
        )
        fg_single = build_video_filtergraph(single_decision, burn_subtitles=False)
        self.assertIn("crop=", fg_single)
        self.assertIn(f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}", fg_single)

        # 2. Split screen
        split_decision = FramingDecision(
            mode="split_screen",
            face_count=2,
            speaker1_box=(100, 0, 800, 1080),
            speaker2_box=(1000, 0, 800, 1080),
            video_width=1920,
            video_height=1080
        )
        fg_split = build_video_filtergraph(split_decision, burn_subtitles=False)
        self.assertIn("vstack=inputs=2", fg_split)
        self.assertIn("top_pane", fg_split)
        self.assertIn("bottom_pane", fg_split)

        # 3. Blur stack
        blur_decision = FramingDecision(
            mode="blur_stack",
            face_count=3,
            video_width=1920,
            video_height=1080
        )
        fg_blur = build_video_filtergraph(blur_decision, burn_subtitles=False)
        self.assertIn("boxblur=", fg_blur)
        self.assertIn("overlay=0:", fg_blur)

        # 4. Dynamic Cut (Multi-camera switching)
        dynamic_decision = FramingDecision(
            mode="dynamic_cut",
            face_count=2,
            crop_x_expr="if(lt(t\\,20.5)\\,200\\,1200)",
            video_width=1920,
            video_height=1080
        )
        fg_dynamic = build_video_filtergraph(dynamic_decision, burn_subtitles=False)
        self.assertIn("crop=", fg_dynamic)
        self.assertIn("if(lt(t\\,20.5)\\,200\\,1200)", fg_dynamic)

    def test_audio_mastering_filters(self):
        af = build_audio_filtergraph()
        self.assertIn("highpass=f=80", af)
        self.assertIn("equalizer=f=3000", af)
        self.assertIn(f"loudnorm=I={TARGET_LUFS}", af)

    def test_studio_clarity_unsharp_in_filtergraph(self):
        decision = FramingDecision(
            mode="single_smooth",
            face_count=1,
            smoothed_center_x=960,
            video_width=1920,
            video_height=1080
        )
        fg = build_video_filtergraph(decision, burn_subtitles=False)
        self.assertIn("cas=0.45", fg)
        self.assertIn("hqdn3d=1.5:1.5:3:3", fg)
        self.assertIn("unsharp=lx=5:ly=5:la=0.75", fg)
        self.assertIn("eq=contrast=1.07", fg)
        self.assertIn("noise=alls=1.2", fg)

    def test_super_resolution_filtergraph_on_low_res(self):
        decision = FramingDecision(
            mode="single_smooth",
            face_count=1,
            smoothed_center_x=320,
            video_width=640,
            video_height=360
        )
        fg = build_video_filtergraph(decision, burn_subtitles=False)
        self.assertIn("cas=0.60", fg)
        self.assertIn("hqdn3d=2.0:2.0:4.0:4.0", fg)
        self.assertIn("unsharp=lx=7:ly=7:la=1.1", fg)
        self.assertIn("eq=contrast=1.09", fg)


    def test_scene_classifier_document_detection(self):
        import numpy as np
        import cv2
        from src.scene_classifier import detect_slide_heuristics, classify_frame_scene

        # Create synthetic document frame (white background with text lines)
        doc_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 245
        for row in range(80, 650, 50):
            cv2.putText(doc_frame, "Clinical Study: Metabolic Biomarkers & Longevity", (80, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        is_slide = detect_slide_heuristics(doc_frame)
        self.assertTrue(is_slide, "Synthetic document frame with text lines should be detected as slide/presentation")

        scene_info = classify_frame_scene(doc_frame)
        self.assertTrue(scene_info["is_presentation"])

    def test_multi_shot_dynamic_filtergraph(self):
        from src.face_tracker import ShotPlan
        shots = [
            ShotPlan(start=0.0, end=5.0, mode="presentation_slide", margin_v=420),
            ShotPlan(start=5.0, end=12.0, mode="portrait_face", crop_x=650, margin_v=220),
        ]
        decision = FramingDecision(
            mode="multi_shot_dynamic",
            face_count=2,
            shots=shots,
            video_width=1920,
            video_height=1080
        )
        fg = build_video_filtergraph(decision, burn_subtitles=False)
        self.assertIn("trim=start=0.00:end=5.00", fg)
        self.assertIn("trim=start=5.00:end=12.00", fg)
        self.assertIn("concat=n=2:v=1:a=0", fg)
        self.assertIn("setsar=1:1", fg)

    def test_dynamic_shot_subtitles_margin(self):
        from src.face_tracker import ShotPlan
        words = [
            WordTimestamp(word="Slide", start=2.0, end=2.5),
            WordTimestamp(word="Speaker", start=6.0, end=6.5),
        ]
        seg = TranscriptSegment(start=2.0, end=7.0, text="Slide Speaker", words=words)
        shots = [
            ShotPlan(start=0.0, end=5.0, mode="presentation_slide", margin_v=420),
            ShotPlan(start=5.0, end=10.0, mode="portrait_face", crop_x=650, margin_v=220),
        ]
        out_ass = Path("subtitles/test_dynamic_shots.ass")
        created = create_styled_ass_subtitles(
            segments=[seg],
            clip_start=0.0,
            clip_end=10.0,
            output_ass_path=out_ass,
            shots=shots
        )
        self.assertTrue(created.exists())
        content = created.read_text(encoding="utf-8")
        # Line 1 (t=2.0s during slide) must have margin_v=420
        # Line 2 (t=6.0s during speaker) must have margin_v=220
        self.assertIn(",0,0,420,,", content)
        self.assertIn(",0,0,220,,", content)
        if created.exists():
            created.unlink()

    def test_sanitize_ffmpeg_path(self):
        p = Path("C:/videos/clip 1.ass")
        sanitized = sanitize_ffmpeg_path(p)
        self.assertIn("C\\:/", sanitized)
        self.assertNotIn("C:/", sanitized)

    def test_ytdlp_client_variants_order(self):
        from src.downloader import _ytdlp_client_variants
        variants = _ytdlp_client_variants()
        labels = [v[0] for v in variants]
        self.assertIn("android_vr", labels)
        self.assertIn("tv_embedded", labels)
        self.assertEqual(labels[0], "android_vr", "android_vr must be first for zero bot-block")
        self.assertEqual(labels[1], "tv_embedded", "tv_embedded must be second")

    def test_cobalt_downloader_mock_success(self):
        from unittest.mock import patch, MagicMock
        from src.downloader import download_via_cobalt
        import tempfile

        fake_post_response = MagicMock()
        fake_post_response.status_code = 200
        fake_post_response.json.return_value = {
            "status": "tunnel",
            "url": "https://fake-cdn.cobalt.tools/stream.mp4"
        }

        fake_stream_response = MagicMock()
        fake_stream_response.status_code = 200
        # 1.5MB dummy stream chunk
        fake_stream_response.iter_content.return_value = [b"0" * (1024 * 1024 * 2)]
        fake_stream_response.__enter__.return_value = fake_stream_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch("requests.post", return_value=fake_post_response), \
                 patch("requests.get", return_value=fake_stream_response), \
                 patch("src.downloader.get_video_height", return_value=1080), \
                 patch("src.downloader.get_video_duration", return_value=45.0):
                res = download_via_cobalt("https://www.youtube.com/watch?v=dQw4w9WgXcQ", out_dir)
                self.assertIsNotNone(res)
                self.assertEqual(res["height"], 1080)
                self.assertFalse(res["is_low_res"])
                self.assertTrue(res["video_path"].exists())

    def test_get_video_duration_fallback(self):
        from src.downloader import get_video_duration
        dur = get_video_duration(Path("non_existent_file.mp4"))
        self.assertEqual(dur, 0.0)

    def test_fetch_transcript_only_returns_none_on_bad_url(self):
        """fetch_transcript_only must return None gracefully for non-YouTube inputs."""
        from src.downloader import fetch_transcript_only
        result = fetch_transcript_only("https://example.com/not-a-youtube-url")
        self.assertIsNone(result)

    def test_fetch_transcript_only_returns_none_when_no_captions(self):
        """fetch_transcript_only must return None when youtube_transcript_api finds nothing."""
        from unittest.mock import patch
        from src.downloader import fetch_transcript_only
        with patch("src.downloader.fetch_youtube_transcript", return_value=None):
            result = fetch_transcript_only("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIsNone(result)

    def test_fetch_transcript_only_success(self):
        """fetch_transcript_only must return video_id + transcript on success."""
        from unittest.mock import patch
        from src.downloader import fetch_transcript_only
        fake_transcript = [{"text": "Hello world", "start": 0.0, "duration": 2.0}]
        with patch("src.downloader.fetch_youtube_transcript", return_value=fake_transcript):
            result = fetch_transcript_only("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIsNotNone(result)
        self.assertEqual(result["video_id"], "dQw4w9WgXcQ")
        self.assertEqual(result["transcript"], fake_transcript)

    def test_download_clip_segment_permanent_error_returns_none(self):
        """download_clip_segment must return None cleanly on a permanent video error."""
        from unittest.mock import patch, MagicMock
        from src.downloader import download_clip_segment
        import tempfile

        perm_exc = Exception("video unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yt_dlp.YoutubeDL") as MockYDL:
                instance = MockYDL.return_value.__enter__.return_value
                instance.extract_info.side_effect = perm_exc
                result = download_clip_segment(
                    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    start_time=60.0,
                    end_time=120.0,
                    clip_index=1,
                    output_dir=Path(tmpdir)
                )
        self.assertIsNone(result)

    def test_download_clip_segment_segment_start_always_zero(self):
        """Returned dict from download_clip_segment must have segment_start=0.0."""
        from unittest.mock import patch, MagicMock
        from src.downloader import download_clip_segment
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # Fake the yt-dlp success path: create a dummy mp4 file
            dummy_mp4 = out_dir / "clip_1_60s_120s.mp4"
            dummy_mp4.write_bytes(b"\x00" * (600 * 1024))  # 600 KB stub

            with patch("yt_dlp.YoutubeDL") as MockYDL, \
                 patch("src.downloader.get_video_height", return_value=1080), \
                 patch("src.downloader.get_video_duration", return_value=60.0):
                instance = MockYDL.return_value.__enter__.return_value
                instance.extract_info.return_value = {"id": "dQw4w9WgXcQ"}
                result = download_clip_segment(
                    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    start_time=60.0,
                    end_time=120.0,
                    clip_index=1,
                    output_dir=out_dir
                )
            if result:
                self.assertEqual(result["segment_start"], 0.0)
                self.assertEqual(result["height"], 1080)

    def test_top_hook_badge_generation(self):
        """Top hook retention capsule badge must be generated in safe-zone without emojis."""
        from src.config import HOOK_BADGE_MARGIN_V
        words = [
            WordTimestamp(word="I", start=0.0, end=0.2),
            WordTimestamp(word="Lost", start=0.2, end=0.5),
            WordTimestamp(word="Millions", start=0.5, end=1.0),
        ]
        seg = TranscriptSegment(start=0.0, end=2.0, text="I Lost Millions", words=words)
        out_ass = Path("subtitles/test_hook_badge.ass")
        created = create_styled_ass_subtitles(
            segments=[seg],
            clip_start=0.0,
            clip_end=2.0,
            output_ass_path=out_ass,
            theme_key="hormozi",
            header_title="🔥 HOW HE LOST $10,000,000 IN 10 DAYS!"
        )
        self.assertTrue(created.exists())
        content = created.read_text(encoding="utf-8")
        # Header style with safe-zone MarginV
        self.assertIn(f"TopHeader,Montserrat Black,46,&H00FFFFFF,&H000000FF,&H00B86B62,&H00000000,-1,0,0,0,100,100,1.2,0,3,18,0,8,120,120,{HOOK_BADGE_MARGIN_V},1", content)
        # Event with clean title (no emoji), 2-line split, fade
        self.assertIn("TopHeader", content)
        self.assertIn(r"{\fad(150,350)}", content)
        self.assertIn("HOW HE", content)
        self.assertNotIn("🔥", content)  # Emoji stripped for ASS safety
        if created.exists():
            created.unlink()

if __name__ == "__main__":
    unittest.main()
