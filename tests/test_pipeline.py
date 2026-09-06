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

    def test_sanitize_ffmpeg_path(self):
        p = Path("C:/videos/clip 1.ass")
        sanitized = sanitize_ffmpeg_path(p)
        self.assertIn("C\\:/", sanitized)
        self.assertNotIn("C:/", sanitized)

if __name__ == "__main__":
    unittest.main()
