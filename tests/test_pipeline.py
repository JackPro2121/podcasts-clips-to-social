import unittest
from pathlib import Path
from src.config import TARGET_LUFS, OUTPUT_WIDTH, OUTPUT_HEIGHT
from src.downloader import extract_youtube_id
from src.transcriber import is_english_language_code, looks_like_english_text, parse_native_transcript, TranscriptSegment, WordTimestamp
from src.viral_detector import fallback_rule_based_detector, parse_clips_json
from src.face_tracker import FramingDecision
from src.subtitle_generator import (
    format_ass_timestamp,
    generate_ass_header,
    create_styled_ass_subtitles,
    normalize_caption_text,
    escape_ass_text,
    sanitize_watermark
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

    def test_english_language_codes_are_accepted(self):
        self.assertTrue(is_english_language_code("en"))
        self.assertTrue(is_english_language_code("en-US"))
        self.assertFalse(is_english_language_code("es"))
        self.assertFalse(is_english_language_code(""))

    def test_transcript_language_corroboration(self):
        english = (
            "I think the market is going to crash and the people who are in the "
            "luxury car business should be careful because they do not know what is "
            "coming for them this year."
        )
        self.assertTrue(looks_like_english_text(english))
        self.assertFalse(looks_like_english_text("আজকের বাজারে অনেক চড়াচড়ি হয়েছে।"))
        self.assertFalse(looks_like_english_text("hola mundo como estas amigo mio hoy"))
        self.assertFalse(looks_like_english_text("too short"))

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
        self.assertTrue(segments[0].words[0].is_estimated)

    def test_fallback_viral_detector(self):
        segments = [
            TranscriptSegment(start=0.0, end=30.0, text="Intro part 1", words=[]),
            TranscriptSegment(start=30.0, end=80.0, text="Deep debate part 2", words=[]),
            TranscriptSegment(start=80.0, end=150.0, text="Closing part 3", words=[])
        ]
        candidates = fallback_rule_based_detector(segments, num_clips=2)
        self.assertEqual(len(candidates), 2)
        for c in candidates:
            self.assertTrue(30.0 <= c.duration <= 140.0)
            self.assertTrue(len(c.hashtags) > 0)
            self.assertTrue(c.viral_score > 0)

    def test_parse_clips_json_enforces_duration_contract(self):
        import json
        segments = [TranscriptSegment(0.0, 100.0, "Long transcript", [])]
        raw = json.dumps({
            "clips": [
                {"title": "Short", "start_time": 20, "end_time": 45, "viral_score": 90},
                {"title": "Long", "start_time": 20, "end_time": 90, "viral_score": 80},
            ]
        })
        clips = parse_clips_json(raw, segments, num_clips=3)
        self.assertEqual([(c.start_time, c.end_time) for c in clips], [(20.0, 50.0), (20.0, 90.0)])
        self.assertTrue(all(30.0 <= c.duration <= 140.0 for c in clips))

    def test_buffer_results_preserve_partial_dispatch(self):
        from unittest.mock import patch, MagicMock
        from src.buffer_client import BufferClient

        success = MagicMock(status_code=200)
        success.json.return_value = {"data": {"createPost": {"post": {"id": "post-1"}}}}
        failure = MagicMock(status_code=200)
        failure.json.return_value = {"data": {"createPost": {"message": "rejected"}}}

        with patch(
            "src.buffer_client.BufferClient.get_channels",
            return_value=[{"id": "a", "service": "youtube"}, {"id": "b", "service": "youtube"}],
        ), patch("src.buffer_client.requests.post", side_effect=[success, failure]):
            results = BufferClient("token").schedule_video_post(
                "https://example.com/video.mp4",
                "caption",
                channel_ids=["a", "b"],
            )

        self.assertEqual(len(results), 2)
        self.assertIn("response", results[0])
        self.assertIn("response", results[1])

    def test_buffer_channel_query_uses_organization_id_scalar(self):
        from unittest.mock import MagicMock, patch
        from src.buffer_client import BufferClient

        org_response = MagicMock(status_code=200)
        org_response.json.return_value = {
            "data": {"account": {"organizations": [{"id": "org-1", "name": "Org"}]}}
        }
        channel_response = MagicMock(status_code=200)
        channel_response.json.return_value = {
            "data": {"channels": [{"id": "channel-1", "name": "TikTok", "service": "tiktok"}]}
        }

        with patch("src.buffer_client.requests.post", side_effect=[org_response, channel_response]) as post:
            channels = BufferClient("token").get_channels()

        self.assertEqual(channels[0]["service"], "tiktok")
        channel_query = post.call_args_list[1].kwargs["json"]["query"]
        self.assertIn("$organizationId: OrganizationId!", channel_query)

    def test_niche_fallback_uses_selected_hashtags(self):
        segments = [TranscriptSegment(0.0, 80.0, "AI automation discussion", [])]
        clips = fallback_rule_based_detector(segments, num_clips=1, niche="ai_tech")
        self.assertEqual(clips[0].hashtags[0], "#ai")

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

    def test_clip_window_extends_to_sentence_boundaries(self):
        from src.viral_detector import _normalize_clip_window

        segments = [
            TranscriptSegment(
                start=19.0,
                end=20.0,
                text="This is context.",
                words=[WordTimestamp(word="context.", start=19.5, end=20.0)],
            ),
            TranscriptSegment(
                start=20.5,
                end=56.2,
                text="The key idea starts now and continues through the complete explanation.",
                words=[
                    WordTimestamp(word="The", start=20.5, end=20.8),
                    WordTimestamp(word="payoff.", start=55.8, end=56.2),
                ],
            ),
        ]
        self.assertEqual(_normalize_clip_window(20.5, 55.5, segments), (20.0, 56.2))

    def test_clip_window_rejects_incomplete_endpoint_without_sentence_boundary(self):
        from src.viral_detector import _normalize_clip_window

        segments = [
            TranscriptSegment(
                start=0.0,
                end=100.0,
                text="An incomplete thought that never closes",
                words=[
                    WordTimestamp(word="An", start=0.0, end=1.0),
                    WordTimestamp(word="incomplete", start=10.0, end=11.0),
                    WordTimestamp(word="what's-", start=20.0, end=21.0),
                ],
            )
        ]
        self.assertIsNone(_normalize_clip_window(0.0, 40.0, segments))

    def test_clip_window_repairs_incomplete_endpoint_to_next_sentence(self):
        from src.viral_detector import _normalize_clip_window

        segments = [
            TranscriptSegment(
                start=0.0,
                end=50.0,
                text="A complete thought",
                words=[
                    WordTimestamp(word="A", start=0.0, end=1.0),
                    WordTimestamp(word="complete.", start=10.0, end=11.0),
                    WordTimestamp(word="But", start=20.0, end=21.0),
                    WordTimestamp(word="it's", start=21.0, end=22.0),
                    WordTimestamp(word="those", start=22.0, end=23.0),
                    WordTimestamp(word="micro", start=23.0, end=24.0),
                    WordTimestamp(word="purchases,", start=24.0, end=25.0),
                    WordTimestamp(word="what's-", start=25.0, end=26.0),
                    WordTimestamp(word="the", start=26.0, end=27.0),
                    WordTimestamp(word="problem.", start=40.0, end=41.0),
                ],
            )
        ]
        self.assertEqual(_normalize_clip_window(0.0, 26.5, segments), (0.0, 41.0))

    def test_ass_text_normalization_and_escaping(self):
        self.assertEqual(
            normalize_caption_text("C++ 100% café 🚀\n$5"),
            "C++ 100% café $5",
        )
        self.assertEqual(escape_ass_text("{bad}\\N"), r"\{bad\}\\N")
        self.assertEqual(sanitize_watermark("@{test}\\n"), r"@\{test\}\\n")

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

    def test_subtitle_boundaries_and_estimated_marker(self):
        words = [
            WordTimestamp(word="Before", start=0.0, end=0.5),
            WordTimestamp(word="Inside", start=0.5, end=1.5, is_estimated=True),
            WordTimestamp(word="After", start=1.5, end=2.0),
        ]
        segment = TranscriptSegment(start=0.0, end=2.0, text="Before Inside After", words=words)
        out_ass = Path("subtitles/test_boundaries.ass")
        created = create_styled_ass_subtitles(
            segments=[segment],
            clip_start=0.5,
            clip_end=1.5,
            output_ass_path=out_ass,
        )
        content = created.read_text(encoding="utf-8")
        self.assertIn("Default,estimated,0,0,460", content)
        self.assertIn("INSIDE", content)
        self.assertNotIn("BEFORE", content)
        self.assertNotIn("AFTER", content)
        created.unlink()

    def test_subtitle_drops_word_crossing_complete_endpoint(self):
        out_ass = Path("subtitles/test_cross_boundary_word.ass")
        create_styled_ass_subtitles(
            segments=[TranscriptSegment(0.0, 1.0, "what's-", [WordTimestamp("what's-", 0.8, 1.1)])],
            clip_start=0.0,
            clip_end=1.0,
            output_ass_path=out_ass,
        )
        content = out_ass.read_text(encoding="utf-8")
        self.assertNotIn("WHAT", content)
        out_ass.unlink()

    def test_subtitle_groups_words_into_short_phrases(self):
        words = [
            WordTimestamp(word="One", start=0.0, end=1.0),
            WordTimestamp(word="Two", start=1.0, end=2.0),
            WordTimestamp(word="Three", start=2.0, end=3.0),
            WordTimestamp(word="Four", start=3.0, end=4.0),
        ]
        out_ass = Path("subtitles/test_phrase_grouping.ass")
        create_styled_ass_subtitles(
            segments=[TranscriptSegment(0.0, 4.0, "One Two Three Four", words)],
            clip_start=0.0,
            clip_end=4.0,
            output_ass_path=out_ass,
        )
        lines = out_ass.read_text(encoding="utf-8").splitlines()
        self.assertTrue(any("ONE" in line and "TWO" in line for line in lines))
        self.assertTrue(any("THREE" in line and "FOUR" in line for line in lines))
        out_ass.unlink()

    def test_subtitle_uses_three_word_phrases_when_available(self):
        words = [
            WordTimestamp(word="One", start=0.0, end=0.8),
            WordTimestamp(word="Two", start=0.8, end=1.6),
            WordTimestamp(word="Three", start=1.6, end=2.4),
            WordTimestamp(word="Four", start=2.4, end=3.2),
            WordTimestamp(word="Five", start=3.2, end=4.0),
            WordTimestamp(word="Six", start=4.0, end=4.8),
        ]
        out_ass = Path("subtitles/test_three_word_phrase.ass")
        create_styled_ass_subtitles(
            segments=[TranscriptSegment(0.0, 4.8, "One Two Three Four Five Six", words)],
            clip_start=0.0,
            clip_end=4.8,
            output_ass_path=out_ass,
        )
        lines = [
            line for line in out_ass.read_text(encoding="utf-8").splitlines()
            if line.startswith("Dialogue: 0,")
        ]
        self.assertEqual(len(lines), 2)
        self.assertIn("ONE", lines[0])
        self.assertIn("TWO", lines[0])
        self.assertIn("THREE", lines[0])
        self.assertIn("FOUR", lines[1])
        self.assertIn("FIVE", lines[1])
        self.assertIn("SIX", lines[1])
        out_ass.unlink()

    def test_subtitle_groups_words_across_native_caption_gaps(self):
        words = [
            WordTimestamp(word="One", start=0.0, end=0.4),
            WordTimestamp(word="Two", start=0.9, end=1.3),
            WordTimestamp(word="Three", start=1.8, end=2.2),
            WordTimestamp(word="Four", start=2.7, end=3.1),
        ]
        out_ass = Path("subtitles/test_phrase_gap_grouping.ass")
        create_styled_ass_subtitles(
            segments=[TranscriptSegment(0.0, 3.1, "One Two Three Four", words)],
            clip_start=0.0,
            clip_end=3.1,
            output_ass_path=out_ass,
        )
        content = out_ass.read_text(encoding="utf-8")
        dialogue_lines = [line for line in content.splitlines() if line.startswith("Dialogue: 0,")]
        self.assertEqual(len(dialogue_lines), 2)
        self.assertTrue(any("ONE" in line and "TWO" in line for line in dialogue_lines))
        self.assertTrue(any("THREE" in line and "FOUR" in line for line in dialogue_lines))
        out_ass.unlink()

    def test_local_transcript_alignment_replaces_estimated_timing(self):
        from unittest.mock import patch
        from src.transcriber import align_clip_transcript

        fallback = [
            TranscriptSegment(
                start=10.0,
                end=12.0,
                text="Estimated words",
                words=[WordTimestamp("Estimated", 10.0, 12.0, is_estimated=True)],
            )
        ]
        aligned = [
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="Aligned words",
                words=[
                    WordTimestamp("Aligned", 0.0, 1.0),
                    WordTimestamp("words", 1.0, 2.0),
                ],
            )
        ]
        with patch("src.transcriber.transcribe_audio_whisper", return_value=aligned) as transcribe:
            result = align_clip_transcript(
                clip_path=Path("clip.mp4"),
                render_start=0.5,
                output_start=10.0,
                output_end=12.0,
                fallback_segments=fallback,
                model_size="base.en",
            )
        transcribe.assert_called_once()
        self.assertEqual([(word.start, word.end) for word in result[0].words], [(10.0, 10.5), (10.5, 11.5)])
        self.assertTrue(all(not word.is_estimated for word in result[0].words))

    def test_subtitle_segment_fallback_when_words_are_missing(self):
        segment = TranscriptSegment(start=0.0, end=1.0, text="Fallback text", words=[])
        out_ass = Path("subtitles/test_segment_fallback.ass")
        created = create_styled_ass_subtitles(
            segments=[segment],
            clip_start=0.0,
            clip_end=1.0,
            output_ass_path=out_ass,
        )
        content = created.read_text(encoding="utf-8")
        self.assertIn("FALLBACK TEXT", content)
        self.assertIn("Default,estimated,0,0,460", content)
        created.unlink()

    def test_short_subtitle_event_uses_shorter_bounce_animation(self):
        words = [WordTimestamp(word="Hi", start=0.0, end=0.12)]
        segment = TranscriptSegment(start=0.0, end=0.12, text="Hi", words=words)
        out_ass = Path("subtitles/test_short_animation.ass")
        created = create_styled_ass_subtitles(
            segments=[segment],
            clip_start=0.0,
            clip_end=0.12,
            output_ass_path=out_ass,
        )
        content = created.read_text(encoding="utf-8")
        self.assertIn(r"\t(0,60,\fscx118\fscy118)\t(60,120,\fscx100\fscy100)", content)
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
        self.assertIn("sin(t*1.8)", fg)
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
        self.assertIn(",0,0,420,,", content)
        self.assertIn(",0,0,380,,", content)
        if created.exists():
            created.unlink()

    def test_dynamic_split_shot_uses_divider_alignment(self):
        from src.face_tracker import ShotPlan
        words = [WordTimestamp(word="Both", start=2.0, end=2.5)]
        seg = TranscriptSegment(start=2.0, end=3.0, text="Both", words=words)
        shot = ShotPlan(
            start=0.0,
            end=5.0,
            mode="split_screen",
            margin_v=0,
            subtitle_placement="divider",
            subtitle_alignment=5,
        )
        out_ass = Path("subtitles/test_split_divider.ass")
        created = create_styled_ass_subtitles(
            segments=[seg],
            clip_start=0.0,
            clip_end=5.0,
            output_ass_path=out_ass,
            layout_mode="multi_shot_dynamic",
            shots=[shot],
        )
        content = created.read_text(encoding="utf-8")
        self.assertIn(",0,0,0,,{\\an5}", content)
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
        with patch("src.downloader.fetch_youtube_transcript", return_value=None), \
             patch("src.transcriber.fetch_transcript_chocodata", return_value=None):
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

    def test_fetch_transcript_only_uses_apify_when_native_captions_missing(self):
        from unittest.mock import patch
        from src.downloader import fetch_transcript_only
        fallback = [{"text": "Timed fallback", "start": 0.0, "duration": 2.0}]
        with patch("src.downloader.fetch_youtube_transcript", return_value=None), \
             patch("src.downloader.fetch_transcript_via_apify", return_value=fallback) as apify:
            result = fetch_transcript_only(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                allow_apify_fallback=True,
            )
        self.assertEqual(result["transcript"], fallback)
        apify.assert_called_once()

    def test_download_clip_segment_permanent_error_returns_none(self):
        """download_clip_segment must return None cleanly on a permanent video error."""
        from unittest.mock import patch
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
        from unittest.mock import patch
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
        self.assertIn(f"TopHeader,Montserrat Black,42,&H00FFFFFF,&H000000FF,&H00B86B62,&H00000000,-1,0,0,0,100,100,1.2,0,3,11,0,8,100,100,{HOOK_BADGE_MARGIN_V},1", content)

    def test_kinetic_subtitles_bounce_and_emoji_injection(self):
        """Kinetic subtitles must include high-energy bounce scaling and contextual emojis."""
        words = [
            WordTimestamp(word="He", start=0.0, end=0.3),
            WordTimestamp(word="Made", start=0.3, end=0.6),
            WordTimestamp(word="Wealth", start=0.6, end=1.0),
        ]
        seg = TranscriptSegment(start=0.0, end=2.0, text="He Made Wealth", words=words)
        out_ass = Path("subtitles/test_kinetic_emoji.ass")
        created = create_styled_ass_subtitles(
            segments=[seg],
            clip_start=0.0,
            clip_end=2.0,
            output_ass_path=out_ass,
            theme_key="hormozi",
            keyword_emojis={"wealth": "💰"}
        )
        self.assertTrue(created.exists())
        content = created.read_text(encoding="utf-8")
        # Check kinetic bounce scaling tag
        self.assertIn(r"\fscx118\fscy118", content)
        # Check clean word rendering without unrenderable emoji tofu boxes
        self.assertIn("WEALTH", content)
        self.assertNotIn("💰", content)
        if created.exists():
            created.unlink()

    def test_director_sfx_cues_and_keyword_emojis_in_fallback(self):
        """AI Director fallback detector must generate valid sfx_cues and keyword_emojis."""
        segments = [
            TranscriptSegment(start=0.0, end=40.0, text="Discussion about money and wealth", words=[])
        ]
        candidates = fallback_rule_based_detector(segments, num_clips=1)
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertTrue(len(c.sfx_cues) > 0)
        self.assertEqual(c.sfx_cues[0][1], "whoosh")
        self.assertIn("money", c.keyword_emojis)
        self.assertEqual(c.keyword_emojis["money"], "💰")

    def test_render_viral_clip_with_sfx_and_ducking(self):
        """render_viral_clip must assemble sidechain compressor and adelay for SFX."""
        from unittest.mock import patch, MagicMock
        from src.face_tracker import FramingDecision
        from src.video_editor import render_viral_clip

        measurement = {
            "input_i": -14.0,
            "input_tp": -1.5,
            "input_lra": 1.0,
            "input_thresh": -24.0,
            "target_offset": 0.0,
        }

        def copy_normalized_output(source_path, destination_path, measured):
            Path(destination_path).write_bytes(Path(source_path).read_bytes())

        with patch("subprocess.run") as mock_run, \
             patch("src.video_editor._measure_loudness", return_value=measurement), \
             patch("src.video_editor._apply_measured_loudness", side_effect=copy_normalized_output), \
             patch("src.video_editor._validate_rendered_output"):
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "audio"
            mock_res.stderr = ""

            def fake_run(cmd, *args, **kwargs):
                if cmd and cmd[0] == "ffmpeg":
                    Path(cmd[-1]).write_bytes(b"rendered")
                return mock_res

            mock_run.side_effect = fake_run

            out_path = Path("clips/test_sfx_render.mp4")
            framing = FramingDecision(mode="single_smooth", face_count=1)

            render_viral_clip(
                source_video_path=Path("downloads/test.mp4"),
                start_time=10.0,
                end_time=25.0,
                output_clip_path=out_path,
                framing=framing,
                sfx_cues=[(0.1, "whoosh"), (5.0, "pop")],
                burn_subtitles=False
            )

            # Find the ffmpeg call
            ffmpeg_calls = [call for call in mock_run.call_args_list if call[0][0][0] == "ffmpeg"]
            self.assertTrue(len(ffmpeg_calls) > 0)
            cmd = ffmpeg_calls[0][0][0]
            cmd_str = " ".join(cmd)

            # Verify sidechain ducking filter
            self.assertIn("sidechaincompress", cmd_str)
            # Verify adelay filter for SFX
            self.assertIn("adelay=", cmd_str)
            # Verify amix
            self.assertIn("amix=", cmd_str)

            # Extract filter_complex argument and verify indices match inputs
            filter_idx = cmd.index("-filter_complex") + 1
            filter_str = cmd[filter_idx]

            # Count how many -i flags were provided in cmd
            input_count = cmd.count("-i")
            # Ensure no referenced audio input [N:a] exceeds the number of -i flags
            import re
            referenced_indices = [int(m) for m in re.findall(r"\[(\d+):a\]", filter_str)]
            for ref_idx in referenced_indices:
                self.assertLess(ref_idx, input_count, f"Input index [{ref_idx}:a] exceeds total input count {input_count}")

            if out_path.exists():
                out_path.unlink()

    def test_whisper_probe_uses_clip_local_framing_range(self):
        from unittest.mock import patch
        from src.viral_detector import ViralClipCandidate
        import main
        import tempfile

        moment = ViralClipCandidate(
            title="Probe clip",
            start_time=10.0,
            end_time=40.0,
            duration=30.0,
            viral_score=90,
            hook_reason="Test hook",
            social_caption="Test caption",
            hashtags=["#podcast"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clip_path = Path(tmpdir) / "probe.mp4"
            clip_path.write_bytes(b"video")
            probe_info = {"video_path": clip_path}
            with patch.object(main, "APIFY_API_TOKEN", "test-token"), \
                 patch.object(main, "CLIP_ONLY_MODE", True), \
                 patch.object(main, "fetch_transcript_only", return_value=None), \
                 patch.object(main, "download_clip_segment", side_effect=[probe_info, probe_info]), \
                 patch.object(main, "transcribe_audio_whisper", return_value=[TranscriptSegment(0.0, 30.0, "test", [])]), \
                 patch.object(main, "verify_audio_language", return_value="en"), \
                 patch.object(main, "detect_viral_moments", return_value=[moment]), \
                 patch.object(main, "analyze_faces_in_clip", return_value=FramingDecision(mode="blur_stack", face_count=0)) as analyze_faces, \
                 patch.object(main, "generate_clip_thumbnail", return_value=None), \
                 patch.object(main, "render_viral_clip", return_value=str(clip_path)), \
                 patch.object(main, "upload_clip_to_github_release", return_value=None), \
                 patch("src.release_cleaner.clean_old_releases"), \
                 patch.object(main, "record_history"), \
                 patch.object(main, "SlackNotifier"):
                main.run_pipeline(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    num_clips=1,
                    framing_mode="auto",
                    subtitles_mode="skip",
                )

            analyze_faces.assert_called_once_with(clip_path, 0.0, 30.0)

if __name__ == "__main__":
    unittest.main()
