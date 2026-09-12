import unittest
from pathlib import Path
import json
import tempfile
from src.slack_notifier import SlackNotifier
from src.channel_discovery import extract_youtube_id, load_history, record_history
from src.transcriber import TranscriptSegment, WordTimestamp
from src.subtitle_generator import create_styled_ass_subtitles

class TestSlackAndHistory(unittest.TestCase):
    def test_extract_youtube_id(self):
        self.assertEqual(extract_youtube_id("https://www.youtube.com/watch?v=pBs9xdfeIxM"), "pBs9xdfeIxM")
        self.assertEqual(extract_youtube_id("https://youtu.be/pBs9xdfeIxM"), "pBs9xdfeIxM")
        self.assertEqual(extract_youtube_id("https://www.youtube.com/shorts/pBs9xdfeIxM"), "pBs9xdfeIxM")
        self.assertEqual(extract_youtube_id("pBs9xdfeIxM"), "pBs9xdfeIxM")

    def test_history_load_and_record(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_history_path = Path(tf.name)
        try:
            # Initially empty
            hist = load_history(temp_history_path)
            self.assertEqual(hist, {})

            # Record an episode
            record_history(
                video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                title="Test Longevity Episode",
                niche="health_longevity",
                history_file_path=temp_history_path
            )

            updated = load_history(temp_history_path)
            self.assertIn("dQw4w9WgXcQ", updated)
            self.assertEqual(updated["dQw4w9WgXcQ"]["title"], "Test Longevity Episode")
            self.assertEqual(updated["dQw4w9WgXcQ"]["niche"], "health_longevity")
        finally:
            if temp_history_path.exists():
                temp_history_path.unlink()

    def test_slack_notifier_blocks(self):
        notifier = SlackNotifier(webhook_url="", bot_token="")
        self.assertFalse(notifier.is_enabled())
        # Should gracefully return False when webhook_url is not set
        res = notifier.send_run_report(
            podcast_title="Test Podcast",
            podcast_url="https://www.youtube.com/watch?v=test1234567",
            niche="finance",
            clips=[{
                "title": "Clip 1 Hook",
                "virality_score": 95,
                "duration": 42.5,
                "download_url": "https://github.com/.../clip1.mp4",
                "buffer_status": "Scheduled"
            }]
        )
        self.assertFalse(res)

    def test_subtitle_top_header(self):
        segments = [
            TranscriptSegment(
                text="The secret to longevity is regular zone 2 cardio",
                start=0.0,
                end=4.0,
                words=[
                    WordTimestamp(word="The", start=0.0, end=0.5),
                    WordTimestamp(word="secret", start=0.5, end=1.0),
                    WordTimestamp(word="to", start=1.0, end=1.5),
                    WordTimestamp(word="longevity", start=1.5, end=2.5)
                ]
            )
        ]
        out_ass = Path("tests/test_hook_output.ass")
        try:
            create_styled_ass_subtitles(
                segments=segments,
                clip_start=0.0,
                clip_end=4.0,
                output_ass_path=out_ass,
                theme_key="hormozi",
                header_title="WHY WE AGE FASTER"
            )
            content = out_ass.read_text(encoding="utf-8")
            self.assertIn("TopHeader", content)
            self.assertIn("&H00B86B62", content)
            self.assertIn("WHY WE\\NAGE FASTER", content)
        finally:
            if out_ass.exists():
                out_ass.unlink()

if __name__ == "__main__":
    unittest.main()
