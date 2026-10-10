import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.streamers.discovery import (
    STREAMER_CHANNELS, extract_video_id_from_url, load_streamer_history,
    record_streamer_history
)
from src.streamers.viral_detector import (
    parse_streamer_clips_json, _heuristic_streamer_detector
)
from src.transcriber import TranscriptSegment


class TestStreamerPipeline(unittest.TestCase):
    def test_streamer_channels_configured(self):
        """Top US streamers should be registered with channel URLs and keywords."""
        expected = ["kai_cenat", "ishowspeed", "jynxzi", "caseoh", "xqc", "adin_ross"]
        for streamer in expected:
            self.assertIn(streamer, STREAMER_CHANNELS)
            self.assertTrue(len(STREAMER_CHANNELS[streamer]["channel_urls"]) > 0)
            self.assertTrue(len(STREAMER_CHANNELS[streamer]["keywords"]) > 0)

    def test_extract_video_id(self):
        """URL parser should extract standard 11-char YouTube IDs."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(extract_video_id_from_url(url), "dQw4w9WgXcQ")
        live_url = "https://www.youtube.com/live/12345678901"
        self.assertEqual(extract_video_id_from_url(live_url), "12345678901")

    def test_streamer_history_isolation(self):
        """Streamer history should persist without touching main history file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_hist = Path(tmpdir) / "streamer_history.json"
            with patch("src.streamers.discovery.STREAMER_HISTORY_FILE", test_hist):
                record_streamer_history("test_video_123", "Test Title", "Kai Cenat")
                history = load_streamer_history()
                self.assertIn("test_video_123", history)

    def test_parse_streamer_clips_json(self):
        """JSON parser should extract candidates and format titles in uppercase without emojis."""
        raw_json = json.dumps({
            "clips": [
                {
                    "title": "🔥 KAI GOES CRAZY ON STREAM!!",
                    "start_time": 10.0,
                    "end_time": 40.0,
                    "duration": 30.0,
                    "viral_score": 95,
                    "hook_reason": "High emotional spike",
                    "social_caption": "Bro was stunned. #shorts",
                    "hashtags": ["#streamer", "#twitch"],
                    "streamer_energy": "rage",
                    "sfx_cues": [[0.1, "whoosh"], [28.0, "ding"]]
                }
            ]
        })
        dummy_segments = [TranscriptSegment(start=0.0, end=60.0, text="hello world bro what are you doing", words=[])]
        parsed = parse_streamer_clips_json(raw_json, dummy_segments, num_clips=1)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].title, "KAI GOES CRAZY ON STREAM")
        self.assertEqual(parsed[0].streamer_energy, "rage")
        self.assertEqual(len(parsed[0].sfx_cues), 2)

    def test_heuristic_streamer_detector(self):
        """Heuristic detector should prioritize shouting and exclamation marks."""
        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="normal quiet conversation right here.", words=[]),
            TranscriptSegment(start=15.0, end=45.0, text="BRO WHAT ARE YOU DOING?! NO WAY! CHAT HE IS COOKED! LET'S GOOO!", words=[]),
            TranscriptSegment(start=50.0, end=60.0, text="okay back to regular chat.", words=[]),
        ]
        candidates = _heuristic_streamer_detector(segments, num_clips=1)
        self.assertTrue(len(candidates) >= 1)
        self.assertGreaterEqual(candidates[0].viral_score, 70)


if __name__ == "__main__":
    unittest.main()
