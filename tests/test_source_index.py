import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.source_index import (
    SourceMediaInfo,
    build_source_index,
    probe_source_media,
    save_source_index,
)


class TestSourceIndex(unittest.TestCase):
    def test_probe_source_media_parses_fractional_fps(self):
        payload = {
            "format": {"duration": "12.5"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1080,
                    "height": 1920,
                    "avg_frame_rate": "30000/1001",
                    "pix_fmt": "yuv420p",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "channels": 2,
                },
            ],
        }
        result = MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch("src.source_index.subprocess.run", return_value=result):
            media = probe_source_media(Path("clip.mp4"))
        self.assertEqual(media.width, 1080)
        self.assertEqual(media.height, 1920)
        self.assertAlmostEqual(media.fps, 29.97002997, places=4)
        self.assertEqual(media.sample_rate, 48000)

    def test_build_source_index_returns_shots_quality_and_serializes(self):
        frame_a = np.zeros((100, 200, 3), dtype=np.uint8)
        frame_b = np.full((100, 200, 3), 255, dtype=np.uint8)
        frame_c = np.full((100, 200, 3), 128, dtype=np.uint8)
        samples = [(0.0, frame_a), (1.0, frame_b), (2.0, frame_b), (3.0, frame_c)]
        media = SourceMediaInfo(
            path="clip.mp4",
            duration=4.0,
            width=200,
            height=100,
            fps=4.0,
            video_codec="h264",
            audio_codec="aac",
            sample_rate=48000,
            channels=2,
        )
        with patch("src.source_index.probe_source_media", return_value=media), \
             patch("src.source_index._sample_frames", return_value=samples), \
             patch("src.source_index.detect_clip_shots", return_value=[(0.0, 1.5), (1.5, 4.0)]), \
             patch("src.source_index.detect_slide_heuristics", return_value=False), \
             patch("src.source_index._face_count", return_value=0.0), \
             patch("src.source_index._audio_events", return_value=[]):
            index = build_source_index(Path("clip.mp4"), sample_fps=2.0)
        self.assertEqual(len(index.shots), 2)
        self.assertEqual(index.quality["sample_count"], 4)
        self.assertTrue(index.shots[0].motion_score > 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "index.json"
            save_source_index(index, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "0.1")
            self.assertEqual(len(payload["shots"]), 2)
            self.assertEqual(payload["media"]["height"], 100)

    def test_source_index_quality_reports_freeze_intervals(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        samples = [(0.0, frame), (1.0, frame.copy()), (2.0, frame.copy())]
        media = SourceMediaInfo("clip.mp4", 3.0, 200, 100, 1.0)
        with patch("src.source_index.probe_source_media", return_value=media), \
             patch("src.source_index._sample_frames", return_value=samples), \
             patch("src.source_index.detect_clip_shots", return_value=[(0.0, 3.0)]), \
             patch("src.source_index.detect_slide_heuristics", return_value=False), \
             patch("src.source_index._face_count", return_value=0.0), \
             patch("src.source_index._audio_events", return_value=[]):
            index = build_source_index(Path("clip.mp4"), sample_fps=1.0)
        self.assertGreaterEqual(index.quality["freeze_interval_count"], 1)
        self.assertEqual(index.quality["freeze_duration"], 2.0)


if __name__ == "__main__":
    unittest.main()
