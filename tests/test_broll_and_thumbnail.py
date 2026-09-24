import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.thumbnail_generator import generate_clip_thumbnail
from src.broll_manager import search_pexels_broll, select_best_video_file, find_broll_cues_for_clip
from src.transcriber import WordTimestamp

class TestBrollAndThumbnail(unittest.TestCase):
    def test_thumbnail_generation_fallback_canvas(self):
        """Thumbnail generator should produce a valid JPEG even if video file is dummy."""
        out_path = Path("clips/test_dummy_thumb.jpg")
        try:
            res = generate_clip_thumbnail(
                clip_path=Path("clips/non_existent_clip.mp4"),
                title="TEST VIRAL HOOK TITLE",
                output_path=out_path,
                badge_text="MUST WATCH",
                watermark="@testchannel"
            )
            self.assertIsNotNone(res)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)
        finally:
            if out_path.exists():
                out_path.unlink()

    def test_broll_select_best_video_file(self):
        """select_best_video_file selects the optimal HD MP4 stream."""
        mock_entry = {
            "video_files": [
                {"file_type": "video/mp4", "width": 540, "height": 960, "link": "https://example.com/sd.mp4"},
                {"file_type": "video/mp4", "width": 1080, "height": 1920, "link": "https://example.com/hd.mp4"},
            ]
        }
        best = select_best_video_file(mock_entry)
        self.assertEqual(best, "https://example.com/hd.mp4")

    @patch("src.broll_manager.search_pexels_broll")
    @patch("src.broll_manager.download_broll_clip")
    def test_find_broll_cues_for_clip(self, mock_download, mock_search):
        """B-roll cues should match finance keywords within safe time bounds."""
        mock_search.return_value = [{"video_files": [{"file_type": "video/mp4", "height": 1920, "width": 1080, "link": "https://example.com/test.mp4"}]}]
        mock_download.return_value = Path("assets/broll/dummy.mp4")

        words = [
            WordTimestamp(word="hello", start=1.0, end=1.5),
            WordTimestamp(word="we", start=2.0, end=2.5),
            WordTimestamp(word="talk", start=3.0, end=3.5),
            WordTimestamp(word="about", start=4.0, end=4.5),
            WordTimestamp(word="money", start=6.0, end=6.5),
            WordTimestamp(word="today", start=7.0, end=7.5)
        ]
        with patch.object(Path, "exists", return_value=True):
            cues = find_broll_cues_for_clip(words, clip_duration=30.0, max_brolls=1)
            self.assertEqual(len(cues), 1)
            self.assertEqual(cues[0][3], "money")
            self.assertEqual(cues[0][0], 6.0)

    def test_video_filtergraph_with_cover_overlay(self):
        """Filtergraph should properly overlay 0.25s cover thumbnail frame."""
        from src.video_editor import build_video_filtergraph
        from src.face_tracker import FramingDecision

        framing = FramingDecision(mode="single_smooth", face_count=1, smoothed_center_x=960)
        fg = build_video_filtergraph(
            framing=framing,
            cover_input_idx=1,
            cover_duration=0.25,
            burn_subtitles=False
        )
        self.assertIn("[1:v]scale=1080:1920", fg)
        self.assertIn("overlay=enable='between(t,0,0.25)'[outv]", fg)

if __name__ == "__main__":
    unittest.main()
