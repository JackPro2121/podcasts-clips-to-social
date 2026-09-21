import unittest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from src.face_tracker import (
    FaceBox, ShotPlan, FramingDecision,
    detect_letterbox_margins, analyze_faces_in_clip
)
from src.config import HOOK_BADGE_MARGIN_V
from src.video_editor import build_video_filtergraph

class TestFramingAndAspect(unittest.TestCase):
    def test_hook_badge_safe_zone(self):
        """Hook badge margin must sit at upper safe zone (<= 140px) above speaker face."""
        self.assertLessEqual(HOOK_BADGE_MARGIN_V, 140)
        self.assertGreaterEqual(HOOK_BADGE_MARGIN_V, 90)

    def test_letterbox_margins_detection(self):
        """Synthetic video with 100px top and 100px bottom black bars must be detected."""
        temp_dir = tempfile.mkdtemp()
        vid_path = Path(temp_dir) / "test_letterbox.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 360))
        
        # Frame with top 60px black, bottom 60px black, middle white
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[60:300, :] = 200  # Active content
        
        for _ in range(15):
            out.write(frame)
        out.release()

        cap = cv2.VideoCapture(str(vid_path))
        x, y, w, h = detect_letterbox_margins(cap, 0, 15, max_samples=5)
        cap.release()
        
        if vid_path.exists():
            vid_path.unlink()
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

        self.assertEqual(y, 60)
        self.assertEqual(h, 240)
        self.assertEqual(w, 640)

    def test_split_screen_aspect_ratio_preservation(self):
        """Split screen bounding boxes must maintain 9:8 aspect ratio."""
        s1 = ShotPlan(
            start=0.0, end=5.0, mode='split_screen',
            speaker1_box=(100, 50, 900, 800),
            speaker2_box=(1000, 50, 900, 800),
            margin_v=0
        )
        decision = FramingDecision(
            mode='multi_shot_dynamic', face_count=2, shots=[s1],
            active_x=0, active_y=50, active_w=1920, active_h=800
        )
        fg = build_video_filtergraph(decision, burn_subtitles=False)
        self.assertIn("vstack=inputs=2", fg)
        self.assertIn("crop=", fg)

    def test_temporal_speaker_anchor_no_dead_center_jump(self):
        """When face detection is lost, speaker position must not blindly reset to width // 2 (960)."""
        temp_dir = tempfile.mkdtemp()
        vid_path = Path(temp_dir) / "test_blank.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (1920, 1080))
        # Plain frames with no faces
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        for _ in range(20):
            out.write(frame)
        out.release()

        decision = analyze_faces_in_clip(vid_path, 0.0, 2.0, sample_fps=2.0)
        
        if vid_path.exists():
            vid_path.unlink()
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

        self.assertIsNotNone(decision)
        self.assertEqual(decision.active_y, 0)
        self.assertEqual(decision.active_h, 1080)

if __name__ == '__main__':
    unittest.main()
