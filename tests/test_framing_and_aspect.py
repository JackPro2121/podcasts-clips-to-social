import unittest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from src.face_tracker import (
    ShotPlan, FramingDecision,
    detect_letterbox_margins, analyze_faces_in_clip
)
from src.config import HOOK_BADGE_MARGIN_V
from src.video_editor import build_video_filtergraph

class TestFramingAndAspect(unittest.TestCase):
    def test_frame_window_uses_exclusive_fractional_boundaries(self):
        from src.face_tracker import _frame_window

        self.assertEqual(_frame_window(0.05, 1.05, 10.0), (1, 11))

    def test_relative_frame_timestamp_prefers_pts(self):
        from unittest.mock import MagicMock
        from src.face_tracker import _relative_frame_timestamp

        cap = MagicMock()
        cap.get.return_value = 1050.0
        self.assertAlmostEqual(
            _relative_frame_timestamp(cap, 0.5, 6, 5, 10.0, 0.55),
            0.55,
        )
        cap.get.return_value = 0.0
        self.assertAlmostEqual(
            _relative_frame_timestamp(cap, 0.5, 6, 5, 10.0, 0.55),
            0.1,
        )

    def test_model_download_is_atomic(self):
        import tempfile
        from unittest.mock import MagicMock, patch
        from src.face_tracker import _download_model_atomically

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "model.onnx"
            response = MagicMock(status_code=200, content=b"x" * 200000)
            with patch("src.face_tracker.requests.get", return_value=response):
                result = _download_model_atomically("https://example.com/model.onnx", destination, 100000)
            self.assertTrue(result)
            self.assertTrue(destination.exists())
            self.assertEqual(list(destination.parent.glob("*.part")), [])

    def test_scene_analysis_releases_capture_on_empty_read(self):
        from unittest.mock import MagicMock, patch
        from src.scene_classifier import analyze_clip_presentation_ratio

        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25.0
        capture.read.return_value = (False, None)
        with patch("src.scene_classifier.cv2.VideoCapture", return_value=capture):
            result = analyze_clip_presentation_ratio(Path("unused.mp4"), 0.0, 1.0)
        self.assertEqual(result, 0.0)
        capture.release.assert_called_once()

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
        self.assertIn("force_original_aspect_ratio=decrease", fg)
        self.assertIn("pad=1080:960:(ow-iw)/2:(oh-ih)/2:color=black", fg)

    def test_single_dynamic_shot_preserves_shot_plan(self):
        from unittest.mock import patch
        from src.face_tracker import FaceBox

        temp_dir = tempfile.mkdtemp()
        vid_path = Path(temp_dir) / "test_single_shot.mp4"
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (320, 240))
        for _ in range(10):
            writer.write(np.ones((240, 320, 3), dtype=np.uint8) * 128)
        writer.release()

        face = FaceBox(x=140, y=70, w=40, h=40, center_x=160, center_y=90)
        with patch("src.face_tracker.detect_clip_shots", return_value=[(0.0, 1.0)]), \
             patch("src.face_tracker.get_face_detector", return_value=object()), \
             patch("src.face_tracker.detect_faces_in_frame", return_value=[face]), \
             patch("src.face_tracker.classify_frame_scene", return_value={"is_presentation": False}):
            decision = analyze_faces_in_clip(vid_path, 0.0, 1.0, sample_fps=2.5)

        if vid_path.exists():
            vid_path.unlink()
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

        self.assertEqual(decision.mode, "multi_shot_dynamic")
        self.assertEqual(len(decision.shots), 1)
        self.assertGreater(decision.shots[0].zoom_factor, 1.01)

    def test_dominant_speaker_branches_return_portrait_shots(self):
        from unittest.mock import patch
        from src.face_tracker import FaceBox

        temp_dir = tempfile.mkdtemp()
        vid_path = Path(temp_dir) / "test_dominant_speaker.mp4"
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (320, 240))
        for _ in range(10):
            writer.write(np.ones((240, 320, 3), dtype=np.uint8) * 128)
        writer.release()
        faces = [
            FaceBox(x=40, y=50, w=40, h=40, center_x=60, center_y=70),
            FaceBox(x=240, y=50, w=40, h=40, center_x=260, center_y=70),
        ]

        for dominant_index in (0, 1):
            motion_values = [
                (20.0, None) if index == dominant_index else (0.0, None)
                for index in range(6)
            ]
            with patch("src.face_tracker.detect_clip_shots", return_value=[(0.0, 1.0)]), \
                 patch("src.face_tracker.get_face_detector", return_value=object()), \
                 patch("src.face_tracker.detect_faces_in_frame", return_value=faces), \
                 patch("src.face_tracker.classify_frame_scene", return_value={"is_presentation": False}), \
                 patch("src.face_tracker._estimate_mouth_motion", side_effect=motion_values):
                decision = analyze_faces_in_clip(vid_path, 0.0, 1.0, sample_fps=2.5)
            self.assertIn(decision.mode, ("single_smooth", "multi_shot_dynamic"))
            if decision.mode == "multi_shot_dynamic":
                self.assertEqual(len(decision.shots), 1)
                self.assertEqual(decision.shots[0].mode, "portrait_face")

        if vid_path.exists():
            vid_path.unlink()
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

    def test_manual_framing_uses_probed_dimensions(self):
        from unittest.mock import patch
        import main

        with patch("main.get_video_dimensions", return_value=(321, 241)):
            split = main._manual_framing(Path("unused.mp4"), "split")
            crop = main._manual_framing(Path("unused.mp4"), "crop")
            blur = main._manual_framing(Path("unused.mp4"), "blur_stack")

        self.assertEqual((split.video_width, split.video_height), (321, 241))
        self.assertEqual((split.active_x, split.active_y, split.active_w, split.active_h), (0, 0, 321, 241))
        self.assertEqual(split.speaker1_box, (25, 0, 135, 120))
        self.assertEqual(split.speaker2_box, (160, 0, 135, 120))
        self.assertEqual((crop.active_w, crop.active_h), (321, 241))
        self.assertEqual((blur.active_w, blur.active_h), (321, 241))

    def test_filtergraph_clamps_odd_low_resolution_crops(self):
        decision = FramingDecision(
            mode="single_smooth",
            face_count=0,
            video_width=321,
            video_height=241,
            active_x=0,
            active_y=0,
            active_w=999,
            active_h=999,
            smoothed_center_x=160,
        )
        fg = build_video_filtergraph(decision, burn_subtitles=False)
        self.assertIn("crop=134:240:93:0", fg)

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
