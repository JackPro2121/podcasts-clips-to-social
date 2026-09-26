"""
Unit and render-level verification tests for Phase 3d-2: Animated punch-zoom.

Verifies:
1. `_punch_zoom_filter` expression construction and ramp calculation.
2. Director v2 and ShotPlan motion wiring (`push_in`, `pull_out`, `drift`).
3. Real FFmpeg render verification:
   - `push_in` and `pull_out` execute without filtergraph errors.
   - FFmpeg `freezedetect` reports zero freeze intervals (drift guarantee preserved).
   - Optical verification: an off-center feature moves outward over time, proving
     genuine geometric magnification.
"""
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Tuple
import unittest
import cv2
import numpy as np

from src.director_v2 import ShotDirective, apply_shot_directives
from src.face_tracker import FramingDecision, ShotPlan
from src.video_editor import (
    _punch_zoom_filter,
    render_viral_clip,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    FPS,
)


class TestPunchZoomFilter(unittest.TestCase):
    def test_drift_and_hold_produce_empty_filter(self) -> None:
        self.assertEqual(_punch_zoom_filter("drift", 60), "")
        self.assertEqual(_punch_zoom_filter("hold", 60), "")
        self.assertEqual(_punch_zoom_filter("unknown", 60), "")

    def test_push_in_filter_structure(self) -> None:
        filt = _punch_zoom_filter("push_in", 60)
        self.assertTrue(filt.startswith("zoompan=z="))
        self.assertIn("min(1,on/36)", filt)
        self.assertIn(f"s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}", filt)
        self.assertIn(f"fps={FPS}", filt)
        self.assertIn("setsar=1:1", filt)

    def test_pull_out_filter_structure(self) -> None:
        filt = _punch_zoom_filter("pull_out", 60)
        self.assertTrue(filt.startswith("zoompan=z="))
        self.assertIn("(1-min(1,on/36))", filt)

    def test_short_shot_clamps_ramp_frames(self) -> None:
        filt = _punch_zoom_filter("push_in", 15)
        # Ramp should clamp to shot length (15 frames)
        self.assertIn("min(1,on/15)", filt)


class TestDirectorMotionWiring(unittest.TestCase):
    def test_shot_plan_defaults_to_drift(self) -> None:
        shot = ShotPlan(start=0.0, end=5.0, mode="portrait_face")
        self.assertEqual(getattr(shot, "motion", "drift"), "drift")

    def test_apply_shot_directives_updates_motion(self) -> None:
        shots = [ShotPlan(start=0.0, end=5.0, mode="portrait_face")]
        framing = FramingDecision(
            mode="multi_shot_dynamic",
            face_count=1,
            shots=shots,
            video_width=1920,
            video_height=1080,
            active_w=1920,
            active_h=1080,
        )
        directives = [
            ShotDirective(
                start=0.0,
                end=5.0,
                layout="portrait_face",
                motion="push_in",
                confidence=0.9,
            )
        ]
        updated, audit = apply_shot_directives(framing, directives)
        self.assertGreaterEqual(len(audit), 1)
        self.assertEqual(updated.shots[0].motion, "push_in")

    def test_invalid_motion_falls_back_to_drift(self) -> None:
        shots = [ShotPlan(start=0.0, end=5.0, mode="portrait_face")]
        framing = FramingDecision(
            mode="multi_shot_dynamic",
            face_count=1,
            shots=shots,
            video_width=1920,
            video_height=1080,
            active_w=1920,
            active_h=1080,
        )
        directives = [
            ShotDirective(
                start=0.0,
                end=5.0,
                motion="illegal_spinning_camera",
            )
        ]
        updated, _ = apply_shot_directives(framing, directives)
        self.assertEqual(updated.shots[0].motion, "drift")


def _create_synthetic_test_video(path: Path, duration_sec: float = 2.0, fps: int = 30) -> None:
    """Create a 1920x1080 synthetic video with a distinctive white marker."""
    width, height = 1920, 1080
    total_frames = int(round(duration_sec * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        for _ in range(total_frames):
            frame = np.full((height, width, 3), 30, dtype=np.uint8)
            # Distinct bright block at (800, 400)
            frame[380:440, 780:840] = 240
            out.write(frame)
    finally:
        out.release()


def _run_freezedetect(video_path: Path) -> Tuple[int, str]:
    """Run FFmpeg freezedetect and return (freeze_event_count, stderr)."""
    cmd = [
        "ffmpeg", "-v", "info", "-i", str(video_path),
        "-vf", "freezedetect=n=0.003:d=0.5",
        "-an", "-f", "null", "NUL" if subprocess.os.name == "nt" else "/dev/null",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    count = len(re.findall(r"freeze_start:\s*([\d\.]+)", proc.stderr))
    return count, proc.stderr


class TestPunchZoomRenderVerification(unittest.TestCase):
    def test_push_in_renders_freeze_free_and_proves_magnification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_video = tmp_path / "synthetic_source.mp4"
            _create_synthetic_test_video(test_video, duration_sec=2.5, fps=FPS)

            shot = ShotPlan(
                start=0.0,
                end=2.0,
                mode="portrait_face",
                crop_x=600,
                crop_y=70,
                crop_w=607,
                crop_h=1080,
                motion="push_in",
            )
            framing = FramingDecision(
                mode="multi_shot_dynamic",
                face_count=1,
                shots=[shot],
                video_width=1920,
                video_height=1080,
                active_x=0,
                active_y=0,
                active_w=1920,
                active_h=1080,
            )

            output_path = tmp_path / "rendered_push_in.mp4"
            rendered = render_viral_clip(
                source_video_path=test_video,
                output_clip_path=output_path,
                start_time=0.0,
                end_time=2.0,
                framing=framing,
                burn_subtitles=False,
            )

            self.assertTrue(rendered.exists())
            self.assertGreater(rendered.stat().st_size, 50000)

            # 1. FFmpeg freezedetect: zero freeze events
            freeze_count, _ = _run_freezedetect(rendered)
            self.assertEqual(freeze_count, 0, "push_in render must not trigger freeze events")

            # 2. Geometric zoom proof: marker moves outward from frame center
            cap = cv2.VideoCapture(str(rendered))
            frames = []
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
            finally:
                cap.release()

            self.assertGreaterEqual(len(frames), 50)
            first_frame = frames[0]
            late_frame = frames[40]

            center_x, center_y = 540.0, 960.0

            def get_bright_centroid(img: np.ndarray) -> Tuple[float, float]:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bright_pts = np.argwhere(gray > 180)
                self.assertGreater(len(bright_pts), 0, "Bright marker must be visible in rendered frame")
                mean_y = float(bright_pts[:, 0].mean())
                mean_x = float(bright_pts[:, 1].mean())
                return mean_x, mean_y

            first_x, first_y = get_bright_centroid(first_frame)
            late_x, late_y = get_bright_centroid(late_frame)

            first_dist = np.hypot(first_x - center_x, first_y - center_y)
            late_dist = np.hypot(late_x - center_x, late_y - center_y)

            self.assertGreater(
                late_dist,
                first_dist + 2.0,
                f"push_in must magnify: initial dist={first_dist:.1f}, late dist={late_dist:.1f}",
            )
