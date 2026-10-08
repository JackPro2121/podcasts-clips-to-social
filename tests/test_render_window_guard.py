"""The render window must live inside the segment file.

Run 37748325314 lost every hooked clip to "Rendered output duration is outside
the allowed tolerance": the hook lead-in started the window before the file,
the video came out short by exactly the missing lead-in, and the run aborted
with "No clips were rendered". The download now carries pre-roll
(HOOK_PREROLL_S) and the call-site clamps the lead-in to the segment head; the
renderer additionally refuses a window that starts before the file with an
error that names the real cause instead of failing later on a duration check.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from src.face_tracker import FramingDecision
from src.hook_engine import HOOK_MAX_LEAD_IN_S, HOOK_PREROLL_S
from src.video_editor import render_viral_clip


def _framing() -> FramingDecision:
    return FramingDecision(mode="single_smooth", face_count=0)


def _call(tmp_path: Path, start_time: float, end_time: float) -> None:
    render_viral_clip(
        source_video_path=tmp_path / "segment.mp4",
        start_time=start_time,
        end_time=end_time,
        output_clip_path=tmp_path / "out.mp4",
        framing=_framing(),
        burn_subtitles=False,
    )


class TestNegativeRenderWindow(unittest.TestCase):
    def test_window_before_the_file_is_rejected_with_the_real_cause(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                _call(Path(tmp), start_time=-1.5, end_time=40.0)
        self.assertIn("before the segment file", str(ctx.exception))
        self.assertIn("pre-roll", str(ctx.exception))

    def test_tiny_float_noise_does_not_trip_the_guard(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                # Only the negative guard can fire for a missing file that
                # starts at ~0: any other failure must not be this one.
                _call(Path(tmp), start_time=-0.0005, end_time=40.0)
        self.assertNotIn("before the segment file", str(ctx.exception))


class TestPreRollGeometry(unittest.TestCase):
    def test_pre_roll_always_covers_the_max_lead_in(self):
        # The download head-room is what makes the lead-in real; if this ever
        # stops being true the hook silently re-breaks the render window.
        self.assertGreater(HOOK_PREROLL_S, HOOK_MAX_LEAD_IN_S)

    def test_moment_offset_with_pre_roll_keeps_the_window_positive(self):
        # The production geometry: the file starts at the integer second
        # before the pre-roll; segment_start = moment offset in the file.
        moment_start = 740.19
        download_start = moment_start - HOOK_PREROLL_S
        int_start = float(int(download_start))
        segment_start = moment_start - int_start
        lead_used = min(HOOK_MAX_LEAD_IN_S, segment_start)
        render_start = segment_start - lead_used
        self.assertGreaterEqual(render_start, 0.0)
        # ...and the file still covers the full render window with the lead.
        file_seconds = 790.0 - int_start
        clip_duration = (789.84 - moment_start) + lead_used
        self.assertLessEqual(render_start + clip_duration, file_seconds + 0.001)


if __name__ == "__main__":
    unittest.main()
