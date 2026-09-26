"""Regression coverage for the static-shot motion guarantee.

Background: CI run 36171365900 rendered a 96s clip that editorial_qa rejected with
`freeze_interval` (severity error), which discarded the only clip and aborted the
run with `RuntimeError: No clips were rendered`. Root cause: a `portrait_face` shot
whose face timeline could not be resolved into a moving crop was emitted as a fixed
integer crop box (`crop=526:938:494:39`) with no `t` term, so statically-held source
content produced a literally frozen output.

`test_static_portrait_shot_emits_time_varying_crop` fails against that old behaviour.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.face_tracker import FramingDecision, ShotPlan
from src.video_editor import build_video_filtergraph

# Imported lazily so that the behavioural regression tests below still run (and
# fail on assertions) against a build of video_editor.py that predates the motion
# guarantee, instead of erroring out during collection.
_drift_axis_expr = None
_crop_position = None
_STATIC_SHOT_DRIFT_RATIO = None

SOURCE_W, SOURCE_H = 1920, 1080


def _load_drift_helpers():
    global _drift_axis_expr, _crop_position, _STATIC_SHOT_DRIFT_RATIO
    from src import video_editor
    _drift_axis_expr = getattr(video_editor, "_drift_axis_expr", None)
    _crop_position = getattr(video_editor, "_crop_position", None)
    _STATIC_SHOT_DRIFT_RATIO = getattr(video_editor, "_STATIC_SHOT_DRIFT_RATIO", None)


def _shot_crop_segment(shot: ShotPlan, active_w: int = SOURCE_W, active_h: int = SOURCE_H) -> str:
    """The crop filter emitted for a single-shot multi_shot_dynamic decision."""
    decision = FramingDecision(
        mode="multi_shot_dynamic",
        face_count=1,
        shots=[shot],
        video_width=SOURCE_W,
        video_height=SOURCE_H,
        active_x=0,
        active_y=0,
        active_w=active_w,
        active_h=active_h,
    )
    graph = build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)
    segments = [part for part in graph.split(";") if "crop=" in part and "v_shot_0" in part]
    assert len(segments) == 1, f"expected exactly one shot filter, got {segments}"
    return segments[0]


_EXPR_RE = re.compile(
    r"^max\((?P<low>-?\d+),min\((?P<centre>-?\d+)\+(?P<amp>\d+)\*"
    r"0\.6366\*asin\(sin\(t\*(?P<freq>[\d.]+)\)\),(?P<high>-?\d+)\)\)$"
)


def _parse_drift(expr):
    """Turn an emitted drift expression back into (low, centre, amplitude, high)."""
    match = _EXPR_RE.match(expr)
    assert match, f"unparseable drift expression: {expr}"
    parts = match.groupdict()
    return (int(parts["low"]), int(parts["centre"]),
            int(parts["amp"]), int(parts["high"]))


class TestDriftAxisExpression(unittest.TestCase):
    """Property tests on the motion guarantee, not string snapshots."""

    @classmethod
    def setUpClass(cls):
        _load_drift_helpers()
        if _drift_axis_expr is None:
            raise unittest.SkipTest("video_editor has no motion-guarantee helpers yet")

    def test_clamp_never_binds_so_the_wave_cannot_flatten(self):
        # A clamp that binds would flatten part of the wave into a static hold,
        # which is the freeze this whole change exists to prevent.
        for extent in (600, 606, 526, 1314):
            for low, high in ((0, 1000), (0, 600), (100, 1400), (0, 40)):
                for base in range(low, high + 1, max(1, (high - low) // 17)):
                    expr = _drift_axis_expr(base, low, high, extent)
                    if expr is None:
                        continue
                    lo, centre, amp, hi = _parse_drift(expr)
                    self.assertGreaterEqual(centre - amp, lo,
                                            f"clamp binds on the low side: {expr}")
                    self.assertLessEqual(centre + amp, hi,
                                         f"clamp binds on the high side: {expr}")

    def test_every_base_with_slack_gets_motion(self):
        # Regression guard for the frame-edge case: a subject pinned to x=0 or
        # x=high previously produced a zero-amplitude (frozen) crop.
        for base in (0, 1, 50, 500, 999, 1000):
            self.assertIsNotNone(
                _drift_axis_expr(base, 0, 1000, 600),
                f"base={base} lost its motion guarantee",
            )

    def test_sweep_is_capped_to_the_crop_extent(self):
        for extent in (600, 526, 2000):
            max_amp = int(extent * _STATIC_SHOT_DRIFT_RATIO)
            for base in (0, 300, 700, 1000):
                expr = _drift_axis_expr(base, 0, 1000, extent)
                if expr is None:
                    continue
                _, _, amp, _ = _parse_drift(expr)
                self.assertLessEqual(amp, max_amp,
                                     f"sweep {amp}px exceeds the {max_amp}px cap")

    def test_centre_never_moves_far_from_the_computed_crop(self):
        extent, ratio = 600, _STATIC_SHOT_DRIFT_RATIO
        for base in (0, 1, 250, 500, 750, 1000):
            expr = _drift_axis_expr(base, 0, 1000, extent)
            if expr is None:
                continue
            _, centre, _, _ = _parse_drift(expr)
            self.assertLessEqual(abs(centre - base), int(extent * ratio),
                                 "framing shifted too far from the tracked subject")

    def test_wave_is_a_triangle_not_a_sine(self):
        # A sine's velocity hits zero at every extremum, which reintroduced
        # sub-threshold runs long enough for freezedetect to report them.
        expr = _drift_axis_expr(500, 0, 1000, 600)
        self.assertIsNotNone(expr)
        self.assertIn("0.6366*asin(sin(t*1.8))", expr)

    def test_returns_none_only_when_the_axis_really_cannot_move(self):
        # high <= low means the crop already spans the whole active region.
        self.assertIsNone(_drift_axis_expr(0, 0, 0, 600))
        self.assertIsNone(_drift_axis_expr(100, 100, 100, 600))
        # a non-positive extent cannot yield a meaningful sweep
        self.assertIsNone(_drift_axis_expr(500, 0, 1000, 0))
        self.assertIsNone(_drift_axis_expr(500, 0, 1000, -10))
        # a crop pinned to the high bound can still drift left, so it keeps motion
        self.assertIsNotNone(_drift_axis_expr(100, 0, 100, 600))

    def test_returns_none_for_out_of_range_base(self):
        self.assertIsNone(_drift_axis_expr(-5, 0, 1000, 600))
        self.assertIsNone(_drift_axis_expr(1200, 0, 1000, 600))

    def test_crop_position_only_quotes_real_expressions(self):
        self.assertEqual("494", _crop_position(494, None))
        self.assertEqual("'max(0,min(1,2))'", _crop_position(494, "max(0,min(1,2))"))


class TestStaticShotMotionGuarantee(unittest.TestCase):
    def test_static_portrait_shot_emits_time_varying_crop(self):
        """REGRESSION: old code emitted a fixed `crop=526:938:494:39` here."""
        segment = _shot_crop_segment(
            ShotPlan(start=0.0, end=2.5, mode="portrait_face", crop_x=454,
                     margin_v=460, zoom_factor=1.15, face_centers_timeline=[])
        )
        self.assertIn("sin(t*1.8)", segment)
        # crop dimensions must stay fixed while position becomes time-varying
        self.assertIn("crop=526:938:", segment)
        self.assertNotIn("crop=526:938:494:39", segment)

    def test_static_portrait_shot_without_zoom_also_drifts(self):
        segment = _shot_crop_segment(
            ShotPlan(start=0.0, end=3.0, mode="portrait_face", crop_x=441,
                     margin_v=460, zoom_factor=1.0, face_centers_timeline=[])
        )
        self.assertIn("sin(t*1.8)", segment)

    def test_drift_clamps_inside_active_region(self):
        segment = _shot_crop_segment(
            ShotPlan(start=0.0, end=3.0, mode="portrait_face", crop_x=0,
                     margin_v=460, zoom_factor=1.0, face_centers_timeline=[])
        )
        # the drift must be wrapped in a clamp against the active-region bounds
        self.assertIn("max(0,min(", segment)
        self.assertIn("sin(t*1.8)", segment)

    def test_resolved_face_panning_branch_is_unchanged(self):
        """Shots that already had motion must keep their timeline-driven crop."""
        decision = FramingDecision(
            mode="multi_shot_dynamic",
            face_count=1,
            video_width=SOURCE_W, video_height=SOURCE_H,
            active_x=0, active_y=0, active_w=SOURCE_W, active_h=SOURCE_H,
            shots=[ShotPlan(start=0.0, end=6.0, mode="portrait_face", crop_x=400,
                            margin_v=460, zoom_factor=1.0,
                            face_centers_timeline=[(0.0, 300), (6.0, 900)])],
        )
        graph = build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)
        self.assertIn("min(1.0,max(0.0,t/6.00))", graph)
        self.assertNotIn("asin(sin(t*1.8))", graph)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg/ffprobe required for the render-level freeze assertion")
class TestRenderedShotIsNeverFrozen(unittest.TestCase):
    """Render-level proof: a static source must not yield a freezedetect event."""

    QA_FREEZE_FILTER = "freezedetect=n=0.003:d=0.5"

    def _render_still_source(self, tmp: Path) -> Path:
        os.environ.setdefault("GITHUB_ACTIONS", "true")
        still = tmp / "still.png"
        src = tmp / "still.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", f"testsrc2=size={SOURCE_W}x{SOURCE_H}:rate=1",
             "-frames:v", "1", str(still)],
            check=True, timeout=300,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30",
             "-t", "3.0", "-i", str(still), "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", str(src)],
            check=True, timeout=300,
        )
        return src

    def test_frozen_source_renders_without_freeze_event(self):
        from src.config import FPS

        tmp = Path(tempfile.mkdtemp())
        try:
            src = self._render_still_source(tmp)
            decision = FramingDecision(
                mode="multi_shot_dynamic", face_count=1,
                video_width=SOURCE_W, video_height=SOURCE_H,
                active_x=0, active_y=0, active_w=SOURCE_W, active_h=SOURCE_H,
                shots=[ShotPlan(start=0.0, end=3.0, mode="portrait_face", crop_x=454,
                                margin_v=460, zoom_factor=1.15, face_centers_timeline=[])],
            )
            graph = build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)
            out = tmp / "out.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(src),
                 "-filter_complex", graph, "-map", "[outv]",
                 "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-r", str(FPS), str(out)],
                check=True, timeout=900,
            )
            probe = subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostats", "-i", str(out),
                 "-vf", self.QA_FREEZE_FILTER, "-an", "-f", "null", "NUL"],
                capture_output=True, text=True, timeout=600,
            )
            violations = [
                line for line in probe.stderr.splitlines()
                if "freeze_duration" in line or "freeze_start" in line
            ]
            self.assertEqual(
                violations, [],
                f"rendered a frozen interval from a static source: {violations}",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
