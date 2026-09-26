"""Tests for the picture-in-picture (host over slide) layout."""
import unittest

from src.face_tracker import FramingDecision, ShotPlan
from src.video_editor import (
    PIP_INSET_HEIGHT,
    PIP_INSET_WIDTH,
    PIP_INSET_X,
    PIP_INSET_Y,
    PIP_SLIDE_CANVAS_RATIO,
    build_video_filtergraph,
)

W, H = 1920, 1080


def _graph(shot, active=(0, 0, W, H)):
    ax, ay, aw, ah = active
    decision = FramingDecision(
        mode="multi_shot_dynamic", face_count=2,
        video_width=W, video_height=H,
        active_x=ax, active_y=ay, active_w=aw, active_h=ah,
        shots=[shot],
    )
    return build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)


class TestPipGeometry(unittest.TestCase):
    def test_inset_is_portrait_and_scaled_off_the_output(self):
        from src.config import OUTPUT_WIDTH
        self.assertEqual(PIP_INSET_WIDTH, int(OUTPUT_WIDTH * 0.30))
        self.assertGreater(PIP_INSET_HEIGHT, PIP_INSET_WIDTH)

    def test_inset_clears_the_caption_band(self):
        from src.config import OUTPUT_HEIGHT, SUBTITLE_MARGIN_MIN_V
        self.assertGreater(PIP_INSET_Y, 240, "must clear the top platform UI")
        self.assertLess(
            PIP_INSET_Y + PIP_INSET_HEIGHT,
            OUTPUT_HEIGHT - SUBTITLE_MARGIN_MIN_V,
            "inset must not sit under the caption band",
        )

    def test_inset_stays_inside_the_output_frame(self):
        from src.config import OUTPUT_HEIGHT, OUTPUT_WIDTH
        self.assertGreaterEqual(PIP_INSET_X, 0)
        self.assertLessEqual(PIP_INSET_X + PIP_INSET_WIDTH, OUTPUT_WIDTH)
        self.assertLessEqual(PIP_INSET_Y + PIP_INSET_HEIGHT, OUTPUT_HEIGHT)

    def test_canvas_ratio_leaves_positional_slack(self):
        self.assertLess(PIP_SLIDE_CANVAS_RATIO, 1.0)
        self.assertGreater(PIP_SLIDE_CANVAS_RATIO, 0.80)


class TestPipFiltergraph(unittest.TestCase):
    def setUp(self):
        self.shot = ShotPlan(start=0.0, end=6.0, mode="pip_slide",
                             speaker1_box=(1200, 120, 480, 480), margin_v=460)
        self.graph = _graph(self.shot)

    def test_emits_a_canvas_and_an_inset_stream(self):
        self.assertIn("p0_canvas", self.graph)
        self.assertIn("p0_host", self.graph)
        self.assertIn("p0_slide", self.graph)
        self.assertIn("p0_blur", self.graph)

    def test_inset_is_overlaid_with_a_border(self):
        self.assertIn(f"overlay={PIP_INSET_X}:{PIP_INSET_Y}", self.graph)
        self.assertIn("drawbox=", self.graph)
        self.assertIn("t=6", self.graph)

    def test_slide_is_centred_over_a_blurred_fill(self):
        self.assertIn("boxblur=30:5", self.graph)
        self.assertIn("force_original_aspect_ratio=increase", self.graph)
        self.assertIn("force_original_aspect_ratio=decrease", self.graph)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", self.graph)

    def test_canvas_crop_is_smaller_than_the_active_region(self):
        cw = int(W * PIP_SLIDE_CANVAS_RATIO)
        self.assertIn(f"crop={cw}:", self.graph)
        self.assertNotIn(f"crop={W}:{H}:", self.graph)

    def test_both_canvas_and_inset_receive_the_motion_guarantee(self):
        drift = "0.6366*asin(sin(t*1.8))"
        parts = self.graph.split(";")
        canvas = [p for p in parts if p.strip().endswith("[p0_slide]")]
        host = [p for p in parts if p.strip().endswith("[p0_host]")]
        self.assertEqual(len(canvas), 1)
        self.assertEqual(len(host), 1)
        self.assertIn(drift, canvas[0], "the slide canvas must drift")
        self.assertIn(drift, host[0], "the host inset must drift")

    def test_no_static_full_region_crop_is_emitted(self):
        """A crop spanning the whole active region has no slack and would freeze."""
        self.assertNotRegex(self.graph, rf"crop={W}:{H}:\d+:")

    def test_output_is_9x16(self):
        from src.config import OUTPUT_HEIGHT, OUTPUT_WIDTH
        self.assertIn(f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}", self.graph)


class TestPipFallbacks(unittest.TestCase):
    def test_pip_shot_without_a_speaker_box_falls_back(self):
        # No speaker box means no host to inset; must not emit a broken composite.
        shot = ShotPlan(start=0.0, end=6.0, mode="pip_slide", crop_x=400, margin_v=460)
        graph = _graph(shot)
        self.assertNotIn("p0_host", graph)
        self.assertIn("v_shot_0", graph)

    def test_pip_inside_a_mixed_shot_sequence(self):
        shots = [
            ShotPlan(start=0.0, end=4.0, mode="presentation_slide", margin_v=420),
            ShotPlan(start=4.0, end=10.0, mode="pip_slide",
                     speaker1_box=(1200, 120, 480, 480), margin_v=460),
            ShotPlan(start=10.0, end=16.0, mode="portrait_face", crop_x=441, margin_v=460),
        ]
        decision = FramingDecision(
            mode="multi_shot_dynamic", face_count=3,
            video_width=W, video_height=H, active_x=0, active_y=0, active_w=W, active_h=H,
            shots=shots,
        )
        graph = build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)
        self.assertIn("p1_host", graph)
        self.assertIn("v_shot_0", graph)
        self.assertIn("v_shot_1", graph)
        self.assertIn("v_shot_2", graph)
        self.assertIn("concat=n=3:v=1:a=0", graph)


if __name__ == "__main__":
    unittest.main()
