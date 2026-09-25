import unittest
from unittest.mock import patch

from src.composition_planner import NormalizedRect, ShotComposition, CompositionPlan, SafeZone
from src.editorial_qa import run_editorial_qa
from src.edit_director import EditPlan, HookCandidate, StoryBeat


class TestEditorialQa(unittest.TestCase):
    def _edit_plan(self) -> EditPlan:
        return EditPlan(
            plan_id="plan_qa",
            start=0.0,
            end=30.0,
            hook=HookCandidate("story", "A strong hook", 0.0, 3.0, 0.9),
            beats=[StoryBeat(0.0, 30.0, "context_or_method", "A strong hook")],
            selected_shot_ids=["shot_1"],
            confidence=0.8,
        )

    def _composition(self) -> CompositionPlan:
        return CompositionPlan(
            plan_id="composition_plan_qa",
            width=1080,
            height=1920,
            safe_zone=SafeZone(0.12, 0.22, 0.08, 0.16),
            shots=[ShotComposition(
                shot_id="shot_1",
                start=0.0,
                end=30.0,
                source_type="human_or_scene",
                crop=NormalizedRect(0.0, 0.0, 1.0, 1.0),
                caption_anchor="lower_center",
                caption_rect=NormalizedRect(0.08, 0.62, 0.76, 0.14),
            )],
        )

    def test_passing_output_allows_publish(self):
        probe = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "avg_frame_rate": "30/1", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
            ],
        }
        with patch("src.editorial_qa._probe_output", return_value=probe), \
             patch("src.editorial_qa._detect_intervals", side_effect=[[], []]), \
             patch("src.editorial_qa._loudness_metrics", return_value={"integrated_lufs": -14.0, "peak_dbfs": -1.8}):
            report = run_editorial_qa("output.mp4", self._edit_plan(), self._composition(), report_id="qa_pass")
        self.assertTrue(report.passed)
        self.assertEqual(report.issues, [])

    def test_incomplete_endpoint_blocks_publish(self):
        probe = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "avg_frame_rate": "30/1", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
            ],
        }
        edit_plan = self._edit_plan()
        edit_plan.warnings.append("endpoint_not_proven_complete")
        with patch("src.editorial_qa._probe_output", return_value=probe), \
             patch("src.editorial_qa._detect_intervals", side_effect=[[], []]), \
             patch("src.editorial_qa._loudness_metrics", return_value={"integrated_lufs": -14.0, "peak_dbfs": -1.8}):
            report = run_editorial_qa("output.mp4", edit_plan, self._composition(), report_id="qa_incomplete")
        self.assertFalse(report.passed)
        self.assertIn("incomplete_endpoint", {issue.code for issue in report.issues})

    def test_fragmented_endpoint_blocks_publish(self):
        probe = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "avg_frame_rate": "30/1", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
            ],
        }
        edit_plan = self._edit_plan()
        edit_plan.warnings.append("endpoint_caption_fragment")
        with patch("src.editorial_qa._probe_output", return_value=probe), \
             patch("src.editorial_qa._detect_intervals", side_effect=[[], []]), \
             patch("src.editorial_qa._loudness_metrics", return_value={"integrated_lufs": -14.0, "peak_dbfs": -1.8}):
            report = run_editorial_qa("output.mp4", edit_plan, self._composition(), report_id="qa_fragment")
        self.assertFalse(report.passed)
        self.assertIn("fragmented_endpoint", {issue.code for issue in report.issues})

    def test_freeze_and_collision_block_publish(self):
        probe = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "avg_frame_rate": "30/1", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
            ],
        }
        composition = self._composition()
        composition.shots[0].protected_regions = [NormalizedRect(0.1, 0.64, 0.5, 0.1)]
        with patch("src.editorial_qa._probe_output", return_value=probe), \
             patch("src.editorial_qa._detect_intervals", side_effect=[[], [{"start": 4.0, "end": 6.0, "duration": 2.0}]]), \
             patch("src.editorial_qa._loudness_metrics", return_value={"integrated_lufs": -14.0, "peak_dbfs": -1.8}):
            report = run_editorial_qa("output.mp4", self._edit_plan(), composition, report_id="qa_fail")
        self.assertFalse(report.passed)
        codes = {issue.code for issue in report.issues}
        self.assertIn("freeze_interval", codes)
        self.assertIn("caption_source_collision", codes)
        self.assertTrue(report.to_dict()["blocks_publish"])


if __name__ == "__main__":
    unittest.main()
