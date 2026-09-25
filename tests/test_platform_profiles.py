import json
import tempfile
import unittest
from pathlib import Path

from src.composition_planner import CompositionPlan, SafeZone
from src.edit_director import EditPlan, HookCandidate, StoryBeat
from src.platform_profiles import (
    PerformanceFeedback,
    build_platform_variant,
    get_platform_profile,
    save_feedback,
)


class TestPlatformProfiles(unittest.TestCase):
    def _edit_plan(self, duration: float = 100.0) -> EditPlan:
        return EditPlan(
            plan_id="plan_platform",
            start=0.0,
            end=duration,
            hook=HookCandidate("story", "A strong hook", 0.0, 3.0, 0.8),
            beats=[StoryBeat(0.0, duration, "context_or_method", "A strong hook")],
            selected_shot_ids=["shot_1"],
            confidence=0.8,
        )

    def _composition(self) -> CompositionPlan:
        return CompositionPlan(
            plan_id="composition_platform",
            width=1080,
            height=1920,
            safe_zone=SafeZone(0.12, 0.22, 0.08, 0.16),
            shots=[],
        )

    def test_platform_profiles_define_distinct_safe_zones_and_limits(self):
        instagram = get_platform_profile("instagram")
        youtube = get_platform_profile("youtube")
        self.assertEqual(instagram.max_duration, 90.0)
        self.assertEqual(instagram.caption_max_words, 3)
        self.assertEqual(youtube.max_duration, 180.0)
        self.assertEqual(youtube.hashtag_max, 3)
        self.assertNotEqual(instagram.safe_zone.bottom, youtube.safe_zone.bottom)

    def test_platform_variant_trims_only_when_required(self):
        plan = self._edit_plan(100.0)
        composition = self._composition()
        instagram = build_platform_variant(plan, composition, "instagram")
        youtube = build_platform_variant(plan, composition, "youtube")
        self.assertEqual(instagram.duration, 90.0)
        self.assertEqual(youtube.duration, 100.0)
        self.assertEqual(instagram.metadata["duration_strategy"], "trim_to_platform_limit")
        self.assertEqual(youtube.metadata["duration_strategy"], "preserve_master")

    def test_feedback_serializes_for_retention_learning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "feedback.json"
            save_feedback(PerformanceFeedback(
                run_id="run_1",
                clip_id="clip_1",
                service="youtube",
                views=1000,
                average_watch_seconds=42.5,
                completion_rate=0.42,
                saves=15,
                shares=8,
                retention_3s_rate=0.81,
            ), output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "run_1")
            self.assertEqual(payload["retention_3s_rate"], 0.81)
            self.assertTrue(payload["recorded_at"])

    def test_unknown_platform_is_rejected(self):
        with self.assertRaises(ValueError):
            get_platform_profile("unknown")


if __name__ == "__main__":
    unittest.main()
