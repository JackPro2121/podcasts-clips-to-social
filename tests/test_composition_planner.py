import unittest

from src.composition_planner import DEFAULT_SAFE_ZONE, build_composition_plan
from src.edit_director import EditPlan, HookCandidate, StoryBeat
from src.source_index import IndexedShot, SourceIndex, SourceMediaInfo, TextRegion


class TestCompositionPlanner(unittest.TestCase):
    def _edit_plan(self) -> EditPlan:
        return EditPlan(
            plan_id="plan_test",
            start=0.0,
            end=30.0,
            hook=HookCandidate("story", "A strong opening", 0.0, 3.0, 0.9),
            beats=[StoryBeat(0.0, 30.0, "context_or_method", "A strong opening")],
            selected_shot_ids=["shot_1"],
            confidence=0.8,
        )

    def _index(self, shot: IndexedShot) -> SourceIndex:
        return SourceIndex(
            media=SourceMediaInfo("clip.mp4", 30.0, 1080, 1920, 30.0),
            shots=[shot],
        )

    def test_source_lower_text_moves_captions_away_from_collision(self):
        shot = IndexedShot(
            "shot_1", 0.0, 30.0, "human_or_scene", 0.4, 0.6, 0.0, 1.0,
            text_regions=[TextRegion(0.05, 0.80, 0.9, 0.12, 0.9)],
        )
        composition = build_composition_plan(self._edit_plan(), self._index(shot))
        self.assertEqual(composition.shots[0].caption_anchor, "upper_center")
        self.assertTrue(composition.shots[0].protected_regions)
        self.assertIn("source_text_requires_protection", composition.shots[0].warnings)

    def test_split_screen_uses_divider_placement(self):
        shot = IndexedShot("shot_1", 0.0, 30.0, "split_screen", 0.4, 0.6, 0.0, 2.0)
        composition = build_composition_plan(self._edit_plan(), self._index(shot))
        self.assertEqual(composition.shots[0].caption_anchor, "split_divider")
        self.assertEqual(composition.shots[0].caption_rect.x, 0.48)

    def test_freeze_shot_gets_motion_plan(self):
        shot = IndexedShot(
            "shot_1", 0.0, 30.0, "human_or_scene", 0.0, 1.0, 0.0, 1.0,
            freeze_intervals=[(10.0, 13.0)],
        )
        composition = build_composition_plan(self._edit_plan(), self._index(shot))
        self.assertEqual(composition.shots[0].broll_mode, "ken_burns_motion")
        self.assertEqual(composition.shots[0].transition, "motion_cut")
        self.assertEqual(composition.shots[0].max_static_hold, 2.5)
        self.assertTrue(composition.overlays[0].allow_dynamic_motion)

    def test_plan_serializes_dynamic_decisions(self):
        shot = IndexedShot("shot_1", 0.0, 30.0, "human_or_scene", 0.4, 0.6, 0.0, 1.0)
        composition = build_composition_plan(self._edit_plan(), self._index(shot))
        payload = composition.to_dict()
        self.assertEqual(payload["safe_zone"]["bottom"], DEFAULT_SAFE_ZONE.bottom)
        self.assertEqual(payload["shots"][0]["caption_anchor"], "lower_center")
        self.assertEqual(payload["schema_version"], "0.1")


if __name__ == "__main__":
    unittest.main()
