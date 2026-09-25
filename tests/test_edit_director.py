import unittest

from src.edit_director import (
    EditPlan,
    HookCandidate,
    build_director_prompt,
    build_semantic_edit_plans,
    validate_edit_plan,
)
from src.source_index import IndexedShot, SourceIndex, SourceMediaInfo
from src.transcriber import TranscriptSegment, WordTimestamp


class TestEditDirector(unittest.TestCase):
    def _segments(self) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(0.0, 15.0, "We were living off of 16,000 a year.", [WordTimestamp("We", 0.0, 0.4), WordTimestamp("house.", 14.0, 15.0)]),
            TranscriptSegment(15.0, 30.0, "We cut the cable off.", [WordTimestamp("We", 15.0, 15.4), WordTimestamp("off.", 29.0, 30.0)]),
            TranscriptSegment(30.0, 45.0, "We cut the internet off.", [WordTimestamp("We", 30.0, 30.4), WordTimestamp("off.", 44.0, 45.0)]),
            TranscriptSegment(45.0, 60.0, "Then we paid off our house.", [WordTimestamp("Then", 45.0, 45.4), WordTimestamp("house.", 59.0, 60.0)]),
        ]

    def test_builds_complete_hook_and_story_plan_from_evidence(self):
        media = SourceMediaInfo("clip.mp4", 60.0, 1080, 1920, 30.0)
        index = SourceIndex(
            media=media,
            shots=[IndexedShot("shot_1", 0.0, 30.0, "human_or_scene", 0.4, 0.6, 0.0, 1.0), IndexedShot("shot_2", 30.0, 60.0, "human_or_scene", 0.5, 0.5, 0.0, 1.0)],
        )
        plans = build_semantic_edit_plans(self._segments(), index, num_clips=1)
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertTrue(30.0 <= plan.duration <= 140.0)
        self.assertTrue(plan.hook.text)
        self.assertTrue(plan.beats)
        self.assertEqual(validate_edit_plan(plan, 60.0), [])

    def test_incomplete_endpoint_is_marked_for_review(self):
        plan = EditPlan(
            plan_id="plan_test",
            start=0.0,
            end=50.0,
            hook=HookCandidate("story", "A complete beginning", 0.0, 3.0, 0.8),
            beats=[],
            selected_shot_ids=[],
            confidence=0.5,
            warnings=["endpoint_requires_review"],
        )
        self.assertIn("endpoint_not_proven_complete", validate_edit_plan(plan, 50.0))

    def test_prompt_contains_transcript_and_visual_evidence(self):
        media = SourceMediaInfo("clip.mp4", 60.0, 1080, 1920, 30.0)
        index = SourceIndex(media=media, shots=[IndexedShot("shot_1", 0.0, 60.0, "presentation", 0.1, 0.9, 1.0, 0.0, freeze_intervals=[(20.0, 21.0)])])
        prompt = build_director_prompt(self._segments(), index, num_clips=1)
        self.assertIn("16,000", prompt)
        self.assertIn("shot_1", prompt)
        self.assertIn("freeze_intervals", prompt)
        self.assertIn("complete start and end boundary", prompt)


if __name__ == "__main__":
    unittest.main()
