"""Tests for the bounded render-repair loop."""
import unittest

from src.composition_planner import (
    DEFAULT_SAFE_ZONE,
    CompositionPlan,
    NormalizedRect,
    ShotComposition,
    build_caption_placements,
)
from src.repair import (
    MAX_MOTION_GAIN,
    UNREPAIRABLE_CODES,
    MAX_REPAIR_ATTEMPTS,
    REPAIRABLE_CODES,
    RenderAdjustment,
    is_repairable,
    plan_repair,
    render_with_repair,
)
from src.video_editor import _effective_drift_ratio, _STATIC_SHOT_DRIFT_RATIO


class TestPlanRepair(unittest.TestCase):
    def test_static_hold_limit_escalates_motion(self):
        nxt, actions = plan_repair(["static_hold_limit"], RenderAdjustment())
        self.assertGreater(nxt.motion_gain, 1.0)
        self.assertTrue(any(a.startswith("motion_gain") for a in actions))

    def test_freeze_is_not_repaired_but_fails_loudly(self):
        """The renderer's motion guarantee is the real fix for freezes."""
        nxt, actions = plan_repair(["freeze_interval"], RenderAdjustment())
        self.assertEqual(actions, [])
        self.assertEqual(nxt.motion_gain, 1.0)
        self.assertIn("freeze_interval", UNREPAIRABLE_CODES)
        self.assertNotIn("freeze_interval", REPAIRABLE_CODES)

    def test_freeze_veto_does_not_trigger_a_second_render(self):
        calls = []

        def render(adjustment):
            calls.append(adjustment)
            return "clip.mp4"

        outcome = render_with_repair(
            render, lambda p: (False, ["freeze_interval"]), log=None
        )
        self.assertFalse(outcome.succeeded)
        self.assertEqual(len(calls), 1)
        self.assertEqual(outcome.final_codes, ["freeze_interval"])

    def test_caption_collision_toggles_the_caption_lever(self):
        nxt, actions = plan_repair(["caption_source_collision"], RenderAdjustment())
        self.assertTrue(nxt.avoid_caption_collisions)
        self.assertIn("avoid_caption_collisions->True", actions)

    def test_unrepairable_codes_produce_no_actions(self):
        nxt, actions = plan_repair(["output_probe", "video_stream_missing"], RenderAdjustment())
        self.assertEqual(actions, [])
        self.assertEqual(nxt.motion_gain, 1.0)

    def test_unknown_codes_are_ignored(self):
        _, actions = plan_repair(["some_future_issue"], RenderAdjustment())
        self.assertEqual(actions, [])

    def test_repairs_are_cumulative_and_capped(self):
        adjustment = RenderAdjustment()
        seen = []
        for _ in range(6):
            adjustment, actions = plan_repair(["static_hold_limit"], adjustment)
            seen.append(adjustment.motion_gain)
            if not actions:
                break
        self.assertEqual(seen, sorted(seen))
        self.assertLessEqual(seen[-1], MAX_MOTION_GAIN)
        self.assertNotEqual(seen[0], seen[1], "gain must actually change between attempts")

    def test_already_maxed_motion_stops_escalating(self):
        at_max = RenderAdjustment(motion_gain=MAX_MOTION_GAIN, avoid_caption_collisions=True)
        _, actions = plan_repair(
            ["static_hold_limit", "caption_source_collision"], at_max
        )
        self.assertEqual(actions, [])

    def test_mixed_codes_move_every_available_lever(self):
        nxt, actions = plan_repair(["static_hold_limit", "caption_source_collision"], RenderAdjustment())
        self.assertGreater(nxt.motion_gain, 1.0)
        self.assertTrue(nxt.avoid_caption_collisions)
        self.assertEqual(len(actions), 2)

    def test_is_repairable(self):
        self.assertTrue(is_repairable(["static_hold_limit"]))
        self.assertFalse(is_repairable(["output_probe"]))
        self.assertFalse(is_repairable(["freeze_interval"]))
        self.assertFalse(is_repairable([]))

    def test_repairable_codes_all_map_to_a_known_lever(self):
        self.assertTrue(set(REPAIRABLE_CODES.values()) <= {"motion", "captions"})

    def test_drift_ratio_is_scaled_and_capped(self):
        self.assertAlmostEqual(
            _effective_drift_ratio(1.0), _STATIC_SHOT_DRIFT_RATIO, places=6)
        self.assertGreater(_effective_drift_ratio(3.6), _STATIC_SHOT_DRIFT_RATIO)
        self.assertLessEqual(_effective_drift_ratio(1000.0), 0.20)

    def test_drift_ratio_rejects_nonsense_gain(self):
        for bad in (0.0, -2.0, float("nan"), float("inf"), "abc", None):
            self.assertAlmostEqual(
                _effective_drift_ratio(bad), _STATIC_SHOT_DRIFT_RATIO, places=6,
                msg=f"gain={bad!r}",
            )


class TestRenderWithRepair(unittest.TestCase):
    def test_passes_on_first_attempt_without_repairing(self):
        adjustments = []

        def render(adjustment):
            adjustments.append(adjustment)
            return "clip.mp4"

        outcome = render_with_repair(render, lambda p: (True, []), log=None)
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.attempts, 1)
        self.assertEqual(outcome.result, "clip.mp4")
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0].motion_gain, 1.0)
        self.assertEqual(outcome.history, [])

    def test_veto_triggers_a_stronger_second_render(self):
        adjustments = []
        results = iter([(False, ["static_hold_limit"]), (True, [])])

        def render(adjustment):
            adjustments.append(adjustment)
            return f"clip-{len(adjustments)}.mp4"

        def qa(path):
            return next(results)

        outcome = render_with_repair(render, qa, log=None)
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(outcome.result, "clip-2.mp4")
        self.assertGreater(adjustments[1].motion_gain, adjustments[0].motion_gain)
        self.assertEqual(len(outcome.history), 1)
        self.assertEqual(outcome.history[0]["issue_codes"], ["static_hold_limit"])

    def test_gives_up_after_the_attempt_cap(self):
        calls = []

        def render(adjustment):
            calls.append(adjustment)
            return "clip.mp4"

        outcome = render_with_repair(
            render, lambda p: (False, ["static_hold_limit"]), log=None
        )
        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.attempts, MAX_REPAIR_ATTEMPTS + 1)
        self.assertEqual(len(calls), MAX_REPAIR_ATTEMPTS + 1)
        self.assertEqual(outcome.final_codes, ["static_hold_limit"])

    def test_unrepairable_failure_stops_immediately(self):
        calls = []

        def render(adjustment):
            calls.append(adjustment)
            return "clip.mp4"

        outcome = render_with_repair(
            render, lambda p: (False, ["output_probe"]), log=None
        )
        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.attempts, 1, "a fatal issue must not be re-rendered")
        self.assertEqual(len(calls), 1)

    def test_render_exceptions_propagate(self):
        def render(adjustment):
            raise RuntimeError("ffmpeg exploded")

        with self.assertRaises(RuntimeError):
            render_with_repair(render, lambda p: (True, []), log=None)

    def test_zero_attempts_allowed_means_no_repair(self):
        calls = []

        def render(adjustment):
            calls.append(adjustment)
            return "clip.mp4"

        outcome = render_with_repair(
            render, lambda p: (False, ["static_hold_limit"]), max_attempts=0, log=None
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(outcome.succeeded)

    def test_negative_attempts_is_rejected(self):
        with self.assertRaises(ValueError):
            render_with_repair(lambda a: "x", lambda p: (True, []), max_attempts=-1)

    def test_outcome_serialises(self):
        outcome = render_with_repair(
            lambda a: "clip.mp4", lambda p: (False, ["freeze_interval"]), max_attempts=1, log=None
        )
        payload = outcome.to_dict()
        self.assertFalse(payload["succeeded"])
        self.assertIn("adjustment", payload)
        self.assertIn("history", payload)


class TestCaptionCollisionRepair(unittest.TestCase):
    def _plan(self, rect, protected):
        return CompositionPlan(
            plan_id="p", width=1080, height=1920, safe_zone=DEFAULT_SAFE_ZONE,
            shots=[ShotComposition(
                shot_id="s1", start=0.0, end=10.0, source_type="presentation",
                crop=NormalizedRect(0, 0, 1, 1), caption_anchor="lower_center",
                caption_rect=rect, protected_regions=protected,
            )],
        )

    def test_overlapping_caption_is_relocated_on_repair(self):
        caption = NormalizedRect(0.08, 0.62, 0.76, 0.14)
        source_text = NormalizedRect(0.10, 0.66, 0.50, 0.10)
        plan = self._plan(caption, [source_text])

        before = build_caption_placements(plan, avoid_collisions=False)[0]
        self.assertEqual(before.alignment, 2)
        self.assertFalse(before.collision_avoidance)

        after = build_caption_placements(plan, avoid_collisions=True)[0]
        self.assertEqual(after.alignment, 8, "should move to the top safe zone")
        self.assertTrue(after.collision_avoidance)
        self.assertEqual(after.anchor, "upper_center")

    def test_non_overlapping_caption_is_left_alone(self):
        caption = NormalizedRect(0.08, 0.62, 0.76, 0.14)
        source_text = NormalizedRect(0.10, 0.20, 0.50, 0.08)
        plan = self._plan(caption, [source_text])
        before = build_caption_placements(plan, avoid_collisions=False)[0]
        after = build_caption_placements(plan, avoid_collisions=True)[0]
        self.assertEqual(before.to_dict(), after.to_dict())

    def test_no_protected_regions_is_a_no_op(self):
        plan = self._plan(NormalizedRect(0.08, 0.62, 0.76, 0.14), [])
        before = build_caption_placements(plan, avoid_collisions=False)[0]
        after = build_caption_placements(plan, avoid_collisions=True)[0]
        self.assertEqual(before.to_dict(), after.to_dict())

    def test_falls_back_to_centre_when_the_top_is_also_blocked(self):
        caption = NormalizedRect(0.08, 0.62, 0.76, 0.14)
        bottom_text = NormalizedRect(0.10, 0.66, 0.50, 0.10)
        top_text = NormalizedRect(0.10, 0.22, 0.50, 0.10)
        plan = self._plan(caption, [bottom_text, top_text])
        after = build_caption_placements(plan, avoid_collisions=True)[0]
        self.assertEqual(after.anchor, "center")
        self.assertEqual(after.alignment, 5)


if __name__ == "__main__":
    unittest.main()
