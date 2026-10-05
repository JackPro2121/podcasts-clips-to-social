"""Tests for the configurable Gemini ladder and the short-form duration gate."""
import os
import unittest
from unittest import mock

from src import config as config_module
from src.edit_director import duration_warnings
from src.viral_detector import query_gemini_models


class TestGeminiModelLadder(unittest.TestCase):
    def test_ladder_is_ordered_and_deduplicated_of_blanks(self):
        self.assertTrue(config_module.GEMINI_MODEL_LADDER)
        for model in config_module.GEMINI_MODEL_LADDER:
            self.assertTrue(model.strip())
            self.assertEqual(model, model.strip())

    def test_no_retired_model_ids(self):
        """Verified live 2026-10-04: 2.0-flash and 2.5-flash-lite are 404 and
        2.5-flash shuts down 2026-10-16; none may stay in a default ladder."""
        for ladder in (
            config_module.GEMINI_MODEL_LADDER,
            config_module.GEMINI_VISION_MODEL_LADDER,
        ):
            for model in ladder:
                self.assertFalse(
                    model.startswith("gemini-2."),
                    f"{model} is retired or scheduled for shutdown",
                )
            self.assertEqual(len(ladder), len(set(ladder)), "ladder has duplicates")

    def test_ladder_has_a_long_lived_tail(self):
        ladder = config_module.GEMINI_MODEL_LADDER
        self.assertIn("gemini-3.5-flash", ladder)
        self.assertIn("gemini-3.5-flash-lite", ladder)

    def test_vision_ladder_is_separate_and_cost_appropriate(self):
        vision = config_module.GEMINI_VISION_MODEL_LADDER
        self.assertTrue(vision)
        self.assertIn("gemini-3.5-flash-lite", vision)
        self.assertNotEqual(vision, config_module.GEMINI_MODEL_LADDER)

    def test_custom_ladder_is_honored(self):
        calls = []

        class _FakeResponse:
            text = ('{"clips": [{"start_time": 10.0, "end_time": 55.0, '
                    '"hook": "a real hook"}]}')

        class _FakeModels:
            def generate_content(self, model, contents, config):
                calls.append(model)
                return _FakeResponse()

        class _FakeClient:
            def __init__(self, api_key, http_options):
                self.models = _FakeModels()

        fake_genai = mock.Mock()
        fake_genai.Client = _FakeClient
        fake_types = mock.Mock()
        fake_types.HttpOptions = mock.Mock()
        fake_types.GenerateContentConfig = mock.Mock()

        with mock.patch("src.viral_detector.HAS_NEW_GENAI", True), \
             mock.patch("src.viral_detector.genai", fake_genai), \
             mock.patch("src.viral_detector.genai_types", fake_types), \
             mock.patch("src.viral_detector.GEMINI_MODEL_LADDER", ["global-model"]), \
             mock.patch("src.viral_detector.time.sleep", lambda *_: None):
            result = query_gemini_models("prompt", "key", model_ladder=["only-model"])

        self.assertIsNotNone(result)
        self.assertEqual(calls, ["only-model"])

    def test_ladder_is_configurable_from_the_environment(self):
        with mock.patch.dict(os.environ, {"GEMINI_MODEL_LADDER": "a-model, b-model ,"}, clear=False):
            ladder = [
                m.strip() for m in os.environ["GEMINI_MODEL_LADDER"].split(",") if m.strip()
            ]
        self.assertEqual(ladder, ["a-model", "b-model"])

    def test_rotation_walks_the_ladder_and_stops_at_the_first_success(self):
        calls = []

        class _FakeResponse:
            # query_gemini_models rejects responses under 20 chars, so the stub
            # has to return a realistically sized payload.
            text = ('{"clips": [{"start_time": 10.0, "end_time": 55.0, '
                    '"hook": "a real hook"}]}')

        class _FakeModels:
            def generate_content(self, model, contents, config):
                calls.append(model)
                if model == "second-model":
                    raise RuntimeError("503 UNAVAILABLE high demand")
                return _FakeResponse()

        class _FakeClient:
            def __init__(self, api_key, http_options):
                self.models = _FakeModels()

        fake_genai = mock.Mock()
        fake_genai.Client = _FakeClient
        fake_types = mock.Mock()
        fake_types.HttpOptions = mock.Mock()
        fake_types.GenerateContentConfig = mock.Mock()

        with mock.patch("src.viral_detector.HAS_NEW_GENAI", True), \
             mock.patch("src.viral_detector.genai", fake_genai), \
             mock.patch("src.viral_detector.genai_types", fake_types), \
             mock.patch("src.viral_detector.GEMINI_MODEL_LADDER", ["first-model", "second-model"]), \
             mock.patch("src.viral_detector.time.sleep", lambda *_: None):
            result = query_gemini_models("prompt", "key")

        self.assertIsNotNone(result)
        self.assertEqual(calls, ["first-model"])

    def test_unavailable_model_is_retried_before_moving_on(self):
        calls = []

        class _FakeResponse:
            text = ('{"clips": [{"start_time": 10.0, "end_time": 55.0, '
                    '"hook": "a real hook"}]}')

        class _FakeModels:
            def generate_content(self, model, contents, config):
                calls.append(model)
                if model == "busy-model":
                    raise RuntimeError("503 UNAVAILABLE high demand")
                return _FakeResponse()

        class _FakeClient:
            def __init__(self, api_key, http_options):
                self.models = _FakeModels()

        fake_genai = mock.Mock()
        fake_genai.Client = _FakeClient
        fake_types = mock.Mock()
        fake_types.HttpOptions = mock.Mock()
        fake_types.GenerateContentConfig = mock.Mock()

        with mock.patch("src.viral_detector.HAS_NEW_GENAI", True), \
             mock.patch("src.viral_detector.genai", fake_genai), \
             mock.patch("src.viral_detector.genai_types", fake_types), \
             mock.patch("src.viral_detector.GEMINI_MODEL_LADDER", ["busy-model", "ok-model"]), \
             mock.patch("src.viral_detector.time.sleep", lambda *_: None):
            result = query_gemini_models("prompt", "key")

        self.assertIsNotNone(result)
        # 3 attempts against the busy model, then a single success on the next
        self.assertEqual(calls.count("busy-model"), 3)
        self.assertEqual(calls[-1], "ok-model")

    def test_returns_none_when_every_model_fails(self):
        class _FakeModels:
            def generate_content(self, model, contents, config):
                raise RuntimeError("401 invalid key")

        class _FakeClient:
            def __init__(self, api_key, http_options):
                self.models = _FakeModels()

        fake_genai = mock.Mock()
        fake_genai.Client = _FakeClient
        fake_types = mock.Mock()
        fake_types.HttpOptions = mock.Mock()
        fake_types.GenerateContentConfig = mock.Mock()

        with mock.patch("src.viral_detector.HAS_NEW_GENAI", True), \
             mock.patch("src.viral_detector.genai", fake_genai), \
             mock.patch("src.viral_detector.genai_types", fake_types), \
             mock.patch("src.viral_detector.GEMINI_MODEL_LADDER", ["a", "b"]), \
             mock.patch("src.viral_detector.time.sleep", lambda *_: None):
            self.assertIsNone(query_gemini_models("prompt", "bad-key"))


class TestDurationWarnings(unittest.TestCase):
    def test_long_clip_is_flagged_against_the_short_form_window(self):
        warnings = duration_warnings(0.0, 96.0)
        self.assertTrue(any(w.startswith("exceeds_short_form_window") for w in warnings))
        self.assertTrue(any("96.0s" in w for w in warnings))

    def test_clip_inside_the_window_is_clean(self):
        self.assertEqual(duration_warnings(0.0, 47.5), [])

    def test_short_clip_is_flagged(self):
        warnings = duration_warnings(0.0, 12.0)
        self.assertTrue(any(w.startswith("below_min_clip_duration") for w in warnings))

    def test_boundary_of_sixty_seconds_is_not_flagged(self):
        self.assertEqual(duration_warnings(0.0, 60.0), [])


if __name__ == "__main__":
    unittest.main()
