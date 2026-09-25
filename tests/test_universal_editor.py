import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.source_index import IndexedShot, SourceIndex, SourceMediaInfo
from src.transcriber import TranscriptSegment, WordTimestamp
from src.universal_editor import build_clip_editor_artifacts


class TestUniversalEditor(unittest.TestCase):
    def test_builds_shadow_artifacts_from_clip_relative_transcript(self):
        media = SourceMediaInfo("clip.mp4", 30.0, 1080, 1920, 30.0)
        index = SourceIndex(
            media=media,
            shots=[IndexedShot("shot_1", 0.0, 30.0, "human_or_scene", 0.4, 0.6, 0.0, 1.0)],
        )
        segments = [TranscriptSegment(10.0, 40.0, "A complete story about a house.", [WordTimestamp("A", 10.0, 10.5), WordTimestamp("house.", 39.0, 40.0)])]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.universal_editor.build_source_index", return_value=index):
                artifacts = build_clip_editor_artifacts(
                    Path("clip.mp4"),
                    segments,
                    10.0,
                    40.0,
                    Path(tmpdir),
                    "run_1",
                    1,
                )
            self.assertEqual(artifacts.edit_plan.start, 0.0)
            self.assertEqual(artifacts.edit_plan.end, 30.0)
            self.assertTrue(artifacts.edit_plan.hook.text.startswith("A"))
            self.assertTrue((Path(tmpdir) / "clip_1_source_index.json").exists())
            payload = json.loads((Path(tmpdir) / "clip_1_composition_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["plan_id"], "composition_run_1_clip_1")


if __name__ == "__main__":
    unittest.main()
