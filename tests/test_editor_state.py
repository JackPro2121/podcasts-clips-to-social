import json
import tempfile
import unittest
from pathlib import Path

from src.editor_models import StageStatus
from src.run_state import RunStateStore


class TestRunStateStore(unittest.TestCase):
    def test_manifest_persists_safe_settings_and_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStateStore(Path(tmpdir))
            manifest = store.create(
                run_id="run-2026-001",
                source_url="https://www.youtube.com/watch?v=example",
                source_id="example",
                settings={
                    "num_clips": 1,
                    "niche": "finance",
                    "post_to_buffer": True,
                    "GEMINI_API_KEY": "must-not-be-stored",
                    "nested": {"token": "must-not-be-stored"},
                },
            )

            path = store.manifest_path(manifest.run_id)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["settings"], {
                "niche": "finance",
                "num_clips": 1,
                "post_to_buffer": True,
            })
            self.assertNotIn("GEMINI_API_KEY", path.read_text(encoding="utf-8"))
            self.assertNotIn("must-not-be-stored", path.read_text(encoding="utf-8"))

            store.set_stage(manifest.run_id, "analysis", StageStatus.RUNNING)
            store.set_stage(manifest.run_id, "analysis", StageStatus.COMPLETED)
            updated = store.record_artifact(manifest.run_id, "analysis_report", Path("reports/analysis.json"))
            self.assertEqual(updated.stages["analysis"], "completed")
            self.assertEqual(updated.artifacts["analysis_report"], str(Path("reports/analysis.json")))
            self.assertEqual(store.load(manifest.run_id).status, "created")

    def test_failure_persists_error_without_creating_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStateStore(Path(tmpdir))
            manifest = store.create("run-failure", "local://clip.mp4")
            failed = store.mark_failed(manifest.run_id, "qa_failed: ending incomplete")
            loaded = store.load(manifest.run_id)
            self.assertEqual(loaded.status, "failed")
            self.assertEqual(loaded.error, "qa_failed: ending incomplete")
            self.assertEqual(loaded.updated_at, failed.updated_at)

    def test_invalid_run_id_is_rejected_before_path_join(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStateStore(Path(tmpdir))
            with self.assertRaises(ValueError):
                store.create("../outside", "local://clip.mp4")
            self.assertFalse((Path(tmpdir) / "outside").exists())


if __name__ == "__main__":
    unittest.main()
