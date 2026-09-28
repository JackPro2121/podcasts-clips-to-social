"""Tests for the publication gate in ``main.py``.

The gate is the point of the whole exercise. Everything in ``src/verification``
exists to produce a verdict, and this is where that verdict stops a bad clip from
reaching Buffer.

Measured context, run ``36274579761`` on the public repo:

    [+] Gemini identified 1 viral clips:
    [render_metadata] {"output_duration": 102.12, ...}
    [+] Clip rendered successfully: clip_1_PAID_OFF_60000... (59.08 MB)
    [+] Editorial QA report for clip #1: passed=True
    ALL CLIPS PROCESSED SUCCESSFULLY!

One clip of three requested, 102 seconds against a 55-second maximum, published
and reported as complete success. And in run ``36277030701``, a clip containing
1.63s of frozen video inside a frame that was 50% dead black also passed QA.

These tests pin the behaviour that would have caught both.

Note what is *not* tested here: that the gate blocks. That requires a real render,
and ``TestGoldenMaster*`` in ``test_visual_regressions.py`` covers it against the
actual published artifacts. What is tested here is the wiring, which is the part
that a refactor can silently remove.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent / "main.py"


def _source() -> str:
    return MAIN.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _functions() -> dict:
    module = _tree()
    return {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
    }


class TestPixelGateIsWired(unittest.TestCase):
    """Every render branch must be gated, or a clip can bypass the verdict."""

    def setUp(self) -> None:
        self.source = _source()

    def test_gate_function_exists(self) -> None:
        self.assertIn("_evaluate_rendered_pixels", _functions())

    def test_all_three_render_branches_call_the_gate(self) -> None:
        """Three branches append to rendered_clips: segment, probe, full-download.

        Counted rather than hard-coded per site, because the failure mode is a
        fourth branch appearing without a gate, which a per-site assertion would
        not notice.
        """
        appends = self.source.count("rendered_clips.append")
        gates = self.source.count("pixel_ok, _pixel_codes = _evaluate_rendered_pixels")
        self.assertEqual(
            appends,
            3,
            "expected three render branches appending to rendered_clips; a new "
            "branch needs a gate too",
        )
        self.assertEqual(
            gates,
            appends,
            "every rendered_clips.append must be preceded by a pixel verdict gate",
        )

    def test_gate_is_not_environment_switchable(self) -> None:
        """No env var may disable the pixel gate.

        `UNIVERSAL_EDITOR_ENFORCE_QA` gates the *plan* check and defaults to
        false in config.py. If the pixel gate were similarly switchable it would
        be a report, not a gate -- and the first CI run of this work already
        demonstrated the failure mode, when the caption check silently did nothing
        on a green build.

        Checked against the AST rather than the source text, because the function's
        docstring *mentions* `UNIVERSAL_EDITOR_ENFORCE_QA` in order to explain why
        the gate is deliberately different from it. A substring test would flag
        the explanation as the violation.
        """
        gate = _functions()["_evaluate_rendered_pixels"]
        offending: list[str] = []
        for node in ast.walk(gate):
            # os.getenv(...) / os.environ[...] anywhere in the body
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "getenv":
                    offending.append(f"os.getenv at line {node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                offending.append(f"os.environ at line {node.lineno}")
            if isinstance(node, ast.Name) and "ENFORCE_QA" in node.id:
                offending.append(f"{node.id} at line {node.lineno}")
        self.assertEqual(
            offending,
            [],
            "the pixel gate must not read an environment variable; a gate that can "
            "be switched off by a repository variable silently stops verifying",
        )

    def test_gate_writes_its_evidence_before_deciding(self) -> None:
        """A rejection must be inspectable after the fact."""
        functions = _functions()
        body = ast.get_source_segment(self.source, functions["_evaluate_rendered_pixels"]) or ""
        self.assertIn("verdict.json", body)
        self.assertIn("record_artifact", body)
        write_at = body.index("result.write(verdict_path)")
        decide_at = body.index("if result.passed:")
        self.assertLess(
            write_at,
            decide_at,
            "the verdict must be persisted before the pass/fail branch, or a "
            "rejected clip leaves no evidence",
        )


class TestClipCountContract(unittest.TestCase):
    """AGENTS.md 3.5: exactly the requested number of clips, and never a lie."""

    def setUp(self) -> None:
        self.source = _source()

    def test_shortfall_is_reported_and_recorded(self) -> None:
        self.assertIn("CLIP COUNT SHORTFALL", self.source)
        self.assertIn("delivered_clips", self.source)
        self.assertIn("record_settings", self.source)

    def test_unqualified_success_banner_is_gone(self) -> None:
        """`ALL CLIPS PROCESSED SUCCESSFULLY!` must not print unconditionally.

        Mutation guard: restoring the bare print fails this test, and with it the
        exact log line that turned a contract violation into a green checkmark.
        """
        self.assertNotIn(
            'print("✨ ALL CLIPS PROCESSED SUCCESSFULLY!")',
            self.source,
            "the unqualified success banner is what made a 1-of-3, 102-second run "
            "read as a success",
        )
        self.assertIn("ALL {delivered_clips} CLIPS PROCESSED SUCCESSFULLY", self.source)
        self.assertIn("COMPLETED WITH SHORTFALL", self.source)

    def test_zero_clips_still_aborts(self) -> None:
        """The original guard must survive the shortfall change."""
        self.assertIn(
            "No clips were rendered. Refusing to report a successful run without publishable output.",
            self.source,
        )


class TestRunStateSettings(unittest.TestCase):
    """A shortfall has to land in the manifest, not only in a scrolling log."""

    def test_record_settings_exists_and_merges(self) -> None:
        from src.run_state import RunStateStore

        self.assertTrue(hasattr(RunStateStore, "record_settings"))
        signature = RunStateStore.record_settings.__doc__ or ""
        self.assertIn("merge", signature.lower())

    def test_record_settings_does_not_clobber_existing_keys(self) -> None:
        import inspect

        from src.run_state import RunStateStore

        source = inspect.getsource(RunStateStore.record_settings)
        self.assertIn(".update(", source, "must merge, not replace")
        self.assertNotIn("manifest.settings = values", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
