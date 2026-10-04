"""Loudness grading: the ceiling blocks, genuine quiet warns.

A real run blocked a clip at -2.6 dBFS true peak. That is a safe peak (the
ceiling is -1.5); platforms normalise loudness anyway, so quiet audio must not
cost a clip. The failure mode worth catching is the audio chain not engaging at
all, and that is a warning.
"""

from __future__ import annotations

import unittest

from src.verification import loudness


class TestGrade(unittest.TestCase):
    def test_in_band_passes_clean(self) -> None:
        report = loudness.grade(-14.1, -1.7)
        self.assertTrue(report.ok)
        self.assertFalse(report.warn_reason)
        self.assertEqual(report.severity, "ok")

    def test_lufs_error_blocks(self) -> None:
        report = loudness.grade(-16.0, -1.8)
        self.assertFalse(report.ok)
        self.assertIn("integrated loudness", report.reason)

    def test_peak_exactly_at_tolerance_passes(self) -> None:
        # -1.2 vs the -1.5 ceiling is exactly the 0.3 tolerance; float
        # subtraction makes it 0.30000000000000004. Run 37208970507's clip #1
        # measured -14.0 LUFS with this peak and was blocked by the rounding.
        report = loudness.grade(-14.0, -1.2)
        self.assertTrue(report.ok, report.reason)
        self.assertEqual(report.reason, "")

    def test_ceiling_is_strict(self) -> None:
        report = loudness.grade(-14.0, -1.0)
        self.assertFalse(report.ok)
        self.assertIn("ceiling", report.reason)

    def test_safe_quiet_peak_passes_clean(self) -> None:
        report = loudness.grade(-14.2, -2.6)
        self.assertTrue(report.ok, "safe quiet audio must not block a clip")
        self.assertFalse(report.warn_reason, "-2.6 is above the quiet floor; nothing to report")
        self.assertEqual(report.severity, "ok")

    def test_implausibly_quiet_peak_warns(self) -> None:
        report = loudness.grade(-14.0, -6.0)
        self.assertTrue(report.ok)
        self.assertIn("floor", report.warn_reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
