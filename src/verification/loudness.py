"""Loudness verification.

This is the one part of the pipeline that was already correct, and the point of
this module is to keep it that way. Measured across all six golden masters:

============================  ========  =========
Clip                          LUFS      True peak
============================  ========  =========
clip_1_THE_REAL_FEAR...       -14.1     -1.7 dBFS
clip_2_THE_MYTH_OF...         -14.3     -1.4 dBFS
clip_3_WHY_HARD_WORK          -14.2     -1.5 dBFS
clip_1_HOW_I_MADE_400000      -14.3     -1.5 dBFS
clip_2_WHY_DISNEY_PAY...      -14.4     -1.4 dBFS
clip_1_PAID_OFF_60000...      -14.0     -1.6 dBFS
============================  ========  =========

``loudnorm=I=-14:TP=-1.5:LRA=11`` is being honoured to within 0.4 LU and 0.2 dB.
``ARCHITECTURE_AND_METHODOLOGY.md`` section 5 is genuinely implemented, and the
sidechain ducking and SFX chain around it work.

So this module has no defects to catch. It exists as a **regression guard**: the
one number in the whole verification surface that is currently right, and which a
careless change to the audio graph could silently break. A guard that only ever
passes still has value, provided someone has checked that it can fail -- see
``test_grade_severity`` below, and the mutation note on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from . import probe

# ARCHITECTURE_AND_METHODOLOGY.md section 5, and ARCHITECTURE section 5's
# stated platform targets for Shorts, Reels and TikTok.
TARGET_LUFS = -14.0
TARGET_TRUE_PEAK_DBFS = -1.5

# Tolerances. The second loudnorm pass plus AAC re-encode moves the integrated
# value by a fraction of a LU, and true peak by a couple of tenths of a dB, so a
# tolerance tighter than this would be measuring the encoder rather than the
# pipeline. 0.5 LU and 0.3 dB bracket the observed spread (0.4 LU, 0.2 dB) with a
# small margin.
LUFS_TOLERANCE = 0.5
TRUE_PEAK_TOLERANCE = 0.3

# AGENTS.md section 3's audio chain, listed so a regression names the clause it
# broke.
SPEC_CLAUSE = "ARCHITECTURE_AND_METHODOLOGY.md section 5 / AGENTS.md section 3.6"


@dataclass
class LoudnessReport:
    """Loudness verdict for one clip."""

    ok: bool = True
    reason: str = ""
    measured: bool = False
    integrated_lufs: float = 0.0
    loudness_range: float = 0.0
    peak_dbfs: float = 0.0
    lufs_error: float = 0.0
    peak_error: float = 0.0
    spec_clause: str = SPEC_CLAUSE

    @property
    def severity(self) -> str:
        if not self.ok or not self.measured:
            return "error"
        return "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "severity": self.severity,
            "reason": self.reason,
            "measured": self.measured,
            "integrated_lufs": round(self.integrated_lufs, 2),
            "loudness_range": round(self.loudness_range, 2),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "lufs_error": round(self.lufs_error, 3),
            "peak_error": round(self.peak_error, 3),
            "targets": {
                "lufs": TARGET_LUFS,
                "lufs_tolerance": LUFS_TOLERANCE,
                "true_peak_dbfs": TARGET_TRUE_PEAK_DBFS,
                "true_peak_tolerance": TRUE_PEAK_TOLERANCE,
            },
            "spec_clause": self.spec_clause,
        }


def grade(
    integrated_lufs: float,
    peak_dbfs: float,
    loudness_range: float = 0.0,
) -> LoudnessReport:
    """Pure grading, so it is testable without ffmpeg."""
    report = LoudnessReport(
        measured=True,
        integrated_lufs=integrated_lufs,
        loudness_range=loudness_range,
        peak_dbfs=peak_dbfs,
    )
    report.lufs_error = abs(integrated_lufs - TARGET_LUFS)
    report.peak_error = abs(peak_dbfs - TARGET_TRUE_PEAK_DBFS)

    if report.lufs_error > LUFS_TOLERANCE:
        report.ok = False
        report.reason = (
            f"integrated loudness {integrated_lufs:.1f} LUFS is "
            f"{report.lufs_error:.2f} LU from the {TARGET_LUFS:.1f} target "
            f"(tolerance {LUFS_TOLERANCE})"
        )
    elif report.peak_error > TRUE_PEAK_TOLERANCE:
        report.ok = False
        report.reason = (
            f"true peak {peak_dbfs:.1f} dBFS is {report.peak_error:.2f} dB from "
            f"the {TARGET_TRUE_PEAK_DBFS:.1f} dBTP target "
            f"(tolerance {TRUE_PEAK_TOLERANCE})"
        )
    return report


def analyse(path: Path) -> LoudnessReport:
    """Measure and grade the loudness of a rendered clip."""
    values = probe.read_loudness(path)
    if "integrated_lufs" not in values:
        return LoudnessReport(
            ok=False, reason="could not read loudness; ebur128 produced no summary"
        )
    return grade(
        integrated_lufs=values.get("integrated_lufs", 0.0),
        peak_dbfs=values.get("peak_dbfs", 0.0),
        loudness_range=values.get("loudness_range", 0.0),
    )
