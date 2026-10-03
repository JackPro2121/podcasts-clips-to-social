"""Camera-sway measurement for rendered clips.

The user-visible complaint was shaking: the composed frame oscillated
horizontally and vertically at the motion guarantee's own frequency. Nothing in
the verifier measured it, so both the 107px per-branch drift and the later
15-37px double-oscillator stack shipped.

Method
------
Decode the clip at 15fps, measure the global translation between consecutive
frames with phase correlation, split the trajectory at camera cuts, remove any
linear pan per segment, and read the amplitude of the motion policy's own
frequency band (0.28Hz +/- 0.08). Content-driven movement is broadband and does
not produce a sharp peak in that band, so measuring the band, not total travel,
is what separates policy oscillation from a person leaning.

Thresholds
----------
``SWAY_BLOCK_PX`` currently catches the pre-fix class (45-74px) and leaves the
renderer-verified 15-37px clips as warnings. It tightens once a full pipeline
run renders with the single-mechanism motion policy; the number will come from
that measurement, not from a guess.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

CODE_SWAY = "sway"
SWAY_WARN_PX = 14.0
# The policy oscillator is 12px; a repair attempt can raise its gain and the
# renderer clamps that at roughly 20px, so 25px is the first value that is
# unambiguously a regression rather than an escalated repair.
SWAY_BLOCK_PX = 25.0
# Broadband gate for gross camera shake (the 107px per-branch class) regardless
# of frequency. Content motion almost never translates the whole frame by 60px;
# the previous single band flagged a speaker's own 0.3Hz sway as camera shake.
BROADBAND_BLOCK_PX = 60.0
BROADBAND_WARN_PX = 30.0
BROADBAND_BAND_HZ = (0.15, 1.5)

ANALYSIS_FPS = 15.0
SCALE_W, SCALE_H = 270, 480
# The metric must be tuned to the oscillator the renderer actually uses, or it
# measures the subject instead: corpus content sway at 0.22-0.33Hz was being
# reported as policy drift while the policy runs at 0.5Hz.
try:
    from src.creative_spec import MOTION_POLICY

    POLICY_BAND_HZ: tuple = (
        MOTION_POLICY.FREQUENCY_HZ - 0.10,
        MOTION_POLICY.FREQUENCY_HZ + 0.10,
    )
except Exception:  # pragma: no cover - defensive
    POLICY_BAND_HZ = (0.40, 0.60)
MIN_SEGMENT_S = 2.0
CUT_JUMP_PX = 30.0
CUT_RESPONSE = 0.05


@dataclass
class TrajectorySway:
    """Sway of one axis' trajectory, per segment; worst segment reported."""

    p2p_px: float = 0.0
    freq_hz: float = 0.0
    broadband_p2p_px: float = 0.0
    broadband_freq_hz: float = 0.0
    start_s: float = 0.0
    end_s: float = 0.0
    segments: int = 0
    reason: str = ""


@dataclass
class SwayReport:
    ok: bool = True
    reason: str = ""
    measurable: bool = False
    worst_p2p_px: float = 0.0
    worst_freq_hz: float = 0.0
    worst_broadband_p2p_px: float = 0.0
    worst_broadband_freq_hz: float = 0.0
    worst_axis: str = ""
    worst_start_s: float = 0.0
    worst_end_s: float = 0.0
    segments: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "measurable": self.measurable,
            "worst_p2p_px": round(self.worst_p2p_px, 2),
            "worst_freq_hz": round(self.worst_freq_hz, 4),
            "worst_broadband_p2p_px": round(self.worst_broadband_p2p_px, 2),
            "worst_broadband_freq_hz": round(self.worst_broadband_freq_hz, 4),
            "policy_band_hz": [round(value, 3) for value in POLICY_BAND_HZ],
            "worst_axis": self.worst_axis,
            "worst_start_s": round(self.worst_start_s, 2),
            "worst_end_s": round(self.worst_end_s, 2),
            "segments": self.segments,
        }


def _band_peak(freqs, spectrum, low: float, high: float) -> Tuple[float, float]:
    import numpy as np

    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        return 0.0, 0.0
    index = int(np.argmax(spectrum[band]))
    return float(spectrum[band][index]), float(freqs[band][index])


def _segments_from_jumps(
    trajectory: List[float], fps: float
) -> List[Tuple[int, int]]:
    cuts = [0]
    for index in range(1, len(trajectory)):
        if abs(trajectory[index] - trajectory[index - 1]) > CUT_JUMP_PX:
            cuts.append(index)
    cuts.append(len(trajectory))
    minimum = int(MIN_SEGMENT_S * fps)
    return [
        (start, end)
        for start, end in zip(cuts, cuts[1:])
        if end - start >= minimum
    ]


def analyse_trajectory(trajectory: List[float], fps: float = ANALYSIS_FPS) -> TrajectorySway:
    """Policy-band and broadband sway of one axis. Pure, so it is unit-testable."""
    import numpy as np

    if len(trajectory) < int(MIN_SEGMENT_S * fps):
        return TrajectorySway(reason="too short to measure")

    segments = _segments_from_jumps(trajectory, fps)
    if not segments:
        return TrajectorySway(reason="no segment long enough after cut splitting")

    worst = TrajectorySway(segments=len(segments))
    for start, end in segments:
        values = np.asarray(trajectory[start:end], dtype=np.float64)
        values = values - np.linspace(values[0], values[-1], len(values))
        windowed = values * np.hanning(len(values))
        spectrum = np.abs(np.fft.rfft(windowed)) / len(values) * 2.0
        freqs = np.fft.rfftfreq(len(values), d=1.0 / fps)
        amp, freq = _band_peak(freqs, spectrum, *POLICY_BAND_HZ)
        if amp * 2.0 > worst.p2p_px:
            worst.p2p_px = amp * 2.0
            worst.freq_hz = freq
            worst.start_s = start / fps
            worst.end_s = end / fps
        broad_amp, broad_freq = _band_peak(freqs, spectrum, *BROADBAND_BAND_HZ)
        if broad_amp * 2.0 > worst.broadband_p2p_px:
            worst.broadband_p2p_px = broad_amp * 2.0
            worst.broadband_freq_hz = broad_freq
    if worst.p2p_px <= 0.0 and worst.broadband_p2p_px <= 0.0:
        worst.reason = "no band peak found"
    return worst


def _decode_gray(path: Path) -> Tuple[List[Any], str]:
    try:
        import numpy as np
    except ImportError:
        return [], "numpy is not installed"
    command = [
        "ffmpeg", "-v", "error", "-i", str(path.resolve()),
        "-vf", f"fps={ANALYSIS_FPS},scale={SCALE_W}:{SCALE_H}:flags=area,format=gray",
        "-f", "rawvideo", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], f"decode failed: {error}"
    if result.returncode != 0 or not result.stdout:
        return [], "decode produced no frames"
    frame_bytes = SCALE_W * SCALE_H
    count = len(result.stdout) // frame_bytes
    if count < 2:
        return [], "fewer than two frames decoded"
    return list(
        np.frombuffer(result.stdout[: count * frame_bytes], dtype=np.uint8).reshape(
            count, SCALE_H, SCALE_W
        )
    ), ""


def analyse(path: Path) -> SwayReport:
    """Measure camera sway for one rendered clip. Never raises."""
    if not path.exists():
        return SwayReport(ok=False, reason=f"file not found: {path}")
    try:
        import cv2
    except ImportError:
        return SwayReport(ok=False, reason="OpenCV is not installed")

    frames, reason = _decode_gray(path)
    if not frames:
        return SwayReport(ok=False, reason=reason)

    window = cv2.createHanningWindow((SCALE_W, SCALE_H), cv2.CV_32F)
    dx = [0.0]
    dy = [0.0]
    previous = frames[0].astype("float32")
    for frame in frames[1:]:
        current = frame.astype("float32")
        (shift_x, shift_y), response = cv2.phaseCorrelate(previous, current, window)
        if response >= CUT_RESPONSE and abs(shift_x) <= CUT_JUMP_PX and abs(shift_y) <= CUT_JUMP_PX:
            dx.append(dx[-1] + shift_x)
            dy.append(dy[-1] + shift_y)
        else:
            # A camera cut: restart both trajectories so the jump is not read as motion.
            dx.append(dx[-1])
            dy.append(dy[-1])
            dx[-1] = 0.0
            dy[-1] = 0.0
        previous = current

    scale = 1080.0 / SCALE_W
    worst = SwayReport(measurable=True)
    for axis, trajectory in (("x", dx), ("y", dy)):
        axis_report = analyse_trajectory([value * scale for value in trajectory])
        if axis_report.p2p_px > worst.worst_p2p_px:
            worst.worst_p2p_px = axis_report.p2p_px
            worst.worst_freq_hz = axis_report.freq_hz
            worst.worst_axis = axis
            worst.worst_start_s = axis_report.start_s
            worst.worst_end_s = axis_report.end_s
            worst.segments = axis_report.segments
        if axis_report.broadband_p2p_px > worst.worst_broadband_p2p_px:
            worst.worst_broadband_p2p_px = axis_report.broadband_p2p_px
            worst.worst_broadband_freq_hz = axis_report.broadband_freq_hz
    if not worst.worst_axis and worst.worst_broadband_p2p_px <= 0.0:
        return SwayReport(ok=True, measurable=False, reason="no band peak on either axis")
    if not worst.worst_axis:
        worst.worst_axis = "x"
    worst.measurable = True
    return worst
