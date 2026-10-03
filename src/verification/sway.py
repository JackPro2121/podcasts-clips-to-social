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
SWAY_BLOCK_PX = 40.0

ANALYSIS_FPS = 15.0
SCALE_W, SCALE_H = 270, 480
POLICY_BAND_HZ = (0.20, 0.36)
MIN_SEGMENT_S = 2.0
CUT_JUMP_PX = 30.0
CUT_RESPONSE = 0.05


@dataclass
class TrajectorySway:
    """Sway of one axis' trajectory, per segment; worst segment reported."""

    p2p_px: float = 0.0
    freq_hz: float = 0.0
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
            "worst_axis": self.worst_axis,
            "worst_start_s": round(self.worst_start_s, 2),
            "worst_end_s": round(self.worst_end_s, 2),
            "segments": self.segments,
        }


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
    """Policy-band sway of one axis. Pure, so the arithmetic is unit-testable."""
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
        band = (freqs >= POLICY_BAND_HZ[0]) & (freqs <= POLICY_BAND_HZ[1])
        if not band.any():
            continue
        index = int(np.argmax(spectrum[band]))
        peak_amp = float(spectrum[band][index])
        if peak_amp * 2.0 > worst.p2p_px:
            worst.p2p_px = peak_amp * 2.0
            worst.freq_hz = float(freqs[band][index])
            worst.start_s = start / fps
            worst.end_s = end / fps
    if worst.p2p_px <= 0.0:
        worst.reason = "no policy-band peak found"
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
    if not worst.worst_axis:
        return SwayReport(ok=True, measurable=False, reason="no policy-band peak on either axis")
    return worst
