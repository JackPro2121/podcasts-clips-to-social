"""Selfie-flip (mirrored) source detection.

A published clip in the corpus is horizontally flipped: the "SHURE SM7B"
microphone label reads right-to-left, and the subject and background are
consistently mirrored. There is no flip filter anywhere in the renderer, so the
flip is inherited from the source, and nothing in the pipeline noticed. Editorial
QA passed the clip.

**A single frame cannot tell you whether it is mirrored.** There is no universal
"correct" orientation for arbitrary footage, so any absolute heuristic --
bilateral symmetry, face-shape priors, lighting direction -- will fire on
legitimate content. The one thing that *does* carry the information is text,
because text has a canonical reading direction.

So this detector is **self-normalising**: it runs OCR on the frame and on the
horizontally flipped frame, and compares.

* Correctly oriented text is recognised in the original and not in the mirror.
* Mirrored text is the reverse.

Whichever orientation OCR reads better is the correct one. There is no absolute
threshold to tune and no content that can produce a false positive, because the
comparison is against the same content in both orientations.

Requirements and limits, stated plainly:

* **Needs OCR.** Without RapidOCR the answer is "not measurable". It is not
  guessed from symmetry, because a guessed answer here is worse than none.
* **Needs text in frame.** A clip with no text at all is not measurable, and that
  is most podcast footage. This is a best-effort check, not a gate on its own.
* **Reuses** :mod:`src.text_detection` rather than calling RapidOCR directly.
  That module already contains the fix for HANDOFF section 4 bug 12, where the
  OCR response was parsed as ``(box, text, confidence)`` but ``entry[1]`` -- the
  recognised *text* -- was read as the confidence. ``float("MOST PEOPLE")``
  raised, the handler swallowed it, and the detector silently produced zero
  regions while appearing to work. Reusing the adapter means that bug cannot be
  reintroduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import probe

# Sample count. Enough to average out per-frame OCR noise, few enough that the
# check stays cheap; OCR is the expensive part, not the decoding.
DEFAULT_SAMPLES = 6

# Decode size for OCR. RapidOCR is tuned for roughly this scale and going larger
# only costs time.
OCR_WIDTH = 960
OCR_HEIGHT = 1280

# A frame counts as "readable" only above this mean confidence. Below it the
# frame is not evidence either way.
MIN_READABLE_CONFIDENCE = 0.55

# The mirrored orientation must beat the original by this ratio, on the
# aggregate, for the clip to be called mirrored. A ratio rather than an absolute
# gap, because confidence scales differ between engines and content.
MIRROR_DOMINANCE_RATIO = 1.15

# A clip needs at least this many readable frames before it is judged.
MIN_READABLE_FRAMES = 2


@dataclass
class MirrorReport:
    """Mirror verdict for one clip."""

    ok: bool = True
    measurable: bool = False
    reason: str = ""
    mirrored: bool = False
    samples: int = 0
    readable_frames: int = 0
    original_confidence: float = 0.0
    mirrored_confidence: float = 0.0
    dominance_ratio: float = 1.0
    boxes_original: int = 0
    boxes_mirrored: int = 0
    detector: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def severity(self) -> str:
        if self.mirrored:
            return "error"
        if not self.measurable:
            return "info"
        return "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "measurable": self.measurable,
            "severity": self.severity,
            "reason": self.reason,
            "mirrored": self.mirrored,
            "samples": self.samples,
            "readable_frames": self.readable_frames,
            "original_confidence": round(self.original_confidence, 4),
            "mirrored_confidence": round(self.mirrored_confidence, 4),
            "dominance_ratio": round(self.dominance_ratio, 4),
            "boxes_original": self.boxes_original,
            "boxes_mirrored": self.boxes_mirrored,
            "detector": self.detector,
            "notes": self.notes,
        }


def _mean_confidence(boxes: List[Any]) -> float:
    if not boxes:
        return 0.0
    return sum(float(box.confidence) for box in boxes) / len(boxes)


def classify(
    original_boxes: List[Any],
    mirrored_boxes: List[Any],
    original_confidence: float,
    mirrored_confidence: float,
    samples: int,
) -> MirrorReport:
    """Pure decision, so the threshold behaviour is testable without OCR.

    ``original_boxes`` / ``mirrored_boxes`` are the detected regions in each
    orientation; they are used for reporting and as a tiebreak, not for the
    primary signal.
    """
    report = MirrorReport(
        samples=samples,
        readable_frames=0,
        original_confidence=original_confidence,
        mirrored_confidence=mirrored_confidence,
        boxes_original=len(original_boxes),
        boxes_mirrored=len(mirrored_boxes),
    )
    best_original = max(original_confidence, 0.0)
    best_mirrored = max(mirrored_confidence, 0.0)

    if best_original < MIN_READABLE_CONFIDENCE and best_mirrored < MIN_READABLE_CONFIDENCE:
        report.reason = (
            f"no readable text in either orientation "
            f"(original {best_original:.2f}, mirrored {best_mirrored:.2f}); "
            f"mirror state is not determinable from this clip"
        )
        return report

    report.readable_frames = 1
    report.measurable = True
    report.dominance_ratio = (
        best_mirrored / best_original if best_original > 1e-6 else float("inf")
    )
    report.mirrored = report.dominance_ratio >= MIRROR_DOMINANCE_RATIO
    if report.mirrored:
        report.reason = (
            f"OCR reads the horizontally flipped frame {report.dominance_ratio:.2f}x "
            f"better ({best_mirrored:.2f} vs {best_original:.2f}), which indicates "
            f"selfie-flipped source footage"
        )
    else:
        report.reason = (
            f"OCR reads the original frame better "
            f"({best_original:.2f} vs {best_mirrored:.2f}); orientation is correct"
        )
    return report


def analyse(path: Path, samples: int = DEFAULT_SAMPLES) -> MirrorReport:
    """Detect selfie-flipped source footage by comparing OCR across orientations."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return MirrorReport(ok=False, reason="numpy is not installed")

    try:
        from src.text_detection import detect_text_regions, get_text_detector
    except Exception as error:  # pragma: no cover - import guard
        return MirrorReport(ok=False, reason=f"text detection unavailable: {error}")

    try:
        detector = get_text_detector()
    except Exception as error:
        return MirrorReport(ok=False, reason=f"no text detector: {error}")

    frames, reason = probe.decode_gray_frames(
        path, OCR_WIDTH, OCR_HEIGHT, max_frames=samples * 8
    )
    if not frames:
        return MirrorReport(ok=False, reason=reason or "no frames decoded")

    detector_name = getattr(detector, "name", "unknown")
    if "ocr" not in str(detector_name).lower():
        return MirrorReport(
            ok=False,
            reason=(
                f"active text detector is '{detector_name}', not OCR. Mirror "
                f"detection needs OCR; set TEXT_DETECTOR=ocr to enable it."
            ),
            detector=str(detector_name),
        )

    # Evenly pick readable samples from what was decoded.
    if len(frames) <= samples:
        chosen = frames
    else:
        step = len(frames) / float(samples)
        chosen = [frames[int(index * step)] for index in range(samples)]

    original_scores: List[float] = []
    mirrored_scores: List[float] = []
    original_boxes: List[Any] = []
    mirrored_boxes: List[Any] = []

    for gray in chosen:
        # decode_gray_frames returns greyscale; OCR wants BGR.
        bgr = np.stack([gray] * 3, axis=-1)
        flipped = bgr[:, ::-1, :]
        try:
            forward = list(detect_text_regions(bgr.copy(), force="ocr"))
            reverse = list(detect_text_regions(flipped.copy(), force="ocr"))
        except Exception:
            # A detector that throws is not evidence in either direction.
            continue
        original_boxes.extend(forward)
        mirrored_boxes.extend(reverse)
        original_scores.append(_mean_confidence(forward))
        mirrored_scores.append(_mean_confidence(reverse))

    if not original_scores:
        return MirrorReport(
            ok=False,
            reason="OCR produced no results on any sampled frame",
            detector=str(detector_name),
        )

    report = classify(
        original_boxes,
        mirrored_boxes,
        max(original_scores) if original_scores else 0.0,
        max(mirrored_scores) if mirrored_scores else 0.0,
        len(chosen),
    )
    report.detector = str(detector_name)
    if report.measurable and report.readable_frames < MIN_READABLE_FRAMES:
        report.notes.append(
            f"only {report.readable_frames} sample(s) carried readable text; "
            f"the verdict rests on a thin margin"
        )
    return report
