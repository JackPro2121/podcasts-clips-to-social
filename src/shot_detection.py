"""
Shot boundary detection: TransNetV2 (primary) with PySceneDetect (fallback).

`scene_classifier.detect_clip_shots` relies on PySceneDetect's AdaptiveDetector,
which is content-adaptive and therefore *misses hard cuts on low-motion
content* -- precisely the talking-head podcast case this pipeline exists for. A
cut between two similar shots produces almost no pixel delta, so an adaptive
content threshold never fires.

TransNetV2 is a small convolutional network trained specifically for shot
boundaries, so it does not have that blind spot. It is also a genuinely cheap
model (~few MB, 48x27 input), unlike a video LLM.

**The weights are not downloaded.** There is no verified canonical URL for a
TransNetV2 ONNX export, and this project does not invent URLs. Place the file at
`src/models/transnetv2.onnx` (exported from the soCzech/TransNetV2 reference
implementation) and the ONNX path engages automatically. Until then every call
falls through to PySceneDetect and behaviour is byte-identical to before.

The post-processing -- clamping to the requested window, merging sub-minimum
shots -- lives here and is shared by both detectors, so the fallback cannot
drift away from the primary over time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import (
    SHOT_DETECTOR,
    TRANSNET_MODEL_PATH,
    TRANSNET_MIN_CUT_CONFIDENCE,
    TRANSNET_WINDOW,
)

ShotList = List[Tuple[float, float]]


@dataclass
class ShotBoundaryResult:
    shots: ShotList
    source: str
    transition_count: int = 0
    mean_confidence: float = 0.0
    notes: List[str] = field(default_factory=list)


def normalise_shots(
    raw: Sequence[Tuple[float, float]],
    start_sec: float,
    end_sec: float,
    min_shot_duration: float = 0.4,
) -> ShotList:
    """
    Clamp boundaries to the window and merge shots shorter than the minimum.

    Shared by every detector so the fallback cannot diverge from the primary.
    A shot shorter than `min_shot_duration` is folded into its predecessor
    rather than dropped, because dropping it would punch a hole in the timeline
    and leave the renderer with a gap.
    """
    duration = max(0.1, end_sec - start_sec)
    single = [(round(start_sec, 2), round(end_sec, 2))]

    kept: ShotList = []
    for shot_start, shot_end in raw:
        clamped_start = max(start_sec, float(shot_start))
        clamped_end = min(end_sec, float(shot_end))
        if clamped_end - clamped_start > 0.1:
            kept.append((round(clamped_start, 2), round(clamped_end, 2)))
    if not kept:
        return single

    if kept[0][0] > start_sec:
        kept[0] = (round(start_sec, 2), kept[0][1])
    if kept[-1][1] < end_sec:
        kept[-1] = (kept[-1][0], round(end_sec, 2))

    merged: ShotList = []
    for shot_start, shot_end in kept:
        if not merged:
            merged.append((shot_start, shot_end))
        elif (shot_end - shot_start) < min_shot_duration:
            previous_start, _ = merged[-1]
            merged[-1] = (previous_start, shot_end)
        else:
            merged.append((shot_start, shot_end))

    # Merging can still leave a short trailing shot, and clamping guarantees the
    # total span, so the result is a strict partition of the window.
    if not merged:
        return single
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_shot_duration:
        head_start, _ = merged[-2]
        merged = merged[:-2] + [(head_start, merged[-1][1])]
    covered = sum(shot_end - shot_start for shot_start, shot_end in merged)
    if covered <= 0 or abs(covered - duration) > 0.5:
        return single
    return merged


def _detect_with_pyscenedetect(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    min_shot_duration: float,
) -> Tuple[ShotList, str]:
    """PySceneDetect AdaptiveDetector. The pre-existing behaviour, unchanged."""
    default_shot = [(round(start_sec, 2), round(end_sec, 2))]
    try:
        from scenedetect import (
            AdaptiveDetector,
            FrameTimecode,
            SceneManager,
            open_video,
        )
        video = open_video(str(video_path))
        fps = video.frame_rate
        if not fps or fps <= 0:
            return default_shot, "scenedetect"
        start_tc = FrameTimecode(timecode=start_sec, fps=fps)
        dur_tc = FrameTimecode(timecode=end_sec - start_sec, fps=fps)
        video.seek(start_tc)
        manager = SceneManager()
        manager.add_detector(AdaptiveDetector(
            adaptive_threshold=2.5,
            min_scene_len=max(2, int(fps * min_shot_duration)),
        ))
        manager.detect_scenes(video, frame_skip=2, duration=dur_tc)
        scenes = manager.get_scene_list()
        if not scenes:
            return default_shot, "scenedetect"
        raw = [
            (max(start_sec, float(scene[0].seconds)), min(end_sec, float(scene[1].seconds)))
            for scene in scenes
        ]
        return normalise_shots(raw, start_sec, end_sec, min_shot_duration), "scenedetect"
    except Exception as error:
        print(f"[-] Scene detection fallback (error: {error}). Using monolithic segment.")
        return [(round(start_sec, 2), round(end_sec, 2))], "single"


class TransNetV2Detector:
    """
    TransNetV2 over ONNX Runtime, with an OpenCV DNN fallback for runtimes
    without onnxruntime installed.

    The network sees 48x27 RGB frames in windows of 100 and emits three
    logits per frame: hard cut, gradual/fade, and no transition. Both transition
    classes are considered, because a dissolve mid-interview is still a shot
    boundary the renderer should not pan across.
    """

    INPUT_WIDTH = 48
    INPUT_HEIGHT = 27

    def __init__(self, session: Any, fps: float) -> None:
        self._session = session
        self._fps = max(1.0, float(fps))

    @classmethod
    def try_create(cls, fps: float) -> Optional["TransNetV2Detector"]:
        model_path = Path(TRANSNET_MODEL_PATH)
        if not model_path.exists():
            return None
        try:
            import onnxruntime  # type: ignore
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            session = onnxruntime.InferenceSession(
                str(model_path), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            return cls(session, fps)
        except Exception as error:
            print(f"[-] TransNetV2 via onnxruntime unavailable ({error}); trying OpenCV DNN.")
        try:
            import cv2
            net = cv2.dnn.readNetFromONNX(str(model_path))
            return cls(net, fps)
        except Exception as error:
            print(f"[-] TransNetV2 model unusable ({error}); using PySceneDetect.")
            return None

    def _infer(self, batch: Any) -> Any:
        try:
            return self._session.run(None, {self._session.get_inputs()[0].name: batch})[0]
        except AttributeError:
            # OpenCV DNN: no get_inputs(), and run() takes positional args.
            self._session.setInput(batch)
            return self._session.forward()

    @staticmethod
    def _prepare(frame: Any) -> Any:
        import cv2
        import numpy as np

        resized = cv2.resize(
            frame, (TransNetV2Detector.INPUT_WIDTH, TransNetV2Detector.INPUT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= 255.0
        return rgb

    def _transition_strengths(self, frames: Sequence[Any]) -> List[float]:
        """
        Per-frame probability that a shot boundary sits on that frame.

        Windows overlap by half so a boundary near a window edge is still seen
        whole by at least one window; predictions are averaged, which smooths the
        double-counted region instead of biasing it.
        """
        import numpy as np

        total = len(frames)
        if total < 2:
            return [0.0] * total
        window = max(2, int(TRANSNET_WINDOW))
        stride = max(1, window // 2)
        window = min(window, total)
        stride = max(1, min(stride, window - 1)) if window > 1 else 1

        accumulated = np.zeros(total, dtype=np.float64)
        counts = np.zeros(total, dtype=np.float64)
        positions = list(range(0, max(1, total - window + 1), stride)) or [0]
        if positions[-1] + window < total:
            positions.append(total - window)

        for start in positions:
            chunk = frames[start:start + window]
            if len(chunk) < 2:
                continue
            batch = np.stack([self._prepare(frame) for frame in chunk])[None, ...]
            output = self._infer(batch)
            probabilities = np.asarray(output)
            if probabilities.ndim == 3:
                probabilities = probabilities[0]
            if probabilities.ndim != 2 or probabilities.shape[0] < 2:
                continue
            # Rows are [cut, gradual, static]; both transition classes count.
            transitions = probabilities[:, :2].max(axis=1)
            transitions = np.clip(transitions, 0.0, 1.0)
            accumulated[start:start + len(transitions)] += transitions
            counts[start:start + len(transitions)] += 1.0

        return [
            float(accumulated[i] / counts[i]) if counts[i] else 0.0
            for i in range(total)
        ]

    def shots(
        self,
        frames: Sequence[Any],
        start_sec: float,
        end_sec: float,
        min_shot_duration: float,
    ) -> Tuple[ShotList, int, float]:
        strengths = self._transition_strengths(frames)
        threshold = max(0.0, min(1.0, TRANSNET_MIN_CUT_CONFIDENCE))
        boundaries: List[float] = []
        confidences: List[float] = []
        for index, strength in enumerate(strengths):
            # Frame 0 can never be a cut: the window starts there.
            if index == 0 or strength < threshold:
                continue
            boundaries.append(start_sec + (index - 1) / self._fps)
            confidences.append(strength)

        raw: List[Tuple[float, float]] = []
        cursor = start_sec
        for boundary in boundaries:
            if boundary - cursor < min_shot_duration:
                continue
            raw.append((cursor, boundary))
            cursor = boundary
        raw.append((cursor, end_sec))
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return (
            normalise_shots(raw, start_sec, end_sec, min_shot_duration),
            len(raw) - 1,
            mean_confidence,
        )


_detector_lock = threading.Lock()
_reported: Dict[str, bool] = {}


def detect_shot_boundaries(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    min_shot_duration: float = 0.4,
    sample_fps: float = 4.0,
) -> ShotBoundaryResult:
    """
    Resolve shot boundaries, preferring TransNetV2 and falling back cleanly.

    The ONNX path is only attempted when explicitly requested or when the
    weights are present, so a machine without them behaves exactly as it did
    before this module existed.
    """
    duration = max(0.1, end_sec - start_sec)
    requested = (SHOT_DETECTOR or "auto").strip().lower()

    if requested in ("auto", "transnet"):
        result = _try_transnet(
            video_path, start_sec, end_sec, min_shot_duration, sample_fps
        )
        if result is not None:
            return result
        if requested == "transnet" and not _reported.get("transnet_missing"):
            print("[-] SHOT_DETECTOR=transnet but no usable model; using PySceneDetect.")
            _reported["transnet_missing"] = True

    shots, source = _detect_with_pyscenedetect(
        video_path, start_sec, end_sec, min_shot_duration
    )
    return ShotBoundaryResult(
        shots=shots,
        source=source,
        transition_count=max(0, len(shots) - 1),
        notes=[f"duration={duration:.2f}s"],
    )


def _try_transnet(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    min_shot_duration: float,
    sample_fps: float,
) -> Optional[ShotBoundaryResult]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            return None
        detector = TransNetV2Detector.try_create(fps)
        if detector is None:
            return None

        step = max(1, int(round(fps / max(0.1, sample_fps))))
        start_frame = max(0, int(start_sec * fps))
        end_frame = max(start_frame + 2, int(end_sec * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: List[Any] = []
        index = start_frame
        while index < end_frame and len(frames) < 900:
            success, frame = capture.read()
            if not success or frame is None:
                break
            frames.append(frame)
            index += step
    finally:
        capture.release()

    if len(frames) < 2:
        return None

    # Frames were sampled every `step` frames, so the index-to-seconds mapping
    # inside shots() must use the *sampling* rate, not the source frame rate.
    effective_fps = fps / step
    if not _reported.get("transnet_ok"):
        print(f"[+] TransNetV2 shot detection active ({len(frames)} sampled frames).")
        _reported["transnet_ok"] = True

    detector._fps = effective_fps
    shots, cuts, confidence = detector.shots(
        frames, start_sec, end_sec, min_shot_duration
    )
    return ShotBoundaryResult(
        shots=shots,
        source="transnetv2",
        transition_count=cuts,
        mean_confidence=round(confidence, 4),
        notes=[
            f"frames={len(frames)}",
            f"sample_fps={effective_fps:.2f}",
            f"mean_confidence={confidence:.3f}",
        ],
    )


def reset_shot_detector_cache() -> None:
    with _detector_lock:
        _reported.clear()
