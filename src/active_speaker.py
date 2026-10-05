"""ASD-Lite: which visible face is speaking, per time window.

Evidence (run 37347481817, clip_2 "THE LONELINESS EPIDEMIC..."): the clip
showed one face for 498/499 frames whose mouth-region motion correlated with
the voice at only +0.09, and a +-5s lag scan peaked at |0.17| with no
alignment -- so the picture was not time-shifted, the visible face simply was
not the speaker. The show is multi-speaker (8 faces detected, zero cuts) and
the composition picks crops on face geometry, which is speech-blind.

This module is the speech signal the composition lacks, CPU-only (YuNet faces,
mouth-region frame difference, audio RMS envelope). The decision logic is pure
(``scores_from_series`` / ``windowed_speakers``) so it is unit-testable with
synthetic series where the talker is known by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

WINDOW_S = 1.0
FRAME_WIDTH, FRAME_HEIGHT = 480, 270
# A face must clear both: some correlation with the voice, and real mouth
# movement (a still photo or a listener cannot win by accident).
SPEAKER_MIN_CORR = 0.15
SPEAKER_MIN_MOTION = 0.8


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    import numpy as np

    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    first = np.asarray(a[:n], dtype=np.float64)
    second = np.asarray(b[:n], dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else 0.0


def scores_from_series(
    face_motion: Sequence[Sequence[float]], envelope: Sequence[float]
) -> List[float]:
    """Correlation of each face's mouth motion with the voice envelope."""
    return [pearson(motion, envelope) for motion in face_motion]


def pick_speaker(face_motion: Sequence[Sequence[float]], envelope: Sequence[float]) -> int:
    """Index of the speaking face, or -1 when no face clears the thresholds."""
    import numpy as np

    best_index, best_score = -1, 0.0
    for index, motion in enumerate(face_motion):
        if len(motion) < 8 or float(np.mean(motion)) < SPEAKER_MIN_MOTION:
            continue
        score = pearson(motion, envelope)
        if score > best_score:
            best_index, best_score = index, score
    if best_score < SPEAKER_MIN_CORR:
        return -1
    return best_index


def windowed_speakers(
    face_motion: Sequence[Sequence[float]],
    envelope: Sequence[float],
    fps: float,
    window_s: float = WINDOW_S,
) -> List[Tuple[float, int, float]]:
    """Per-window ``(start_s, speaker_index, score)``; -1 when no talker.

    This is what the composition should consult: the speaker can change between
    windows, and a window with no confident talker (music, laughter, both
    silent) must not force a switch.
    """
    hop = max(1, int(round(fps * window_s)))
    frame_count = min(
        [len(motion) for motion in face_motion] + [len(envelope)]
    ) if face_motion else 0
    results: List[Tuple[float, int, float]] = []
    for start in range(0, frame_count - hop + 1, hop):
        window_motion = [list(motion[start:start + hop]) for motion in face_motion]
        window_envelope = list(envelope[start:start + hop])
        index = pick_speaker(window_motion, window_envelope)
        score = (
            pearson(window_motion[index], window_envelope) if index >= 0 else 0.0
        )
        results.append((start / fps, index, score))
    return results


def _extract_frames(clip: Path, fps: int) -> List[Any]:
    import numpy as np
    import subprocess

    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(clip),
            "-vf", f"fps={fps},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        capture_output=True,
    )
    raw = result.stdout
    size = FRAME_WIDTH * FRAME_HEIGHT * 3
    return [
        np.frombuffer(raw[i * size:(i + 1) * size], dtype=np.uint8).reshape(
            FRAME_HEIGHT, FRAME_WIDTH, 3
        )
        for i in range(len(raw) // size)
    ]


def _audio_envelope(clip: Path, fps: int) -> List[float]:
    import numpy as np

    from src.verification import av_sync

    samples = av_sync._decode_mono(clip, 16000)
    if samples is None or samples.size == 0:
        return []
    hop = max(1, int(16000 / fps))
    return [
        float(np.sqrt(np.mean(samples[i:i + hop] ** 2)))
        for i in range(0, max(1, samples.size - hop), hop)
    ]


def _mouth_patch(grey: Any, box: Any) -> Any:
    import cv2
    import numpy as np

    x0 = int(max(0, box.x))
    y0 = int(max(0, box.y + box.height * 0.55))
    x1 = int(min(grey.shape[1], box.x + box.width))
    y1 = int(min(grey.shape[0], box.y + box.height))
    patch = grey[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((10, 20), dtype=np.float32)
    return cv2.resize(patch, (20, 10)).astype(np.float32)


def track_mouth_motion(clip: Path, fps: int = 10) -> List[List[float]]:
    """Per-face-track mouth-motion series (frames-indexed, greedy nearest)."""
    import numpy as np

    from src.verification import faces as faces_mod

    frames = _extract_frames(clip, fps)
    detector, _reason = faces_mod._load_yunet()
    if not frames or detector is None:
        return []
    tracks: List[dict] = []
    for frame in frames:
        grey = frame.mean(axis=2)
        boxes = faces_mod.detect_in_frame(frame, detector) or []
        used: set[int] = set()
        for track in tracks:
            last = track["last"]
            best, best_dist = None, 1e9
            for box_index, box in enumerate(boxes):
                if box_index in used:
                    continue
                dist = abs((box.x + box.width / 2) - (last.x + last.width / 2)) + abs(
                    (box.y + box.height / 2) - (last.y + last.height / 2)
                )
                if dist < best_dist:
                    best, best_dist = box_index, dist
            if best is not None and best_dist < 0.25 * FRAME_WIDTH:
                used.add(best)
                box = boxes[best]
                patch = _mouth_patch(grey, box)
                track["motion"].append(float(np.mean(np.abs(patch - track["prev"]))))
                track["prev"] = patch
                track["last"] = box
        for box_index, box in enumerate(boxes):
            if box_index not in used and len(tracks) < 4:
                tracks.append(
                    {"motion": [], "prev": _mouth_patch(grey, box), "last": box}
                )
    return [track["motion"] for track in tracks]


def analyse(clip: Path, fps: int = 10, window_s: float = WINDOW_S) -> List[Tuple[float, int, float]]:
    """Per-window speaker index for a (source or rendered) clip."""
    motion = track_mouth_motion(clip, fps=fps)
    if not motion:
        return []
    envelope = _audio_envelope(clip, fps)
    if not envelope:
        return []
    return windowed_speakers(motion, envelope, float(fps), window_s)
