import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    center_x: int
    center_y: int

@dataclass
class FramingDecision:
    mode: str  # 'single_smooth', 'split_screen', 'blur_stack'
    face_count: int
    speaker1_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) for top or single
    speaker2_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) for bottom (split-screen)
    smoothed_center_x: int = 0
    video_width: int = 1920
    video_height: int = 1080

def get_face_cascade() -> Optional[Any]:
    """Loads OpenCV's default frontal face Haar cascade with safe fallback."""
    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                return cascade
    except Exception as e:
        print(f"[-] Could not load face cascade: {e}")
    return None

def analyze_faces_in_clip(
    video_path: Path,
    start_time: float,
    end_time: float,
    sample_fps: float = 3.0
) -> FramingDecision:
    """
    Samples frames at sample_fps (e.g. 3 frames/sec) across the clip duration.
    Detects faces, determines layout type (1 face, 2 faces split, or group),
    and calculates smoothed camera coordinates.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[-] Could not open video {video_path}. Defaulting to blur_stack.")
        return FramingDecision(mode='blur_stack', face_count=0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    cascade = get_face_cascade()
    if cascade is None:
        print("[-] Face detector cascade unavailable. Defaulting to blur_stack framing.")
        cap.release()
        return FramingDecision(mode='blur_stack', face_count=0, video_width=width, video_height=height)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_step = max(1, int(fps / sample_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    face_counts: List[int] = []
    speaker1_centers_x: List[int] = []
    speaker2_centers_x: List[int] = []
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - start_frame) % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize for super-fast detection on CPU
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            detected = cascade.detectMultiScale(
                small_gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Map coordinates back to original resolution
            faces: List[FaceBox] = []
            for (x, y, w, h) in detected:
                rx, ry, rw, rh = x * 2, y * 2, w * 2, h * 2
                faces.append(FaceBox(
                    x=rx, y=ry, w=rw, h=rh,
                    center_x=rx + rw // 2,
                    center_y=ry + rh // 2
                ))

            # Sort faces from left to right
            faces.sort(key=lambda f: f.center_x)
            face_counts.append(len(faces))

            if len(faces) == 1:
                speaker1_centers_x.append(faces[0].center_x)
            elif len(faces) >= 2:
                speaker1_centers_x.append(faces[0].center_x)
                speaker2_centers_x.append(faces[1].center_x)

        current_frame += 1

    cap.release()

    if not face_counts:
        return FramingDecision(mode='blur_stack', face_count=0, video_width=width, video_height=height)

    # Median face count to filter out momentary false positives
    avg_face_count = int(round(np.median(face_counts)))

    # Decision Matrix:
    # 1 Face: Single speaker crop with camera smoothing
    if avg_face_count == 1:
        avg_cx = int(np.mean(speaker1_centers_x)) if speaker1_centers_x else width // 2
        # Ensure crop window fits within video bounds
        target_crop_w = int(height * (9 / 16))
        left_bound = max(0, min(avg_cx - target_crop_w // 2, width - target_crop_w))
        return FramingDecision(
            mode='single_smooth',
            face_count=1,
            smoothed_center_x=left_bound + target_crop_w // 2,
            video_width=width,
            video_height=height
        )

    # 2 Faces side-by-side in wide shot: Dynamic Split Screen
    elif avg_face_count == 2 and speaker1_centers_x and speaker2_centers_x:
        s1_cx = int(np.mean(speaker1_centers_x))
        s2_cx = int(np.mean(speaker2_centers_x))
        
        # Crop width for half-height pane (1080x960 -> aspect ratio 1080/960 = 9:8)
        half_crop_w = int(height * 0.9)
        s1_x = max(0, min(s1_cx - half_crop_w // 2, width - half_crop_w))
        s2_x = max(0, min(s2_cx - half_crop_w // 2, width - half_crop_w))

        return FramingDecision(
            mode='split_screen',
            face_count=2,
            speaker1_box=(s1_x, 0, half_crop_w, height),
            speaker2_box=(s2_x, 0, half_crop_w, height),
            video_width=width,
            video_height=height
        )

    # 3+ Faces (Panel discussion) or 0 Faces: Blurred Stack Failsafe
    else:
        return FramingDecision(
            mode='blur_stack',
            face_count=avg_face_count,
            video_width=width,
            video_height=height
        )
