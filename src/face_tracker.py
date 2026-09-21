import cv2
import numpy as np
import os
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from src.scene_classifier import classify_frame_scene, detect_clip_shots

MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    center_x: int
    center_y: int

@dataclass
class ShotPlan:
    start: float  # Relative start seconds within clip (0.0 to clip duration)
    end: float    # Relative end seconds within clip
    mode: str     # 'presentation_slide', 'portrait_face', 'split_screen'
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
    center_y: int = 0
    # Timeline of center_x for each sample in this shot to enable LERP panning
    face_centers_timeline: List[Tuple[float, int]] = field(default_factory=list)
    speaker1_box: Optional[Tuple[int, int, int, int]] = None
    speaker2_box: Optional[Tuple[int, int, int, int]] = None
    margin_v: int = 460

@dataclass
class FramingDecision:
    mode: str  # 'multi_shot_dynamic', 'single_smooth', 'dynamic_cut', 'split_screen', 'blur_stack'
    face_count: int
    shots: List[ShotPlan] = field(default_factory=list)
    speaker1_box: Optional[Tuple[int, int, int, int]] = None
    speaker2_box: Optional[Tuple[int, int, int, int]] = None
    smoothed_center_x: int = 0
    smoothed_center_y: int = 0
    crop_x_expr: Optional[str] = None
    video_width: int = 1920
    video_height: int = 1080
    active_x: int = 0
    active_y: int = 0
    active_w: int = 1920
    active_h: int = 1080

def detect_letterbox_margins(cap: cv2.VideoCapture, start_frame: int, end_frame: int, max_samples: int = 15) -> Tuple[int, int, int, int]:
    """
    Detects baked-in black letterbox bars (cinematic 2:1 or 2.39:1 aspect ratio inside 16:9).
    Returns (active_x, active_y, active_w, active_h).
    """
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    span = max(1, end_frame - start_frame)
    step = max(1, span // max_samples)

    top_candidates = []
    bot_candidates = []

    saved_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    for fn in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row_means = np.mean(gray, axis=1)

        # Check top letterbox
        top = 0
        while top < h // 3 and row_means[top] < 12:
            top += 1

        # Check bottom letterbox
        bot = 0
        while bot < h // 3 and row_means[h - 1 - bot] < 12:
            bot += 1

        top_candidates.append(top)
        bot_candidates.append(bot)

    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

    if not top_candidates:
        return 0, 0, w, h

    med_top = int(np.median(top_candidates))
    med_bot = int(np.median(bot_candidates))

    # Ignore minor padding (< 12px)
    if med_top < 12:
        med_top = 0
    if med_bot < 12:
        med_bot = 0

    active_y = med_top
    active_h = max(100, h - med_top - med_bot)
    return 0, active_y, w, active_h

_yunet_attempted = False
_yunet_detector = None

def get_face_detector():
    """Initializes Face Detection with robust multi-tiered fallbacks (YuNet -> MediaPipe -> Haar Cascade)."""
    global _yunet_attempted, _yunet_detector
    
    # 1. Primary: OpenCV YuNet Face Detector (High-accuracy, profile faces, light-weight 232KB)
    if not _yunet_attempted:
        _yunet_attempted = True
        try:
            if not YUNET_MODEL_PATH.exists():
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                print(f"[*] Downloading high-precision YuNet face detector model...")
                r = requests.get(YUNET_URL, allow_redirects=True, timeout=15)
                if r.status_code == 200 and len(r.content) > 100000:
                    with open(YUNET_MODEL_PATH, "wb") as f:
                        f.write(r.content)
                    print(f"[+] YuNet model downloaded successfully ({len(r.content)/1024:.1f} KB).")
            if YUNET_MODEL_PATH.exists() and hasattr(cv2, 'FaceDetectorYN'):
                _yunet_detector = cv2.FaceDetectorYN.create(
                    str(YUNET_MODEL_PATH), "", (320, 320), 0.5, 0.3, 5000
                )
                print("[+] Initialized OpenCV YuNet multi-angle face detector.")
        except Exception as e:
            print(f"[-] YuNet initialization warning: {e}. Trying MediaPipe...")

    if _yunet_detector is not None:
        return _yunet_detector

    # 2. Secondary: MediaPipe Solutions (if available in environment)
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
            return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.55)
    except Exception:
        pass

    # 3. Tertiary: OpenCV Haar Cascade fallback
    try:
        local_cascade = 'haarcascade_frontalface_default.xml'
        if os.path.exists(local_cascade):
            cascade_path = local_cascade
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            return cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        print(f"[-] All face detectors failed: {e}.")
    return None

def detect_faces_in_frame(detector, frame: np.ndarray, width: int, height: int) -> List[FaceBox]:
    """Performs face detection across different detector backends."""
    faces: List[FaceBox] = []
    if detector is None or frame is None:
        return faces

    # Case A: OpenCV FaceDetectorYN (YuNet)
    if hasattr(cv2, 'FaceDetectorYN') and isinstance(detector, cv2.FaceDetectorYN):
        try:
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, detections = detector.detect(frame)
            if detections is not None:
                for d in detections:
                    fx = int(d[0])
                    fy = int(d[1])
                    fw = int(d[2])
                    fh = int(d[3])
                    conf = float(d[-1])
                    if conf >= 0.40 and fw >= 25 and fh >= 25:
                        faces.append(FaceBox(
                            x=max(0, fx), y=max(0, fy),
                            w=fw, h=fh,
                            center_x=max(0, fx) + fw // 2,
                            center_y=max(0, fy) + fh // 2
                        ))
            return faces
        except Exception:
            pass

    # Case B: MediaPipe Solutions
    if hasattr(detector, 'process'):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)
            if results and results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    fx = int(bbox.xmin * width)
                    fy = int(bbox.ymin * height)
                    fw = int(bbox.width * width)
                    fh = int(bbox.height * height)
                    faces.append(FaceBox(
                        x=max(0, fx), y=max(0, fy),
                        w=fw, h=fh,
                        center_x=max(0, fx) + fw // 2,
                        center_y=max(0, fy) + fh // 2
                    ))
            return faces
        except Exception:
            pass

    # Case C: Haar Cascade
    if hasattr(detector, 'detectMultiScale'):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, w_box, h_box) in rects:
                faces.append(FaceBox(
                    x=x, y=y, w=w_box, h=h_box,
                    center_x=x + w_box // 2,
                    center_y=y + h_box // 2
                ))
        except Exception:
            pass

    return faces

def analyze_faces_in_clip(
    video_path: Path,
    start_time: float,
    end_time: float,
    sample_fps: float = 2.5
) -> FramingDecision:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[-] Could not open video {video_path}. Defaulting to blur_stack.")
        return FramingDecision(mode='blur_stack', face_count=0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_step = max(1, int(fps / sample_fps))

    # Detect letterbox active content area
    active_x, active_y, active_w, active_h = detect_letterbox_margins(cap, start_frame, end_frame)
    if active_y > 0 or active_h < height:
        print(f"[*] Active Content Detected: Y={active_y}..{active_y + active_h} (Height: {active_h}px, stripped {height - active_h}px black bars)")

    target_crop_w = int(active_h * (9 / 16))
    if target_crop_w > active_w:
        target_crop_w = active_w

    detector = get_face_detector()
    detected_cuts = detect_clip_shots(video_path, start_time, end_time, min_shot_duration=0.4)
    print(f"[*] Visual Shot Segmentation: Detected {len(detected_cuts)} distinct camera cuts.")

    timeline_samples: List[Tuple[float, List[FaceBox], bool]] = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (current_frame - start_frame) % frame_step == 0:
            rel_t = (current_frame - start_frame) / fps
            faces: List[FaceBox] = []
            is_slide = False

            if (current_frame - start_frame) % (frame_step * 2) == 0:
                try:
                    scene_info = classify_frame_scene(frame)
                    if scene_info.get("is_presentation"):
                        is_slide = True
                except Exception:
                    pass

            faces = detect_faces_in_frame(detector, frame, width, height)
            faces.sort(key=lambda f: f.center_x)
            timeline_samples.append((rel_t, faces, is_slide))

        current_frame += 1

    cap.release()

    if not timeline_samples:
        return FramingDecision(
            mode='blur_stack', face_count=0, video_width=width, video_height=height,
            active_x=active_x, active_y=active_y, active_w=active_w, active_h=active_h
        )

    # Compute global anchor across entire clip for robust temporal anchoring
    all_cx = [f.center_x for _, flist, _ in timeline_samples for f in flist if active_x <= f.center_x <= active_x + active_w]
    global_anchor_cx = int(np.median(all_cx)) if all_cx else (active_x + active_w // 2)

    def is_valid_two_speaker_frame(face_pair: List[FaceBox]) -> bool:
        if len(face_pair) != 2:
            return False
        f1, f2 = face_pair[0], face_pair[1]
        x_dist = abs(f1.center_x - f2.center_x)
        y_dist = abs(f1.center_y - f2.center_y)
        area1 = f1.w * f1.h
        area2 = f2.w * f2.h
        if area1 == 0 or area2 == 0:
            return False
        area_ratio = area1 / area2
        return (
            x_dist >= active_w * 0.20 and
            y_dist <= active_h * 0.30 and
            0.30 <= area_ratio <= 3.3
        )

    clip_duration = round(end_time - start_time, 2)
    shot_plans: List[ShotPlan] = []
    last_known_cx = None

    for s_start, s_end in detected_cuts:
        rel_s = round(max(0.0, s_start - start_time), 2)
        rel_e = round(min(clip_duration, s_end - start_time), 2)
        if rel_e - rel_s < 0.2:
            continue

        shot_samples = [s for s in timeline_samples if rel_s <= s[0] <= rel_e]
        if not shot_samples:
            shot_samples = timeline_samples

        pres_count = sum(1 for s in shot_samples if s[2])
        pres_ratio = pres_count / max(1, len(shot_samples))

        valid_face_samples = [s[1] for s in shot_samples if s[1]]
        two_speaker_samples = [f for f in valid_face_samples if is_valid_two_speaker_frame(f)]

        # 1. Verified Presentation Slide (strictly requiring visual confirmation, not just lack of frontal faces)
        if pres_ratio >= 0.50:
            shot_plans.append(ShotPlan(
                start=rel_s,
                end=rel_e,
                mode='presentation_slide',
                crop_x=active_x,
                crop_y=active_y,
                crop_w=active_w,
                crop_h=active_h,
                margin_v=400
            ))
        # 2. Dynamic Split-Screen (Aspect-ratio preserving 9:8 pane crops)
        elif valid_face_samples and (len(two_speaker_samples) / len(valid_face_samples) >= 0.40):
            s1_cx = int(np.median([f[0].center_x for f in two_speaker_samples]))
            s2_cx = int(np.median([f[1].center_x for f in two_speaker_samples]))
            s1_cy = int(np.median([f[0].center_y for f in two_speaker_samples]))
            s2_cy = int(np.median([f[1].center_y for f in two_speaker_samples]))

            target_aspect = 9.0 / 8.0  # 1080x960 pane aspect ratio
            pane_h = active_h
            pane_w = int(pane_h * target_aspect)
            if pane_w > width:
                pane_w = width
                pane_h = int(pane_w / target_aspect)

            s1_x = max(0, min(s1_cx - pane_w // 2, width - pane_w))
            s1_y = active_y + max(0, min(s1_cy - pane_h // 2, active_h - pane_h))
            s2_x = max(0, min(s2_cx - pane_w // 2, width - pane_w))
            s2_y = active_y + max(0, min(s2_cy - pane_h // 2, active_h - pane_h))

            shot_plans.append(ShotPlan(
                start=rel_s,
                end=rel_e,
                mode='split_screen',
                speaker1_box=(s1_x, s1_y, pane_w, pane_h),
                speaker2_box=(s2_x, s2_y, pane_w, pane_h),
                margin_v=0  # Centered right on middle divider with \an5
            ))
        # 3. Portrait Solo Face (with temporal anchor memory)
        else:
            eye_level_y = active_y + active_h * 0.35
            timed_single_faces = [(s[0], min(s[1], key=lambda f: abs(f.center_y - eye_level_y))) for s in shot_samples if s[1]]
            
            single_faces = [f[1] for f in timed_single_faces] if timed_single_faces else []
            if not single_faces and valid_face_samples:
                single_faces = [min(faces, key=lambda f: abs(f.center_y - eye_level_y)) for faces in valid_face_samples]

            if single_faces:
                avg_cx = int(np.median([f.center_x for f in single_faces]))
                avg_cy = int(np.median([f.center_y for f in single_faces]))
                last_known_cx = avg_cx
            else:
                # Persistent speaker memory: Never default to dead-center width // 2!
                avg_cx = last_known_cx if last_known_cx is not None else global_anchor_cx
                avg_cy = int(eye_level_y)

            crop_x = max(active_x, min(avg_cx - target_crop_w // 2, active_x + active_w - target_crop_w))
            timeline = [(s, f.center_x) for s, f in timed_single_faces if isinstance(f, FaceBox)] if timed_single_faces else []

            shot_plans.append(ShotPlan(
                start=rel_s,
                end=rel_e,
                mode='portrait_face',
                crop_x=crop_x,
                crop_y=active_y,
                crop_w=target_crop_w,
                crop_h=active_h,
                center_y=avg_cy,
                face_centers_timeline=timeline,
                margin_v=460
            ))

    if shot_plans:
        shot_plans[0].start = 0.0
        shot_plans[-1].end = clip_duration
        for idx in range(len(shot_plans) - 1):
            shot_plans[idx].end = shot_plans[idx + 1].start

    modes = set(s.mode for s in shot_plans)
    if len(shot_plans) > 1 and (len(modes) > 1 or len(set(s.crop_x for s in shot_plans if s.mode == 'portrait_face')) > 1):
        return FramingDecision(
            mode='multi_shot_dynamic',
            face_count=len(shot_plans),
            shots=shot_plans,
            video_width=width,
            video_height=height,
            active_x=active_x,
            active_y=active_y,
            active_w=active_w,
            active_h=active_h
        )

    if modes == {'presentation_slide'}:
        return FramingDecision(
            mode='blur_stack', face_count=0, video_width=width, video_height=height,
            active_x=active_x, active_y=active_y, active_w=active_w, active_h=active_h
        )

    if modes == {'split_screen'}:
        s = shot_plans[0]
        return FramingDecision(
            mode='split_screen',
            face_count=2,
            speaker1_box=s.speaker1_box,
            speaker2_box=s.speaker2_box,
            video_width=width,
            video_height=height,
            active_x=active_x,
            active_y=active_y,
            active_w=active_w,
            active_h=active_h
        )

    first_shot = shot_plans[0] if shot_plans else None
    crop_x = first_shot.crop_x if first_shot else (active_x + (active_w - target_crop_w) // 2)
    return FramingDecision(
        mode='single_smooth',
        face_count=1,
        smoothed_center_x=crop_x + target_crop_w // 2,
        smoothed_center_y=first_shot.center_y if first_shot else (active_y + active_h // 3),
        video_width=width,
        video_height=height,
        active_x=active_x,
        active_y=active_y,
        active_w=active_w,
        active_h=active_h
    )
