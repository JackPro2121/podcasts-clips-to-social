import cv2
import numpy as np
import os
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from src.scene_classifier import classify_frame_scene, detect_clip_shots

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
    center_y: int = 0
    speaker1_box: Optional[Tuple[int, int, int, int]] = None
    speaker2_box: Optional[Tuple[int, int, int, int]] = None
    margin_v: int = 220  # Safe zone for subtitles (420 for presentation_slide, 220 for portrait_face)

@dataclass
class FramingDecision:
    mode: str  # 'multi_shot_dynamic', 'single_smooth', 'dynamic_cut', 'split_screen', 'blur_stack'
    face_count: int
    shots: List[ShotPlan] = field(default_factory=list)
    speaker1_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) for top or single
    speaker2_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) for bottom (split-screen)
    smoothed_center_x: int = 0
    smoothed_center_y: int = 0  # Intelligent face-height tracking
    crop_x_expr: Optional[str] = None  # Dynamic FFmpeg expression for multi-camera angle switching
    video_width: int = 1920
    video_height: int = 1080

def get_face_detector(width: int, height: int) -> Tuple[str, Any]:
    """
    Initializes the most accurate face detector available:
    1. Primary: OpenCV YuNet Deep Learning Detector (Fast, accurate on profiles/glasses/dark lighting)
    2. Fallback: Haar Cascade Classifier
    3. Fallback: None (triggers blur_stack)
    """
    model_dir = Path(__file__).resolve().parent / "models"
    model_path = model_dir / "face_detection_yunet_2023mar.onnx"

    # Auto-download YuNet ONNX model if not already present
    if not model_path.exists():
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            print(f"[*] Downloading modern YuNet face detector model to {model_path}...")
            r = requests.get(url, allow_redirects=True, timeout=20)
            if r.status_code == 200 and len(r.content) > 100000:
                with open(model_path, "wb") as f:
                    f.write(r.content)
                print(f"[+] YuNet model downloaded successfully ({len(r.content) / 1024:.1f} KB).")
        except Exception as e:
            print(f"[-] Could not auto-download YuNet model: {e}")

    # 1. Try YuNet
    if model_path.exists() and hasattr(cv2, "FaceDetectorYN_create"):
        try:
            detector = cv2.FaceDetectorYN_create(str(model_path), "", (width, height), score_threshold=0.70)
            return "yunet", detector
        except Exception as e:
            print(f"[-] YuNet init failed: {e}. Trying Haar fallback...")

    # 2. Try Haar Cascade
    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    return "haar", cascade
    except Exception as e:
        print(f"[-] Haar cascade init failed: {e}")

    return "none", None

def analyze_faces_in_clip(
    video_path: Path,
    start_time: float,
    end_time: float,
    sample_fps: float = 2.5
) -> FramingDecision:
    """
    Universal multi-camera & shot-adaptive face analysis:
    1. Samples faces at sample_fps across the entire clip.
    2. Identifies if shot is single-speaker, multi-camera switching, or side-by-side wide angle.
    3. Generates precise FFmpeg crop expressions or split-screen layouts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[-] Could not open video {video_path}. Defaulting to blur_stack.")
        return FramingDecision(mode='blur_stack', face_count=0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    target_crop_w = int(height * (9 / 16))

    detector_type, detector = get_face_detector(width, height)
    if detector_type == "none":
        print("[-] No face detector available. Defaulting to blur_stack framing.")
        cap.release()
        return FramingDecision(mode='blur_stack', face_count=0, video_width=width, video_height=height)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_step = max(1, int(fps / sample_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Detect visual shot boundaries using PySceneDetect (0.4s catches sub-second reaction cuts)
    detected_cuts = detect_clip_shots(video_path, start_time, end_time, min_shot_duration=0.4)
    print(f"[*] Visual Shot Segmentation: Detected {len(detected_cuts)} distinct camera cuts.")

    timeline_samples: List[Tuple[float, List[FaceBox], bool]] = []
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - start_frame) % frame_step == 0:
            rel_t = (current_frame - start_frame) / fps
            faces: List[FaceBox] = []
            is_slide = False

            # Check for presentation/slides every 2nd sample (~1.25 Hz)
            if (current_frame - start_frame) % (frame_step * 2) == 0:
                try:
                    scene_info = classify_frame_scene(frame)
                    if scene_info.get("is_presentation"):
                        is_slide = True
                except Exception:
                    pass

            if detector_type == "yunet":
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, det_faces = detector.detect(frame)
                if det_faces is not None:
                    for f in det_faces:
                        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        faces.append(FaceBox(
                            x=fx, y=fy, w=fw, h=fh,
                            center_x=fx + fw // 2,
                            center_y=fy + fh // 2
                        ))

            elif detector_type == "haar":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                detected = detector.detectMultiScale(small_gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in detected:
                    rx, ry, rw, rh = x * 2, y * 2, w * 2, h * 2
                    faces.append(FaceBox(
                        x=rx, y=ry, w=rw, h=rh,
                        center_x=rx + rw // 2,
                        center_y=ry + rh // 2
                    ))

            faces.sort(key=lambda f: f.center_x)
            timeline_samples.append((rel_t, faces, is_slide))

        current_frame += 1

    cap.release()

    if not timeline_samples:
        return FramingDecision(mode='blur_stack', face_count=0, video_width=width, video_height=height)

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
            x_dist >= width * 0.22 and
            y_dist <= height * 0.22 and
            0.35 <= area_ratio <= 2.8
        )

    clip_duration = round(end_time - start_time, 2)
    shot_plans: List[ShotPlan] = []

    for s_start, s_end in detected_cuts:
        rel_s = round(max(0.0, s_start - start_time), 2)
        rel_e = round(min(clip_duration, s_end - start_time), 2)
        if rel_e - rel_s < 0.2:
            continue

        # Find samples belonging to this shot
        shot_samples = [s for s in timeline_samples if rel_s <= s[0] <= rel_e]
        if not shot_samples:
            # Fallback to general samples
            shot_samples = timeline_samples

        pres_count = sum(1 for s in shot_samples if s[2])
        pres_ratio = pres_count / max(1, len(shot_samples))

        valid_face_samples = [s[1] for s in shot_samples if s[1]]
        face_presence = len(valid_face_samples) / max(1, len(shot_samples))
        two_speaker_samples = [f for f in valid_face_samples if is_valid_two_speaker_frame(f)]

        # Classify this individual shot
        if pres_ratio >= 0.35 or face_presence < 0.35 or not valid_face_samples:
            # Shot is a presentation slide / document / chart
            shot_plans.append(ShotPlan(
                start=rel_s,
                end=rel_e,
                mode='presentation_slide',
                margin_v=420
            ))
            print(f"    Shot [{rel_s:.1f}s - {rel_e:.1f}s]: Presentation Slide/Infographic (Pres ratio: {pres_ratio*100:.0f}%, Face presence: {face_presence*100:.0f}%)")

        elif valid_face_samples and (len(two_speaker_samples) / len(valid_face_samples) >= 0.40):
            # Shot has 2 genuine speakers
            s1_cx = int(np.median([f[0].center_x for f in two_speaker_samples]))
            s2_cx = int(np.median([f[1].center_x for f in two_speaker_samples]))
            half_crop_w = int(height * 0.9)
            s1_x = max(0, min(s1_cx - half_crop_w // 2, width - half_crop_w))
            s2_x = max(0, min(s2_cx - half_crop_w // 2, width - half_crop_w))
            shot_plans.append(ShotPlan(
                start=rel_s,
                end=rel_e,
                mode='split_screen',
                speaker1_box=(s1_x, 0, half_crop_w, height),
                speaker2_box=(s2_x, 0, half_crop_w, height),
                margin_v=220
            ))
            print(f"    Shot [{rel_s:.1f}s - {rel_e:.1f}s]: 2-Speaker Split Screen")

        else:
            # Shot is a single human speaker talking
            eye_level_y = height * 0.35
            timed_single_faces = [(s[0], min(s[1], key=lambda f: abs(f.center_y - eye_level_y))) for s in shot_samples if s[1]]
            
            # Check for intra-shot face position jumps (sub-second camera shifts)
            jump_indices = []
            for j in range(1, len(timed_single_faces)):
                prev_cx = timed_single_faces[j-1][1].center_x
                curr_cx = timed_single_faces[j][1].center_x
                if abs(curr_cx - prev_cx) > width * 0.14:
                    jump_indices.append(j)

            if jump_indices and len(timed_single_faces) >= 4:
                # Sub-segment this shot dynamically based on face shifts
                split_points = [0] + jump_indices + [len(timed_single_faces)]
                for k in range(len(split_points) - 1):
                    sub_faces = timed_single_faces[split_points[k]:split_points[k+1]]
                    if not sub_faces:
                        continue
                    sub_s = round(sub_faces[0][0], 2) if k > 0 else rel_s
                    sub_e = round(sub_faces[-1][0], 2) if k < len(split_points) - 2 else rel_e
                    if sub_e <= sub_s:
                        sub_e = sub_s + 0.2
                    sub_cx = int(np.median([f[1].center_x for f in sub_faces]))
                    sub_cy = int(np.median([f[1].center_y for f in sub_faces]))
                    sub_crop_x = max(0, min(sub_cx - target_crop_w // 2, width - target_crop_w))
                    shot_plans.append(ShotPlan(
                        start=sub_s,
                        end=sub_e,
                        mode='portrait_face',
                        crop_x=sub_crop_x,
                        center_y=sub_cy,
                        margin_v=220
                    ))
                    print(f"    Sub-Shot [{sub_s:.1f}s - {sub_e:.1f}s]: Portrait Speaker (Center X: {sub_cx}, Crop X: {sub_crop_x})")
            else:
                single_faces = [f[1] for f in timed_single_faces] if timed_single_faces else [min(faces, key=lambda f: abs(f.center_y - eye_level_y)) for faces in valid_face_samples]
                avg_cx = int(np.median([f.center_x for f in single_faces]))
                avg_cy = int(np.median([f.center_y for f in single_faces]))
                crop_x = max(0, min(avg_cx - target_crop_w // 2, width - target_crop_w))
                shot_plans.append(ShotPlan(
                    start=rel_s,
                    end=rel_e,
                    mode='portrait_face',
                    crop_x=crop_x,
                    center_y=avg_cy,
                    margin_v=220
                ))
                print(f"    Shot [{rel_s:.1f}s - {rel_e:.1f}s]: Full Portrait Speaker (Center X: {avg_cx}, Crop X: {crop_x})")

    # Ensure shot plans span the entire duration without gaps
    if shot_plans:
        shot_plans[0].start = 0.0
        shot_plans[-1].end = clip_duration
        for idx in range(len(shot_plans) - 1):
            shot_plans[idx].end = shot_plans[idx + 1].start

    # Determine overall framing mode
    modes = set(s.mode for s in shot_plans)
    if len(shot_plans) > 1 and (len(modes) > 1 or len(set(s.crop_x for s in shot_plans if s.mode == 'portrait_face')) > 1):
        # Multiple shots with different layouts or camera angle switches -> Dynamic Multi-Shot!
        print(f"[+] Multi-Shot Dynamic Framing Activated: {len(shot_plans)} shots will transition dynamically!")
        return FramingDecision(
            mode='multi_shot_dynamic',
            face_count=len(shot_plans),
            shots=shot_plans,
            video_width=width,
            video_height=height
        )

    # Monolithic fallbacks if all shots are identical
    if modes == {'presentation_slide'}:
        return FramingDecision(mode='blur_stack', face_count=0, video_width=width, video_height=height)

    if modes == {'split_screen'}:
        s = shot_plans[0]
        return FramingDecision(
            mode='split_screen',
            face_count=2,
            speaker1_box=s.speaker1_box,
            speaker2_box=s.speaker2_box,
            video_width=width,
            video_height=height
        )

    # Default to single smooth portrait
    first_shot = shot_plans[0] if shot_plans else None
    crop_x = first_shot.crop_x if first_shot else (width - target_crop_w) // 2
    return FramingDecision(
        mode='single_smooth',
        face_count=1,
        smoothed_center_x=crop_x + target_crop_w // 2,
        smoothed_center_y=first_shot.center_y if first_shot else height // 3,
        video_width=width,
        video_height=height
    )
