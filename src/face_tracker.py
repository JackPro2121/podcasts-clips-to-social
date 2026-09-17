import cv2
import numpy as np
import os
import mediapipe as mp
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
    # Timeline of center_x for each sample in this shot to enable LERP panning
    face_centers_timeline: List[Tuple[float, int]] = field(default_factory=list)
    speaker1_box: Optional[Tuple[int, int, int, int]] = None
    speaker2_box: Optional[Tuple[int, int, int, int]] = None
    margin_v: int = 220

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

def get_face_detector():
    """Initializes Face Detection with robust fallbacks."""
    try:
        import mediapipe.solutions.face_detection as mp_fd
        return mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    except Exception as e:
        print(f"[-] MediaPipe Solutions failed: {e}. Trying OpenCV fallback...")
        try:
            # Prioritize local project root for the cascade file to avoid site-packages issues
            local_cascade = 'haarcascade_frontalface_default.xml'
            if os.path.exists(local_cascade):
                cascade_path = local_cascade
            else:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
            return cv2.CascadeClassifier(cascade_path)
        except Exception as e2:
            print(f"[-] All face detectors failed: {e2}.")
            return None

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
    target_crop_w = int(height * (9 / 16))

    detector = get_face_detector()
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_step = max(1, int(fps / sample_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
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

            if (current_frame - start_frame) % (frame_step * 2) == 0:
                try:
                    scene_info = classify_frame_scene(frame)
                    if scene_info.get("is_presentation"):
                        is_slide = True
                except Exception:
                    pass

            # MediaPipe Detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if detector is None:
                current_frame += 1
                continue

            if hasattr(detector, 'process'):
                results = detector.process(rgb_frame)
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        fx = int(bbox.xmin * width)
                        fy = int(bbox.ymin * height)
                        fw = int(bbox.width * width)
                        fh = int(bbox.height * height)
                        faces.append(FaceBox(
                            x=fx, y=fy, w=fw, h=fh,
                            center_x=fx + fw // 2,
                            center_y=fy + fh // 2
                        ))
            elif hasattr(detector, 'detectMultiScale'):
                # Haar Cascade fallback
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in rects:
                    faces.append(FaceBox(
                        x=x, y=y, w=w, h=h,
                        center_x=x + w // 2,
                        center_y=y + h // 2
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

        shot_samples = [s for s in timeline_samples if rel_s <= s[0] <= rel_e]
        if not shot_samples:
            shot_samples = timeline_samples

        pres_count = sum(1 for s in shot_samples if s[2])
        pres_ratio = pres_count / max(1, len(shot_samples))

        valid_face_samples = [s[1] for s in shot_samples if s[1]]
        face_presence = len(valid_face_samples) / max(1, len(shot_samples))
        two_speaker_samples = [f for f in valid_face_samples if is_valid_two_speaker_frame(f)]

        if pres_ratio >= 0.35 or face_presence < 0.35 or not valid_face_samples:
            shot_plans.append(ShotPlan(
                start=rel_s,
                end=rel_e,
                mode='presentation_slide',
                margin_v=420
            ))
        elif valid_face_samples and (len(two_speaker_samples) / len(valid_face_samples) >= 0.40):
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
        else:
            eye_level_y = height * 0.35
            timed_single_faces = [(s[0], min(s[1], key=lambda f: abs(f.center_y - eye_level_y))) for s in shot_samples if s[1]]
            
            jump_indices = []
            for j in range(1, len(timed_single_faces)):
                prev_cx = timed_single_faces[j-1][1].center_x
                curr_cx = timed_single_faces[j][1].center_x
                if abs(curr_cx - prev_cx) > width * 0.14:
                    jump_indices.append(j)

            if jump_indices and len(timed_single_faces) >= 4:
                split_points = [0] + jump_indices + [len(timed_single_faces)]
                for k in range(len(split_points) - 1):
                    sub_faces = timed_single_faces[split_points[k]:split_points[k+1]]
                    if not sub_faces:
                        continue
                    sub_s = round(sub_faces[0][0], 2) if k > 0 else rel_s
                    sub_e = round(sub_faces[-1][0], 2) if k < len(split_points) - 2 else rel_e
                    if sub_e <= sub_s:
                        sub_e = sub_s + 0.2
                    
                    sub_cx_median = int(np.median([f[1].center_x for f in sub_faces]))
                    sub_cy_median = int(np.median([f[1].center_y for f in sub_faces]))
                    sub_crop_x = max(0, min(sub_cx_median - target_crop_w // 2, width - target_crop_w))
                    
                    # Record timeline for LERP panning
                    timeline = [(f[0], f[1].center_x) for f in sub_faces if isinstance(f, tuple) and hasattr(f[1], 'center_x')]
                    
                    shot_plans.append(ShotPlan(
                        start=sub_s,
                        end=sub_e,
                        mode='portrait_face',
                        crop_x=sub_crop_x,
                        center_y=sub_cy_median,
                        face_centers_timeline=timeline,
                        margin_v=220
                    ))
            else:
                single_faces = [f[1] for f in timed_single_faces] if timed_single_faces else []
                if not single_faces and valid_face_samples:
                    single_faces = [min(faces, key=lambda f: abs(f.center_y - eye_level_y)) for faces in valid_face_samples]
                
                if not single_faces:
                    avg_cx, avg_cy = width // 2, height // 3
                else:
                    avg_cx = int(np.median([f.center_x for f in single_faces]))
                    avg_cy = int(np.median([f.center_y for f in single_faces]))
                
                crop_x = max(0, min(avg_cx - target_crop_w // 2, width - target_crop_w))
                timeline = [(s, f.center_x) for s, f in timed_single_faces if isinstance(f, FaceBox)] if timed_single_faces else []
                
                shot_plans.append(ShotPlan(
                    start=rel_s,
                    end=rel_e,
                    mode='portrait_face',
                    crop_x=crop_x,
                    center_y=avg_cy,
                    face_centers_timeline=timeline,
                    margin_v=220
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
            video_height=height
        )

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
