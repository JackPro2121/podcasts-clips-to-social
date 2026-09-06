import cv2
import numpy as np
import os
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from src.scene_classifier import classify_frame_scene

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
    mode: str  # 'single_smooth', 'dynamic_cut', 'split_screen', 'blur_stack'
    face_count: int
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

    timeline_samples: List[Tuple[float, List[FaceBox]]] = []
    presentation_votes = 0
    scene_check_count = 0
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - start_frame) % frame_step == 0:
            rel_t = (current_frame - start_frame) / fps
            faces: List[FaceBox] = []

            # Check for presentation/slides every 2nd sample (~1.25 Hz)
            if (current_frame - start_frame) % (frame_step * 2) == 0:
                scene_check_count += 1
                try:
                    scene_info = classify_frame_scene(frame)
                    if scene_info.get("is_presentation"):
                        presentation_votes += 1
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
            timeline_samples.append((rel_t, faces))

        current_frame += 1

    cap.release()

    if not timeline_samples:
        return FramingDecision(mode='blur_stack', face_count=0, video_width=width, video_height=height)

    # 1. Evaluate genuine 2-speaker wide frames (Host + Guest seated side-by-side)
    # A true 2-speaker shot requires:
    # - Substantial horizontal separation: >= width * 0.22
    # - Similar eye/seated level: abs(y1 - y2) <= height * 0.22 (filters out coffee cups/hands)
    # - Comparable face size: 0.35 <= (w1*h1) / (w2*h2) <= 2.8
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

    two_face_samples = [faces for (_, faces) in timeline_samples if is_valid_two_speaker_frame(faces)]
    total_valid_samples = [faces for (_, faces) in timeline_samples if len(faces) > 0]
    
    if total_valid_samples and (len(two_face_samples) / len(total_valid_samples) >= 0.40):
        # High prevalence of 2 genuine speakers in shot -> Dynamic Split Screen
        s1_centers = [f[0].center_x for f in two_face_samples]
        s2_centers = [f[1].center_x for f in two_face_samples]
        s1_cx = int(np.median(s1_centers))
        s2_cx = int(np.median(s2_centers))

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

    # 2. Single Speaker vs Document / Presentation / Infographic Detection
    # If the video shows slides, charts, research papers, or screenshares for most of the clip,
    # cropping into a 9:16 portrait viewport destroys the diagram.
    # Dual-signal verification:
    # 1. Genuine human faces present < 50% of the clip, OR
    # 2. Scene classifier detects document/diagram/presentation in >= 35% of samples.
    single_samples: List[Tuple[float, int, int]] = []
    for rel_t, faces in timeline_samples:
        if faces:
            # Pick the face closest to eye level (Y: ~35% of frame) to ignore desk objects/mugs
            best_face = min(faces, key=lambda f: abs(f.center_y - height * 0.35))
            single_samples.append((rel_t, best_face.center_x, best_face.center_y))

    face_presence_ratio = len(single_samples) / max(1, len(timeline_samples))
    presentation_ratio = (presentation_votes / max(1, scene_check_count)) if scene_check_count > 0 else 0.0

    if face_presence_ratio < 0.50 or presentation_ratio >= 0.35 or not single_samples:
        # Presentation detected: Automatically preserve 100% of the diagram in blur_stack mode!
        print(f"[*] Visual graphics / slide detected (Face presence: {face_presence_ratio*100:.1f}%, Presentation ratio: {presentation_ratio*100:.1f}%). Activating intelligent Blur-Stack presentation framing!")
        return FramingDecision(mode='blur_stack', face_count=0, video_width=width, video_height=height)

    # Detect camera angle switches (clusters of face centers separated by significant X shift)
    # Threshold for camera angle shift: 20% of video width (e.g. 384px in 1920p)
    shift_threshold = width * 0.20

    shots: List[Dict[str, Any]] = []
    current_shot_centers_x = [single_samples[0][1]]
    current_shot_centers_y = [single_samples[0][2]]
    current_shot_start = single_samples[0][0]

    for i in range(1, len(single_samples)):
        t_cur, cx_cur, cy_cur = single_samples[i]
        median_cx = np.median(current_shot_centers_x)
        
        if abs(cx_cur - median_cx) > shift_threshold:
            # Camera cut detected
            shots.append({
                "start": current_shot_start,
                "end": t_cur,
                "cx": int(median_cx),
                "cy": int(np.median(current_shot_centers_y))
            })
            current_shot_start = t_cur
            current_shot_centers_x = [cx_cur]
            current_shot_centers_y = [cy_cur]
        else:
            current_shot_centers_x.append(cx_cur)
            current_shot_centers_y.append(cy_cur)

    # Append last shot
    shots.append({
        "start": current_shot_start,
        "end": (end_time - start_time),
        "cx": int(np.median(current_shot_centers_x)),
        "cy": int(np.median(current_shot_centers_y))
    })

    # Filter out momentary glitch shots (< 1.2 seconds)
    filtered_shots = []
    for s in shots:
        dur = s["end"] - s["start"]
        if dur >= 1.2 or not filtered_shots:
            filtered_shots.append(s)
        else:
            # Merge with previous shot
            filtered_shots[-1]["end"] = s["end"]

    # If only 1 shot or camera angles are all close:
    if len(filtered_shots) <= 1:
        avg_cx = filtered_shots[0]["cx"] if filtered_shots else width // 2
        avg_cy = filtered_shots[0]["cy"] if filtered_shots else height // 3
        left_bound = max(0, min(avg_cx - target_crop_w // 2, width - target_crop_w))
        return FramingDecision(
            mode='single_smooth',
            face_count=1,
            smoothed_center_x=left_bound + target_crop_w // 2,
            smoothed_center_y=avg_cy,
            video_width=width,
            video_height=height
        )

    # Multi-camera switching detected: Build piecewise FFmpeg crop expression!
    def build_crop_expr(shot_list: List[Dict[str, Any]]) -> str:
        def get_x(cx: int) -> int:
            return max(0, min(cx - target_crop_w // 2, width - target_crop_w))

        if len(shot_list) == 1:
            return str(get_x(shot_list[0]["cx"]))

        cur = str(get_x(shot_list[-1]["cx"]))
        for s in reversed(shot_list[:-1]):
            t_switch = s["end"]
            x_val = get_x(s["cx"])
            cur = f"if(lt(t\\,{t_switch:.2f})\\,{x_val}\\,{cur})"
        return cur

    crop_expr = build_crop_expr(filtered_shots)
    first_cx = filtered_shots[0]["cx"]
    first_x = max(0, min(first_cx - target_crop_w // 2, width - target_crop_w))

    print(f"[+] Multi-Camera shot switching detected! {len(filtered_shots)} shots framed dynamically.")
    for idx, s in enumerate(filtered_shots):
        print(f"    Shot #{idx+1}: [{s['start']:.1f}s - {s['end']:.1f}s] Center X: {s['cx']}")

    return FramingDecision(
        mode='dynamic_cut',
        face_count=len(filtered_shots),
        smoothed_center_x=first_x + target_crop_w // 2,
        crop_x_expr=crop_expr,
        video_width=width,
        video_height=height
    )
