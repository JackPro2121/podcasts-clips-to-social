import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import requests

MODEL_DIR = Path(__file__).resolve().parent / "models"
YOLO_MODEL_PATH = MODEL_DIR / "yolov8n.onnx"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Classes that signify a presentation, slide, screen-share, or document
PRESENTATION_CLASSES = {"tv", "laptop", "book", "cell phone"}

_yolo_net = None
_yolo_attempted = False

def get_yolo_net() -> Optional[Any]:
    """Loads YOLOv8-Nano ONNX model into OpenCV DNN, auto-downloading if missing."""
    global _yolo_net, _yolo_attempted
    if _yolo_attempted:
        return _yolo_net
    _yolo_attempted = True

    if not YOLO_MODEL_PATH.exists():
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[*] Downloading pre-trained YOLOv8-Nano ONNX model ({YOLO_URL})...")
            r = requests.get(YOLO_URL, allow_redirects=True, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000000:
                with open(YOLO_MODEL_PATH, "wb") as f:
                    f.write(r.content)
                print(f"[+] YOLOv8-Nano ONNX downloaded successfully ({len(r.content)/(1024*1024):.2f} MB).")
        except Exception as e:
            print(f"[-] Could not download YOLOv8 model: {e}. Using high-precision visual document analyzer.")
            return None

    if YOLO_MODEL_PATH.exists():
        try:
            _yolo_net = cv2.dnn.readNetFromONNX(str(YOLO_MODEL_PATH))
            _yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            _yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return _yolo_net
        except Exception as e:
            print(f"[-] Error loading YOLOv8 ONNX: {e}")
            return None
    return None

def detect_slide_heuristics(frame: np.ndarray) -> bool:
    """
    High-accuracy visual text, diagram, and presentation slide detector:
    1. Analyzes high-frequency edge distribution (text lines and diagram borders).
    2. Measures document contrast and uniform background areas.
    3. Detects whether the frame contains structured text/charts vs organic human scene.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge and text density analysis via Sobel horizontal gradients (reading lines)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    _, thresh = cv2.threshold(abs_grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Morphological closing along horizontal line direction to connect words into text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count horizontal text blocks typical of research papers, slides, and medical reports
    text_line_count = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(1, ch)
        # Text line features: wider than tall, reasonably sized
        if aspect >= 2.5 and cw >= w * 0.10 and ch <= h * 0.15:
            text_line_count += 1
            
    # 2. Background uniformity (white papers, black slides, presentation decks)
    # Presentation slides typically have a large dominant background (>40% of pixels in same luminance range)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    dominant_luminance_ratio = hist.max() / hist.sum()
    
    is_slide = (text_line_count >= 5 and dominant_luminance_ratio >= 0.25) or (text_line_count >= 8)
    return is_slide

def classify_frame_scene(frame: np.ndarray) -> Dict[str, Any]:
    """
    Classifies a frame as 'presentation' (slide/chart/paper) or 'human' (speaker/interview).
    Combines YOLOv8 deep learning detection with document text density heuristics.
    """
    h, w = frame.shape[:2]
    
    # Heuristic text check
    has_slide_layout = detect_slide_heuristics(frame)
    
    net = get_yolo_net()
    yolo_detected_presentation = False
    person_detected = False
    
    if net is not None:
        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            preds = net.forward() # shape (1, 84, 8400)
            
            # YOLOv8 format: preds[0] is (84, 8400) -> transpose to (8400, 84)
            preds = preds[0].T
            boxes = preds[:, :4]
            scores = preds[:, 4:]
            
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            mask = confidences > 0.40
            valid_class_ids = class_ids[mask]
            
            for cid in valid_class_ids:
                if cid < len(COCO_CLASSES):
                    cname = COCO_CLASSES[cid]
                    if cname in PRESENTATION_CLASSES:
                        yolo_detected_presentation = True
                    elif cname == "person":
                        person_detected = True
        except Exception:
            pass
            
    is_presentation = has_slide_layout or yolo_detected_presentation
    return {
        "is_presentation": is_presentation,
        "person_detected": person_detected,
        "has_slide_layout": has_slide_layout,
        "yolo_presentation": yolo_detected_presentation
    }

def analyze_clip_presentation_ratio(video_path: Path, start_sec: float, end_sec: float) -> float:
    """
    Samples frames across a clip to determine what fraction of the clip is a slide/presentation.
    Returns ratio from 0.0 (100% human speaker) to 1.0 (100% slides/charts/presentation).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # Sample every 1.0s
    step = max(1, int(fps * 1.0))
    
    presentation_frames = 0
    total_sampled = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur = start_frame
    while cur <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (cur - start_frame) % step == 0:
            total_sampled += 1
            info = classify_frame_scene(frame)
            if info["is_presentation"]:
                presentation_frames += 1
        cur += 1
        
    cap.release()
    if total_sampled == 0:
        return 0.0
    return presentation_frames / total_sampled

def detect_clip_shots(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    min_shot_duration: float = 0.4
) -> List[Tuple[float, float]]:
    """
    Splits video segment [start_sec, end_sec] into discrete camera cuts / shot intervals.
    Uses PySceneDetect with AdaptiveDetector.
    Gracefully falls back to full segment if no scene cuts found or if scenedetect is unavailable.
    Merges any micro-glitches shorter than min_shot_duration (0.4s handles sub-second podcast reaction cuts).
    """
    duration = max(0.1, end_sec - start_sec)
    default_shot = [(round(start_sec, 2), round(end_sec, 2))]
    
    try:
        from scenedetect import open_video, SceneManager, AdaptiveDetector, FrameTimecode
        video = open_video(str(video_path))
        fps = video.frame_rate
        if not fps or fps <= 0:
            return default_shot
            
        start_tc = FrameTimecode(timecode=start_sec, fps=fps)
        dur_tc = FrameTimecode(timecode=duration, fps=fps)
        
        video.seek(start_tc)
        sm = SceneManager()
        # Adaptive threshold 2.5 & min_scene_len 0.4s to reliably catch quick reaction cuts
        sm.add_detector(AdaptiveDetector(adaptive_threshold=2.5, min_scene_len=max(2, int(fps * min_shot_duration))))
        sm.detect_scenes(video, frame_skip=2, duration=dur_tc)
        scene_list = sm.get_scene_list()
        
        if not scene_list:
            return default_shot
            
        raw_shots = []
        for s in scene_list:
            s_start = max(start_sec, float(s[0].seconds))
            s_end = min(end_sec, float(s[1].seconds))
            if s_end - s_start > 0.1:
                raw_shots.append((round(s_start, 2), round(s_end, 2)))
                
        if not raw_shots:
            return default_shot
            
        # Ensure beginning and ending bounds align
        if raw_shots[0][0] > start_sec:
            raw_shots[0] = (round(start_sec, 2), raw_shots[0][1])
        if raw_shots[-1][1] < end_sec:
            raw_shots[-1] = (raw_shots[-1][0], round(end_sec, 2))
            
        # Filter and merge shots that are too short (< min_shot_duration)
        merged_shots: List[Tuple[float, float]] = []
        for s_start, s_end in raw_shots:
            if not merged_shots:
                merged_shots.append((s_start, s_end))
            elif (s_end - s_start) < min_shot_duration:
                # Merge into preceding shot
                prev_start, _ = merged_shots[-1]
                merged_shots[-1] = (prev_start, s_end)
            else:
                merged_shots.append((s_start, s_end))
                
        return merged_shots if merged_shots else default_shot
        
    except Exception as e:
        print(f"[-] Scene detection fallback (error: {e}). Using monolithic segment.")
        return default_shot

