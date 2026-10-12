"""Streamer Layout Compositor & Facecam ROI Detection.

Intelligent stream composition engine:
1. Auto-detects stream type:
   - 'stacked_webcam': Streamer webcam box overlaid on top of gameplay / YouTube reaction video.
   - 'fullscreen_room': Streamer is in full-screen camera (IRL room, podcasts, challenges).
2. For 'stacked_webcam':
   - Automatically detects webcam ROI coordinates (e.g. top-right, bottom-left).
   - Generates an auto-stacked 9:16 layout:
       - Top 50%: Cropped and centered Streamer Webcam reaction.
       - Bottom 50%: Cropped main content (gameplay / reaction video).
3. For 'fullscreen_room':
   - Routes to StreamerIdentityTracker in src/streamers/face_tracker.py for
     Kai Cenat identity locking, LERP panning, and smart off-screen wide fallback!
"""

import math
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from src.face_tracker import FaceBox, ShotPlan, FramingDecision, detect_letterbox_margins
from src.streamers.face_tracker import (
    analyze_streamer_clip_framing, StreamerIdentityTracker, MODELS_DIR
)


@dataclass
class StreamLayoutInfo:
    layout_type: str  # 'stacked_webcam', 'multi_person_room', or 'fullscreen_room'
    webcam_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    content_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    speaker1_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) Top Pane (Host / Streamer)
    speaker2_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) Bottom Pane (Guest)
    confidence: float = 0.0


def detect_stream_layout(
    video_path: Path,
    sample_duration: float = 8.0,
    num_samples: int = 16,
    streamer_name: str = "Kai Cenat"
) -> StreamLayoutInfo:
    """Samples video frames to classify layout into 'stacked_webcam', 'multi_person_room', or 'fullscreen_room'."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return StreamLayoutInfo(layout_type="fullscreen_room", confidence=0.5)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    max_frame = min(total_frames, int(sample_duration * video_fps))
    step = max(1, max_frame // num_samples)

    tracker = StreamerIdentityTracker(w, h, det_max_dim=640, streamer_name=streamer_name)

    detected_faces: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)
    face_sim_pairs: List[Tuple[Tuple[int, int, int, int], float]] = []
    multi_face_frames = 0
    valid_face_frames = 0

    f_idx = 0
    while f_idx < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        scaled_faces = tracker.detect_and_scale(frame)
        if scaled_faces is not None and len(scaled_faces) > 0:
            valid_faces = [f for f in scaled_faces if float(f[14]) >= 0.45]
            if valid_faces:
                valid_face_frames += 1
                if len(valid_faces) >= 2:
                    multi_face_frames += 1
                for face in valid_faces:
                    fbox = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
                    detected_faces.append(fbox)
                    feat = tracker.extract_face_embedding(frame, face)
                    sim = tracker.match_similarity(feat) if feat is not None else 0.0
                    face_sim_pairs.append((fbox, sim))

        f_idx += step

    cap.release()

    if not detected_faces:
        return StreamLayoutInfo(layout_type="fullscreen_room", confidence=0.6)

    # Analyze face sizes relative to frame
    # In a webcam-over-content layout:
    # 1. Face box is small: w < 0.22 of frame width AND h < 0.28 of frame height
    # 2. Face positions are spatially clustered in one corner/region with low variance
    face_widths = [f[2] for f in detected_faces]
    face_heights = [f[3] for f in detected_faces]
    med_fw = np.median(face_widths)
    med_fh = np.median(face_heights)

    rel_fw = med_fw / w
    rel_fh = med_fh / h

    cxs = [f[0] + f[2] // 2 for f in detected_faces]
    cys = [f[1] + f[3] // 2 for f in detected_faces]
    std_cx = np.std(cxs) if len(cxs) > 1 else 0.0
    std_cy = np.std(cys) if len(cys) > 1 else 0.0

    # Low standard deviation (< 8% of frame width) and small face (< 18% of frame width)
    # indicates a stationary webcam box!
    is_webcam_box = (
        rel_fw < 0.20 and
        rel_fh < 0.28 and
        (std_cx / w) < 0.08 and
        (std_cy / h) < 0.08
    )

    if is_webcam_box:
        # Stationary webcam detected!
        avg_cx = int(np.median(cxs))
        avg_cy = int(np.median(cys))
        box_w = int(med_fw * 2.8)  # Expand to capture streamer's webcam frame
        box_h = int(box_w * (9 / 16))  # 16:9 crop around streamer

        wx = max(0, min(avg_cx - box_w // 2, w - box_w))
        wy = max(0, min(avg_cy - box_h // 2, h - box_h))
        webcam_box = (wx, wy, box_w, box_h)

        # Content box: Main gameplay / video (typically centered on non-webcam area)
        cw = int(w * 0.75)
        ch = int(cw * (9 / 16))
        # Offset away from webcam side
        cx = 0 if wx > w // 2 else int(w - cw)
        cy = max(0, (h - ch) // 2)
        content_box = (cx, cy, cw, ch)

        print(f"[+] Detected 'stacked_webcam' stream layout: Webcam at ({wx}, {wy}, {box_w}x{box_h})")
        return StreamLayoutInfo(
            layout_type="stacked_webcam",
            webcam_box=webcam_box,
            content_box=content_box,
            confidence=0.88
        )

    # Check for multi-person room stream (Host + Guest collab, e.g. Kai Cenat x MrBeast, Speed x Adin)
    multi_face_ratio = (multi_face_frames / max(1, valid_face_frames)) if valid_face_frames > 0 else 0.0
    min_cx, max_cx = min(cxs), max(cxs)
    is_horizontally_split = (max_cx - min_cx) >= (0.18 * w)

    if (multi_face_ratio >= 0.20 or is_horizontally_split) and len(detected_faces) >= 3:
        # Partition faces into two spatial clusters via 2-means
        c1, c2 = min_cx, max_cx
        g1, g2 = [], []
        for _ in range(6):
            g1 = [f for f in detected_faces if abs((f[0] + f[2] // 2) - c1) <= abs((f[0] + f[2] // 2) - c2)]
            g2 = [f for f in detected_faces if abs((f[0] + f[2] // 2) - c1) > abs((f[0] + f[2] // 2) - c2)]
            if g1:
                c1 = int(np.median([f[0] + f[2] // 2 for f in g1]))
            if g2:
                c2 = int(np.median([f[0] + f[2] // 2 for f in g2]))

        if g1 and g2 and abs(c1 - c2) >= int(0.15 * w):
            g1_cx = int(np.median([f[0] + f[2] // 2 for f in g1]))
            g1_cy = int(np.median([f[1] + f[3] // 2 for f in g1]))
            g2_cx = int(np.median([f[0] + f[2] // 2 for f in g2]))
            g2_cy = int(np.median([f[1] + f[3] // 2 for f in g2]))

            # Identity match: Host / Streamer on Top pane, Guest on Bottom pane
            g1_sims = [sim for box, sim in face_sim_pairs if abs((box[0] + box[2] // 2) - c1) < abs((box[0] + box[2] // 2) - c2)]
            g2_sims = [sim for box, sim in face_sim_pairs if abs((box[0] + box[2] // 2) - c2) < abs((box[0] + box[2] // 2) - c1)]
            g1_sim = float(np.mean(g1_sims)) if g1_sims else 0.0
            g2_sim = float(np.mean(g2_sims)) if g2_sims else 0.0

            if g2_sim > g1_sim + 0.05:
                # Group 2 is primary streamer (Top), Group 1 is Guest (Bottom)
                p1_cx, p1_cy = g2_cx, g2_cy
                p2_cx, p2_cy = g1_cx, g1_cy
            else:
                # Group 1 is primary streamer (Top), Group 2 is Guest (Bottom)
                p1_cx, p1_cy = g1_cx, g1_cy
                p2_cx, p2_cy = g2_cx, g2_cy

            # Compute 9:8 pane dimensions to fill 1080x960 panes with ZERO blurry bars
            pane_h = h
            pane_w = int(pane_h * 9.0 / 8.0)
            if pane_w > w:
                pane_w = w - (w % 2)
                pane_h = int(pane_w * 8.0 / 9.0)
            pane_w -= pane_w % 2
            pane_h -= pane_h % 2

            s1_x = max(0, min(p1_cx - pane_w // 2, w - pane_w))
            s1_y = max(0, min(int(p1_cy - pane_h * 0.35), h - pane_h))
            speaker1_box = (s1_x, s1_y, pane_w, pane_h)

            s2_x = max(0, min(p2_cx - pane_w // 2, w - pane_w))
            s2_y = max(0, min(int(p2_cy - pane_h * 0.35), h - pane_h))
            speaker2_box = (s2_x, s2_y, pane_w, pane_h)

            print(
                f"[+] Detected 'multi_person_room' collab: Split-Screen Stacked layout!\n"
                f"    Top Pane (Streamer): ({s1_x}, {s1_y}, {pane_w}x{pane_h})\n"
                f"    Bottom Pane (Guest): ({s2_x}, {s2_y}, {pane_w}x{pane_h})"
            )
            return StreamLayoutInfo(
                layout_type="multi_person_room",
                speaker1_box=speaker1_box,
                speaker2_box=speaker2_box,
                confidence=0.93
            )

    print(f"[*] Detected 'fullscreen_room' single streamer layout (rel_face_w: {rel_fw:.2f}, std_cx: {std_cx:.1f}px)")
    return StreamLayoutInfo(
        layout_type="fullscreen_room",
        confidence=0.92
    )


def build_streamer_framing(
    video_path: Path,
    start_time: float,
    end_time: float,
    streamer_name: str = "Kai Cenat",
    force_layout: str = "auto"
) -> FramingDecision:
    """Builds optimal FramingDecision for streamer clip.

    Args:
        video_path: Path to video segment
        start_time: Relative or absolute start time
        end_time: Relative or absolute end time
        streamer_name: Name of primary streamer
        force_layout: 'auto', 'stacked_webcam', 'multi_person_room', 'split_screen', 'fullscreen_room', 'single_smooth'
    """
    clip_dur = max(0.1, end_time - start_time)

    layout_info = StreamLayoutInfo(layout_type="fullscreen_room")
    if force_layout == "stacked_webcam":
        layout_info = detect_stream_layout(video_path, sample_duration=min(6.0, clip_dur), streamer_name=streamer_name)
        layout_info.layout_type = "stacked_webcam"
    elif force_layout in ("multi_person_room", "split_screen"):
        layout_info = detect_stream_layout(video_path, sample_duration=min(6.0, clip_dur), streamer_name=streamer_name)
        layout_info.layout_type = "multi_person_room"
    elif force_layout in ("fullscreen_room", "single_smooth"):
        layout_info.layout_type = "fullscreen_room"
    elif force_layout == "auto":
        layout_info = detect_stream_layout(video_path, sample_duration=min(6.0, clip_dur), streamer_name=streamer_name)

    if layout_info.layout_type == "stacked_webcam" and layout_info.webcam_box and layout_info.content_box:
        # Split-screen Auto-Stacked 9:16 layout
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap.release()

        shot = ShotPlan(
            start=0.0,
            end=clip_dur,
            mode="split_screen",
            speaker1_box=layout_info.webcam_box,
            speaker2_box=layout_info.content_box,
            crop_x=0,
            crop_y=0,
            crop_w=w,
            crop_h=h
        )

        return FramingDecision(
            mode="split_screen",
            face_count=2,
            shots=[shot],
            speaker1_box=layout_info.webcam_box,
            speaker2_box=layout_info.content_box,
            video_width=w,
            video_height=h,
            active_x=0,
            active_y=0,
            active_w=w,
            active_h=h
        )

    elif layout_info.layout_type == "multi_person_room":
        # Dual Split-Screen (Top: Host / Bottom: Guest)
        # Fills 100% of 9:16 mobile canvas with ZERO blurry padding and large clear reactions!
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap.release()

        pane_h = h
        pane_w = int(pane_h * 9.0 / 8.0)
        if pane_w > w:
            pane_w = w - (w % 2)
            pane_h = int(pane_w * 8.0 / 9.0)
        pane_w -= pane_w % 2
        pane_h -= pane_h % 2

        s1_box = layout_info.speaker1_box or (0, 0, pane_w, pane_h)
        s2_box = layout_info.speaker2_box or (max(0, w - pane_w), 0, pane_w, pane_h)

        shot = ShotPlan(
            start=0.0,
            end=clip_dur,
            mode="split_screen",
            speaker1_box=s1_box,
            speaker2_box=s2_box,
            crop_x=0,
            crop_y=0,
            crop_w=w,
            crop_h=h
        )

        return FramingDecision(
            mode="split_screen",
            face_count=2,
            shots=[shot],
            speaker1_box=s1_box,
            speaker2_box=s2_box,
            video_width=w,
            video_height=h,
            active_x=0,
            active_y=0,
            active_w=w,
            active_h=h
        )

    # Fullscreen single-camera layout: use StreamerIdentityTracker with off-screen fallback
    return analyze_streamer_clip_framing(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        streamer_name=streamer_name
    )

