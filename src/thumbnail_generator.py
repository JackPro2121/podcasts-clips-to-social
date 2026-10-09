import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional, List
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

from src.config import FONTS_DIR, OUTPUT_WIDTH, OUTPUT_HEIGHT


def calculate_smart_title_y(
    faces: Optional[List[Any]],
    total_text_h: int,
    top_margin: int = 240,
    bottom_margin: int = 1680,
    default_y: int = 1200
) -> int:
    """Calculates an optimal, collision-free vertical position (start_y) for thumbnail title text.

    - If no faces are detected: falls back to the balanced lower-middle region.
    - If a face is detected:
      - Computes headroom (space above skull/hairline) and chest space (space below chin).
      - Selects the zone with generous clearance to prevent covering eyes, mouth, chin, or microphone.
      - Clamps strictly within platform safe zones [top_margin, bottom_margin - total_text_h].
    """
    if not faces:
        return max(top_margin + 20, min(default_y - (total_text_h // 2), bottom_margin - total_text_h))

    primary = max(faces, key=lambda f: getattr(f, "w", 0) * getattr(f, "h", 0))
    face_top = getattr(primary, "y", 0)
    face_h = getattr(primary, "h", 0)
    face_bottom = face_top + face_h

    # Account for upper skull, hair, and chin boundaries
    head_top = max(0, face_top - int(face_h * 0.35))
    chin_bottom = face_bottom + int(face_h * 0.15)

    headroom = head_top - top_margin
    chest_space = bottom_margin - chin_bottom

    chin_clearance = 80
    headroom_viable = headroom >= (total_text_h + 30)
    chest_viable = chest_space >= (total_text_h + chin_clearance)

    if chest_viable and not headroom_viable:
        start_y = chin_bottom + chin_clearance
    elif headroom_viable and not chest_viable:
        start_y = max(top_margin + 20, (top_margin + head_top - total_text_h) // 2)
    elif chest_viable and headroom_viable:
        if chest_space >= headroom + 100:
            start_y = chin_bottom + chin_clearance
        else:
            start_y = max(top_margin + 20, (top_margin + head_top - total_text_h) // 2)
    else:
        if chest_space >= headroom:
            start_y = bottom_margin - total_text_h
        else:
            start_y = top_margin + 20

    return max(top_margin + 20, min(start_y, bottom_margin - total_text_h))

def extract_frame_at_time(video_path: Path, timestamp: float = 2.0) -> Optional[Image.Image]:
    """Extracts a crisp frame from the video at specified timestamp using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.2f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
        if proc.returncode == 0 and len(proc.stdout) > 1000:
            import io
            return Image.open(io.BytesIO(proc.stdout)).convert("RGBA")
    except Exception as e:
        print(f"[-] Frame extraction error: {e}")
    return None

def _probe_video_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return max(0.0, float(result.stdout.strip()))
    except (OSError, TypeError, ValueError, subprocess.SubprocessError):
        pass
    return 0.0


def find_best_font(size: int = 72) -> Any:
    """Finds best available font from assets/fonts or falls back to default."""
    candidates = [
        FONTS_DIR / "Montserrat-Black.ttf",
        FONTS_DIR / "Anton-Regular.ttf",
        FONTS_DIR / "BebasNeue-Regular.ttf",
        Path("C:/Windows/Fonts/impact.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf")
    ]
    for font_path in candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                continue
    return ImageFont.load_default()

def wrap_text(text: str, font: Any, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    """Wraps text into lines that fit within max_width."""
    words = text.split()
    lines = []
    current_line: List[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def generate_clip_thumbnail(
    clip_path: Path,
    title: str,
    output_path: Optional[Path] = None,
    extract_time: float = 2.0,
    badge_text: str = "MUST WATCH",
    watermark: Optional[str] = "@allinonepodcastsss"
) -> Optional[Path]:
    """
    Generates a viral, high-CTR 9:16 thumbnail/cover for the clip:
    - Extracts frame from video.
    - Enhances contrast, saturation, and applies subtle vignette.
    - Draws high-visibility hook title with stroke and drop shadow.
    - Adds vibrant top pill badge and channel watermark.
    """
    out_path = output_path or (clip_path.parent / f"{clip_path.stem}_thumb.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(
        f".{out_path.stem}.{uuid.uuid4().hex}.part{out_path.suffix or '.jpg'}"
    )

    frame = extract_frame_at_time(clip_path, timestamp=extract_time)
    if not frame:
        duration = _probe_video_duration(clip_path)
        fallback_time = duration / 2 if duration > 0 else 0.5
        frame = extract_frame_at_time(clip_path, timestamp=fallback_time)

    if not frame:
        # Create gradient canvas if frame extract fails
        frame = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (20, 20, 25, 255))
    else:
        frame = frame.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
        # Enhance visual pop
        contrast_enhancer = ImageEnhance.Contrast(frame.convert("RGB"))
        frame = contrast_enhancer.enhance(1.15).convert("RGBA")
        color_enhancer = ImageEnhance.Color(frame.convert("RGB"))
        frame = color_enhancer.enhance(1.20).convert("RGBA")

    # Detect face to dynamically determine optimal title position (chest vs headroom)
    faces = None
    try:
        from src.face_tracker import get_face_detector, detect_faces_in_frame
        detector = get_face_detector()
        if detector is not None:
            frame_bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
            faces = detect_faces_in_frame(detector, frame_bgr, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    except Exception:
        faces = None

    # Measure viral hook title dimensions first
    clean_title = re.sub(r'[^\w\s\-\'\,\.\?]', '', title).strip().upper()
    title_font = find_best_font(76)
    max_title_width = OUTPUT_WIDTH - 160
    dummy_draw = ImageDraw.Draw(frame)
    lines = wrap_text(clean_title, title_font, max_title_width, dummy_draw)[:5]

    line_height = 92
    total_text_h = len(lines) * line_height
    start_y = calculate_smart_title_y(faces, total_text_h)

    # Dark gradient overlays for ultra-readable text
    overlay = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
    d_overlay = ImageDraw.Draw(overlay)

    # Top gradient for top badge
    for y in range(350):
        alpha = int(180 * (1 - y / 350))
        d_overlay.line([(0, y), (OUTPUT_WIDTH, y)], fill=(0, 0, 0, alpha))

    # Dynamic dark vignette backing behind wherever the title is positioned
    pad_y = 50
    tb_top = max(0, start_y - pad_y)
    tb_bottom = min(OUTPUT_HEIGHT, start_y + total_text_h + pad_y)
    for y in range(tb_top, tb_bottom):
        dist = min(y - tb_top, tb_bottom - y)
        alpha = int(190 * min(1.0, dist / 40.0))
        d_overlay.line([(0, y), (OUTPUT_WIDTH, y)], fill=(0, 0, 0, alpha))

    canvas = Image.alpha_composite(frame, overlay)
    draw = ImageDraw.Draw(canvas)

    # 1. Top Hook Badge / Pill
    if badge_text:
        badge_font = find_best_font(36)
        clean_badge = badge_text.strip().upper()
        b_bbox = draw.textbbox((0, 0), clean_badge, font=badge_font)
        bw = b_bbox[2] - b_bbox[0]
        bh = b_bbox[3] - b_bbox[1]
        
        pad_x, pad_y = 35, 16
        pill_w = min(OUTPUT_WIDTH - 40, bw + pad_x * 2)
        pill_h = bh + pad_y * 2
        pill_x = max(20, (OUTPUT_WIDTH - pill_w) // 2)
        pill_y = 140

        # Draw red/orange vibrant pill capsule
        draw.rounded_rectangle(
            [(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)],
            radius=pill_h // 2,
            fill=(255, 30, 45, 255),
            outline=(255, 255, 255, 220),
            width=3
        )
        draw.text(
            (pill_x + pad_x, pill_y + pad_y - 2),
            clean_badge,
            font=badge_font,
            fill=(255, 255, 255, 255)
        )

    # 2. Main Viral Hook Title
    colors = [(255, 230, 0), (255, 255, 255)]  # Alternating Gold / White

    for i, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = line_bbox[2] - line_bbox[0]
        lx = (OUTPUT_WIDTH - lw) // 2
        ly = start_y + (i * line_height)

        text_color = colors[i % len(colors)]

        # Heavy black drop shadow and outline for ultra-high contrast
        outline_range = 5
        for ox in range(-outline_range, outline_range + 1):
            for oy in range(-outline_range, outline_range + 1):
                if ox != 0 or oy != 0:
                    draw.text((lx + ox, ly + oy), line, font=title_font, fill=(0, 0, 0, 255))

        # Main text
        draw.text((lx, ly), line, font=title_font, fill=text_color)

    # 3. Watermark
    if watermark:
        wm_font = find_best_font(28)
        wm_text = watermark.strip()
        wm_bbox = draw.textbbox((0, 0), wm_text, font=wm_font)
        wm_w = wm_bbox[2] - wm_bbox[0]
        draw.text(((OUTPUT_WIDTH - wm_w) // 2, OUTPUT_HEIGHT - 120), wm_text, font=wm_font, fill=(220, 220, 220, 200))

    final_img = canvas.convert("RGB")
    try:
        final_img.save(temp_path, "JPEG", quality=95, optimize=True)
        with Image.open(temp_path) as verification_image:
            verification_image.verify()
        os.replace(temp_path, out_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise
    print(f"[+] Auto-generated viral thumbnail: {out_path.name}")
    return out_path
