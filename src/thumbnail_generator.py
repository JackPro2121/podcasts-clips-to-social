import os
import re
import subprocess
from pathlib import Path
from typing import Optional, List
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

from src.config import FONTS_DIR, OUTPUT_WIDTH, OUTPUT_HEIGHT, CLIPS_DIR

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
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if proc.returncode == 0 and len(proc.stdout) > 1000:
            import io
            return Image.open(io.BytesIO(proc.stdout)).convert("RGBA")
    except Exception as e:
        print(f"[-] Frame extraction error: {e}")
    return None

def find_best_font(size: int = 72) -> ImageFont.FreeTypeFont:
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

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    """Wraps text into lines that fit within max_width."""
    words = text.split()
    lines = []
    current_line = []

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
                current_line = []
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

    frame = extract_frame_at_time(clip_path, timestamp=extract_time)
    if not frame:
        # Fallback to middle frame
        frame = extract_frame_at_time(clip_path, timestamp=0.5)

    if not frame:
        # Create gradient canvas if frame extract fails
        frame = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (20, 20, 25, 255))
    else:
        frame = frame.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
        # Enhance visual pop
        enhancer = ImageEnhance.Contrast(frame.convert("RGB"))
        frame = enhancer.enhance(1.15).convert("RGBA")
        enhancer = ImageEnhance.Color(frame.convert("RGB"))
        frame = enhancer.enhance(1.20).convert("RGBA")

    # Dark gradient overlays at top and bottom for ultra-readable text
    overlay = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
    d_overlay = ImageDraw.Draw(overlay)

    # Top gradient
    for y in range(400):
        alpha = int(180 * (1 - y / 400))
        d_overlay.line([(0, y), (OUTPUT_WIDTH, y)], fill=(0, 0, 0, alpha))

    # Center-bottom gradient for hook title
    for y in range(900, OUTPUT_HEIGHT):
        alpha = int(220 * min(1.0, (y - 900) / 450))
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
        pill_w = bw + pad_x * 2
        pill_h = bh + pad_y * 2
        pill_x = (OUTPUT_WIDTH - pill_w) // 2
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
    clean_title = re.sub(r'[^\w\s\-\'\,\.\?]', '', title).strip().upper()
    title_font = find_best_font(76)
    max_title_width = OUTPUT_WIDTH - 160
    lines = wrap_text(clean_title, title_font, max_title_width, draw)

    # Position title in the middle-lower region (Y: ~1150px)
    line_height = 92
    total_text_h = len(lines) * line_height
    start_y = 1200 - (total_text_h // 2)

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
    final_img.save(out_path, "JPEG", quality=95, optimize=True)
    print(f"[+] Auto-generated viral thumbnail: {out_path.name}")
    return out_path
