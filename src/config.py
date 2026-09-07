import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
CLIPS_DIR = BASE_DIR / "clips"
SUBTITLES_DIR = BASE_DIR / "subtitles"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
AUDIO_ASSETS_DIR = ASSETS_DIR / "audio"
HISTORY_FILE = DATA_DIR / "history.json"

# Ensure directories exist
DOWNLOADS_DIR.mkdir(exist_ok=True, parents=True)
CLIPS_DIR.mkdir(exist_ok=True, parents=True)
SUBTITLES_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_ASSETS_DIR.mkdir(exist_ok=True, parents=True)

# API Keys & Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
BUFFER_ACCESS_TOKEN = (
    os.getenv("BUFFER_ACCESS_TOKEN") or
    os.getenv("BUFFER_API_KEY") or
    ""
)
BUFFER_CHANNEL_IDS = [
    cid.strip() for cid in os.getenv("BUFFER_CHANNEL_IDS", "").split(",") if cid.strip()
]
CHANNEL_WATERMARK = os.getenv("CHANNEL_WATERMARK", "@allinonepodcastsss")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")
APIFY_API_TOKEN = (
    os.getenv("APIFY_API_TOKEN") or
    os.getenv("APIFY_KEY") or
    os.getenv("apify_key") or
    os.getenv("APIFY_API_TOKEN_NEW") or
    ""
)
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY") or
    os.getenv("groq_api_key") or
    ""
)
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY") or
    os.getenv("openrouter_api_key") or
    ""
)
CHOCODATA_API_KEY = (
    os.getenv("CHOCODATA_API_KEY") or
    os.getenv("chocodata_api_key") or
    ""
)
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "")
YTDLP_PROXY = (
    os.getenv("YTDLP_PROXY") or
    os.getenv("HTTP_PROXY") or
    os.getenv("PROXY_URL") or
    ""
)
RAPIDAPI_KEY = (
    os.getenv("RAPIDAPI_KEY") or
    os.getenv("rapidapi_key") or
    ""
)
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "podcast-clip")

# Visual Retention & Audio Features
ENABLE_PUNCH_ZOOM = os.getenv("ENABLE_PUNCH_ZOOM", "true").lower() in ("true", "1", "yes")
ENABLE_TOP_HOOK_BADGE = os.getenv("ENABLE_TOP_HOOK_BADGE", "true").lower() in ("true", "1", "yes")
ENABLE_BGM = os.getenv("ENABLE_BGM", "false").lower() in ("true", "1", "yes")

# Video & Format Defaults (Ultra HD 60FPS Broadcast Studio)
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
FPS = 60
VIDEO_CRF = 16  # Ultra high visual fidelity (near-lossless)
AUDIO_BITRATE = "256k"

# Social Media UI Safe Zone Margins (TikTok, Reels, Shorts)
SAFE_ZONE_TOP = 240       # Reserved for header/search
SAFE_ZONE_BOTTOM = 380    # Reserved for creator handle, captions, audio disc
SAFE_ZONE_RIGHT = 120     # Reserved for like, comment, share icons

# Audio Mastering (Social Broadcast Standard)
TARGET_LUFS = -14.0       # EBU R128 standard for Instagram/TikTok/Shorts
TARGET_TRUE_PEAK = -1.5   # Prevents compression clipping distortion
HIGHPASS_FREQ = 80        # Cut microphone room rumble
VOCAL_PRESENCE_FREQ = 3000 # Enhance voice clarity
VOCAL_AIR_FREQ = 10000    # High-frequency studio microphone sheen

# Subtitle Color & Aesthetic Themes (ASS color codes: &HAABBGGRR)
# Note: ASS hex format is &H[Alpha][Blue][Green][Red]
SUBTITLE_THEMES: Dict[str, Dict[str, Any]] = {
    "hormozi": {
        "name": "Hormozi Viral",
        "font_name": "Montserrat Black",
        "fallback_font": "Arial Black",
        "font_size": 52,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H0000E6FF",    # Electric Yellow (&H00BBGGRR: Blue 00, Green E6, Red FF)
        "outline_color": "&H00000000",      # Deep Black Outline
        "outline_width": 4.5,
        "shadow_color": "&H80000000",       # Soft Black Shadow
        "shadow_depth": 2.5,
        "max_words_per_line": 3,
        "uppercase": True
    },
    "neon_green": {
        "name": "Toxic Neon",
        "font_name": "Montserrat Black",
        "fallback_font": "Arial Black",
        "font_size": 52,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H0066FF00",    # Neon Toxic Green
        "outline_color": "&H00000000",
        "outline_width": 4.0,
        "shadow_color": "&H90000000",
        "shadow_depth": 2.0,
        "max_words_per_line": 3,
        "uppercase": True
    },
    "luxury_gold": {
        "name": "Luxury Mindset",
        "font_name": "Arial",
        "fallback_font": "Helvetica",
        "font_size": 50,
        "primary_color": "&H00F5F5F5",      # Ivory White
        "highlight_color": "&H0000D7FF",    # Warm Gold
        "outline_color": "&H001A1A1A",      # Charcoal Outline
        "outline_width": 3.5,
        "shadow_color": "&H70000000",
        "shadow_depth": 3.0,
        "max_words_per_line": 4,
        "uppercase": False
    },
    "cyber_cyan": {
        "name": "Cyber Cyan",
        "font_name": "Montserrat Black",
        "fallback_font": "Arial Black",
        "font_size": 52,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H00FFFF00",    # Pure Cyan
        "outline_color": "&H00000000",
        "outline_width": 4.0,
        "shadow_color": "&H90000000",
        "shadow_depth": 2.5,
        "max_words_per_line": 3,
        "uppercase": True
    }
}
