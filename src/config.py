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
SFX_ASSETS_DIR = ASSETS_DIR / "sfx"
FONTS_DIR = ASSETS_DIR / "fonts"
BROLL_DIR = ASSETS_DIR / "broll"
HISTORY_FILE = DATA_DIR / "history.json"

# Ensure directories exist
DOWNLOADS_DIR.mkdir(exist_ok=True, parents=True)
CLIPS_DIR.mkdir(exist_ok=True, parents=True)
SUBTITLES_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_ASSETS_DIR.mkdir(exist_ok=True, parents=True)
SFX_ASSETS_DIR.mkdir(exist_ok=True, parents=True)
FONTS_DIR.mkdir(exist_ok=True, parents=True)
BROLL_DIR.mkdir(exist_ok=True, parents=True)

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
# The maintained Streamers actor is retained for explicitly requested full
# downloads. It does not support time ranges, so it must never be used for
# clip extraction. The segment actor accepts startTime/endTime and is invoked
# only after the viral detector selects a short range.
APIFY_FULL_DOWNLOAD_ACTOR_ID = os.getenv(
    "APIFY_FULL_DOWNLOAD_ACTOR_ID", "streamers/youtube-video-downloader"
).strip()
APIFY_SEGMENT_ACTOR_ID = os.getenv(
    "APIFY_SEGMENT_ACTOR_ID", "vidkraken/youtube-video-audio-downloader-reliable"
).strip()
APIFY_TRANSCRIPT_ACTOR_ID = os.getenv(
    "APIFY_TRANSCRIPT_ACTOR_ID", "om_kh/video-transcript-api"
).strip()
CLIP_ONLY_MODE = os.getenv("CLIP_ONLY_MODE", "true").lower() in ("true", "1", "yes")
MAX_DISCOVERY_CANDIDATES = max(1, int(os.getenv("MAX_DISCOVERY_CANDIDATES", "12")))
MAX_TRANSCRIPT_FALLBACKS = max(1, int(os.getenv("MAX_TRANSCRIPT_FALLBACKS", "3")))
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
OLLAMA_API_KEY = (
    os.getenv("OLLAMA_API_KEY") or
    os.getenv("ollama_api_key") or
    "2187c3d426224f23a8728263cf66e98a.aLOIml1Yo-_FvhhKQn7FnXn9"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "gemma4:31b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "https://ollama.com"
PEXELS_API_KEY = (
    os.getenv("PEXELS_API_KEY") or
    os.getenv("pexels_api_key") or
    "F38Z5qgMKdU9NUNEmmuELhQGdu4WDTOAz1X3fh1oGpnCkYHkP2HJ6CBW"
)
ENABLE_BROLL = os.getenv("ENABLE_BROLL", "true").lower() in ("true", "1", "yes")
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
COBALT_API_URL = os.getenv("COBALT_API_URL", "https://api.cobalt.tools")
COBALT_INSTANCES = [
    url.strip() for url in (
        os.getenv("COBALT_INSTANCES") or
        COBALT_API_URL
    ).split(",") if url.strip()
]
# Leave the local Proof-of-Origin service optional. Sending yt-dlp to a dead
# localhost service makes every extractor client fail before it can try its
# normal authenticated/cookie-based path.
YTDLP_POT_PROVIDER_URL = os.getenv("YTDLP_POT_PROVIDER_URL", "").strip()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "podcast-clip")

# Video Ingestion Standards (Strict Quality Gate)
MIN_VIDEO_HEIGHT = 720
MAX_VIDEO_HEIGHT = 1080

# Visual Retention & Anti-Fingerprint Features
ENABLE_PUNCH_ZOOM = os.getenv("ENABLE_PUNCH_ZOOM", "true").lower() in ("true", "1", "yes")
ENABLE_TOP_HOOK_BADGE = os.getenv("ENABLE_TOP_HOOK_BADGE", "true").lower() in ("true", "1", "yes")
HOOK_BADGE_DURATION = float(os.getenv("HOOK_BADGE_DURATION", "4.0"))  # 4-second initial hook retention badge
HOOK_BADGE_MARGIN_V = int(os.getenv("HOOK_BADGE_MARGIN_V", "95"))     # Safe zone (upper safe zone 90-140px, above speaker forehead)
ENABLE_BGM = os.getenv("ENABLE_BGM", "true").lower() in ("true", "1", "yes")  # Breaks audio fingerprinting
ENABLE_SFX = os.getenv("ENABLE_SFX", "true").lower() in ("true", "1", "yes")  # Transitions & keyword sound design
ENABLE_DYNAMIC_DUCKING = os.getenv("ENABLE_DYNAMIC_DUCKING", "true").lower() in ("true", "1", "yes")  # Sidechain BGM ducking
ENABLE_FILM_GRAIN = os.getenv("ENABLE_FILM_GRAIN", "true").lower() in ("true", "1", "yes")  # Breaks visual pHash

# Video & Format Defaults (Ultra HD 60FPS Broadcast Studio)
IS_CI = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

if IS_CI:
    # Optimized for GitHub Actions: Lower CPU/RAM/Disk usage to prevent OOM/Crash
    OUTPUT_WIDTH = 1080
    OUTPUT_HEIGHT = 1920
    FPS = 30
    VIDEO_CRF = 23  # Standard quality (saves massive disk space vs 16)
    AUDIO_BITRATE = "128k"
else:
    OUTPUT_WIDTH = 1080
    OUTPUT_HEIGHT = 1920
    FPS = 60
    VIDEO_CRF = 16  # Ultra high visual fidelity
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
        "font_size": 66,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H0000E6FF",    # Electric Yellow (&H00BBGGRR: Blue 00, Green E6, Red FF)
        "outline_color": "&H00000000",      # Deep Black Outline
        "outline_width": 6.5,
        "shadow_color": "&H90000000",       # Deep Soft Drop Shadow
        "shadow_depth": 3.5,
        "max_words_per_line": 2,            # Fast-paced punchy 2-word flash
        "uppercase": True
    },
    "neon_green": {
        "name": "Toxic Neon",
        "font_name": "Montserrat Black",
        "fallback_font": "Arial Black",
        "font_size": 66,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H0033FF22",    # Toxic Lime Green
        "outline_color": "&H00000000",
        "outline_width": 6.5,
        "shadow_color": "&HA0000000",
        "shadow_depth": 3.5,
        "max_words_per_line": 2,
        "uppercase": True
    },
    "luxury_gold": {
        "name": "Luxury Mindset",
        "font_name": "Bebas Neue",
        "fallback_font": "Helvetica",
        "font_size": 68,
        "primary_color": "&H00F5F5F5",      # Ivory White
        "highlight_color": "&H0000D7FF",    # Warm Gold
        "outline_color": "&H00101010",      # Charcoal Deep Outline
        "outline_width": 6.0,
        "shadow_color": "&H80000000",
        "shadow_depth": 3.0,
        "max_words_per_line": 3,
        "uppercase": True
    },
    "cyber_cyan": {
        "name": "Cyber Cyan",
        "font_name": "Anton",
        "fallback_font": "Arial Black",
        "font_size": 66,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H00FFFF00",    # Pure Cyan
        "outline_color": "&H00000000",
        "outline_width": 6.5,
        "shadow_color": "&HA0000000",
        "shadow_depth": 3.5,
        "max_words_per_line": 2,
        "uppercase": True
    }
}
