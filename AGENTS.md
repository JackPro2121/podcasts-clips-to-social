# AGENTS.md — Autonomous AI Media Engine (V1 Podcast & V2 Streamer Pipelines)

> **CRITICAL DIRECTIVE FOR ALL AGENTS & DEVELOPERS:**
> This repository houses **TWO INDEPENDENT, ISOLATED PRODUCTION PIPELINES**:
> 1. **V1 Pipeline**: High-CPM Long-Form Podcasts (Personal Finance, Tech, Interviews)
> 2. **V2 Pipeline**: High-Energy US Live Streamers (Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc)
> 
> **RULE OF ISOLATION**: Changes to the V2 Streamer pipeline must **NEVER** break, alter, or regress the V1 Podcast pipeline. Each pipeline has its own entrypoint, workflow, and dedicated Buffer publishing credentials.

---

## 🎯 System Architecture Overview

```
                               ┌─────────────────────────────────────────────────────────┐
                               │           PODCASTS-CLIPS-TO-SOCIAL ENGINE               │
                               └────────────────────────────┬────────────────────────────┘
                                                            │
                            ┌───────────────────────────────┴───────────────────────────────┐
                            ▼                                                               ▼
        ┌───────────────────────────────────────┐                       ┌───────────────────────────────────────┐
        │       V1: PODCAST CLIPPER             │                       │       V2: STREAMER CLIPPER            │
        │   (Wealth, Finance, Long-Form)        │                       │   (Kai Cenat, Speed, US Streamers)    │
        ├───────────────────────────────────────┤                       ├───────────────────────────────────────┤
        │ • CLI: main.py                        │                       │ • CLI: main_streamer.py               │
        │ • CI: podcast_clipper.yml             │                       │ • CI: streamer_clipper.yml            │
        │ • Core: src/                          │                       │ • Core: v2/streamers/                │
        │ • Tone: Finance / Serious / Interview │                       │ • Tone: Unhinged / Chaotic / Funny    │
        │ • Subtitles: Hormozi / Center Pop     │                       │ • Subtitles: TikTok Meme Box + Lower  │
        │ • Face: YuNet Active Speaker          │                       │ • Face: SFace ArcFace ID Locking      │
        │ • Framing: Split-Screen (Host/Guest)  │                       │ • Framing: 3-Tier State Machine +     │
        │ • Buffer: BUFFER_ACCESS_TOKEN         │                       │            Auto-Stacked (Webcam/Game) │
        │                                       │                       │ • Buffer: STREAMER_BUFFER_ACCESS_TOKEN│
        └───────────────────────────────────────┘                       └───────────────────────────────────────┘
```

---

## 🧭 Deep Dive: Pipeline V1 (Podcast Viral Clipper)

### Purpose & Niche
Automated clipping of 1-to-3 hour podcasts focused on Personal Finance, Wealth Creation, and High-CPM Niches (Caleb Hammer *Financial Audit*, Dave Ramsey, Graham Stephan, Humphrey Yang, Huberman Lab).

### Technical Specifications
* **Entrypoint**: [`main.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/main.py)
* **GitHub Actions Workflow**: [`.github/workflows/podcast_clipper.yml`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/.github/workflows/podcast_clipper.yml)
* **Viral Detector**: [`core/viral_detector.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/core/viral_detector.py) (Gemini Flash tuned for debt drama, net worth confessions, business secrets).
* **Face Tracking & Layout**:
  - [`core/face_tracker.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/core/face_tracker.py): OpenCV YuNet on CPU tracks active mouth motion.
  - Generates single 9:16 portrait or dual-speaker split screen (Top 1080x960 / Bottom 1080x960) with zero black bars.
* **Audio Engineering**:
  - 80Hz rumble cut, +2.5dB vocal presence boost at 3kHz.
  - Dynamic sidechain ducking on background music (`sidechaincompress`).
  - EBU R128 (-14.0 LUFS / -1.5 dBTP) broadcast normalization.
* **Subtitles**:
  - [`core/subtitle_generator.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/core/subtitle_generator.py): Word-by-word spring bounce animation (`hormozi`, `neon_green`, `luxury_gold`).
* **Publishing**:
  - Uses `BUFFER_ACCESS_TOKEN` and `BUFFER_CHANNEL_IDS` to schedule to podcast-specific social channels.

---

## ⚡ Deep Dive: Pipeline V2 (Streamer Viral Clipper)

### Purpose & Niche
Extracts high-retention TikTok/Shorts clips from 3-to-5 hour live streams of top US creators (Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc, Adin Ross). Designed to compete directly with top plug channels (*Onlyclipin*, *Prada Clips*).

### Technical Specifications
* **Entrypoint**: [`main_streamer.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/main_streamer.py)
* **GitHub Actions Workflow**: [`.github/workflows/streamer_clipper.yml`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/.github/workflows/streamer_clipper.yml)
* **Zero-Download Discovery**:
  - [`v2/streamers/discovery.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/v2/streamers/discovery.py): Checks YouTube channel RSS feeds (60s latency, $0 API cost) and Firecrawl. Skips previously clipped streams via [`data/streamer_history.json`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/data/streamer_history.json).
  - Fetches native timed text without downloading the full 15GB video file.
* **Streamer Viral Intelligence**:
  - [`v2/streamers/viral_detector.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/v2/streamers/viral_detector.py): Gemini 3.8/2.5 Flash finds explosive screaming, rage, comedy fails, and desk slams. Generates high-CTR all-caps hook titles.
  - Enforces strict hashtag bounds: **2 to 4 hashtags** (e.g. `#kaicenat #streamer #funny #shorts`).
* **SFace Streamer Identity Locking & 3-Tier State Machine**:
  - [`v2/streamers/face_tracker.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/v2/streamers/face_tracker.py):
    - **SFace (SphereFace / ArcFace ONNX)** + YuNet: Loads pre-computed ground-truth embeddings from `core/models/streamer_profiles/` (`kai_cenat.npy`, `ishowspeed.npy`, `xqc.npy`). Rejects guests (MrBeast, Kevin Hart, AMP crew) from hijacking the camera.
    - **Tier 1 (Locked)**: Streamer centered with smooth spring-damper LERP panning and safe headroom above the face.
    - **Tier 2 (Occluded / Turning / Running)**: Momentary face loss ($<1.5\text{s}$) holds last known position without jerky movements.
    - **Tier 3 (Off-Screen Exit $>1.5\text{s}$)**: Streamer leaves chair/room $\rightarrow$ Smooth autonomous fallback to **Wide Room View** (`presentation_slide` mode with blurred backdrop). Instant snap-back when the streamer returns!
* **Stream Layout Intelligence Engine**:
  - [`v2/streamers/layout_compositor.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/v2/streamers/layout_compositor.py):
    - **Mode 1 (`stacked_webcam`)**: Detects stationary corner webcam ROI. Generates auto-stacked 9:16 layout (Top 50%: Facecam Reaction / Bottom 50%: Gameplay/Video).
    - **Mode 2 (`multi_person_room`)**: Detects 2+ prominent people in room/collab streams (Kai Cenat x MrBeast, Speed x Adin). Uses SFace ArcFace identity recognition to place Host/Streamer on the Top pane (1080x960) and Guest on the Bottom pane (1080x960) with zero blurry bars, zero letterboxing, and 100% full-screen vertical immersion!
    - **Mode 3 (`single_streamer_room`)**: Single-creator desk/room framing with SFace identity lock.
* **Audio Scream & Energy Analyzer**:
  - [`v2/streamers/audio_analyzer.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/v2/streamers/audio_analyzer.py): Detects decibel volume jumps ($>6\text{ dB}$ above median) via FFmpeg raw PCM to trigger dynamic punch-zooms (1.2x zoompan).
* **Live Chat Velocity Engine**:
  - [`v2/streamers/chat_analyzer.py`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/v2/streamers/chat_analyzer.py): Calculates Messages Per Second (MPS) and weights viral emotes (`💀`, `LMAO`, `COOKED`, `W`, `L`).
* **Aesthetics & In-App TikTok Meme Hook Sticker**:
  - Persistent multi-line connected white meme container box placed in upper safe-zone (`MarginV = 130`) hugging bold black text with 8px rounded corners, subtle drop shadow, and viral emojis (`💀`, `😭`, `🔥`).
  - High-energy kinetic subtitles (`tiktok_streamer`: Montserrat Black 58pt, uppercase, 2-word bursts, Electric Yellow pop `#00E6FF`) in the lower safe-zone.
* **Publishing Isolation**:
  - Uses `STREAMER_BUFFER_ACCESS_TOKEN` and `STREAMER_BUFFER_CHANNEL_IDS`. **Zero crossover** with podcast accounts.

---

## 📁 Repository Directory Structure

```
PODCASTS-CLIPS-TO-SOCIAL/
├── .github/workflows/
│   ├── podcast_clipper.yml          # [V1] Podcast production workflow
│   ├── streamer_clipper.yml         # [V2] Streamer production workflow
│   └── cleanup_old_releases.yml     # Storage hygiene (5-day release purge)
├── src/
│   ├── config.py                    # Shared configs, fonts, safe-zones, audio targets
│   ├── creative_spec.py             # Visual metrics, tolerances, motion policies
│   ├── downloader.py                # Targeted range downloader & Apify proxy
│   ├── transcriber.py               # faster-whisper & YouTube timed text parser
│   ├── jump_cutter.py               # Dead-air trimming and audio compaction
│   ├── face_tracker.py              # [V1] Active mouth tracker & podcast framing
│   ├── subtitle_generator.py        # Styled ASS subtitle generator (V1 & V2 themes)
│   ├── video_editor.py              # FFmpeg filtergraph engine, studio color grade, loudnorm
│   ├── thumbnail_generator.py       # Universal CTR thumbnail engine (V1 & V2 styles)
│   ├── buffer_client.py             # [V1] Podcast Buffer publisher
│   ├── github_uploader.py           # Release uploader for direct asset URLs
│   ├── models/
│   │   ├── face_detection_yunet.onnx         # 232KB tracked face detector
│   │   └── face_recognition_sface_2021dec.onnx # Auto-downloaded SFace recognizer
│   └── streamers/                   # [V2 EXCLUSIVE ENGINE]
│       ├── __init__.py
│       ├── pipeline.py              # V2 Orchestrator
│       ├── discovery.py             # YouTube RSS & Firecrawl streamer discoverer
│       ├── viral_detector.py        # Streamer viral moment finder (Gemini 3.8 Flash)
│       ├── face_tracker.py          # SFace Identity Locking & 3-Tier State Machine
│       ├── layout_compositor.py     # Auto-stacked webcam/gameplay compositor
│       ├── audio_analyzer.py        # Decibel scream & audio peak detector
│       └── chat_analyzer.py         # Live chat velocity (MPS) & emote analyzer
├── data/
│   └── streamer_history.json        # [V2] Processed streamer VOD registry
├── clips/
│   └── streamers/                   # Output directory for rendered V2 clips
├── main.py                          # [V1] CLI Entrypoint
├── main_streamer.py                 # [V2] CLI Entrypoint
├── requirements.txt                 # Dependencies
├── AGENTS.md                        # Canonical architecture & operational guide
└── README.md                        # Project documentation
```

---

## 🔑 Environment Variables & Secrets Reference

| Secret Name | Pipeline | Required? | Purpose |
| :--- | :---: | :---: | :--- |
| `GEMINI_API_KEY` | V1 & V2 | **Yes** | Google AI Studio key for viral moment extraction. |
| `APIFY_API_TOKEN` | V1 & V2 | Optional | Residential proxy fallback for YouTube datacenter bypass. |
| `FIRECRAWL_API_KEY` | V2 | Optional | Real-time streamer VOD search engine. |
| `BUFFER_ACCESS_TOKEN` | **V1 Only** | Optional | Social publishing token for Podcast accounts. |
| `BUFFER_CHANNEL_IDS` | **V1 Only** | Optional | Specific podcast social channels in Buffer. |
| `STREAMER_BUFFER_ACCESS_TOKEN` | **V2 Only** | Optional | **Isolated** social token for Streamer TikTok/Reels accounts. |
| `STREAMER_BUFFER_CHANNEL_IDS` | **V2 Only** | Optional | **Isolated** streamer social channels in Buffer. |
| `STREAMER_WATERMARK` | **V2 Only** | Optional | Channel watermark text (default: `@streamersclips`). |

---

## 💻 CLI Commands & Usage

### 🎙️ Running Pipeline V1 (Podcasts)
```bash
# Process a podcast episode (default 3 clips, auto framing, Hormozi subtitles):
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3

# Force Dual-Speaker Split Screen with Luxury Gold subtitles:
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --framing split --subtitle-style luxury_gold

# Post directly to Podcast Buffer account:
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --post-to-buffer
```

### ⚡ Running Pipeline V2 (Streamers)
```bash
# Auto-discover fresh stream across top US streamers:
python main_streamer.py

# Target specific streamer (Kai Cenat, Speed, Jynxzi, CaseOh, xQc):
python main_streamer.py --streamer kai_cenat --num-clips 3
python main_streamer.py --streamer ishowspeed --num-clips 3

# Process specific stream VOD URL:
python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3

# Force specific framing mode (auto, single_smooth, split_screen, stacked_webcam, fullscreen_room):
python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --framing auto
```

---

## 🛡️ Critical Commandments for AI Agents

1. **V1 Pipeline Inviolability**: Do NOT touch or edit `core/face_tracker.py`, `main.py`, or `core/viral_detector.py` when developing V2 features. All streamer-specific improvements belong strictly in `v2/streamers/`.
2. **Buffer Isolation**: Never reuse `BUFFER_ACCESS_TOKEN` for streamer clips. Streamer publishing must strictly read `STREAMER_BUFFER_ACCESS_TOKEN`.
3. **Hashtag Strictness**: Streamer social captions must contain **between 2 and 4 hashtags**. Never generate 5+ hashtags or 0 hashtags.
4. **Anti-Detection & Safe Zones**:
   - Subtitles must never collide with mobile UI safe-zones (bottom 20%, top 15%, right 12%).
   - The TikTok white meme hook box must always be placed in the upper safe-zone with drop shadow and rounded corners.
5. **No Full-Video Downloads**: Never download 15GB stream VODs. Always fetch transcripts first and perform range downloads of 30–45s windows via `yt-dlp`.
6. **Zero Secrets in Git**: Never log, hardcode, or commit API keys or tokens into git commits or documentation.
