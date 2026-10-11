# Autonomous AI Viral Video Clipper & Social Publisher

[![Streamer Viral Clips (V2)](https://github.com/JackPro2121/podcasts-clips-to-social/actions/workflows/streamer_clipper.yml/badge.svg)](https://github.com/JackPro2121/podcasts-clips-to-social/actions/workflows/streamer_clipper.yml)
[![Podcast Viral Clips (V1)](https://github.com/JackPro2121/podcasts-clips-to-social/actions/workflows/podcast_clipper.yml/badge.svg)](https://github.com/JackPro2121/podcasts-clips-to-social/actions/workflows/podcast_clipper.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, **$0-budget autonomous AI media engine** that converts long-form content (podcasts, interviews, and live streams) into viral 9:16 vertical videos for **TikTok, YouTube Shorts, and Instagram Reels**.

---

## 🚀 Dual-Engine Architecture (V1 vs. V2)

This repository features **two completely isolated production pipelines**, each engineered for its specific content dynamics:

```
                            ┌───────────────────────────────────────────────┐
                            │        PODCASTS-CLIPS-TO-SOCIAL ENGINE        │
                            └───────────────────────┬───────────────────────┘
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
    ┌───────────────────────────────┐                                 ┌───────────────────────────────┐
    │     PIPELINE V1: PODCASTS     │                                 │     PIPELINE V2: STREAMERS    │
    │  (Personal Finance & Wealth)  │                                 │   (Kai Cenat, Speed, Gaming)  │
    ├───────────────────────────────┤                                 ├───────────────────────────────┤
    │ • Long-form interviews & pods │                                 │ • High-energy chaotic streams │
    │ • Caleb Hammer, Ramsey, etc.  │                                 │ • Kai Cenat, IShowSpeed, etc. │
    │ • Active mouth tracking (CPU) │                                 │ • SFace ArcFace ID locking    │
    │ • Dynamic split-screen (9:8)  │                                 │ • 3-Tier off-screen fallback  │
    │ • Hormozi spring subtitles    │                                 │ • Auto-stacked (Webcam/Game)  │
    │ • Sidechain ducked BGM        │                                 │ • TikTok meme box + SFX       │
    │ • CLI: python main.py         │                                 │ • CLI: python main_streamer.py│
    │ • CI: podcast_clipper.yml     │                                 │ • CI: streamer_clipper.yml    │
    └───────────────────────────────┘                                 └───────────────────────────────┘
```

---

## 🌟 Feature Breakdown

### 🎙️ Pipeline V1: Podcast Viral Clipper
* **Hyper-Focused Niche**: Tuned for high-CPM personal finance, debt confessions, net worth reveals, and entrepreneurship.
* **Dual-Speaker Split Screen**: Active mouth motion detection dynamically stacks host and guest vertically (1080x960 each) with zero black side bars.
* **Studio Audio Mastering**: EBU R128 (-14.0 LUFS / -1.5 dBTP), 80Hz rumble cut, +2.5dB vocal presence boost at 3kHz, and dynamic sidechain compression on background music.
* **Kinetic Subtitles**: Word-by-word spring-bounce animation (`hormozi`, `neon_green`, `luxury_gold`).

### ⚡ Pipeline V2: Streamer Viral Clipper
* **Target Streamers**: Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc, Adin Ross.
* **SFace Streamer Identity Locking**: Uses OpenCV SFace (SphereFace / ArcFace ONNX) to lock onto the primary streamer's 128-D face embedding, ignoring guests (MrBeast, Kevin Hart, AMP crew) or background posters.
* **3-Tier Off-Screen State Machine**:
  - *Tier 1 (Locked)*: Centered 9:16 portrait tracking with spring-damper easing.
  - *Tier 2 (Occluded / Running)*: Maintains position anchor when streamer jumps or turns around.
  - *Tier 3 (Off-Screen Exit $>1.5\text{s}$)*: Automatically transitions to a stylish **Wide Room Shot** (`presentation_slide` mode with blurred backdrop) so chaos is never missed. Instantly re-locks when the streamer returns!
* **Auto-Stacked Layout Detection**: Automatically classifies stream layout into stationary corner webcam over gameplay vs. full-screen room camera.
* **Audio Energy & Scream Detection**: Real-time raw PCM waveform analyzer spots decibel jumps ($>6\text{ dB}$ above median) to cue dynamic 1.2x punch-zooms.
* **Authentic In-App TikTok Meme Hook Sticker**: Multi-line connected white container box hugging text with 8px rounded corners, subtle drop shadow, and viral emojis (`💀`, `😭`, `🔥`).
* **Strict Tag Management**: Enforces exactly 2 to 4 hashtags for optimal algorithmic distribution.
* **Publishing Isolation**: Dedicated streamer credentials, completely isolated from podcast social accounts.

---

## 🛠️ Quick Start (Running via GitHub Actions - $0)

No local setup or expensive GPU servers required! Runs entirely in GitHub Actions.

### 1. Add Repository Secrets
In your GitHub repo: **Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions**:

| Secret Name | Required By | Description |
| :--- | :---: | :--- |
| `GEMINI_API_KEY` | V1 & V2 | Free API key from [Google AI Studio](https://aistudio.google.com/app/apikey). |
| `APIFY_API_TOKEN` | Optional | Residential proxy fallback for YouTube datacenter blocks. |
| `FIRECRAWL_API_KEY` | Optional | Real-time streamer stream discovery. |
| `BUFFER_ACCESS_TOKEN` | V1 Only | Buffer token for Podcast accounts. |
| `STREAMER_BUFFER_ACCESS_TOKEN` | V2 Only | Buffer token for Streamer accounts. |

### 2. Trigger Workflow
1. Navigate to the **Actions** tab in GitHub.
2. Select either:
   - **Streamer Viral Clips to Social (V2)** for live streamers.
   - **Podcast Viral Clips to Social (Buffer)** for podcasts.
3. Click **Run workflow**, paste the video URL (or leave blank to auto-discover), and choose the number of clips (`1` to `5`).

---

## 💻 Local Installation & Usage

### Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in system PATH.

```bash
git clone https://github.com/JackPro2121/podcasts-clips-to-social.git
cd podcasts-clips-to-social
pip install -r requirements.txt
```

### Running Pipeline V1 (Podcasts)
```bash
# Extract 3 viral clips with Hormozi subtitles:
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3

# Force Dual-Speaker Split Screen:
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --framing split
```

### Running Pipeline V2 (Streamers)
```bash
# Auto-discover fresh streams across top US creators:
python main_streamer.py

# Target specific streamer:
python main_streamer.py --streamer kai_cenat --num-clips 3
python main_streamer.py --streamer ishowspeed --num-clips 3

# Process specific stream VOD URL:
python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3
```

---

## 📖 Operational Documentation

* [`AGENTS.md`](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/AGENTS.md) — Canonical rules, architecture, and guidelines for AI agents and developers.
