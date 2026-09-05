# 🎙️ Autonomous AI Podcast Viral Clipper & Buffer Social Publisher ($0 Budget)

An automated, broadcast-grade pipeline that extracts high-retention viral moments from long-form podcasts, applies AI multi-speaker face tracking, dynamic 9:16 framing (TikTok/Reels/Shorts), studio audio mastering, and ultra-premium animated subtitles, hosts generated clips permanently on GitHub Releases for $0, and automatically publishes/schedules them to social media via **Buffer**.

---

## 🌟 Key Capabilities

1. **$0 Budget Architecture**:
   - **AI Brain**: Google Gemini 1.5/2.5 Flash Free Tier (15 RPM, 1M tokens/min free).
   - **Instant Transcripts**: Native YouTube caption retrieval (0 compute) + `faster-whisper` CPU int8 fallback.
   - **Face Tracking & Layouts**: Runs on lightweight CPU OpenCV, no expensive GPU servers needed.
   - **Free Hosting for Buffer**: Solves Buffer's public URL requirement by uploading clips to **GitHub Releases** using `${{ secrets.GITHUB_TOKEN }}`.
   - **Zero-Cost Cloud Runner**: Runs entirely on **GitHub Actions** (2,000 free minutes/month).

2. **Studio-Grade Multi-Speaker & Framing Engine**:
   - **Single Speaker**: Smooth camera tracking with exponential moving average (no jitter).
   - **Two Speakers in Wide Shot (Host & Guest)**: Automatically creates a **Dynamic Split-Screen** (Host top 1080x960, Guest bottom 1080x960) with a modern divider bar — the signature viral podcast format.
   - **Panel / Group Discussions**: Dynamic **Blurred-Stack Layout** (crisp 16:9 centered over an ambient blurred and darkened background, preserving all participants without head cutoffs).

3. **Universal Social Media Safe-Zone Compliance**:
   - Strictly optimized for **TikTok, Instagram Reels, Facebook Reels, and YouTube Shorts** (1080x1920).
   - Captions and focal action are placed in the safe zone (avoiding like/share buttons, account handles, search bars, and audio discs).

4. **Studio Audio Mastering**:
   - **80Hz High-Pass Filter**: Removes room rumble and desk vibrations.
   - **3000Hz Voice Presence EQ**: Lifts vocal clarity and intelligibility.
   - **EBU R128 (-14 LUFS / -1.5 dBTP)**: Broadcast-standard loudness normalization matching official Reels and TikTok specifications.

5. **Ultra-Premium Animated Subtitle Themes**:
   - **High-Retention Pacing**: Displays only 2–4 words at a time.
   - **Word-Level Karaoke Animation**: The spoken word pops with color as it is uttered.
   - Themes: `hormozi` (Yellow & White), `neon_green`, `luxury_gold`, and `cyber_cyan`.

---

## 🚀 Quick Start (Running via GitHub Actions - $0)

You do **not** need to install anything on your personal machine to run this!

### Step 1: Fork or Push this Repository to GitHub
Push this codebase to your own GitHub repository (public or private).

### Step 2: Add Repository Secrets
Navigate to your GitHub repository:
**Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions** $\rightarrow$ **New repository secret**:

| Secret Name | How to get it (100% Free) |
| :--- | :--- |
| `GEMINI_API_KEY` | Grab your free API key at [Google AI Studio](https://aistudio.google.com/app/apikey). |
| `BUFFER_ACCESS_TOKEN` | Grab your access token from [Buffer Developer Portal](https://publish.buffer.com) or Buffer Settings. |
| `BUFFER_CHANNEL_IDS` | *(Optional)* Comma-separated channel IDs. Leave empty to automatically broadcast to all connected channels! |

> Note: `GITHUB_TOKEN` is automatically provided by GitHub Actions with release write permissions.

### Step 3: Run the Workflow
1. Go to the **Actions** tab in your GitHub repository.
2. Click **Podcast Viral Clips to Social (Buffer)** in the left sidebar.
3. Click **Run workflow**:
   - **Video URL**: Paste any YouTube link, podcast URL, or direct MP4 link.
   - **Number of clips**: Choose `1` to `5`.
   - **Framing**: `auto` (smart face detection), `split`, `blur_stack`, or `crop`.
   - **Subtitle Style**: `hormozi`, `neon_green`, `luxury_gold`, or `cyber_cyan`.
   - **Post to Buffer**: Check to schedule to your social queue automatically.
4. Click **Run workflow**.

---

## 💻 Running Locally

### 1. Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and in your system PATH.

### 2. Installation
```bash
git clone https://github.com/your-username/podcasts-clips-to-social.git
cd podcasts-clips-to-social
pip install -r requirements.txt
cp .env.example .env
```
Fill in your `GEMINI_API_KEY` and `BUFFER_ACCESS_TOKEN` in `.env`.

### 3. Usage Examples

#### Extract 3 viral clips with auto face tracking and Hormozi subtitles:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --num-clips 3
```

#### Force Split-Screen layout with Neon Green subtitles:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --framing split --subtitle-style neon_green
```

#### Skip subtitle burn-in if source video already has baked-in captions:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --subtitles skip
```

#### Auto-publish generated clips directly to Buffer:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --post-to-buffer
```

#### Dry-run test (analyze viral moments and hooks without video rendering):
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --dry-run
```

---

## 📁 Project Architecture

```
PODCASTS-CLIPS-TO-SOCIAL/
├── .github/
│   └── workflows/
│       └── podcast_clipper.yml   # Complete GitHub Actions automation
├── src/
│   ├── config.py                 # Safe zones, audio targets, subtitle presets
│   ├── downloader.py             # yt-dlp & instant YouTube transcript extractor
│   ├── transcriber.py            # faster-whisper CPU fallback transcriber
│   ├── viral_detector.py         # Google Gemini Flash viral moment analyzer
│   ├── face_tracker.py           # OpenCV face tracking & split-screen decision engine
│   ├── subtitle_generator.py     # ASS karaoke subtitle generator
│   ├── video_editor.py           # FFmpeg 9:16 layout, audio mastering & burn-in
│   ├── github_uploader.py        # Permanent $0 release asset hosting for Buffer
│   └── buffer_client.py          # Buffer GraphQL API scheduler
├── tests/
│   └── test_pipeline.py          # Automated unit & integration tests
├── main.py                       # Unified CLI entrypoint
├── requirements.txt              # Minimal lightweight dependencies
├── .env.example                  # Environment credentials template
└── README.md                     # Documentation
```
