# AGENTS.md — Autonomous AI Media Engine (V1 Podcast Pipeline)

> **CRITICAL DIRECTIVE FOR ALL AGENTS & DEVELOPERS:**
> This repository is the **V1 Podcast pipeline** only — high-CPM long-form podcasts
> (Personal Finance, Tech, Interviews).
>
> The **V2 Streamer pipeline** (Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc) lives in
> its own repository:
> **[`JackPro2121/streamer-clips-to-social`](https://github.com/JackPro2121/streamer-clips-to-social)**
> (private mirror: `ranajawadai/streamer-clips-to-social`).
>
> **RULE OF ISOLATION**: this repo mounts the `clips-core` engine as a git
> submodule at `core/`; the streamer repo carries its **own vendored copy** of the
> same upstream (see its `SYNC.md`). Apart from that shared origin, everything is
> separate: separate entrypoints, separate workflows, separate Buffer credentials.
> A change made here is **not** a change to V2, and vice versa. Never let work on
> one pipeline regress the other.

---

## 🎯 System Architecture Overview

```
                          ┌─────────────────────────────────────────┐
                          │      TWO REPOS, ONE UPSTREAM ENGINE     │
                          └────────────────────┬────────────────────┘
                                               │
                             ┌─────────────────┴──────────────────┐
                             │   clips-core (upstream engine repo) │
                             │   submodule in V1, vendored in V2   │
                             └─────────────────┬──────────────────┘
                  ┌────────────────────────────┴────────────────────────────┐
                  ▼                                                         ▼
   ┌─────────────────────────────────┐                    ┌─────────────────────────────────┐
   │  THIS REPO: V1 PODCAST CLIPPER  │                    │  V2 STREAMER CLIPPER (own repo) │
   │   (Wealth, Finance, Long-Form)  │                    │   (Kai Cenat, Speed, US Streams)│
   ├─────────────────────────────────┤                    ├─────────────────────────────────┤
   │ • CLI: main.py                  │                    │ • CLI: main_streamer.py         │
   │ • CI: podcast_clipper.yml       │                    │ • CI: streamer_clipper.yml      │
   │ • Code: core/ + v1/             │                    │ • Code: core/ + streamers/      │
   │ • Tone: Finance / Serious       │                    │ • Tone: Unhinged / Chaotic      │
   │ • Subtitles: Hormozi / Center Pop│                   │ • Subtitles: TikTok Meme Box    │
   │ • Face: YuNet Active Speaker    │                    │ • Face: SFace ArcFace ID Lock   │
   │ • Framing: Split-Screen         │                    │ • Framing: 3-Tier State Machine │
   │ • Buffer: BUFFER_ACCESS_TOKEN   │                    │ • Buffer: STREAMER_BUFFER_*     │
   └─────────────────────────────────┘                    └─────────────────────────────────┘
```

---

## 🧭 Deep Dive: Pipeline V1 (Podcast Viral Clipper)

### Purpose & Niche
Automated clipping of 1-to-3 hour podcasts focused on Personal Finance, Wealth Creation, and High-CPM Niches (Caleb Hammer *Financial Audit*, Dave Ramsey, Graham Stephan, Humphrey Yang, Huberman Lab).

### Technical Specifications
* **Entrypoint**: [`main.py`](main.py) (thin shim -> `v1.main`)
* **GitHub Actions Workflow**: [`.github/workflows/podcast_clipper.yml`](.github/workflows/podcast_clipper.yml)
* **Viral Detector**: [`core/viral_detector.py`](core/viral_detector.py) (Gemini Flash tuned for debt drama, net worth confessions, business secrets).
* **Face Tracking & Layout**:
  - [`core/face_tracker.py`](core/face_tracker.py): OpenCV YuNet on CPU tracks active mouth motion.
  - Generates single 9:16 portrait or dual-speaker split screen (Top 1080x960 / Bottom 1080x960) with zero black bars.
* **Audio Engineering**:
  - 80Hz rumble cut, +2.5dB vocal presence boost at 3kHz.
  - Dynamic sidechain ducking on background music (`sidechaincompress`).
  - EBU R128 (-14.0 LUFS / -1.5 dBTP) broadcast normalization.
* **Subtitles**:
  - [`core/subtitle_generator.py`](core/subtitle_generator.py): Word-by-word spring bounce animation (`hormozi`, `neon_green`, `luxury_gold`).
* **Publishing**:
  - Uses `BUFFER_ACCESS_TOKEN` and `BUFFER_CHANNEL_IDS` to schedule to podcast-specific social channels.

---

## 🧩 Shared Engine — `clips-core` (git submodule)

`core/` is **not** tracked in this repository. It is a **git submodule** pointing
at [`JackPro2121/clips-core`](https://github.com/JackPro2121/clips-core), the
pipeline-agnostic engine both pipelines build on. This repo consumes it as a
submodule; the V2 streamer repo instead carries its **own vendored copy** of the
same upstream, so an engine fix must be landed upstream once and then pulled into
both.

```bash
# after cloning this repo, or after someone bumps the submodule
git submodule update --init --recursive

# pull a newer engine revision
git -C core fetch origin && git -C core checkout <tag-or-sha>
git add core && git commit -m "chore: bump clips-core to <sha>"
```

* GitHub Actions checkouts **must** set `submodules: true` (every workflow here
  already does).
* Large model weights (`SFace`, `S3FD/SyncNet`, `TransNetV2`) are gitignored in
  `clips-core` and fetched by `core/tools/fetch_models.py`. Only the 232 KB YuNet
  detector ships in git here; the V2 streamer identity profiles ship in the
  streamer repo's vendored copy of the engine.
* Paths resolve at runtime and can be overridden with `CLIPS_CORE_ROOT`,
  `CLIPS_CORE_MODELS_DIR`, and `CLIPS_CORE_ASSETS_DIR`.
* **Engine changes must reach both pipelines.** V2 no longer mounts this
  submodule — it keeps its own vendored copy — so an engine fix landed here has to
  be applied upstream in `clips-core` and then synced into
  [`streamer-clips-to-social`](https://github.com/JackPro2121/streamer-clips-to-social)
  (its `SYNC.md` documents the drift check).

---

## 📁 Repository Directory Structure

```
PODCASTS-CLIPS-TO-SOCIAL/                 # V1 Podcast pipeline (this repo)
├── .github/workflows/
│   ├── podcast_clipper.yml          # [V1] Podcast production workflow
│   ├── visual_verification.yml      # [V1] Render/visual regression verification
│   ├── model_smoke.yml              # ML model-weight smoke checks
│   ├── export_transnetv2.yml        # TransNetV2 ONNX export (parity + SHA pinned)
│   ├── syncnet_calibration.yml      # SyncNet lip-sync calibration
│   └── cleanup_old_releases.yml     # Storage hygiene (5-day release purge)
├── core/                            # SHARED ENGINE (git submodule -> clips-core)
│   ├── config.py                    # Shared configs, fonts, safe-zones, audio targets
│   ├── creative_spec.py             # Visual metrics, tolerances, motion policies
│   ├── downloader.py                # Targeted range downloader & Apify proxy
│   ├── transcriber.py               # faster-whisper & YouTube timed text parser
│   ├── jump_cutter.py               # Dead-air trimming and audio compaction
│   ├── face_tracker.py              # Shared face tracker & framing primitives
│   ├── subtitle_generator.py        # Styled ASS subtitle generator (V1 & V2 themes)
│   ├── video_editor.py              # FFmpeg filtergraph engine, studio color grade, loudnorm
│   ├── thumbnail_generator.py       # Universal CTR thumbnail engine (V1 & V2 styles)
│   ├── github_uploader.py           # Release uploader for direct asset URLs
│   ├── viral_detector.py            # Shared Gemini viral-moment helpers
│   ├── ... (endpoint, preflight, scene_classifier, shot_detection, active_speaker,
│   │        caption_emphasis, text_detection)
│   ├── tools/                       # Shared dev/CI tools (fetch, smoke, calibration)
│   ├── verification/                # Pixel/geometry/loudness verification package
│   └── models/
│       ├── face_detection_yunet.onnx         # 232KB tracked face detector
│       └── face_recognition_sface_2021dec.onnx # Auto-downloaded SFace recognizer
├── v1/                              # [V1 EXCLUSIVE] Podcast pipeline
│   ├── main.py                      # V1 orchestrator
│   ├── buffer_client.py             # [V1] Podcast Buffer publisher
│   └── ... (edit_director, director_v2, broll_manager, hook_engine, repair,
│            run_state, editorial_qa, universal_editor, ...)
├── data/
│   └── history.json                 # [V1] Processed episode registry
├── tools/
│   └── trace_caption_placement.py   # [V1] Caption tracer (shared tools live in core/tools/)
├── tests/                           # V1 + shared-engine test suite
├── clips/                           # Rendered V1 output
├── main.py                          # [V1] CLI Entrypoint (shim -> v1.main)
├── requirements.txt                 # Runtime deps — install via core/tools/install_deps.sh
├── requirements-dev.txt             # Dev/test tooling (pytest, ruff, mypy)
├── requirements-ml.txt              # ML tool tier (model smoke / exports / calibration)
├── requirements-ocr.txt             # Optional OCR tier override
├── AGENTS.md                        # Canonical architecture & operational guide
└── README.md                        # Project documentation
```

> The **V2 streamer pipeline is not in this repository.** Its layout
> (`main_streamer.py`, `streamers/`, `data/streamer_history.json`,
> `.github/workflows/streamer_clipper.yml`) lives in
> [`streamer-clips-to-social`](https://github.com/JackPro2121/streamer-clips-to-social).

---

## 🔑 Environment Variables & Secrets Reference (V1)

| Secret Name | Pipeline | Required? | Purpose |
| :--- | :---: | :---: | :--- |
| `GEMINI_API_KEY` | V1 | **Yes** | Google AI Studio key for viral moment extraction. |
| `GEMINI_API_KEY_2`..`_4` | V1 | Optional | Additional keys for LLM ladder rotation. |
| `APIFY_API_TOKEN` | V1 | Optional | Residential proxy fallback for YouTube datacenter bypass. |
| `BUFFER_ACCESS_TOKEN` | **V1 Only** | Optional | Social publishing token for Podcast accounts. |
| `BUFFER_CHANNEL_IDS` | **V1 Only** | Optional | Specific podcast social channels in Buffer. |
| `GROQ_API_KEY`, `OLLAMA_API_KEY`, `OPENROUTER_API_KEY` | V1 | Optional | LLM fallback tiers. |
| `PEXELS_API_KEY` | V1 | Optional | B-roll source (`ENABLE_BROLL`). |
| `YOUTUBE_COOKIES` | V1 | Optional | YouTube resilience when datacenter IPs are blocked (gated by the `USE_YOUTUBE_COOKIES` variable). |
| `COBALT_API_URL` | V1 | Optional | Tier-2 download engine; `COBALT_INSTANCES` overrides with a comma-separated list. |
| `RAPIDAPI_KEY` | V1 | Optional | Tier-3 1080p download fallback (rotatable, comma-separated keys). |
| `CHOCODATA_API_KEY` | V1 | Optional | Tier-3 transcript fallback. |
| `YTDLP_PROXY` | V1 | Optional | Proxy for yt-dlp downloads (also honors `HTTP_PROXY` / `PROXY_URL`). |
| `SLACK_WEBHOOK_URL` | V1 | Optional | Block-Kit run reports and error alerts (`v1/slack_notifier.py`). |
| `CHANNEL_WATERMARK` | V1 | Optional | Default burn-in watermark handle when `--watermark` is omitted. |

`GITHUB_TOKEN` is provided automatically by Actions. Optional repository **variables**
(`ENABLE_BROLL`, `OLLAMA_MODEL`, `USE_YOUTUBE_COOKIES`, `ENABLE_WARP`, `APIFY_*_ACTOR_ID`,
`UNIVERSAL_EDITOR_*`) tune optional features and all carry workflow defaults.

> **V2 secrets** — `STREAMER_BUFFER_ACCESS_TOKEN`, `STREAMER_BUFFER_CHANNEL_IDS`,
> `STREAMER_WATERMARK`, `FIRECRAWL_API_KEY` — belong to the
> [streamer repo](https://github.com/JackPro2121/streamer-clips-to-social). Never
> reuse the V1 `BUFFER_*` names as streamer credentials, or vice versa.

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
Moved out of this repository. Clone and run it from there:

```bash
git clone https://github.com/JackPro2121/streamer-clips-to-social.git
cd streamer-clips-to-social
bash core/tools/install_deps.sh           # same deterministic installer (vendored copy)

python main_streamer.py                                  # auto-discover a fresh stream
python main_streamer.py --streamer kai_cenat --num-clips 3
python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3
```

---

## 🛡️ Critical Commandments for AI Agents

1. **Pipeline isolation**: all V1 work lives in `v1/` (+ shared `core/`); all V2 work lives in the
   [streamer repo](https://github.com/JackPro2121/streamer-clips-to-social) under `streamers/`. Never
   add streamer-specific code to this repository.
2. **Shared engine discipline**: `core/` here is `clips-core` (a submodule); V2 keeps its own vendored
   copy of the same upstream. An engine change therefore affects **both** pipelines — land it in
   `clips-core`, then bump the submodule pointer here *and* sync V2 (see V2's `SYNC.md`).
3. **Buffer isolation**: V1 publishing reads `BUFFER_ACCESS_TOKEN` / `BUFFER_CHANNEL_IDS` only. Streamer
   publishing reads `STREAMER_BUFFER_ACCESS_TOKEN` / `STREAMER_BUFFER_CHANNEL_IDS` only. **Zero crossover.**
4. **Anti-Detection & Safe Zones**:
   - Subtitles must never collide with mobile UI safe-zones (bottom 20%, top 15%, right 12%).
5. **No Full-Video Downloads**: Never download multi-GB source VODs. Fetch transcripts first and perform
   range downloads of only the 30–45s windows you need via `yt-dlp`.
6. **Zero Secrets in Git**: Never log, hardcode, or commit API keys or tokens into git commits or documentation.
