# Future Editing & Subtitle Remediation Plan

## Document Status

- Scope: Core video editing, subtitle generation, framing, audio, B-roll, thumbnails, and render validation.
- Audit type: Read-only audit.
- Code changes made while preparing this document: None.
- Baseline reviewed: `main` branch after the Buffer service-mapping fix.
- Primary goal: Convert the current prototype editing path into a deterministic, safe, testable production pipeline.

## Executive Summary

The project has a strong editing foundation:

- Transcript-driven clip selection
- Targeted segment downloads
- 9:16 output configuration
- ASS subtitle generation
- Face detection and active-speaker framing
- Scene and shot detection
- BGM, SFX, sidechain ducking, and loudness filters
- B-roll and thumbnail overlays
- Atomic rendered-video replacement
- Basic FFmpeg graph tests

The audit found that the highest risks are not cosmetic. The following issues can create incorrect, incomplete, slow, or non-terminating output:

1. B-roll can keep FFmpeg running indefinitely.
2. B-roll timestamps do not seek the B-roll media.
3. Audio-less media can lose BGM/SFX or fail rendering.
4. Automatic split-screen panes can be horizontally stretched.
5. Face/scene shot intelligence is discarded in common cases.
6. Requested duration is not reconciled with actual media duration.
7. Native transcript word timestamps are approximate rather than true word boundaries.
8. Transcript and watermark text are not safely escaped for ASS.
9. Safe-zone rules are inconsistent and not enforced from one source of truth.
10. There is no real end-to-end FFmpeg/libass integration test.

No changes should be made directly from this document until the implementation phases below are reviewed.

---

# 1. Current Editing Architecture

## 1.1 Main render path

`main.py` currently coordinates:

```text
Source selection
  -> transcript fetch
  -> viral moment detection
  -> targeted/full video acquisition
  -> face/scene analysis
  -> ASS subtitle generation
  -> B-roll cue preparation
  -> FFmpeg render
  -> GitHub Release upload
  -> Buffer publishing
```

Important locations:

- Pipeline orchestration: `main.py:33-582`
- Targeted native-caption path: `main.py:126-266`
- Whisper probe path: `main.py:238-374`
- Full-download path: `main.py:358-496`
- Hosting and Buffer publishing: `main.py:503-582`

## 1.2 Video and audio graph

`src/video_editor.py` builds the production FFmpeg graph.

Important locations:

- Path sanitization: `src/video_editor.py:13-20`
- Video filter construction: `src/video_editor.py:24-245`
- Input indexing: `src/video_editor.py:250-328`
- Audio probe and graph: `src/video_editor.py:298-377`
- Atomic FFmpeg execution: `src/video_editor.py:379-425`

The input counter is deterministic and should be retained. The graph must only be changed where the audit identifies correctness issues.

## 1.3 Subtitle path

`src/subtitle_generator.py` generates the ASS file.

Important locations:

- Theme and margin defaults: `src/subtitle_generator.py:11-31`
- ASS timestamp formatting: `src/subtitle_generator.py:40-53`
- ASS header generation: `src/subtitle_generator.py:56-108`
- Word selection and normalization: `src/subtitle_generator.py:110-223`
- Hook badge and watermark: `src/subtitle_generator.py:233-258`
- ASS file writing: `src/subtitle_generator.py:260-267`

The subtitle generator currently supports:

- Word-level caption events
- Highlight color changes
- Theme selection
- Safe-zone margins
- Hook badge
- Watermark
- Shot-specific margins
- Layout-specific alignment

---

# 2. Audit Findings: Subtitle System

## S-01 — Native transcript word timings are fabricated

**Severity:** High
**Status:** Open
**Files:** `src/transcriber.py:50-77`, `src/subtitle_generator.py:130-157`

Native transcript snippets are divided into equal word durations:

```python
word_duration = duration / len(words)
```

The subtitle renderer then treats these as actual word-level boundaries.

### Impact

Captions can drift during pauses, fast speech, long numbers, or uneven sentence pacing. The system presents approximate timings as precise karaoke timing.

### Required plan

- Preserve real word timestamps when available.
- Mark equal-duration estimates explicitly.
- Consider forced alignment for selected clips.
- Fall back to segment-level captions when word boundaries are unreliable.
- Add tests for pauses, uneven rates, overlapping words, and boundary timestamps.

## S-02 — Raw text is not ASS-escaped

**Severity:** High
**Status:** Open
**Files:** `src/subtitle_generator.py:213-217`, `src/subtitle_generator.py:241-258`

Transcript words, hook titles, and custom watermarks are inserted into ASS Dialogue lines without dedicated ASS literal escaping.

### Impact

Braces, backslashes, `\N`, newlines, and control sequences can be interpreted as ASS formatting rather than literal text.

### Required plan

Create separate helpers:

- `normalize_caption_text`
- `escape_ass_text`
- `sanitize_watermark`

Add tests for braces, backslashes, newline injection, control bytes, and custom watermark values.

## S-03 — Safe-zone constants are not the source of truth

**Severity:** High
**Status:** Open
**Files:** `src/config.py:154-157`, `src/subtitle_generator.py:11-16`, `src/subtitle_generator.py:77-100`, `src/subtitle_generator.py:168-178`

Configured zones are defined in `config.py`, but the ASS generator uses hard-coded margins and allows arbitrary per-shot margins.

### Impact

Changing `SAFE_ZONE_*` does not necessarily change generated captions. Per-shot values can violate bottom and right UI reserves.

### Required plan

- Derive ASS margins from one safe-zone calculation.
- Clamp all shot-specific positions.
- Define one canonical right-side reserve.
- Add pixel-level frame checks rather than only string checks.

## S-04 — Hook and watermark conflict with the declared top reserve

**Severity:** High
**Status:** Open
**Files:** `src/config.py:154-157`, `src/subtitle_generator.py:99-100`, `src/subtitle_generator.py:247-258`

The hook and watermark are positioned near the top while `SAFE_ZONE_TOP` reserves 240 pixels.

### Impact

Platform headers or search controls can obscure branding and the hook.

### Required plan

Choose one policy:

- Move hook/watermark below the reserved top area, or
- Change the configured top reserve to the actual design target.

Then enforce the chosen policy in code and tests.

## S-05 — Dynamic split-screen subtitles are not placed on the divider

**Severity:** High
**Status:** Open
**Files:** `src/face_tracker.py:552-559`, `src/subtitle_generator.py:166-178`, `src/subtitle_generator.py:233-239`

A split-screen shot uses `margin_v=0` to represent center placement. The subtitle generator treats zero as missing and substitutes the default margin.

### Impact

Dynamic split-screen captions appear in the lower third instead of at the center divider.

### Required plan

Represent placement explicitly:

```text
placement = lower_third | divider | custom
alignment = bottom_center | center
margin_v = integer
```

Add a test for a split-screen shot inside `multi_shot_dynamic`.

## S-06 — Unicode and currency sanitization is lossy

**Severity:** Medium
**Status:** Open
**Files:** `src/subtitle_generator.py:137-141`, `src/subtitle_generator.py:248-255`

ASCII-oriented filtering can remove or alter:

- `$`
- `%`
- `C++`
- Accented characters
- Non-Latin text
- Internal emoji

### Impact

Financial meaning and international text can be lost.

### Required plan

Use Unicode-aware normalization, preserve legitimate currency and punctuation, and explicitly remove unsupported emoji from all word paths.

## S-07 — Subtitle events can exceed clip boundaries

**Severity:** Medium
**Status:** Open
**Files:** `src/subtitle_generator.py:134-157`

Words are selected with inclusive boundary checks, and the monotonic timing pass can extend the final event.

### Impact

ASS events can continue after the rendered clip ends.

### Required plan

- Use strict overlap checks.
- Clamp every event to the clip duration.
- Add boundary, overlap, and cumulative-drift tests.

## S-08 — Missing ASS files are silently skipped

**Severity:** Medium
**Status:** Open
**Files:** `src/video_editor.py:209-225`

If subtitle burn-in is requested but the file is absent, rendering continues without captions.

### Impact

A clip can be published without dialogue subtitles while appearing successful.

### Required plan

Fail the render when a requested subtitle file is missing. Treat subtitle burn-in as a required output contract.

## S-09 — Real libass integration is not tested

**Severity:** High test gap
**Status:** Open
**Files:** `tests/test_core_loop.py:235-275`, `tests/test_pipeline.py:160-209`

Current graph tests generally disable subtitles.

### Required plan

Add a real FFmpeg/libass integration test using a tiny media fixture:

1. Generate a real ASS file.
2. Render a short clip.
3. Verify subtitle pixels.
4. Verify font loading.
5. Verify missing-file failure.

## S-10 — Empty word lists can produce captionless clips

**Severity:** Medium
**Status:** Open
**Files:** `src/subtitle_generator.py:130-267`, `main.py:194-209`, `main.py:329-343`, `main.py:446-461`

A segment can contain text but no usable word list. ASS generation still succeeds.

### Required plan

Choose one explicit policy:

- Segment-level fallback
- Render error
- Intentional visual-only mode

Add tests for text segments with empty word lists.

## S-11 — `auto` subtitle mode does not detect baked-in captions

**Severity:** Medium
**Status:** Open
**Files:** `main.py:628-632`, `main.py:194`, `main.py:446`

The implementation treats `auto` as burn-in.

### Required plan

Implement detection or update the CLI contract to state that `auto` means default burn behavior.

## S-12 — `keyword_emojis` is a dead subtitle API

**Severity:** Low
**Status:** Contract cleanup required
**Files:** `src/subtitle_generator.py:107-118`, `main.py:207-209`, `src/viral_detector.py:51-64`

The value is accepted and generated but intentionally not burned into captions.

### Required plan

Remove it from subtitle-generation calls or document it as non-rendered metadata. Rename tests and documentation to avoid claiming emoji injection.

## S-13 — Short word events cannot complete the bounce animation

**Severity:** Low/Medium
**Status:** Open
**Files:** `src/subtitle_generator.py:152-155`, `src/subtitle_generator.py:211-215`

Word events can be 120 ms while the animation is designed for 140 ms.

### Required plan

Use a shorter animation for short events or disable bounce below the minimum animation duration.

## S-14 — Font fallback and Unicode coverage are not enforced

**Severity:** Medium
**Status:** Open
**Files:** `src/config.py:168-224`, `src/subtitle_generator.py:64-65`, `src/subtitle_generator.py:98-100`, `src/video_editor.py:209-215`

Fallback fonts are declared but not actively resolved. `FONTS_DIR.exists()` does not prove fonts are present.

### Required plan

Validate required font files, define a real fallback order, and test rendered glyphs.

## S-15 — Cover overlay can hide the opening hook

**Severity:** Low
**Status:** Open quality issue
**Files:** `src/video_editor.py:209-235`

The cover image is overlaid after subtitles and remains active for the first 0.25 seconds.

### Required plan

Decide whether the cover should sit above or below the hook and add a first-frame visual test.

---

# 3. Audit Findings: Core Video Editing

## E-01 — B-roll can prevent render termination

**Severity:** Critical
**Status:** Open
**Files:** `src/video_editor.py:282-287`, `src/video_editor.py:204-206`, `src/video_editor.py:405-411`

Infinite B-roll inputs have no reliable finite output constraint.

### Required plan

Use `shortest=1`, explicit output duration, or finite B-roll preprocessing. Add a real termination test.

## E-02 — B-roll cue time does not seek B-roll media

**Severity:** High
**Status:** Open
**Files:** `src/video_editor.py:204-206`, `src/broll_manager.py:147-180`

Cue time only controls overlay visibility.

### Required plan

Add explicit source offset semantics or `setpts`/trim offset. Test with timestamped markers.

## E-03 — Audio-less media is not handled safely

**Severity:** High
**Status:** Open
**Files:** `src/video_editor.py:298-305`, `src/video_editor.py:330-377`

No-audio sources can lose BGM/SFX or cause `[0:a]` failures.

### Required plan

Distinguish no-audio from probe failure and generate a silent voice bed when needed.

## E-04 — Automatic split-screen panes are stretched

**Severity:** High
**Status:** Open
**Files:** `src/face_tracker.py:527-542`, `src/video_editor.py:83-92`, `src/video_editor.py:161-171`

The source panes are approximately 9:16 but destination panels are 9:8.

### Required plan

Use aspect-preserving scale and pad/crop. Add rendered geometry tests.

## E-05 — Meaningful shot plans are collapsed

**Severity:** High
**Status:** Open
**Files:** `src/face_tracker.py:640-688`

Only mode and `crop_x` influence whether dynamic shots are retained.

### Required plan

Preserve complete shot descriptors and propagate single-shot zoom/timeline data.

## E-06 — Actual media duration is not authoritative

**Severity:** High
**Status:** Open
**Files:** `main.py:181-190`, `main.py:318-326`, `main.py:362-373`, `src/video_editor.py:267-271`, `src/video_editor.py:395-419`

The requested transcript window is used even when the downloaded segment is shorter.

### Required plan

Probe source duration, clamp all timeline elements, and post-render validate actual output duration.

## E-07 — Manual/fallback framing can generate invalid crops

**Severity:** High
**Status:** Open
**Files:** `src/face_tracker.py:42-56`, `main.py:34-63`, `src/video_editor.py:58-61`

`active_w/active_h` can retain 1920×1080 defaults for smaller sources.

### Required plan

Initialize active dimensions from actual media and validate every crop.

## E-08 — Frame timing assumes constant FPS

**Severity:** High
**Status:** Open
**Files:** `src/face_tracker.py:352-358`, `src/face_tracker.py:376-410`, `src/face_tracker.py:446-456`

VFR sources and fractional cut boundaries can drift.

### Required plan

Use actual timestamps, exclusive end frames, and frame-native shot boundaries.

## E-09 — Output validation only checks nonzero file size

**Severity:** Medium
**Status:** Open
**Files:** `src/video_editor.py:379-425`

A nonempty but invalid or semantically wrong file can be accepted.

### Required plan

Validate ffprobe readability, video/audio streams, dimensions, codec, frame rate, and duration before atomic replacement.

## E-10 — Audio graph is not explicitly duration-bounded

**Severity:** High
**Status:** Open
**Files:** `src/video_editor.py:334-371`, `src/video_editor.py:395-419`

Audio uses `amix=duration=first`, while video has separate trimming behavior.

### Required plan

Normalize sample rate/layout, reset timestamps, trim/pad streams, and enforce one target duration.

## E-11 — One-pass loudness is not verified

**Severity:** Medium
**Status:** Open
**Files:** `src/video_editor.py:246-250`, `src/video_editor.py:334-371`

`loudnorm` runs once without measurement or verification.

### Required plan

Use measurement plus application, then verify integrated LUFS and true peak with `ebur128` or equivalent.

## E-12 — Zoom is top-left anchored

**Severity:** Medium
**Status:** Open
**Files:** `src/video_editor.py:186-195`

`zoompan` has no explicit center expressions.

### Required plan

Add centered `x` and `y` expressions and verify the subject remains centered.

## E-13 — Film grain is duplicated

**Severity:** Low/Medium
**Status:** Open
**Files:** `src/video_editor.py:40-56`

The same grain filter is appended twice.

### Required plan

Keep one grain stage and test exact graph contents.

## E-14 — Peak zoom is disabled in CI

**Severity:** Medium
**Status:** Open intentional degradation
**Files:** `src/video_editor.py:186`, `src/config.py:136-144`

Local and production output differ.

### Required plan

Document the reduced CI mode or implement a lighter CI version.

## E-15 — B-roll cache files are not atomic or validated

**Severity:** High
**Status:** Open
**Files:** `src/broll_manager.py:103-139`

Only file size is checked.

### Required plan

Use stable SHA-256 names, temporary files, media validation, byte limits, time limits, and atomic replacement.

## E-16 — B-roll cache names are process-randomized

**Severity:** Medium
**Status:** Open
**Files:** `src/broll_manager.py:112-117`

Python string `hash()` changes between processes.

### Required plan

Use stable URL or provider IDs.

## E-17 — B-roll is not prepared in all ingestion paths

**Severity:** Medium
**Status:** Open
**Files:** `main.py:230-243`, `main.py:304-374`, `main.py:435-492`

Whisper and full-download branches do not use the same B-roll preparation.

### Required plan

Factor B-roll cue preparation into a shared per-clip path.

## E-18 — B-roll preparation is outside render isolation

**Severity:** Medium
**Status:** Open
**Files:** `main.py:230-245`

B-roll errors can abort the entire targeted batch before render exception handling.

### Required plan

Make B-roll optional and independently bounded.

## E-19 — Thumbnail output is not atomic

**Severity:** Medium
**Status:** Open
**Files:** `src/thumbnail_generator.py:186-189`

Final JPEG is written directly.

### Required plan

Write to a temporary file, validate, and atomically replace.

## E-20 — Thumbnail parent directories are not always created

**Severity:** Medium
**Status:** Open
**Files:** `src/thumbnail_generator.py:68-85`

Custom output directories can fail.

### Required plan

Create parent directories before saving.

## E-21 — Thumbnail fallback is not the actual middle frame

**Severity:** Low/Medium
**Status:** Open
**Files:** `src/thumbnail_generator.py:85-89`

The fallback is hard-coded to 0.5 seconds.

### Required plan

Use probed duration and select `duration / 2` with EOF safety.

## E-22 — Scene detector fallback resources are not fully closed

**Severity:** Low/Medium
**Status:** Open
**Files:** `src/scene_classifier.py:152-187`, `src/scene_classifier.py:204-256`

Some video captures are not released on every path.

### Required plan

Use `try/finally` and validate complete, ordered shot coverage.

## E-23 — YuNet model download is not atomic or retryable

**Severity:** Medium
**Status:** Open
**Files:** `src/face_tracker.py:112-161`, `src/scene_classifier.py:27-59`

A partial model can poison the cache.

### Required plan

Download to temporary storage, validate, atomically replace, and retry initialization failures.

## E-24 — Face identity tracking is not association-based

**Severity:** Medium/High
**Status:** Open
**Files:** `src/face_tracker.py:396-405`, `src/face_tracker.py:479-526`

Faces are sorted by x-coordinate and stored in fixed list positions.

### Required plan

Associate detections by IoU, center distance, size, and mouth-patch continuity.

## E-25 — Face boxes are not fully clamped

**Severity:** Medium
**Status:** Open
**Files:** `src/face_tracker.py:195-215`, `src/face_tracker.py:219-237`, `src/face_tracker.py:242-253`

Right and bottom box boundaries can exceed the frame.

### Required plan

Clamp and validate boxes before crop construction.

## E-26 — Letterbox detection can remove real dark content

**Severity:** Medium
**Status:** Open
**Files:** `src/face_tracker.py:58-110`

Black rows are assumed to be bars without distinguishing fades and dark scenes. Horizontal bars are not handled.

### Required plan

Use multi-sample bar detection, continuity checks, and aspect validation.

## E-27 — Repeated detector work increases runtime

**Severity:** Medium
**Status:** Open
**Files:** `src/face_tracker.py:115-161`, `src/face_tracker.py:388-397`, `src/scene_classifier.py:99-143`

The same frame can be face-detected multiple times and fallback detectors can be repeatedly allocated.

### Required plan

Cache detector instances, reuse face results, and avoid decoding every frame when sampling is sufficient.

## E-28 — YOLO path is inactive

**Severity:** Medium feature gap
**Status:** Open
**Files:** `src/scene_classifier.py:7-59`, `src/scene_classifier.py:145-150`

YOLO is defined but not used by the active classifier.

### Required plan

Either implement and test it or remove dead model-loading code and clearly document heuristic-only behavior.

---

# 4. Current Test Gaps

The current tests are useful for string-level and synthetic graph behavior, but they do not prove full production rendering.

## Missing high-value tests

1. Real end-to-end FFmpeg render.
2. Real ASS/libass burn-in.
3. Audio-less source.
4. B-roll finite termination.
5. B-roll timestamp offset.
6. Split-screen aspect preservation.
7. Dynamic split-screen subtitle placement.
8. Single-shot zoom/timeline preservation.
9. Source/output duration reconciliation.
10. VFR and fractional timestamps.
11. FFmpeg timeout cleanup.
12. Atomic output failure behavior.
13. Long/Unicode subtitle rendering.
14. B-roll cache corruption.
15. Stable B-roll cache naming.
16. Detector lifecycle/performance.
17. Scene-cut integration.

## Existing protections to preserve

- Explicit FFmpeg input counter: `src/video_editor.py:250-328`
- Per-clip render exception isolation: `main.py:213-234`, `main.py:304-377`, `main.py:436-496`
- Atomic MP4 replacement: `src/video_editor.py:379-425`
- FFmpeg timeout: `src/video_editor.py:405-411`
- Basic filtergraph execution: `tests/test_core_loop.py:235-275`
- Segment offset propagation: `main.py:181-190`

---

# 5. Implementation Plan

## Phase 0 — Freeze the editing contract

**Purpose:** Prevent future fixes from changing behavior accidentally.

### Files

- `src/video_editor.py`
- `src/subtitle_generator.py`
- `main.py`
- `tests/test_pipeline.py`
- `tests/test_core_loop.py`

### Changes

Define explicit contracts for:

- Output dimensions
- Target FPS
- Target duration
- Audio stream policy
- Subtitle policy
- Safe-zone policy
- B-roll timing semantics
- B-roll failure behavior
- Source/output duration tolerance

### Impact

No immediate output change. All later changes become testable against one documented contract.

---

## Phase 1 — Fix P0 render safety

### 1.1 Make B-roll finite

**Files:**

- `src/video_editor.py`
- `tests/test_broll_and_thumbnail.py`
- `tests/test_core_loop.py`

**Changes:**

- Add finite B-roll duration handling.
- Add explicit output duration.
- Add `shortest=1` or equivalent.
- Add a real termination test.

**Impact:**

- Prevents 900-second hangs.
- Reduces CI resource consumption.
- May make long B-roll cues shorter or clip-safe.

### 1.2 Fix B-roll time semantics

**Files:**

- `src/video_editor.py`
- `src/broll_manager.py`
- `tests/test_broll_and_thumbnail.py`

**Changes:**

- Define whether cue time means source offset or overlay time.
- Add `setpts`, `trim`, or explicit source start.
- Add timestamp-marker test.

**Impact:**

- B-roll begins at the intended moment.
- Visual meaning matches transcript timing.

### 1.3 Handle no-audio media

**Files:**

- `src/video_editor.py`
- `tests/test_pipeline.py`
- `tests/test_core_loop.py`

**Changes:**

- Add a silent voice-bed input for no-audio sources.
- Distinguish no-audio from probe failure.
- Mix BGM/SFX against the chosen bed.

**Impact:**

- Silent videos still receive the configured sound design.
- Probe failures become explicit and safe.

---

## Phase 2 — Fix framing geometry and shot preservation

### 2.1 Aspect-preserving split-screen

**Files:**

- `src/face_tracker.py`
- `src/video_editor.py`
- `tests/test_framing_and_aspect.py`

**Changes:**

- Use scale with aspect preservation.
- Add pad/crop geometry.
- Validate actual rendered pane dimensions.

**Impact:**

- Faces are no longer stretched.
- Split-screen output becomes stable across source resolutions.

### 2.2 Preserve complete shot plans

**Files:**

- `src/face_tracker.py`
- `src/video_editor.py`
- `tests/test_framing_and_aspect.py`

**Changes:**

- Compare complete shot descriptors.
- Preserve single-shot zoom/timeline data.
- Preserve split boxes and vertical crop changes.

**Impact:**

- Dynamic framing becomes truly dynamic.
- Punch zoom and panning survive more often.

### 2.3 Actual source dimensions and crop validation

**Files:**

- `src/face_tracker.py`
- `main.py`
- `src/video_editor.py`
- `tests/test_framing_and_aspect.py`

**Changes:**

- Initialize `active_x/y/w/h` from probed dimensions.
- Clamp all boxes.
- Normalize dimensions to encoder-safe values.
- Add odd-size, 720p, portrait, and low-resolution tests.

**Impact:**

- Prevents invalid FFmpeg crop graphs.
- Makes lower-resolution inputs reliable.

### 2.4 Frame-native timing

**Files:**

- `src/face_tracker.py`
- `src/video_editor.py`
- `tests/test_framing_and_aspect.py`

**Changes:**

- Use exclusive end-frame logic.
- Use actual PTS where available.
- Reduce decimal boundary rounding.

**Impact:**

- Reduces VFR drift.
- Improves cut and subtitle alignment.

---

## Phase 3 — Fix duration and audio correctness

### 3.1 Actual media duration authority

**Files:**

- `main.py`
- `src/downloader.py`
- `src/video_editor.py`
- `tests/test_pipeline.py`

**Changes:**

- Probe source duration.
- Clamp requested windows.
- Use actual source duration as render authority.
- Reject impossible windows.

**Impact:**

- No subtitle/SFX/B-roll activity after EOF.
- Output duration matches the actual source.

### 3.2 Unified A/V duration contract

**Files:**

- `src/video_editor.py`
- `tests/test_pipeline.py`
- `tests/test_core_loop.py`

**Changes:**

- Reset timestamps after seek.
- Normalize audio to 48 kHz stereo.
- Trim/pad streams to target duration.
- Validate output with ffprobe.

**Impact:**

- Prevents A/V drift.
- Makes all output variants predictable.

### 3.3 Loudness verification

**Files:**

- `src/video_editor.py`
- `tests/test_pipeline.py`

**Changes:**

- Add measurement pass.
- Apply measured normalization.
- Verify LUFS and true peak.

**Impact:**

- Actual output meets the documented audio target.

---

## Phase 4 — Fix subtitle correctness and safety

### 4.1 ASS escaping and Unicode normalization

**Files:**

- `src/subtitle_generator.py`
- `src/transcriber.py`
- `tests/test_pipeline.py`
- `tests/test_core_loop.py`

**Changes:**

- Add plain-text normalization.
- Add ASS literal escaping.
- Preserve currency and Unicode where supported.
- Remove internal emoji consistently.

**Impact:**

- External transcript content cannot break ASS formatting.
- Financial symbols and international text remain readable.

### 4.2 Safe-zone single source of truth

**Files:**

- `src/config.py`
- `src/subtitle_generator.py`
- `src/face_tracker.py`
- `tests/test_core_loop.py`
- `tests/test_framing_and_aspect.py`

**Changes:**

- Define one safe rectangle.
- Clamp all default and shot-specific placements.
- Resolve hook/watermark policy.

**Impact:**

- Captions and branding obey one documented policy.

### 4.3 Dynamic split-screen placement

**Files:**

- `src/subtitle_generator.py`
- `src/face_tracker.py`
- `tests/test_pipeline.py`

**Changes:**

- Represent placement explicitly.
- Support center-divider alignment.
- Distinguish zero from missing values.

**Impact:**

- Split-screen captions appear at the intended divider position.

### 4.4 Word timing and boundary behavior

**Files:**

- `src/transcriber.py`
- `src/subtitle_generator.py`
- `tests/test_pipeline.py`

**Changes:**

- Mark estimated word timings.
- Clamp events to clip duration.
- Use strict overlap.
- Add segment fallback policy.

**Impact:**

- Fewer timing drift and EOF subtitle errors.
- More honest caption quality.

### 4.5 Real libass integration

**Files:**

- `tests/test_pipeline.py`
- `tests/fixtures/` or generated test media
- `.github/workflows/podcast_clipper.yml`

**Changes:**

- Render real ASS through FFmpeg.
- Verify subtitle pixels and font behavior.
- Verify missing-file failure.

**Impact:**

- CI proves subtitle burn-in works in the actual renderer.

---

## Phase 5 — Fix B-roll, thumbnails, and model assets

### 5.1 B-roll cache integrity

**Files:**

- `src/broll_manager.py`
- `tests/test_broll_and_thumbnail.py`

**Changes:**

- Stable SHA-256 cache names.
- Temporary download files.
- ffprobe validation.
- Byte/time limits.
- Response cleanup.

**Impact:**

- Corrupt B-roll cannot poison future renders.
- Cache behaves consistently across processes.

### 5.2 Thumbnail integrity

**Files:**

- `src/thumbnail_generator.py`
- `tests/test_broll_and_thumbnail.py`

**Changes:**

- Atomic thumbnail writes.
- Parent directory creation.
- Correct middle-frame fallback.
- Watermark default resolution.
- Long-title bounds.

**Impact:**

- Thumbnails are reliable and visually bounded.

### 5.3 Model and detector lifecycle

**Files:**

- `src/face_tracker.py`
- `src/scene_classifier.py`
- `tests/test_framing_and_aspect.py`

**Changes:**

- Atomic model download.
- Retry partial initialization.
- Cache detector instances.
- Release all captures.
- Add structured backend metadata.

**Impact:**

- Lower CPU/RAM usage.
- More consistent face/scene decisions.

---

## Phase 6 — Rendering validation and observability

### 6.1 Post-render validation

**Files:**

- `src/video_editor.py`
- `tests/test_pipeline.py`

**Changes:**

- Validate codec, dimensions, frame rate, duration, and streams.
- Validate audio presence policy.
- Preserve previous final output on failure.

**Impact:**

- Invalid or partial MP4s cannot be published.

### 6.2 Structured render metadata

**Files:**

- `main.py`
- `src/video_editor.py`
- `src/face_tracker.py`
- `src/broll_manager.py`

**Changes:**

Record:

```text
face_backend
scene_backend
detected_shots
broll_count
broll_failure_reason
source_duration
requested_duration
output_duration
```

**Impact:**

- Production failures become diagnosable without changing user-facing behavior.

---

# 6. File-Level Change Matrix

| File | Planned responsibility | Main impact |
|---|---|---|
| `main.py` | Shared duration contract, B-roll path, render validation, metadata | Consistent behavior across ingestion paths |
| `src/video_editor.py` | Finite B-roll, audio bed, aspect-preserving graph, duration, ffprobe | Correct playable renders |
| `src/subtitle_generator.py` | ASS escaping, safe zones, Unicode, placement, boundaries | Reliable and readable subtitles |
| `src/transcriber.py` | Word-timing reliability markers and fallback behavior | Less caption drift |
| `src/face_tracker.py` | Active dimensions, shot preservation, face identity, crop safety | Correct dynamic framing |
| `src/scene_classifier.py` | Resource cleanup, detector reuse, YOLO decision | Faster and more reliable scene decisions |
| `src/broll_manager.py` | Cache integrity, stable IDs, B-roll limits, phrase matching | Reliable optional B-roll |
| `src/thumbnail_generator.py` | Atomic output, middle-frame fallback, bounds, watermark | Reliable thumbnails |
| `src/config.py` | Safe-zone and rendering policy constants | Single source of truth |
| `tests/test_pipeline.py` | ASS, render, duration, audio, atomic output tests | Catches regressions |
| `tests/test_core_loop.py` | FFmpeg graph, B-roll, framing, duration tests | Catches graph defects |
| `tests/test_broll_and_thumbnail.py` | B-roll and thumbnail integrity tests | Protects optional enhancements |
| `.github/workflows/podcast_clipper.yml` | Run new test matrix and static checks | Prevents untested changes |

---

# 7. Test Matrix

The eventual implementation should test this matrix:

```text
layout:
  single_smooth
  blur_stack
  split_screen
  multi_shot_dynamic

subtitles:
  none
  real ASS
  missing ASS

B-roll:
  none
  one cue
  multiple cues
  short media
  invalid media

audio:
  source audio
  no source audio
  probe failure
  BGM
  SFX
  sidechain ducking

media:
  16:9
  4:3
  portrait
  letterbox
  odd dimensions
  VFR
  fractional source offset

output:
  success
  FFmpeg non-zero
  timeout
  invalid output
  atomic replacement failure
```

---

# 8. Rollout Strategy

## Stage 1 — Safe, non-visual fixes

- ASS escaping
- Safe-zone calculations
- B-roll cache validation
- Thumbnail atomic output
- Duration validation
- Timeouts
- Test coverage

## Stage 2 — Geometry and timing

- Aspect-preserving split-screen
- Shot-plan preservation
- Frame-native boundaries
- Source duration reconciliation
- A/V duration contract

## Stage 3 — Audio and visual quality

- Loudness measurement
- Zoom centering
- Duplicate grain removal
- Font fallback
- B-roll timing
- Thumbnail bounds

## Stage 4 — Production hardening

- Post-render ffprobe validation
- Structured metadata
- Model download locking
- Detector caching
- B-roll byte/time budgets
- CI runtime monitoring

---

# 9. Expected Impact After Completion

## Editing quality

- No stretched split-screen panes
- Correct B-roll timing
- No infinite B-roll renders
- Correct A/V duration
- No captions after EOF
- Center-divider captions in dynamic split-screen
- Better punch zoom and face panning

## Subtitle quality

- Safer ASS output
- Better Unicode and currency handling
- Consistent safe zones
- More honest word timing
- Real FFmpeg burn-in verification
- Clear fallback when word data is unavailable

## Reliability

- B-roll and thumbnails cannot be corrupted silently
- Invalid renders fail before publishing
- Detector and model initialization is repeatable
- Failures are observable in logs/metadata

## Performance

- Reduced redundant face detection
- Finite B-roll processing
- Bounded optional downloads
- More predictable GitHub Actions runtime

## Test confidence

- Real media rendering coverage
- Audio/no-audio coverage
- Subtitle/libass coverage
- B-roll and thumbnail integrity coverage
- Duration and timestamp regression coverage

---

# 10. Completion Criteria

The future editing work is complete only when:

- [ ] B-roll cannot cause unbounded render time.
- [ ] B-roll cue timing is visually correct.
- [ ] No-audio media renders intentionally.
- [ ] Split-screen output preserves aspect ratio.
- [ ] Shot plans retain zoom, panning, and speaker boxes.
- [ ] Actual source duration controls output duration.
- [ ] Video/audio durations match within an agreed tolerance.
- [ ] ASS text is escaped and Unicode-safe.
- [ ] Safe zones have one enforceable source of truth.
- [ ] Dynamic split-screen captions use divider placement.
- [ ] Real libass burn-in tests pass.
- [ ] Thumbnail and B-roll files are atomic and validated.
- [ ] FFmpeg outputs pass ffprobe validation.
- [ ] Timeout and failure paths leave no partial final artifacts.
- [ ] CI runs unit, static, media, subtitle, and integration checks.

---

## Final Audit Conclusion

The current editing pipeline is promising but not yet safe to call broadcast-grade. The recommended order is:

1. B-roll termination and timing
2. No-audio handling
3. Split-screen geometry
4. Duration and timestamp validation
5. ASS escaping and safe zones
6. Real subtitle/render integration tests
7. Detector, cache, thumbnail, and observability hardening

No code changes should be made from this document without implementing and testing each phase separately.
