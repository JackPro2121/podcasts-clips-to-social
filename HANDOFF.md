# HANDOFF — Autonomous AI Podcast Clipper

**Written:** 2026-09-26 · **For:** any agent/human resuming this work
**Repo:** `JackPro2121/podcasts-clips-to-social` (public) · **Branch:** `main` · **HEAD:** `ff01a1b`

> **READ THIS FIRST.** This file is the only memory that survives a session
> boundary. If you are resuming work here, read this file before touching code.

---

## 0. Read this before you start

This project is **not** a greenfield build. It is a working, production Python
pipeline. The working principle for every change here is:

> Understand before modifying. Reproduce before fixing. Prove the root cause.
> Make the smallest safe change. Add a regression test. Verify nothing else broke.

**Hard constraints (from the repo's own docs and the user):**

| Constraint | Detail |
|---|---|
| **$0 budget** | Free tiers only. Never add a paid dependency. |
| **CPU-only, GitHub Actions** | No GPU. `ubuntu-latest` = **4 vCPU, 16 GB RAM, 14 GB SSD**. Job cap **6 h**. Standard runners are free & unlimited on **public** repos. |
| **14 GB disk is the binding limit** | Not CPU. PyTorch costs ~3–5 GB installed — a third of remaining disk for one model. |
| **Never install heavy deps locally** | The user installs on CI only. Anything I cannot verify locally must be flagged **UNVERIFIED** in the report. |
| **Docs live in a worktree** | All `*.md` are under `.kilo/worktrees/decorous-pearl/`, **not** the repo root. Read them there. |

---

## 1. Architecture — the most important thing to understand

The pipeline had a **two-brain split**, which was the root cause of a long
string of CI failures. Both brains still exist; one is now partially wired to
the other.

```
                       ┌── BRAIN A (legacy) ──► renders the actual pixels ─────┐
transcript ─► Gemini ─►│  face_tracker.analyze_faces_in_clip()               │
   detect_viral_moments│        ↓  FramingDecision                            │
                       │  video_editor.build_video_filtergraph(framing)       │
                       │        ↓  FFmpeg                                    │
                       └─────────────────────────────────────────────────────┘
                       ┌── BRAIN B ("universal editor", originally SHADOW) ──┐
                       │  universal_editor.build_clip_editor_artifacts()    │
                       │        ↓                                            │
                       │  SourceIndex → EditPlan → CompositionPlan           │
                       │        ↓  (+ caption_placements, added now)         │
                       │  editorial_qa.run_editorial_qa(rendered, plan)     │
                       │        ↓                                            │
                       │  can VETO the render (ENFORCE_QA=true)              │
                       └─────────────────────────────────────────────────────┘
```

**Key facts (verified in code):**
- `main.py` renders from `framing` only; `render_viral_clip()` had no plan param.
- Brain B produced a full plan that was serialised to JSON and **never read**.
- QA judged Brain A's pixels using Brain B's plan, and could only **veto** —
  never tell the renderer what to change. That is why failures read
  `fix → fail → fix → fail`.
- **Both brains independently ran face detection on the same frames.**
- `editorial_qa.py:25-26` — only `error`/`fatal` severities block. A log line
  printing all codes under "blocked" was actively misleading (it hid that
  `incomplete_endpoint` was only a warning).

**Now partially closed:** `CompositionPlan.caption_anchor` (derived from
text-collision analysis) reaches the renderer via `CaptionPlacement`.
Still open: the renderer does not yet consume the *layout* half of the plan.

---

## 2. The original CI failure and its root cause (fully solved)

**Symptom** — GitHub Actions run `36171365900` (on `ff01a1b`) failed:

```
[+] Clip rendered successfully: .../clip_1_MILLIONAIRES_WEALTH_SHARING_SE.mp4 (Size: 27.12 MB)
[-] Editorial QA blocked clip #1: ['freeze_interval', 'incomplete_endpoint']
RuntimeError: No clips were rendered. Refusing to report a successful run without publishable output.
```

`RuntimeError` was **not** the bug — it is the guardrail at `main.py` firing
because zero clips survived. The real blocker was `freeze_interval`
(severity `error`). `incomplete_endpoint` is only a **warning** and never blocked.

**Root cause:** `build_video_filtergraph` emitted a **fixed integer crop box**
for `portrait_face` shots it could not resolve into motion:

```
[0:v]trim=...,crop=526:938:494:39,scale=1080:1920,...   ← all constants, no `t`
```

On statically-held source content this renders a **literally frozen** output.

**Proof chain (all reproduced locally with real ffmpeg):**
1. Downloaded the actual 27 MB artifact from the failed run.
2. `freezedetect=n=0.003:d=0.5` → `freeze_start 0.266667, duration 0.867667`
   (the freeze window sits inside shot 0, `0.00-2.50s portrait_face`).
3. Frame-level diff over that window: mean **0.09–0.52 / 255** — visually frozen.
4. A/B on the real artifact: static crop → FAIL (0.867 s); drift → **zero freezes**.

**Six of eight recent failures were this one bug**; two were unrelated
(true-peak loudness, non-English audio). Earlier commits `dc3cee0` / `6057a71`
had added `sin(t*1.8)` drift to the `presentation_slide` branch only — which is
why they did not help.

### The fix

`video_editor.py` now guarantees motion on every static branch via a
**triangle wave** (`asin(sin(t*1.8))`, not a sine — see below):

```python
_STATIC_SHOT_DRIFT_FREQ = 1.8
_STATIC_SHOT_DRIFT_RATIO = 0.05
_STATIC_SHOT_DRIFT_WAVE = "0.6366*asin(sin(t*{freq}))"
```

Helper contract (`_drift_axis_expr(base, low, high, extent, ratio)`):
1. **The clamp must never bind.** The oscillation centre is placed so the full
   amplitude fits between the bounds. A binding clamp flattens part of the wave
   into a static hold — reintroducing the freeze.
2. Sweep capped at `ratio * extent` of the crop dimension.
3. A subject against a frame edge still gets motion (centre nudged inward by at
   most the cap) instead of collapsing to zero amplitude.

**Why a triangle and not a sine:** a sine's velocity reaches zero at every
extremum, which left sub-threshold runs up to **0.533 s**. A triangle holds
near-constant speed, dropping the longest run to **~0.07 s** — below
freezedetect's own `d=0.5` report floor.

**Verified:** 10/10 crop base positions (both zoom paths, including frame edges)
render with zero freeze events and no unterminated freeze.

**Regression test proves itself:** `tests/test_shot_motion_guarantee.py` — on the
pre-fix code **4 tests fail** (including a render-level one showing
`freeze_start: 0`); on the fixed code all pass.

---

## 3. Work completed (all uncommitted)

`git status` — modified: `main.py`, `src/composition_planner.py`, `src/config.py`,
`src/editorial_qa.py`, `src/edit_director.py`, `src/face_tracker.py`,
`src/scene_classifier.py`, `src/source_index.py`, `src/subtitle_generator.py`,
`src/universal_editor.py`, `src/video_editor.py`, `src/viral_detector.py`
New: `src/director_v2.py`, `src/engagement.py`, `src/repair.py`,
`src/shot_detection.py`, `src/text_detection.py`, `requirements-ocr.txt` + 12 new test files.
Nothing committed.

Current verification: **368 tests passed**, ruff clean, mypy clean (29 files).
Baseline before any of this work was 134 tests.

### Phase 0 — freeze root cause ✅
- Motion guarantee on all static crop branches (`video_editor.py`).
- Regression test proven to fail on old code.
- Removed a **duplicated** `ENABLE_FILM_GRAIN` block that appended
  `noise=alls=1.2:allf=t` twice (`video_editor.py`, was lines 208-209 + 218-219).

### Phase 1a — VAD ✅ (nothing to build)
`vad_filter=True` with `min_silence_duration_ms=500` was **already** present in
both Whisper call sites (`transcriber.py:200`, `:88`) — Silero, already a
transitive dep via `faster_whisper`. Verified, not added.

### Phase 1d — engagement / energy signal ✅
New `src/engagement.py`. Speech rate, speech ratio, longest pause, dead air,
deliberate-pause bonus, emphasis + question rate, filler ratio → 0..1 score.
Plus `engagement_curve`, `rank_candidates` (duration gate is the **primary** sort
key), `best_energy_peak`, `summarise`. Stdlib only — no new dep.
Wired into `edit_director._score_plan` at 0.20 weight; the old terms were
rebalanced, none dropped.

### Phase 3a — Gemini model ladder ✅
`config.GEMINI_MODEL_LADDER`, env-overridable via `GEMINI_MODEL_LADDER`.
Default ladder is 2026 models first:
`gemini-3.8-flash, 3.7, 3.6, 3.5-flash-lite, 3.1-flash-lite, 2.5-flash, 2.5-flash-lite, 2.0-flash`.
Pre-existing 503/429 rotation preserved and now unit-tested.

### Phase 3c — Brain B → Brain A bridge ✅ (partial)
- `composition_planner.build_caption_placements()` converts normalised plan rects
  into pixel ASS positions.
- **Found a real ASS limitation:** alignment lives in the *style*, not the
  Dialogue line, so per-shot alignment was impossible. Added `DefaultMid` (5) and
  `DefaultTop` (8) styles + `style_name_for_alignment()`. `Default` (2) output is
  byte-identical to before when no placements are passed.
- `caption_placements` threaded through all 3 `create_styled_ass_subtitles`
  call sites in `main.py` and carried on `ClipEditorArtifacts`.

### Phase 4 — bounded repair loop ✅
New `src/repair.py`. A QA veto becomes an actionable render instruction.
- `REPAIRABLE_CODES`: `static_hold_limit`/`black_interval` → motion gain
  (×1.8/attempt, capped 3.6); `caption_source_collision` → relocate captions.
- `UNREPAIRABLE_CODES`: `output_probe`, `*_stream_missing`, `video_codec`,
  `audio_stream`, `composition_missing`, `output_duration`, **`freeze_interval`**.
- `MAX_REPAIR_ATTEMPTS = 2` → max 3 renders total. Verified bounded: gains
  `[1.0, 1.8, 3.24]`, exactly 3 renders, clean failure.
- Render exceptions deliberately **not** caught (ffmpeg failures are
  deterministic; retrying would hide a bug).
- Wired at all 3 render sites via `_render_clip_with_repair`.
- `motion_gain` threaded through `build_video_filtergraph` / `render_viral_clip`.

> **Why `freeze_interval` was removed from the repairable set:** the motion
> guarantee already prevents frozen output, verified against the real CI
> artifact and every crop position. Escalating gain on a freeze verdict would
> only mask a regression in that guarantee. A freeze now **fails the run loudly**.
> The motion guarantee itself was deliberately **kept** — it is the fix, not a layer.

### Phase 3d — PiP layout ✅
- New shot mode `pip_slide` + renderer branch in `video_editor.py`.
  Slide canvas (over a blurred fill so any aspect survives) + host inset
  bottom-right with a white border, inside platform + caption safe zones.
- `PIP_SLIDE_CANVAS_RATIO = 0.94` — **required**. A crop spanning the whole
  active region has zero slack, so no drift can be emitted and the slide
  renders frozen (this was caught by rendering, not by reading).
- **Both** canvas and inset receive the drift.
- Detection: `face_tracker._picture_in_picture_host_box()` — median box across
  frames holding exactly one face, ≥50 % persistence (`PIP_MIN_PERSISTENCE`),
  area ≤12 % of frame (`PIP_MAX_HOST_AREA_RATIO`). Two faces or a large face →
  existing layouts, unchanged.
- `has_pip_shot` added to the mode dispatch so a **single** PiP shot is not
  silently downgraded to `blur_stack` (real bug, caught by reasoning then fixed).

### Phase 3b — Director v2: the director finally gets authority over pixels ✅
New `src/director_v2.py`. This is the first point in the pipeline where the
director's opinion changes the rendered output rather than only vetoing it.
Ships **off by default** (`DIRECTOR_V2_ENABLED=false`).

Flow: sample keyframes → compose a labelled contact sheet → send sheet +
transcript + per-shot evidence + engagement scores to Gemini → parse a shot
decision list → execute it on the `FramingDecision` the renderer consumes.

- **Contact sheet** built with **OpenCV, not ffmpeg `tile`+`drawtext`.** The
  ffmpeg path needs a specific font file present on the runner, and a missing
  font silently drops every timestamp label. OpenCV always has a built-in
  Hershey font, and the sheet becomes testable without invoking ffmpeg. Frames
  come from the new `source_index.sample_frames_at()`, which reuses the index's
  existing decode path instead of adding a second ffmpeg invocation with
  different seeking behaviour. A legend (`r0c0=0.0s r0c1=8.1s ...`) is always
  supplied in the prompt text, so the sheet is usable even if labels blur.
- **Directives are matched to shots by time overlap, not by id.** The two brains
  segment shots independently — `face_tracker` drives the renderer,
  `SourceIndex` drives the plan — so their shot ids do not correspond. Time
  overlap is the only reliable join key. Requires ≥0.5 s overlap or it is ignored.
- **The model is treated as untrusted input.** Per-field validation, not
  all-or-nothing, so one stray enum does not cost the whole plan:
  unknown `layout`/`crop_target`/`motion`/`caption_anchor` fall back to safe
  defaults and the rest of the directive survives.
- **Executed:** `layout` (switch a shot's renderer branch), `keep` (drop a shot),
  `crop_target` (re-point the framing box), `caption_anchor` (via
  `apply_caption_directives` → `CompositionPlan` → ASS).
- **Executed: `motion` (unblocked in Phase 3d-2).** The per-frame `zoompan` path
  (`_punch_zoom_filter`) executes `push_in` and `pull_out` additive ramps. Fully verified with
  FFmpeg freezedetect (zero freeze events) and optical magnification proofs in
  `tests/test_punch_zoom.py`.
- `viral_detector.query_gemini_models` gained an optional `image_path`, attaching
  the sheet **in the same request** (not a second call) so free-tier quota is
  unaffected. Works on both the `google.genai` and legacy SDK paths.
- Every failure mode — disabled, no key, no contact sheet, API error, malformed
  reply, empty result — returns an empty review and leaves framing untouched.
- Decision log written to `clip_N_director_v2.json` plus the contact sheet JPEG,
  both recorded as run artifacts, so a human can audit what it chose.

**Guards, each mutation-tested to prove the test bites:**
| Guard | Mutation → failures |
|---|---|
| `pip_slide`/`split_screen` rejected without the required `FaceBox` | 2 |
| `"keep": "false"` string not treated as truthy (`bool("false") is True`) | 1 |
| Dropping *every* shot restores the first | 1 |

**End-to-end verified with a real render**, not just a graph string:
`parse → apply_shot_directives → render_viral_clip` produced a **1080×1920** clip
with a genuine PiP chain and **zero freeze events**; the input `FramingDecision`
was confirmed unmutated (the repair loop re-renders, so a mutated input would
apply directives twice).

### Phase 1b — text-region detection, and a QA check that can finally fail ✅
New `src/text_detection.py`. `caption_source_collision` is a **blocking** QA code
but nothing ever populated real `protected_regions` — `source_index` used a
Sobel edge heuristic, so the check could not fire. Now it can.

- **Tiered detection**: `auto | ocr | edge` (`TEXT_DETECTOR`). `auto` uses OCR
  when the package imports, otherwise the edge heuristic. `edge` pins the
  original behaviour for A/B. `ocr` asks explicitly and says so on stderr if it
  cannot load, so a broken setup never degrades silently.
- **RapidOCR adapter** (`requirements-ocr.txt`, deliberately *not* in
  `requirements.txt` — onnxruntime + weights is ~284 MB and disk is the binding
  constraint, and it is unverified end to end). Every failure path — missing
  package, engine that will not construct, engine that raises mid-inference,
  a response whose shape changed between releases — resolves to "no regions",
  never an exception.
- **Cross-frame IoU merging.** The index used to de-duplicate text regions on
  *exact* coordinates. That happened to work for the edge heuristic's stable
  boxes, but a real detector jitters per frame, so it would have produced ~12
  near-identical protected rects per line and made the collision check fire on
  everything. Now merged by overlap across the whole shot.
- **`editorial_qa` fixed**: it tested only `protected_regions[0]`, so a collision
  on any *later* region passed QA — the one case the check exists to catch. Now
  every region is tested and the message reports how many overlapped.

#### A zero-dependency MSER tier was built, measured, and removed
Kept documented in the `src/text_detection.py` docstring because a negative
result is worth not repeating:
- MSER returns **0 regions** on unblurred rendered text. A Gaussian sigma of 3
  fixes it (~100 regions/frame).
- But those cover only ~20-26 regions per ~35-character line. MSER **silently
  misses glyphs**, leaving fragments 34-132 px apart. No gap threshold bridges
  that without merging whole text blocks. Measured: "Most people misunderstand
  this completely" came back as 2-3 fragments, and one of three lines was lost.
- On a soft lower-third over a busy background: **0 regions** — exactly the case
  the edge heuristic already handles.
- Loosening stability params yields ~1072 regions/frame, which floods the
  protected-region list. Worse than detecting nothing.
- Also note: the OpenCV method is `detectRegions` (camelCase) while the
  constructor takes `min_area`/`max_variation` (snake_case), and neither is in
  the type stubs, so a wrong guess fails only at runtime.

### Phase 1c — TransNetV2 shot detection & timeline partitioning ✅
New `src/shot_detection.py`. Delegated cleanly from `scene_classifier.detect_clip_shots`.
- **Hybrid fallback**: TransNetV2 (ONNX Runtime / OpenCV DNN) when weights exist at `src/models/transnetv2.onnx`; falls back to PySceneDetect `AdaptiveDetector` with zero behavior change if weights are absent.
- **Shared normalisation**: `normalise_shots()` guarantees strict timeline partitioning, clamps to requested clip window, and folds sub-minimum (<0.4s) shots into predecessors so no gaps or holes are left in the filtergraph.
- Unit tested across 11 test cases in `tests/test_shot_detection.py`.

### Phase 3d-2 — Animated punch-zoom & Director motion unblocked ✅
- `_punch_zoom_filter()` in `src/video_editor.py` implements additive FFmpeg `zoompan` ramps (`push_in` / `pull_out`) layered on top of the continuous triangle-wave camera drift.
- Unblocks the Director's `motion` parameter in `src/director_v2.py` and `ShotPlan.motion`.
- Verified end-to-end with real FFmpeg rendering: zero `freezedetect` events, plus optical verification of outward centroid magnification over time (`tests/test_punch_zoom.py`).

### Diagnostics improvements ✅
- QA log now separates blocking from warning codes and prints the message +
  timestamp for each blocking issue.
- `MIN_CLIP_DURATION` / `MAX_CLIP_DURATION` / `SHORT_FORM_MAX_DURATION`
  configurable via env. **Defaults unchanged (30/140)** on purpose: tightening to
  60 s risks the "no clips were rendered" failure. Deviation is surfaced as a
  plan warning via `edit_director.duration_warnings()` instead.

---

## 4. Bugs found in *my own* new code (all caught by tests, not assumed)

Recorded because the same mistakes are easy to repeat:

1. `engagement.py` used `_WORD_RE.sub("", word)` — that **deletes** the word
   instead of extracting it, so every sample returned `no_speech`. Needed a
   separate `_NOISE_RE = re.compile(r"[^a-z']+")`.
2. `CaptionPlacement` was missing its `@dataclass` decorator.
3. The regression test imported new private symbols at module level, so on
   pre-fix code it **errored during collection** instead of failing an
   assertion. Fixed with a lazy import + `setUpClass` skip, which makes the
   behavioural test genuinely run and fail.
4. `WindowCandidate.combined_score` had a `base_score <= 0` short-circuit that
   made a 0.65 lose to a 1.0. Removed in favour of an unconditional blend.
5. `_force_collision_avoidance` hard-coded the anchor to `upper_center` even when
   falling back to the centre band — the reported anchor could lie about the
   geometry actually used. Now returns `(anchor, rect)`.
6. **Structural mistake:** inserting a new function *inside* `face_tracker` split
   `is_valid_two_speaker_frame`'s signature from its body. I made it worse with
   two piecemeal patches before reading the whole damaged region and repairing it
   in one edit. **Lesson: never insert into a function without reading it whole.**
7. Several test-data mistakes of my own: a 40 s window asserted as "too short";
   a 13-char fake Gemini response rejected by a real `len(raw) > 20` guard; a
   500×500 inset asserted inside a 12 %-of-frame budget when it is not;
   `assertEqual` where `assertNotEqual` was meant; searching for lowercase words
   in an output the hormozi theme uppercases.
8. **My own scratch-parser bug:** a freezdetect parser that only recorded events
   with a closing `freeze_duration` silently missed freeze-**to-EOF** cases,
   making a fully frozen clip look like a pass. Any future freezdetect work must
   handle the unterminated case.
9. **A name typo I misdiagnosed as a homoglyph.** I defined
   `plan_shot_directions` but imported `plan_shot_directives`. `ImportError:
   cannot import name`. I then searched the file for the *wrong* word, found
   zero matches, and wrongly concluded a Unicode homoglyph was corrupting the
   identifier. **Lesson: when a symbol "isn't in the file" but grep sees it,
   check that your search string actually matches the definition before
   theorising.** `dir(module)` is the ground truth, not a hand-typed pattern.
10. **A silent no-op mutation.** A `Set-Content` regex "mutation" of the
    string-bool guard did not match, so the test suite reported 55 passed and I
   nearly recorded that guard as untested. Mutations must be applied by a script
   that **asserts its anchor was found and exits non-zero** otherwise, or they
   prove nothing.
11. Eight separate test-data mistakes in `test_director_v2.py` on first run:
    `TranscriptSegment` requires `words=`; `parse_shot_directives` returns a
    tuple, so `x[0][0]` unpacked a `ShotDirective`; `sample_frames_at` passes
    `sample_fps` as a **kwarg** not a positional; a `CONTACT_TILE_WIDTH`
    assertion against a renderer that actually uses `PIP_INSET_WIDTH`; a
    tile-count cap that protects the *request* not the mocked *response*; and a
    `subprocess` list element left as an f-string with no placeholders.
    All eight were test bugs — the module was correct — but each one would have
    looked like a product bug from the failure message alone.
12. **A silent zero-result in the OCR adapter.** RapidOCR entries are
    `(box, text, confidence)`. I read `entry[1]` — the recognised *text* — as the
    confidence. `float("MOST PEOPLE")` raised, the handler caught it, and the box
    was dropped. Net effect: a perfectly good response would have produced **zero
    text regions with nothing logged anywhere**, and Phase 1b would have looked
    like it worked while detecting nothing. Found only because a test asserted
    a box count instead of trusting the adapter. Pinned by
    `test_confidence_comes_from_the_third_element` and
    `test_text_that_looks_numeric_does_not_become_confidence`.
13. **A coverage gap that mutation testing found, not review.** My
    cross-frame-merge test exercised `_merge_shot_text_regions` *directly*, so
    replacing the merge *call site* in `build_source_index` with `list(shot_text)`
    left the entire suite green while the anti-flooding guarantee was gone.
    Replaced with an integration test through the real entry point on a real
    encoded video. **Lesson: a unit test on a helper is not coverage of its call
    site.** Mutate the wiring, not just the logic.
14. **I deleted three source files with a bad PowerShell backup loop.**
    `Copy-Item $files -Destination .` flattened relative paths into the repo
    root, then `Remove-Item $files` deleted the originals under `src/`. Recovered
    from the stray root copies, which happened to be the pre-mutation versions —
    pure luck, not design. Use one explicit `.bak` per file, and never pass a
    relative path array to `Copy-Item -Destination .`.

---

## 5. Verification commands (run these; do not guess)

```bash
cd /d/Workspace/PODCASTS-CLIPS-TO-SOCIAL      # PowerShell
python -m pytest tests -q                     # expect: 368 passed
python -m ruff check src main.py tests --output-format=concise
python -m mypy src                            # expect: Success, 29 files
```

**How to prove a new test actually tests something:** mutate the guard and
confirm a failure. A test that passes against deliberately broken code is worse
than no test, because it reports safety that does not exist. Apply mutations with
a script that asserts its anchor was found, and check the *number* of failures
matches the number of guards you think you have. **Mutate the wiring too** — a
unit test on a helper does not cover its call site (see bug 13).

**CI state at time of writing:** `UNIVERSAL_EDITOR_SHADOW=true`,
`UNIVERSAL_EDITOR_ENFORCE_QA=true`, `ENABLE_FILM_GRAIN=true` — so QA **can**
veto a clip in CI. Baseline before this work was 134 tests.

**How the freeze fix was proven** (reproduce with any static source):
1. Build a 1920×1080 source holding one still frame for 3 s.
2. `build_video_filtergraph` with a `portrait_face` `ShotPlan` whose
   `face_centers_timeline=[]`.
3. Render, then `ffmpeg -i out.mp4 -vf freezedetect=n=0.003:d=0.5 -an -f null NUL`.
4. **Expect no `freeze_start` / `freeze_duration` lines at all.**

Note: freezedetect is the authority. A hand-rolled "mean abs diff" proxy does
**not** track it reliably and will produce false conclusions.

---

## 6. Remaining work

### Recommended next: Phase 2 — identity layer
The biggest capability gap: **the pipeline has no idea who is speaking.**
`face_count` is a count, not an identity; there is no diarization anywhere. Plan:
`pyannote/speaker-diarization-community-1` (v4 `exclusive_speaker_diarization` is built
for STT alignment) **+** MediaPipe FaceLandmarker blendshapes for mouth-motion, correlated
to the diarized speaker. **Needs PyTorch — gate on measured disk usage first.**
Also needs `HF_TOKEN` (gated model); `.env.example` has none today.

### Then, in order
- **Enable OCR on CI.** Install `requirements-ocr.txt` and confirm the log line
  `[+] OCR text detection active (RapidOCR ONNX).` appears. Until then the edge
  heuristic is still in charge, and `caption_source_collision` remains blunter
  than it should be. Watch for **more** collisions firing, not fewer — that
  means the check has started working, and the repair loop's
  `caption_source_collision` relocation will start doing real work.
- **Director v2 rollout.** It is off by default (`DIRECTOR_V2_ENABLED=false`) and has
  never seen a real Gemini response. Before enabling in CI, review a few
  The main risk is not a crash (every path degrades to no-op) but a confidently wrong layout.

### Deliberately NOT recommended
- Video LLMs (VideoChat3, STORM, Kimi K3, GLM 5.x, MiniMax M3) — all need
  GPU-class compute. Their published numbers are GPU numbers.
- Switching ASR to NVIDIA Parakeet TDT v3. Evidence is **conflicted**: the HF
  Open ASR Leaderboard says 6.34 % vs Whisper large-v3's 7.44 % WER, but a
  hands-on eval found **1.63 % → 6.62 % WER regression** on jargon-heavy
  English, mangling exactly this niche's terms (*Sonnet→"Sonic"*, *YOLO→"yellow"*).
  Benchmark on real clips first. If ASR is upgraded at all, `distil-large-v3` or
  `large-v3-turbo` is the low-risk move (the pipeline currently uses **`tiny.en`**,
  which is the accuracy floor of the caption system).
- Gemini 3.1 Pro — breaks the $0 constraint.
- Loosening QA thresholds — this is exactly what produced the failure streak.
- Rewriting `face_tracker`'s existing layouts — they work.

---

## 7. Engineering rules for this repo (learned the hard way)

1. **Reproduce, then fix.** Every bug in this handoff has a reproduction.
2. **The log message is not the bug.** `RuntimeError: No clips were rendered`
   was a guardrail, not the cause.
3. **QA codes have severities.** Only `error`/`fatal` block. Check
   `QaIssue.blocks_publish` before concluding anything from a "blocked" line.
4. **Never soften a QA gate to make a test pass.** Fix the renderer.
5. **Guard against both parities of every edit.** Crop/overlay need even
   dimensions; boxes need integer positions; ffmpeg expression arguments need
   quoting, and `crop` rejects per-frame `w`/`h` expressions.
6. **Render, don't just inspect, when a change touches a filtergraph.** Reading
   the graph would not have caught the zero-slack frozen PiP background.
7. **f-string braces in FFmpeg graph strings need doubling** when nested inside
   f-strings.
8. **Minimal change, then verify the rest.** Baseline was 134 green tests; it
   must stay green.

---

## 8. Open questions / not verified

- **OCR has never executed.** `requirements-ocr.txt` is written and the adapter
  is fully unit-tested against mocked engines, but RapidOCR has never been
  installed or run. Two specifics are unproven: the exact return shape of
  `RapidOCR.__call__` for the pinned version, and the cost of loading the model
  once per process. Confirm the tier engaged (the log line, or
  `get_text_detector(force='ocr').name`) before trusting any output.
- **`ffmpeg drawtext` is not portable.** It needs fontconfig, which is absent on
  this machine, so a fixture built with it fails at fixture time, not test time.
  Prefer OpenCV for anything that draws. This is the same reason the director's
  contact sheet is OpenCV.
- **Director v2 has never seen a real Gemini response.** Everything in
  `src/director_v2.py` is verified against mocked replies and synthetic video.
  The prompt's actual quality is **UNVERIFIED**. The image attachment path
  (`google.genai` `Part.from_bytes` and the legacy dict form) is likewise
  unexercised at runtime.
- **No CPU runtime was measured** for any 2026 model. Everything in the research
  report is an estimate. Benchmark on a real runner before committing to any
  new dependency.
- The runner **pricing** page lists "Linux 2-core" while the **runners** page
  says `ubuntu-latest` = 4 CPU. Possible 2026 restructuring; verify before
  relying on either.
- `pyannote` Community-1 is **gated** on Hugging Face: needs a token + accepted
  terms. CI currently has no `HF_TOKEN` (the log even warns about unauthenticated
  HF Hub requests).
- A 96 s clip reached the renderer despite the documented 30–60 s window. The
  code's 140 s cap is why. Left unchanged deliberately — see §3.
- `opencv-python-headless` is pinned `<5.0.0.0` in `requirements.txt` but CI
  installs OpenCV 5.0.0.93 transitively. Harmless, but worth tidying.
